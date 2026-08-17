# PyTorch
from torch import nn

# Custom
from networks import modules


# パッチを 2 次元配置のまま、同じ行・同じ列だけを見る attention で混ぜ、空間平均を線形読み出しに渡す
# NOTE: CCNet (Huang et al., ICCV 2019) の criss-cross attention を固定ランダム重みにしたもの。
#       位置 (i, j) が参照するのは行 i と列 j の N_h + N_w - 1 箇所だけで、受容野は厳密に十字になる。
#       これは BiReservoir2D 1 層の受容野と同じ形なので、
#       「十字の混合を再帰結合で作る (bi_esn2d) か、内容ベースの重み付けで作る (これ) か」の対照になる。
#       2 層で十字 x 十字 = 窓全体に届く点も BiReservoir2D と同じ
#
# NOTE: 重みは学習せず seed だけで決まる (読み出しだけをリッジ回帰で解く条件を他の classifier と揃えるため)。
#       したがって Q/K は「学習された類似度」ではなくランダム射影上の類似度である
#
# NOTE: temperature の既定値 0.3 は MNIST 10,000 枚・units=512・1 層で掃引して決めた
#       (val acc: T=0.02 -> 0.881, 0.3 -> 0.908, 1.0 -> 0.909, 10 -> 0.880, 100 -> 0.862)。
#       Q/K が未学習なので注意を鋭くしても「ランダムなパッチを選ぶ」だけになり、
#       リザバー系の spectral_radius と違って小さい温度は効かない。0.3-1.0 の平坦な領域が最良
#
# NOTE: units などのハイパーパラメータはリストで層ごとに指定できる (modules.per_layer_hparams 参照)。
#       スカラーで渡した場合は全層で同じ値になる
class CrissCrossAttentionClassifier(modules.Classifier):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        patch_sizes: tuple[int, int] = (4, 4),
        units: int | list[int] = 256,
        n_head: int | list[int] = 4,
        temperature: float | list[float] = 0.3,
        activations: str | list[str] = "tanh",
        pos_encoding: str = "sincos",
        pos_dim: int = 16,
        n_layer: int | None = None,
        seed: int | list[int] = 0,
    ):
        layers = modules.per_layer_hparams(
            n_layer,
            units=units,
            n_head=n_head,
            temperature=temperature,
            activations=activations,
        )

        # 特徴量の次元は最終層の units になる
        super().__init__(feature_dim=layers[-1]["units"], num_classes=num_classes)

        C, H, W = input_shape
        Hp, Wp = patch_sizes
        if H % Hp != 0 or W % Wp != 0:
            raise ValueError(f"input_shape {input_shape} must be divisible by patch_sizes {patch_sizes}.")

        self.patchify = modules.Patchify(patch_sizes)

        input_dim = Hp * Wp * C

        # NOTE: 既定は "sincos" (理由は self_attention の NOTE を参照)。マスクが行・列という位置構造を
        #       持つので self_attention ほど極端ではないが、"none" では行をまるごと入れ替える /
        #       列をまるごと入れ替える操作に対して実測でも不変 (差 1e-7) になる
        #
        # NOTE: MNIST 10,000 枚・units=512・1 層の test acc は none 0.647 -> sincos 0.884。
        #       "sincos" 側が bi_esn2d の 0.883 とほぼ同着になる点がこのモデルを置いた理由で、
        #       受容野が同じ十字なら再帰結合で作っても内容ベースの重み付けで作っても同程度、と読める。
        #       "none" のままだと差の大部分が受容野ではなく位置盲によるものになり、この比較が成立しない
        self.pos_encoding = self._build_pos_encoding(pos_encoding, H // Hp, W // Wp, pos_dim)
        if pos_encoding == "sincos":
            input_dim += pos_dim

        # 1 層あたりの重み (埋め込みと Q/K/V/O) を 1 つの Generator から引くので、seed は 1 つずつ消費する
        seeds = modules.per_layer_seeds(seed, [1] * len(layers))

        # 十字以外を塞ぐマスク。self_attention との違いはこれだけになる
        attn_mask = modules.criss_cross_mask(H // Hp, W // Wp)

        attentions = []
        for hp, layer_seed in zip(layers, seeds):
            attention = modules.FixedMultiheadAttention2D(
                input_dim, hp["units"], hp["n_head"], hp["temperature"], layer_seed, attn_mask
            )

            # NOTE: attention は softmax の分だけ入力に対して非線形だが、V と O の経路は線形のままなので、
            #       活性化を挟まないと層を重ねたときの表現力の伸びが鈍る
            attentions += [attention, modules.get_activation(hp["activations"])]

            input_dim = hp["units"]

        self.attentions = nn.Sequential(*attentions)

        # 固定重み (非訓練) として扱う
        self.attentions.requires_grad_(False)

    @staticmethod
    def _build_pos_encoding(pos_encoding: str, N_h: int, N_w: int, pos_dim: int) -> nn.Module:
        if pos_encoding == "none":
            return nn.Identity()
        if pos_encoding == "sincos":
            return modules.ConcatSinCosPositions2D(N_h, N_w, pos_dim)

        raise ValueError(f"pos_encoding must be 'sincos' or 'none', got {pos_encoding!r}.")

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        x = self.patchify(images)  # (B, N_h, N_w, D)
        x = self.pos_encoding(x)  # (B, N_h, N_w, D + pos_dim)
        x = self.attentions(x)  # (B, N_h, N_w, units)

        return x.mean(dim=(1, 2))  # (B, units)


def build(input_shape, num_classes, **kwargs):
    """networks.build_classifier("criss_cross_attention", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        num_classes: クラス数。
    """
    return CrissCrossAttentionClassifier(input_shape, num_classes, **kwargs)
