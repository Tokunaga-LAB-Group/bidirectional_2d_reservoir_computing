# PyTorch
from torch import nn

# Custom
from networks import modules


# 全パッチ間の scaled dot-product self-attention で混ぜ、空間平均を線形読み出しに渡す
# NOTE: 1 層目から全パッチが相互作用するので、受容野は最初から画像全体になる。
#       esn / bi_esn が系列走査で 1 層目から画像全体を見るのと同じ立ち位置だが、
#       あちらは走査順に沿った減衰を持つのに対し、こちらは距離に関係なく全対全で見る。
#       criss_cross_attention は受容野を十字に絞った条件だが、そちらは CCNet 原典の構成
#       (Q/K のボトルネック、スケーリング無し、出力射影無し) を写しているため、
#       両者の差は受容野だけではない
#
# NOTE: 重みは学習せず seed だけで決まる (読み出しだけをリッジ回帰で解く条件を他の classifier と揃えるため)。
#       したがって Q/K は「学習された類似度」ではなくランダム射影上の類似度である
#
# NOTE: nn.MultiheadAttention を固定重みにしただけの構成にしてある。softmax の温度は
#       1/sqrt(d_head) が実装に内蔵されていて次元から自動的に決まるので、独立の
#       ハイパーパラメータとしては持たない。既定のヘッド数も原典の criss-cross attention に
#       合わせて 1 とする
#
# NOTE: units などのハイパーパラメータはリストで層ごとに指定できる (modules.per_layer_hparams 参照)。
#       スカラーで渡した場合は全層で同じ値になる
class SelfAttentionClassifier(modules.Classifier):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        patch_sizes: tuple[int, int] = (4, 4),
        units: int | list[int] = 256,
        n_head: int | list[int] = 1,
        input_scaling: float | list[float] = 1.0,
        activations: str | list[str] = "tanh",
        pos_encoding: str = "sincos",
        pos_scale: float = 1.0,
        n_layer: int | None = None,
        seed: int | list[int] = 0,
    ):
        layers = modules.per_layer_hparams(
            n_layer,
            units=units,
            n_head=n_head,
            input_scaling=input_scaling,
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

        # NOTE: 他の classifier に位置符号が無いのは、機構自体が位置を見ているので不要なためである
        #       (fcl は位置ごとに別の重み、esn 系は走査順、conv 系は局所窓)。attention だけは
        #       機構的に置換不変なので、位置符号を入れないと attention だけが位置盲になる。
        #       実測でも "none" の features() は行・列・全パッチのどの並び替えに対しても差が
        #       1e-7 (float 誤差) で、意図的な bag of patches 対照である positionwise_fcl と
        #       同じ対称性クラスに落ちる。そのため既定は "sincos" にしてある
        #
        # NOTE: 位置符号は画像に依存しない定数なので入力の情報量は増やさない。効果は不変性を壊すことだけで、
        #       それだけで MNIST 10,000 枚・units=512・1 層の test acc は none 0.539 -> sincos 0.810 になる
        #       (criss_cross は 0.647 -> 0.884、参考 bi_esn2d 0.883、positionwise_fcl 0.589)。
        #       MNIST のクラス情報がパッチの中身ではなく配置にあるためで、パッチ配置をシャッフルすると
        #       bi_esn2d は -45.6 pt 落ちるのに対し "none" の attention は 0.0 pt しか動かない
        #       (= 最初から配置を使っていない)。"none" は置換不変な対照条件として使う

        # 1 層あたりの重み (埋め込みと Q/K/V/O) を 1 つの Generator から引くので、seed は 1 つずつ消費する
        seeds = modules.per_layer_seeds(seed, [1] * len(layers))

        attentions = []
        for hp, layer_seed in zip(layers, seeds):
            pos = self._build_pos(pos_encoding, H // Hp, W // Wp, hp["units"], pos_scale)
            # attn_mask を渡さない = 全パッチが相互に見える通常の self-attention
            attention = modules.FixedMultiheadAttention2D(
                input_dim, hp["units"], hp["n_head"], hp["input_scaling"], layer_seed, None, pos
            )

            # NOTE: attention は softmax の分だけ入力に対して非線形だが、V と O の経路は線形のままなので、
            #       活性化を挟まないと層を重ねたときの表現力の伸びが鈍る
            attentions += [attention, modules.get_activation(hp["activations"])]

            input_dim = hp["units"]

        self.attentions = nn.Sequential(*attentions)

        # 固定重み (非訓練) として扱う
        self.attentions.requires_grad_(False)

    @staticmethod
    def _build_pos(pos_encoding: str, N_h: int, N_w: int, units: int, pos_scale: float) -> nn.Module:
        """埋め込み後に加算する位置符号を作る。'none' は置換不変な対照条件。"""
        if pos_encoding == "none":
            return nn.Identity()
        if pos_encoding == "sincos":
            return modules.AddSinCosPositions2D(N_h, N_w, units, pos_scale)

        raise ValueError(f"pos_encoding must be 'sincos' or 'none', got {pos_encoding!r}.")

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        x = self.patchify(images)  # (B, N_h, N_w, D)
        x = self.attentions(x)  # (B, N_h, N_w, units)

        return x.mean(dim=(1, 2))  # (B, units)


def build(input_shape, num_classes, **kwargs):
    """networks.build_classifier("self_attention", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        num_classes: クラス数。
    """
    return SelfAttentionClassifier(input_shape, num_classes, **kwargs)
