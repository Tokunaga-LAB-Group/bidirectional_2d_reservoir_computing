# PyTorch
import torch
from torch import nn

# Custom
from networks import modules


# 画像全体を 1 本のベクトルに潰し、固定重みの全結合層で非線形変換して線形読み出しに渡す
# NOTE: positionwise_fcl と対になる比較対象。あちらは各位置に同じ重みを当てて空間平均するため
#       位置情報を使えないが、こちらは位置ごとに別の重みが当たるため空間配置を使える。
#       走査 (再帰結合) は持たないので、bi_esn2d との差が「走査による相互作用」の寄与になる
#
# NOTE: 「固定ランダム射影 + 最小二乗で解く線形読み出し」という構造そのもので、
#       Extreme Learning Machine (ELM) に相当する
#
# NOTE: パッチ化はしない。全要素を 1 本に潰す以上、パッチの切り方は入力次元の並び順を変えるだけで、
#       重みが i.i.d. なランダム値である限り分布は変わらないため
#
# NOTE: units などのハイパーパラメータはリストで層ごとに指定できる (modules.per_layer_hparams 参照)。
#       スカラーで渡した場合は全層で同じ値になる
# NOTE: 固定重みは W ~ U(-input_scaling, +input_scaling) で引き、スケールのみを探索対象にする。
#       Glorot の limit = sqrt(6/(fan_in+fan_out)) は fan_out が支配するため、fan_in が小さいと
#       W u (fan_in 項の和) が sqrt(fan_in) に比例して弱まり、tanh が線形領域に入って特徴が縮退する。
#       畳み込みは fan_in = C*K^2 が 9-27 と小さく、実測で MNIST 49.9% -> 86.4% の差が出た。
#       分布の形 (一様・平均 0・独立同分布) は変えず、スケールだけを 1 個の自由度として残す
class FCLClassifier(modules.Classifier):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        units: int | list[int] = 256,
        input_scaling: float | list[float] = 1.0,
        activations: str | list[str] = "tanh",
        n_layer: int | None = None,
        seed: int | list[int] = 0,
    ):
        layers = modules.per_layer_hparams(
            n_layer,
            units=units,
            input_scaling=input_scaling,
            activations=activations,
        )

        # 特徴量の次元は最終層の units になる
        super().__init__(feature_dim=layers[-1]["units"], num_classes=num_classes)

        C, H, W = input_shape

        # nn.Linear は seed を 1 つ消費する
        seeds = modules.per_layer_seeds(seed, [1] * len(layers))

        # 1 層目の入力は画像の全要素
        input_dim = C * H * W
        fcls = []
        for hp, layer_seed in zip(layers, seeds):
            fcl = nn.Linear(input_dim, hp["units"], bias=False)

            # NOTE: nn.Linear の既定の初期化はグローバル RNG に依存するため、seed 引数だけでは再現しない。
            #       Reservoir と同じく Generator から Glorot 一様分布で初期化し直す
            g = torch.Generator().manual_seed(layer_seed)
            with torch.no_grad():
                fcl.weight.copy_((torch.rand(fcl.weight.shape, generator=g) * 2 - 1) * hp["input_scaling"])

            # NOTE: 活性化を入れないと層を重ねても線形写像のままで、表現力が増えない
            fcls += [fcl, modules.get_activation(hp["activations"])]

            input_dim = hp["units"]

        self.fcls = nn.Sequential(*fcls)

        # 固定重み (非訓練) として扱う
        self.fcls.requires_grad_(False)

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        return self.fcls(images.flatten(1))  # (B, units)


def build(input_shape, num_classes, **kwargs):
    """networks.build_classifier("fcl", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        num_classes: クラス数。
    """
    return FCLClassifier(input_shape, num_classes, **kwargs)
