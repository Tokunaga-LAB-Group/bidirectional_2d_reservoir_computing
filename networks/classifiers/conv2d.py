# PyTorch
import torch
from torch import nn

# Custom
from networks import modules


# 固定重みのランダム畳み込みで特徴を作り、空間平均を線形読み出しに渡す
# NOTE: リザバー系の classifier と条件を揃えるための比較対象。畳み込みの重みは学習せず、
#       seed だけで決まるようにする (読み出しだけをリッジ回帰で解く、という条件を同じにするため)
#
# NOTE: filters などのハイパーパラメータはリストで層ごとに指定できる (modules.per_layer_hparams 参照)。
#       スカラーで渡した場合は全層で同じ値になる
# NOTE: pool_size は各層の直後に挟む max pooling の窓 (1 で無効)。層を積んだときに
#       受容野を広げつつ空間次元を落とす役割で、Tanaka & Tamukoh (NOLTA 2022) の原典が
#       conv -> pool -> conv -> pool の構成を取っているのに合わせて用意してある
# NOTE: 固定重みは W ~ U(-input_scaling, +input_scaling) で引き、スケールのみを探索対象にする。
#       Glorot の limit = sqrt(6/(fan_in+fan_out)) は fan_out が支配するため、fan_in が小さいと
#       W u (fan_in 項の和) が sqrt(fan_in) に比例して弱まり、tanh が線形領域に入って特徴が縮退する。
#       畳み込みは fan_in = C*K^2 が 9-27 と小さく、実測で MNIST 49.9% -> 86.4% の差が出た。
#       分布の形 (一様・平均 0・独立同分布) は変えず、スケールだけを 1 個の自由度として残す
class Conv2DClassifier(modules.Classifier):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        filters: int | list[int] = 256,
        kernel_size: int | list[int] = 3,
        input_scaling: float | list[float] = 1.0,
        pool_size: int | list[int] = 1,
        n_layer: int | None = None,
        activations: str | list[str] = "tanh",
        seed: int | list[int] = 0,
    ):
        layer_params = modules.per_layer_hparams(
            n_layer,
            filters=filters,
            kernel_size=kernel_size,
            input_scaling=input_scaling,
            pool_size=pool_size,
            activations=activations,
        )

        # 特徴量の次元は最終層の filters になる
        super().__init__(feature_dim=layer_params[-1]["filters"], num_classes=num_classes)

        C, _, _ = input_shape

        # 畳み込み 1 層あたり乱数の引き方は 1 通りなので、seed は 1 つずつ消費する
        seeds = modules.per_layer_seeds(seed, [1] * len(layer_params))

        in_channels = C
        convs = []
        for i, (hp, layer_seed) in enumerate(zip(layer_params, seeds)):
            conv = nn.Conv2d(in_channels, hp["filters"], kernel_size=hp["kernel_size"], padding="same", bias=False)

            # NOTE: nn.Conv2d の既定の初期化はグローバル RNG に依存するため、seed 引数だけでは再現しない。
            #       Generator から U(-input_scaling, +input_scaling) で引き直す
            g = torch.Generator().manual_seed(layer_seed)
            with torch.no_grad():
                conv.weight.copy_(
                    (torch.rand(conv.weight.shape, generator=g) * 2 - 1) * hp["input_scaling"]
                )

            convs += [conv, modules.get_activation(hp["activations"])]
            if int(hp["pool_size"]) > 1:
                convs.append(nn.MaxPool2d(int(hp["pool_size"])))

            in_channels = hp["filters"]

        self.conv2d = nn.Sequential(*convs)

        # 固定重み (非訓練) として扱う
        self.conv2d.requires_grad_(False)

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        x = self.conv2d(images)  # (B, filters, H, W)

        return x.mean(dim=(2, 3))  # (B, filters)


def build(input_shape, num_classes, **kwargs):
    """networks.build_classifier("conv2d", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        num_classes: クラス数。
    """

    return Conv2DClassifier(input_shape, num_classes, **kwargs)
