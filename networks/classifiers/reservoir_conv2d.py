# PyTorch
from torch import nn

# Custom
from networks import modules


# K x K の窓を縦横のリザバーで走査して特徴マップを作り、空間平均を線形読み出しに渡す
#
# NOTE: units などのハイパーパラメータはリストで層ごとに指定できる (modules.per_layer_hparams 参照)。
#       スカラーで渡した場合は全層で同じ値になる
class ReservoirConv2DClassifier(modules.Classifier):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        num_reservoirs: int | list[int] = 5,
        units: int | list[int] = 12,
        kernel_size: int | list[int] = 3,
        stride: int | list[int] = 1,
        padding: int | tuple[int, int] | str | list = 0,
        connectivity: float | list[float] = 0.5,
        spectral_radius: float | list[float] = 0.95,
        n_layer: int | None = None,
        seed: int | list[int] = 0,
    ):
        layers = modules.per_layer_hparams(
            n_layer,
            num_reservoirs=num_reservoirs,
            units=units,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            connectivity=connectivity,
            spectral_radius=spectral_radius,
        )

        # 特徴量の次元は最終層の 2 * num_reservoirs * units になる。式を二重に持たないよう、
        # レイヤを先に作って output_dim を読む
        C, _, _ = input_shape

        # ReservoirConv2D は縦横それぞれ num_reservoirs 個の Reservoir を持つため、
        # 1 層あたり 2 * num_reservoirs 個の seed を消費する
        seeds = modules.per_layer_seeds(seed, [2 * hp["num_reservoirs"] for hp in layers])

        in_channels = C
        reservoir_conv2ds = []
        for hp, layer_seed in zip(layers, seeds):
            layer = modules.ReservoirConv2D(
                in_channels,
                hp["num_reservoirs"],
                hp["units"],
                hp["kernel_size"],
                hp["stride"],
                hp["padding"],
                hp["connectivity"],
                hp["spectral_radius"],
                layer_seed,
            )
            reservoir_conv2ds.append(layer)
            in_channels = layer.output_dim

        super().__init__(feature_dim=in_channels, num_classes=num_classes)

        self.reservoir_conv2ds = nn.Sequential(*reservoir_conv2ds)

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        # ReservoirConv2D は channel-last を扱う
        x = images.permute(0, 2, 3, 1).contiguous()
        x = self.reservoir_conv2ds(x)  # (B, H_out, W_out, output_dim)

        return x.mean(dim=(1, 2))  # (B, output_dim)


def build(input_shape, num_classes, **kwargs):
    """networks.build_classifier("reservoir_conv2d", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        num_classes: クラス数。
    """
    return ReservoirConv2DClassifier(input_shape, num_classes, **kwargs)
