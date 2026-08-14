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
class Conv2DClassifier(modules.Classifier):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        filters: int | list[int] = 256,
        kernel_size: int | list[int] = 3,
        n_layer: int | None = None,
        activations: str | list[str] = "tanh",
        seed: int | list[int] = 0,
    ):
        layer_params = modules.per_layer_hparams(
            n_layer, filters=filters, kernel_size=kernel_size, activations=activations
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
            #       Reservoir と同じく Generator から Glorot 一様分布で初期化し直す
            g = torch.Generator().manual_seed(layer_seed)
            limit = (6.0 / ((in_channels + hp["filters"]) * hp["kernel_size"] ** 2)) ** 0.5
            with torch.no_grad():
                conv.weight.copy_((torch.rand(conv.weight.shape, generator=g) * 2 - 1) * limit)

            convs += [conv, modules.get_activation(hp["activations"])]

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
