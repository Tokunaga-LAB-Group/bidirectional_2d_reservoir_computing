# PyTorch
from torch import nn

# Custom
from networks import modules


# パッチ列を単方向 ESN で走査し、時間平均を線形読み出しに渡す
#
# NOTE: units などのハイパーパラメータはリストで層ごとに指定できる (modules.per_layer_hparams 参照)。
#       スカラーで渡した場合は全層で同じ値になる
class ESNClassifier(modules.Classifier):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        patch_sizes: tuple[int, int] = (4, 4),
        units: int | list[int] = 256,
        connectivity: float | list[float] = 0.1,
        leaky: float | list[float] = 0.9,
        spectral_radius: float | list[float] = 0.95,
        n_layer: int | None = None,
        seed: int | list[int] = 0,
    ):
        layer_params = modules.per_layer_hparams(
            n_layer,
            units=units,
            connectivity=connectivity,
            leaky=leaky,
            spectral_radius=spectral_radius,
        )

        # 特徴量の次元は最終層の units になる
        super().__init__(feature_dim=layer_params[-1]["units"], num_classes=num_classes)

        C, H, W = input_shape
        Hp, Wp = patch_sizes
        if H % Hp != 0 or W % Wp != 0:
            raise ValueError(f"input_shape {input_shape} must be divisible by patch_sizes {patch_sizes}.")

        self.patchify = modules.Patchify(patch_sizes)

        # Reservoir は seed を 1 つ消費する
        seeds = modules.per_layer_seeds(seed, [1] * len(layer_params))

        input_dim = Hp * Wp * C
        reservoirs = []
        for hp, layer_seed in zip(layer_params, seeds):
            reservoirs.append(
                modules.Reservoir(
                    input_dim, hp["units"], hp["connectivity"], hp["leaky"], hp["spectral_radius"], layer_seed
                )
            )
            input_dim = hp["units"]

        self.reservoirs = nn.Sequential(*reservoirs)

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        x = self.patchify(images)  # (B, N_h, N_w, D)
        B, N_h, N_w, D = x.shape

        x = x.reshape(B, N_h * N_w, D)  # (B, T, D)
        x = self.reservoirs(x)  # (B, T, units)

        return x.mean(dim=1)  # (B, units)


def build(input_shape, num_classes, **kwargs):
    """networks.build_classifier("esn", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        num_classes: クラス数。
    """
    return ESNClassifier(input_shape, num_classes, **kwargs)
