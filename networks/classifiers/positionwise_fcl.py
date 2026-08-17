# PyTorch
import torch
from torch import nn

# Custom
from networks import modules


# パッチごとに固定重みの全結合層で非線形変換し、空間平均を線形読み出しに渡す
# NOTE: bi_esn2d から「行・列方向の再帰結合」だけを取り除いた比較対象。全結合の重みは学習せず、
#       seed だけで決まるようにする (読み出しだけをリッジ回帰で解く、という条件を同じにするため)
#
# NOTE: nn.Linear は最終次元にのみ作用するため、各位置のパッチに同じ重みが独立に当たる
#       (Transformer の position-wise FFN と同じ構造)。その後に空間平均を取るので、
#       features() はパッチの並び替えに対して不変になる = 位置情報を一切使わない
#       「bag of patches」モデルである。fcl (画像全体を 1 本に潰す版) と対になる
#
# NOTE: units などのハイパーパラメータはリストで層ごとに指定できる (modules.per_layer_hparams 参照)。
#       スカラーで渡した場合は全層で同じ値になる
class PositionWiseFCLClassifier(modules.Classifier):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        patch_sizes: tuple[int, int] = (4, 4),
        units: int | list[int] = 256,
        activations: str | list[str] = "tanh",
        n_layer: int | None = None,
        seed: int | list[int] = 0,
    ):
        layers = modules.per_layer_hparams(
            n_layer,
            units=units,
            activations=activations,
        )

        # 特徴量の次元は最終層の units になる
        super().__init__(feature_dim=layers[-1]["units"], num_classes=num_classes)

        C, H, W = input_shape
        Hp, Wp = patch_sizes
        if H % Hp != 0 or W % Wp != 0:
            raise ValueError(f"input_shape {input_shape} must be divisible by patch_sizes {patch_sizes}.")

        self.patchify = modules.Patchify(patch_sizes)

        # nn.Linear は seed を 1 つ消費する
        seeds = modules.per_layer_seeds(seed, [1] * len(layers))

        input_dim = Hp * Wp * C
        fcls = []
        for hp, layer_seed in zip(layers, seeds):
            fcl = nn.Linear(input_dim, hp["units"], bias=False)

            # NOTE: nn.Linear の既定の初期化はグローバル RNG に依存するため、seed 引数だけでは再現しない。
            #       Reservoir と同じく Generator から Glorot 一様分布で初期化し直す
            g = torch.Generator().manual_seed(layer_seed)
            limit = (6.0 / (input_dim + hp["units"])) ** 0.5
            with torch.no_grad():
                fcl.weight.copy_((torch.rand(fcl.weight.shape, generator=g) * 2 - 1) * limit)

            # NOTE: 活性化を入れないと「全結合 -> 空間平均 -> 線形読み出し」が全体で線形写像に潰れ、
            #       さらに空間平均と可換になるため「平均パッチに重みを掛けただけ」まで退化する
            fcls += [fcl, modules.get_activation(hp["activations"])]

            input_dim = hp["units"]

        self.fcls = nn.Sequential(*fcls)

        # 固定重み (非訓練) として扱う
        self.fcls.requires_grad_(False)

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        x = self.patchify(images)  # (B, N_h, N_w, D)
        x = self.fcls(x)  # (B, N_h, N_w, units)

        return x.mean(dim=(1, 2))  # (B, units)


def build(input_shape, num_classes, **kwargs):
    """networks.build_classifier("positionwise_fcl", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        num_classes: クラス数。
    """
    return PositionWiseFCLClassifier(input_shape, num_classes, **kwargs)
