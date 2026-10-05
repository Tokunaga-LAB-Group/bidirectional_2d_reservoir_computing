# PyTorch
import torch
from torch import nn

# Custom
from networks import modules


# パッチ格子の上で、空間混合 (bi) とチャネル混合 (fc)・局所混合 (conv) を任意の順に積む
#
# NOTE: bi_esn2d は 1 層で行・列全体を走査するため受容野が最初から画像全体になり、
#       単純に積んでも新しい空間情報が入らない (実測では CIFAR-10 で L=1 -> L=4 が 50.81 -> 49.92 と劣化)。
#       Transformer が attention (token mixing) と FFN (channel mixing) を交互に置くのと同じ発想で、
#       間に位置ごとのチャネル混合を挟む構成を試すための classifier
#
# NOTE: blocks は "bi-fc-bi-fc" のようにハイフン区切りで指定する。
#         bi   : BiReservoir2D (行・列の双方向走査による空間混合)
#         fc   : 各位置に同一の固定ランダム射影 + 活性化 (チャネル混合)
#         conv : パッチ格子上の K x K 固定ランダム畳み込み + 活性化 (局所的な空間混合)
#       どのブロックも出力次元を units に揃えるので、feature_dim は構成によらず units になる
class PositionWiseFC(nn.Module):
    """各位置に同一の固定ランダム射影をかける。Transformer の FFN に相当する channel mixing。"""

    def __init__(self, input_dim: int, units: int, activation: str, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W", modules.glorot_uniform((input_dim, units), g))
        self.activation = modules.get_activation(activation)

    def forward(self, inputs):
        return self.activation(inputs @ self.W)


class GridConv(nn.Module):
    """パッチ格子上の K x K 固定ランダム畳み込み。局所的な空間混合。"""

    def __init__(self, input_dim: int, units: int, kernel_size: int, activation: str, seed: int):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, units, kernel_size, padding="same", bias=False)

        # NOTE: nn.Conv2d の既定初期化はグローバル RNG に依存するため、Generator から引き直す
        g = torch.Generator().manual_seed(seed)
        limit = (6.0 / ((input_dim + units) * kernel_size**2)) ** 0.5
        with torch.no_grad():
            self.conv.weight.copy_((torch.rand(self.conv.weight.shape, generator=g) * 2 - 1) * limit)
        self.conv.requires_grad_(False)

        self.activation = modules.get_activation(activation)

    def forward(self, inputs):
        # channel-last <-> channel-first を往復する
        x = self.conv(inputs.permute(0, 3, 1, 2))
        return self.activation(x.permute(0, 2, 3, 1).contiguous())


class HybridClassifier(modules.Classifier):
    # 各ブロックが消費する seed の個数
    SEEDS_PER_BLOCK = {"bi": 4, "fc": 1, "conv": 1}

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        blocks: str = "bi-fc",
        patch_sizes: tuple[int, int] = (4, 4),
        units: int = 256,
        kernel_size: int = 3,
        connectivity: float = 0.1,
        leaky: float = 0.9,
        spectral_radius: float = 0.95,
        input_scaling: float = 1.0,
        activations: str = "tanh",
        n_layer: int | None = None,
        seed: int | list[int] = 0,
    ):
        kinds = [b.strip() for b in blocks.split("-") if b.strip()]
        unknown = sorted(set(kinds) - set(self.SEEDS_PER_BLOCK))
        if unknown:
            raise ValueError(f"Unknown block {unknown}. Use {sorted(self.SEEDS_PER_BLOCK)} joined by '-'.")
        if not kinds:
            raise ValueError("blocks must contain at least one block.")

        # NOTE: 深さは blocks の長さで決まる。--n_layer と食い違うと保存パスの L と実態がずれるため弾く
        if n_layer is not None and int(n_layer) != len(kinds):
            raise ValueError(f"n_layer={n_layer} does not match len(blocks)={len(kinds)} ({blocks}).")

        super().__init__(feature_dim=units, num_classes=num_classes)

        C, H, W = input_shape
        Hp, Wp = patch_sizes
        if H % Hp != 0 or W % Wp != 0:
            raise ValueError(f"input_shape {input_shape} must be divisible by patch_sizes {patch_sizes}.")

        self.blocks_spec = blocks
        self.patchify = modules.Patchify(patch_sizes)

        seeds = modules.per_layer_seeds(seed, [self.SEEDS_PER_BLOCK[k] for k in kinds])

        input_dim = Hp * Wp * C
        layers = []
        for kind, layer_seed in zip(kinds, seeds):
            if kind == "bi":
                layers.append(
                    modules.BiReservoir2D(input_dim, units, connectivity, leaky, spectral_radius, input_scaling, layer_seed)
                )
            elif kind == "fc":
                layers.append(PositionWiseFC(input_dim, units, activations, layer_seed))
            else:
                layers.append(GridConv(input_dim, units, kernel_size, activations, layer_seed))
            input_dim = units

        self.layers = nn.Sequential(*layers)

        # 固定重み (非訓練) として扱う
        self.layers.requires_grad_(False)

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        x = self.patchify(images)  # (B, N_h, N_w, D)
        x = self.layers(x)  # (B, N_h, N_w, units)

        return x.mean(dim=(1, 2))  # (B, units)


def build(input_shape, num_classes, **kwargs):
    """networks.build_classifier("hybrid", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        num_classes: クラス数。
    """
    return HybridClassifier(input_shape, num_classes, **kwargs)
