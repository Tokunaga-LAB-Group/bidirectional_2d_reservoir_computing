# PyTorch
import torch
from torch import nn

# Custom
from networks import modules


# BiReservoir2D ブロックを段階的にダウンサンプリングしながら重ね、途中の解像度を取り出して連結する
class FeatureExtractorV2(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
        N_block: int = 5,
        N_subblock: int = 1,
        sample_sizes: tuple[int, ...] = (64, 32, 16),
        base_filters: int = 32,
        connectivity: float = 0.7,
        leaky: float = 0.9,
        spectral_radius: float = 0.9,
        seed: int = 0,
        is_concat: bool = True,
    ):
        super().__init__()

        C, H, W = input_shape
        D_, H_, W_ = output_shape

        if H != W or H_ != W_:
            raise ValueError("Only square images are supported.")
        if D_ % len(sample_sizes) != 0:
            raise ValueError(f"output_shape channels ({D_}) must be divisible by len(sample_sizes).")

        D__ = D_ // len(sample_sizes)

        self.is_concat = is_concat
        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleDict()
        self.tap_blocks = []

        # BiReservoir2D は seed から seed+3 までを消費するため、4 ずつずらして全リザバーを別物にする
        next_seed = seed
        input_dim = C

        for i in range(N_block):
            units = base_filters * 2**i
            layers = []

            for j in range(N_subblock):
                # bottleneck
                if i != 0 and j == 0:
                    layers.append(modules.MaxPool2D((2, 2)))

                layers.append(modules.BiReservoir2D(input_dim, units, connectivity, leaky, spectral_radius, next_seed))
                next_seed += 4
                input_dim = units

            self.blocks.append(nn.Sequential(*layers))

            # このブロックの空間解像度が取り出し対象なら、特徴として登録する
            if H // 2**i in sample_sizes:
                self.tap_blocks.append(i)

                if is_concat:
                    self.heads[str(i)] = nn.Sequential(
                        nn.Linear(units, D__, bias=False),
                        modules.UpSampling2D((H_ * 2**i // H, W_ * 2**i // W)),
                    )

        if not self.tap_blocks:
            raise ValueError(f"No block resolution matched sample_sizes {sample_sizes}.")

        # 連結後のチャネル数 (PaDiM の次元削減など、後段が入力次元を知るために公開する)
        # NOTE: 取り出せた解像度が sample_sizes より少ない場合は D_ より小さくなる
        self.output_dim = D__ * len(self.tap_blocks)

    # (B, C, H, W) -> (B, H_, W_, D_) (channel-last)
    def forward(self, images):
        # BiReservoir2D は channel-last を扱う
        x = images.permute(0, 2, 3, 1).contiguous()

        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)

            if i in self.tap_blocks:
                features.append(self.heads[str(i)](x) if self.is_concat else x)

        if not self.is_concat:
            return features

        return torch.cat(features, dim=-1)


def build(input_shape, output_shape, **kwargs):
    """networks.build_extractor("v2", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        output_shape: 出力特徴マップの形状 (D, H, W)。出力自体は channel-last で返る。
    """
    return FeatureExtractorV2(input_shape, output_shape, **kwargs)
