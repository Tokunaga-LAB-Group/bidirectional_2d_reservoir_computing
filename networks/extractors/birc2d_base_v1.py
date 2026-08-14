# PyTorch
import torch
from torch import nn

# Custom
from networks import modules


# スケールごとにパッチ化し、BiReservoir2D を N 段重ねた特徴を連結する
class FeatureExtractorV1(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int, int],
        sample_sizes: tuple[int, ...] = (64, 32, 16),
        N: int = 10,
        connectivity: float = 0.9,
        leaky: float = 0.8,
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
        self.branches = nn.ModuleList()
        self.upsamplings = nn.ModuleList()

        # 連結後のチャネル数 (PaDiM の次元削減など、後段が入力次元を知るために公開する)
        self.output_dim = D_

        # BiReservoir2D は seed から seed+3 までを消費するため、4 ずつずらして全リザバーを別物にする
        next_seed = seed

        for ss in sample_sizes:
            if H % ss != 0 or H_ % ss != 0:
                raise ValueError(f"sample size {ss} must divide both H ({H}) and H_ ({H_}).")

            ps = H // ss

            # (C, H, W) -> (ss, ss, ps*ps*C)
            layers = [modules.Patchify((ps, ps))]
            input_dim = ps * ps * C

            for _ in range(N):
                # (ss, ss, input_dim) -> (ss, ss, D__)
                layers.append(modules.BiReservoir2D(input_dim, D__, connectivity, leaky, spectral_radius, next_seed))
                next_seed += 4
                input_dim = D__

            self.branches.append(nn.Sequential(*layers))

            # (ss, ss, D__) -> (H_, W_, D__)
            self.upsamplings.append(modules.UpSampling2D((H_ // ss, W_ // ss)))

    # (B, C, H, W) -> (B, H_, W_, D_) (channel-last)
    def forward(self, images):
        features = [branch(images) for branch in self.branches]

        if not self.is_concat:
            return features

        features = [upsampling(feature) for upsampling, feature in zip(self.upsamplings, features)]
        return torch.cat(features, dim=-1)


def build(input_shape, output_shape, **kwargs):
    """networks.build_extractor("v1", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        output_shape: 出力特徴マップの形状 (D, H, W)。出力自体は channel-last で返る。
    """
    return FeatureExtractorV1(input_shape, output_shape, **kwargs)
