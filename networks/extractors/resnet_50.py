# PyTorch
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

# Custom
from networks import modules

# サンプルサイズ -> ResNet50 のステージ番号 (入力 256x256 のとき layer1/2/3 が 64/32/16)
_SAMPLE_SIZE_TO_STAGE = {64: 0, 32: 1, 16: 2}

# ResNet50 の layer1 / layer2 / layer3 の出力チャネル数
_STAGE_DIMS = (256, 512, 1024)


# ImageNet 事前学習済み ResNet50 のマルチスケール特徴を連結する
#
# NOTE: PaDiM のランダムなチャネル間引きは backbone の性質ではなく PaDiM 側の手順なので、
# networks.padim.PaDiM が担当する。ここでは連結した特徴マップをそのまま返す。
class FeatureExtractorCNN(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        sample_sizes: tuple[int, ...] = (64, 32, 16),
        is_concat: bool = True,
        weights=ResNet50_Weights.IMAGENET1K_V1,
    ):
        super().__init__()

        C, H, W = input_shape
        if C != 3:
            raise ValueError(f"ResNet50 backbone expects 3-channel inputs, got {C}.")

        unknown = [ss for ss in sample_sizes if ss not in _SAMPLE_SIZE_TO_STAGE]
        if unknown:
            raise ValueError(f"Unsupported sample sizes {unknown}. Available: {sorted(_SAMPLE_SIZE_TO_STAGE)}.")

        self.tap_stages = sorted(_SAMPLE_SIZE_TO_STAGE[ss] for ss in sample_sizes)
        self.is_concat = is_concat

        # 連結後のチャネル数 (PaDiM の次元削減など、後段が入力次元を知るために公開する)
        self.output_dim = sum(_STAGE_DIMS[i] for i in self.tap_stages)

        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stages = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3][: max(self.tap_stages) + 1])

        # NOTE: keras.applications.resnet50.preprocess_input は caffe 形式 (BGR・平均差し引きのみ) だが、
        # torchvision の重みは RGB を [0, 1] に正規化した入力を前提とするため統計値が異なる。
        self.register_buffer("mean", torch.tensor(weights.transforms().mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(weights.transforms().std).view(1, 3, 1, 1))

        # 一番解像度の高い (最も浅い) 特徴マップに合わせる
        finest = min(self.tap_stages)
        self.upsamplings = nn.ModuleList([modules.UpSampling2D(2 ** (i - finest)) for i in self.tap_stages])

        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # backbone は凍結して使うため、BatchNorm の統計が更新されないよう常に eval に固定する
        return super().train(False)

    # (B, 3, H, W) -> (B, H_, W_, D_) (channel-last)
    @torch.no_grad()
    def forward(self, images):
        x = (images - self.mean) / self.std
        x = self.stem(x)

        feature_maps = []
        for i, stage in enumerate(self.stages):
            x = stage(x)

            if i in self.tap_stages:
                feature_maps.append(x.permute(0, 2, 3, 1).contiguous())

        if not self.is_concat:
            return feature_maps

        feature_maps = [upsampling(fm) for upsampling, fm in zip(self.upsamplings, feature_maps)]
        return torch.cat(feature_maps, dim=-1)


def build(input_shape, output_shape=None, seed=None, **kwargs):
    """networks.build_extractor("resnet_50", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        output_shape: 他の extractor と引数を揃えるためだけに受け取る。出力の形状は backbone と
            sample_sizes で決まるため使用しない。
        seed: 同上。ImageNet 事前学習済みの backbone は乱数を使わないため、この extractor の出力は
            seed に依存しない (seed を使う PaDiM のチャネル間引きは networks.padim.PaDiM が持つ)。
    """
    return FeatureExtractorCNN(input_shape, **kwargs)
