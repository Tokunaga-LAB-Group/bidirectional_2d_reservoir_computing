# PyTorch
import torch
import tqdm
from torch import nn


# チャネル方向をランダムに間引く (PaDiM の次元削減)
class SamplingFeatureMaps(nn.Module):
    def __init__(self, input_dim: int, random_dim_size: int, seed: int = 0):
        super().__init__()
        if random_dim_size > input_dim:
            raise ValueError(f"random_dim_size ({random_dim_size}) must be <= input_dim ({input_dim}).")

        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(input_dim, generator=g)[:random_dim_size]
        self.register_buffer("indices", indices)

    def forward(self, inputs):
        return torch.index_select(inputs, -1, self.indices)


def _iter_batches(images, batch_size: int, device, desc: str, verbose: bool):
    """画像を batch_size ずつ device に載せて流す。images は (B, C, H, W) の tensor / ndarray。"""
    images = torch.as_tensor(images, dtype=torch.float32)

    if images.ndim != 4:
        raise ValueError(f"images must be a 4D (B, C, H, W) array, got shape {tuple(images.shape)}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    steps = range(0, images.shape[0], batch_size)
    for i in tqdm.tqdm(steps, desc=desc, disable=not verbose):
        yield images[i : i + batch_size].to(device)


class PaDiM(nn.Module):
    """特徴抽出器の出力を埋め込みとみなし、位置ごとの正常分布とマハラノビス距離を計算する。

    PaDiM 固有の手順 (チャネルのランダム間引き + 位置ごとのガウス統計) はすべてここに閉じる。
    特徴抽出器側は「画像 -> 特徴マップ」までを担当し、間引きの有無を意識しなくてよい。

    Args:
        extractor: (B, C, H, W) -> (B, H_, W_, D_) (channel-last) を返す特徴抽出器。
            チャネル数 D_ を output_dim 属性で公開している必要がある (is_concat=True で使うこと)。
        random_dim_size: 間引き後のチャネル数。None なら間引きを行わない。
        seed: 間引くチャネルを選ぶ乱数のシード。
        eps: 共分散行列に加える対角項。逆行列の数値的な安定化のため。
    """

    def __init__(self, extractor: nn.Module, random_dim_size: int | None = None, seed: int = 0, eps: float = 0.01):
        super().__init__()

        self.extractor = extractor
        self.eps = eps

        self.sampling = None
        if random_dim_size is not None:
            if not hasattr(extractor, "output_dim"):
                raise AttributeError(f"{type(extractor).__name__} must expose output_dim to use random_dim_size.")

            self.sampling = SamplingFeatureMaps(extractor.output_dim, random_dim_size, seed)

        # fit() で決まる正常分布の統計量
        self.feature_shape = None  # (H_, W_)
        self.register_buffer("mean", None)  # (1, H_*W_, D)
        self.register_buffer("cov_inv", None)  # (H_*W_, D, D)

        self.requires_grad_(False)
        self.eval()

    @property
    def embedding_dim(self) -> int:
        if self.sampling is None:
            return self.extractor.output_dim

        return self.sampling.indices.numel()

    def _device(self):
        for tensor in [*self.parameters(), *self.buffers()]:
            return tensor.device

        return torch.device("cpu")

    # (B, C, H, W) -> (B, H_, W_, D)
    @torch.no_grad()
    def embed(self, images):
        feature_maps = self.extractor(images)

        if not torch.is_tensor(feature_maps):
            raise TypeError(
                f"{type(self.extractor).__name__} returned {type(feature_maps).__name__}; "
                "PaDiM expects a single concatenated feature map (build it with is_concat=True)."
            )

        if self.sampling is None:
            return feature_maps

        return self.sampling(feature_maps)

    @torch.no_grad()
    def fit(self, normal_imgs, batch_size: int = 16, verbose: bool = True):
        """正常画像から位置ごとの平均と共分散の逆行列を求める。"""
        device = self._device()

        count = 0
        sum_ = None  # (HW, D)
        sum_outer = None  # (HW, D, D)

        for batch in _iter_batches(normal_imgs, batch_size, device, "Cal Statistics", verbose):
            embeddings = self.embed(batch)
            B, H, W, D = embeddings.shape

            # NOTE: 総和での桁落ちを避けるため float64 で溜める (np.cov も内部で float64 に昇格する)
            embeddings = embeddings.reshape(B, H * W, D).double()

            if sum_ is None:
                sum_ = torch.zeros(H * W, D, dtype=torch.float64, device=device)
                sum_outer = torch.zeros(H * W, D, D, dtype=torch.float64, device=device)
                self.feature_shape = (H, W)
            elif (H, W) != self.feature_shape:
                raise ValueError(f"Feature map size changed within fit(): {self.feature_shape} -> {(H, W)}.")

            count += B
            sum_ += embeddings.sum(dim=0)
            sum_outer += torch.einsum("nmd,nme->mde", embeddings, embeddings)

        if sum_ is None:
            raise ValueError("normal_imgs is empty.")
        if count < 2:
            raise ValueError(f"At least 2 normal images are required to estimate a covariance, got {count}.")

        mean = sum_ / count

        # 不偏共分散 (Σ x x^T - n μ μ^T) / (n - 1)。巨大な中間テンソルを作らないよう in-place で計算する
        cov = sum_outer.baddbmm_(mean.unsqueeze(2), mean.unsqueeze(1), alpha=-count)
        cov = cov.div_(count - 1)
        cov.diagonal(dim1=-2, dim2=-1).add_(self.eps)

        self.mean = mean.unsqueeze(0).float()
        self.cov_inv = torch.linalg.inv(cov).float()

        return self

    # (B, C, H, W) -> (B, H_, W_) (異常スコアマップ)
    @torch.no_grad()
    def score(self, input_imgs, batch_size: int = 16, verbose: bool = True):
        """マハラノビス距離を計算する。fit() で求めた統計量を使う。"""
        if self.mean is None:
            raise RuntimeError("Call fit() with normal images before score().")

        device = self._device()
        dist_maps = []

        for batch in _iter_batches(input_imgs, batch_size, device, "Cal Distance", verbose):
            embeddings = self.embed(batch)
            B, H, W, D = embeddings.shape

            if H * W != self.mean.shape[1]:
                raise ValueError(
                    f"Feature map size {(H, W)} does not match the fitted size {self.feature_shape} "
                    f"({self.mean.shape[1]} positions)."
                )

            temp = embeddings.reshape(B, H * W, D) - self.mean

            # (B, HW, D) x (HW, D, D) -> (B, HW, D) -> (B, HW)
            dists = torch.einsum("nmd,mde->nme", temp, self.cov_inv).mul_(temp).sum(dim=-1)

            # 数値誤差でごく僅かに負になることがあるため 0 で下切りする
            dist_maps.append(dists.clamp_min_(0).sqrt_().reshape(B, H, W))

        return torch.cat(dist_maps)

    def forward(self, images):
        return self.score(images, verbose=False)
