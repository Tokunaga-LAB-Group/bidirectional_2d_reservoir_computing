# PyTorch
import torch
import torch.nn.functional as F
from torch import nn


# (B, C, H, W) -> (B, N_h, N_w, D)
class Patchify(nn.Module):
    def __init__(self, patch_sizes):
        super().__init__()
        self.Hp, self.Wp = patch_sizes[0], patch_sizes[1]

    def forward(self, images):
        B, C, H, W = images.shape
        N_h, N_w = H // self.Hp, W // self.Wp

        # F.unfold: (B, C*Hp*Wp, N_h*N_w)
        patches = F.unfold(images, kernel_size=(self.Hp, self.Wp), stride=(self.Hp, self.Wp))

        patches = patches.transpose(1, 2)  # (B, N_h*N_w, C*Hp*Wp)
        patches = patches.reshape(B, N_h, N_w, C * self.Hp * self.Wp)
        return patches


# Fixed-weight leaky-integrator reservoir, single direction.
# (B, T, input_dim) -> (B, T, units)
@torch.jit.script
def _scan(
    xW: torch.Tensor,
    W_rec: torch.Tensor,
    leaky: float,
    go_backwards: bool,
) -> torch.Tensor:
    B, T, U = xW.shape
    h = torch.zeros(B, U, device=xW.device, dtype=xW.dtype)
    out = torch.empty(T, B, U, device=xW.device, dtype=xW.dtype)

    for i in range(T):
        t = T - 1 - i if go_backwards else i
        pre = torch.addmm(xW[:, t], h, W_rec)
        h_tilde = torch.tanh(pre)
        h = torch.lerp(h, h_tilde, leaky)
        out[t] = h

    return out.transpose(0, 1).contiguous()


class Reservoir(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: int,
        connectivity: float = 0.1,
        leaky: float = 0.9,
        spectral_radius: float = 0.95,
        seed: int = 0,
    ):
        super().__init__()
        self.units = units
        self.leaky = leaky

        # 乱数ジェネレータを作成
        g = torch.Generator().manual_seed(seed)

        # リザバー入力層の固定重みの初期化
        limit_in = (6.0 / (input_dim + units)) ** 0.5
        W_in = (torch.rand(input_dim, units, generator=g) * 2 - 1) * limit_in

        # リザバー層の固定重みの初期化
        limit_rec = (6.0 / (units + units)) ** 0.5
        W_rec = (torch.rand(units, units, generator=g) * 2 - 1) * limit_rec
        mask = (torch.rand(units, units, generator=g) < connectivity).float()
        W_rec = W_rec * mask

        eigvals = torch.linalg.eigvals(W_rec)
        current_radius = eigvals.abs().max()
        if current_radius > 0:
            W_rec = W_rec * (spectral_radius / current_radius)

        # 固定重みを非訓練パラメータ (バッファ)として登録
        self.register_buffer("W_in", W_in)
        self.register_buffer("W_rec", W_rec)

    def forward(self, x, go_backwards=False):
        with torch.no_grad():
            xW = x @ self.W_in
            return _scan(xW, self.W_rec, float(self.leaky), go_backwards)


# (B, T, input_dim) -> (B, T, output_dim)
class BiReservoir(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        connectivity: float = 0.1,
        leaky: float = 0.9,
        spectral_radius: float = 0.95,
        seed: int | tuple[int, int] = 0,
    ):
        super().__init__()
        # 順方向・逆方向で異なるシードを使い、リザバーの重みが同一になるのを防ぐ
        if isinstance(seed, int):
            f_seed, b_seed = seed, seed + 1
        elif isinstance(seed, (list, tuple)) and len(seed) == 2:
            f_seed, b_seed = seed
        else:
            raise ValueError("seed must be int or list/tuple of two ints.")

        # 切り捨てによる次元の食い違いを防ぐため、偶数のみ許可する
        if output_dim % 2 != 0:
            raise ValueError(f"output_dim must be even, got {output_dim}.")

        half = output_dim // 2
        self.output_dim = output_dim

        self.forward_reservoir = Reservoir(input_dim, half, connectivity, leaky, spectral_radius, f_seed)
        self.backward_reservoir = Reservoir(input_dim, half, connectivity, leaky, spectral_radius, b_seed)

    def forward(self, inputs):
        fwd = self.forward_reservoir(inputs, go_backwards=False)
        bwd = self.backward_reservoir(inputs, go_backwards=True)
        return torch.cat([fwd, bwd], dim=-1)


# (B, N_h, N_w, input_dim) -> (B, N_h, N_w, output_dim)
class BiReservoir2D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        connectivity: float = 0.1,
        leaky: float = 0.9,
        spectral_radius: float = 0.95,
        seed: int | tuple[int, int, int, int] = 0,
    ):
        super().__init__()
        # 4つのリザバー全てに異なるシードを割り当てる
        if isinstance(seed, int):
            v_seed = (seed, seed + 1)
            h_seed = (seed + 2, seed + 3)
        elif isinstance(seed, (list, tuple)) and len(seed) == 4:
            v_seed = (int(seed[0]), int(seed[1]))
            h_seed = (int(seed[2]), int(seed[3]))
        else:
            raise ValueError("seed must be int or list/tuple of four ints.")

        # 縦横で2分割し、さらに各 BiReservoir が双方向で2分割するため 4 の倍数が必要
        if output_dim % 4 != 0:
            raise ValueError(f"output_dim must be divisible by 4, got {output_dim}.")

        self.output_dim = output_dim
        half = output_dim // 2

        self.vertical = BiReservoir(input_dim, half, connectivity, leaky, spectral_radius, v_seed)
        self.horizontal = BiReservoir(input_dim, half, connectivity, leaky, spectral_radius, h_seed)

    def forward(self, inputs):
        B, N_h, N_w, C = inputs.shape
        half = self.output_dim // 2

        # 各行を長さ N_w の系列として処理する
        h = self.horizontal(inputs.reshape(B * N_h, N_w, C))
        h = h.reshape(B, N_h, N_w, half)

        # 各列を長さ N_h の系列として処理する
        v = self.vertical(inputs.permute(0, 2, 1, 3).reshape(B * N_w, N_h, C))
        v = v.reshape(B, N_w, N_h, half).permute(0, 2, 1, 3)

        return torch.cat([h, v], dim=-1)


# ここから下はモデル組み立て用の補助レイヤ
# BiReservoir2D と揃えるため、2D 系のレイヤは全て channel-last (B, H, W, C) を扱う


class UpSampling2D(nn.Module):
    def __init__(self, scale_factor: int | tuple[int, int]):
        super().__init__()
        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor)
        self.scale_factor = (int(scale_factor[0]), int(scale_factor[1]))

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError(f"scale_factor must be >= 1, got {self.scale_factor}.")

    def forward(self, inputs):
        if self.scale_factor == (1, 1):
            return inputs

        # NOTE: F.interpolate は channel-first を要求するため入れ替える
        x = inputs.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous()


class MaxPool2D(nn.Module):
    def __init__(self, pool_size: tuple[int, int] = (2, 2)):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, inputs):
        x = inputs.permute(0, 3, 1, 2)
        x = self.pool(x)
        return x.permute(0, 2, 3, 1).contiguous()


# リッジ回帰などで外部から重みを決める、バイアスなし・非訓練の読み出し層
class LinearReadout(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_classes, input_dim), requires_grad=False)

    def forward(self, features):
        return features @ self.weight.T

    @torch.no_grad()
    def set_kernel(self, kernel):
        """Keras の Dense kernel と同じ (input_dim, num_classes) の並びで重みを設定する。"""
        kernel = torch.as_tensor(kernel, dtype=self.weight.dtype, device=self.weight.device)
        expected = (self.weight.shape[1], self.weight.shape[0])
        if tuple(kernel.shape) != expected:
            raise ValueError(f"kernel must have shape {expected}, got {tuple(kernel.shape)}.")

        self.weight.copy_(kernel.T)


class ReservoirConv2D(nn.Module):
    # (B, N_h, N_w, in_channels) -> (B, H_out, W_out, 2 * num_reservoirs * units)
    def __init__(
        self,
        in_channels: int,
        num_reservoirs: int = 5,
        units: int = 12,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | tuple[int, int] | str = 0,
        connectivity: float = 0.5,
        spectral_radius: float = 0.95,
        seed: int = 0,
    ):
        super().__init__()
        N, K = num_reservoirs, kernel_size
        self.K, self.S = K, stride
        self.pad_h, self.pad_w = self._resolve_padding(padding, K, stride)

        # delta_i = 0.8 * (i - 1) / (N - 1) + 0.1
        leaks = [0.8 * i / (N - 1) + 0.1 for i in range(N)] if N > 1 else [0.5]

        line_dim = in_channels * K  # 1ステップあたりの入力次元
        self.horizontal = nn.ModuleList(
            [Reservoir(line_dim, units, connectivity, lk, spectral_radius, seed + i) for i, lk in enumerate(leaks)]
        )
        self.vertical = nn.ModuleList(
            [Reservoir(line_dim, units, connectivity, lk, spectral_radius, seed + N + i) for i, lk in enumerate(leaks)]
        )
        self.output_dim = 2 * N * units

    @staticmethod
    def _resolve_padding(padding, K: int, stride: int) -> tuple[int, int]:
        if isinstance(padding, str):
            if padding == "valid":
                return 0, 0
            if padding == "same":
                # 出力サイズを入力サイズに保つ設定は stride=1 でのみ成立する
                if stride != 1:
                    raise ValueError("padding='same' requires stride=1.")
                if K % 2 == 0:
                    raise ValueError(f"padding='same' requires an odd kernel_size, got {K}.")
                return K // 2, K // 2
            raise ValueError(f"padding must be 'same', 'valid', int, or tuple, got {padding!r}.")

        if isinstance(padding, int):
            pad_h = pad_w = padding
        elif isinstance(padding, (list, tuple)) and len(padding) == 2:
            pad_h, pad_w = int(padding[0]), int(padding[1])
        else:
            raise ValueError("padding must be 'same', 'valid', int, or tuple of two ints.")

        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"padding must be non-negative, got {(pad_h, pad_w)}.")

        return pad_h, pad_w

    def forward(self, x):
        B, _, _, C = x.shape
        K, S = self.K, self.S
        pad_h, pad_w = self.pad_h, self.pad_w
        output_dim = self.output_dim

        # チャネルが最終軸のため、F.pad の指定は (C, N_w, N_h) の順になる
        # 最終軸から, その軸の前後にパディングする
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, pad_w, pad_w, pad_h, pad_h))

        if x.shape[1] < K or x.shape[2] < K:
            raise ValueError(f"Padded input {tuple(x.shape[1:3])} is smaller than kernel size {K}.")

        # ROI 抽出: (B, H_out, W_out, C, K_h, K_w)
        roi = x.unfold(1, K, S).unfold(2, K, S)
        H_out, W_out = roi.shape[1], roi.shape[2]

        # horizontal: K_w ステップ、各ステップ C * K_h 次元（列を左から右へ）
        seq_h = roi.permute(0, 1, 2, 5, 3, 4).reshape(B * H_out * W_out, K, C * K)
        # vertical: K_h ステップ、各ステップ C * K_w 次元（行を上から下へ）
        seq_v = roi.permute(0, 1, 2, 4, 3, 5).reshape(B * H_out * W_out, K, C * K)

        # 各リザバーの最終状態のみを使う
        feats = [r(seq_h)[:, -1] for r in self.horizontal] + [r(seq_v)[:, -1] for r in self.vertical]

        return torch.cat(feats, dim=-1).reshape(B, H_out, W_out, output_dim)


def glorot_uniform(shape: tuple[int, int], generator: torch.Generator) -> torch.Tensor:
    """分散が 2/(fan_in + fan_out) になる一様分布から重みを引く。

    NOTE: nn.Linear の既定の初期化はグローバル RNG に依存するため、seed 引数だけでは再現しない。
          Reservoir と同じく Generator から明示的に引き直すためのヘルパ。
    """
    limit = (6.0 / (shape[0] + shape[1])) ** 0.5
    return (torch.rand(shape, generator=generator) * 2 - 1) * limit


def lecun_uniform(shape: tuple[int, int], generator: torch.Generator) -> torch.Tensor:
    """分散が 1/fan_in になる一様分布から重みを引く (出力の分散が入力の分散と揃う)。"""
    limit = (3.0 / shape[0]) ** 0.5
    return (torch.rand(shape, generator=generator) * 2 - 1) * limit


def sincos_positions_2d(N_h: int, N_w: int, dim: int) -> torch.Tensor:
    """2 次元の正弦波位置符号 (1, N_h, N_w, dim) を返す。

    前半 dim/2 チャネルが行位置、後半 dim/2 チャネルが列位置を符号化する。
    各軸の内訳は Transformer と同じで、周波数ごとに sin と cos の対を持つため dim は 4 の倍数。
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4, got {dim}.")

    d_axis = dim // 2
    omega = 1.0 / (10000.0 ** (torch.arange(0, d_axis, 2, dtype=torch.float32) / d_axis))

    def encode(n: int) -> torch.Tensor:
        phase = torch.arange(n, dtype=torch.float32).unsqueeze(1) * omega  # (n, d_axis/2)
        return torch.cat([phase.sin(), phase.cos()], dim=-1)  # (n, d_axis)

    enc_h = encode(N_h).reshape(N_h, 1, d_axis).expand(N_h, N_w, d_axis)
    enc_w = encode(N_w).reshape(1, N_w, d_axis).expand(N_h, N_w, d_axis)

    return torch.cat([enc_h, enc_w], dim=-1).unsqueeze(0).contiguous()  # (1, N_h, N_w, dim)


# (B, N_h, N_w, D) -> (B, N_h, N_w, D + dim)
class ConcatSinCosPositions2D(nn.Module):
    """2 次元正弦波位置符号を入力の末尾に連結する。

    NOTE: ViT のように埋め込みへ「加算」するのではなく「連結」する。ここでの入力はパッチの生値で
          スケールが位置符号 ([-1, 1]) と揃っておらず、加算すると位置符号が入力を上書きしてしまうため。
    """

    def __init__(self, N_h: int, N_w: int, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("encoding", sincos_positions_2d(N_h, N_w, dim))

    def forward(self, inputs):
        return torch.cat([inputs, self.encoding.expand(inputs.shape[0], -1, -1, -1)], dim=-1)


def criss_cross_mask(N_h: int, N_w: int) -> torch.Tensor:
    """criss-cross attention 用の attn_mask (N_h*N_w, N_h*N_w) を返す。True が「見ない」。

    位置 (i, j) から見えるのは行 i と列 j の N_h + N_w - 1 箇所だけになる
    (自分自身は行と列の両方に属するが、集合なので 1 回しか数えられない)。
    """
    rows = torch.arange(N_h * N_w) // N_w
    cols = torch.arange(N_h * N_w) % N_w

    same_row = rows.unsqueeze(1) == rows.unsqueeze(0)
    same_col = cols.unsqueeze(1) == cols.unsqueeze(0)

    return ~(same_row | same_col)


# 固定ランダム重みの nn.MultiheadAttention を channel-last の 2 次元格子へ適用する
# (B, N_h, N_w, input_dim) -> (B, N_h, N_w, units)
#
# NOTE: attention の計算そのものは torch の nn.MultiheadAttention をそのまま使い、
#       重みだけを Generator から引いた固定値で上書きする。本リポジトリの他の classifier と同じく
#       「特徴抽出側は非訓練、読み出しだけリッジ回帰」という条件に合わせるための最小限の変更である
#
# NOTE: attn_mask を与えると、許可された位置だけを見る attention になる。criss-cross attention
#       (CCNet, Huang et al. ICCV 2019) は「同じ行・同じ列以外を塞いだ self-attention」と厳密に等価なので、
#       self_attention と criss_cross_attention はマスク以外すべて同一の実装・同一の重みになる。
#       両者の差はそのまま「受容野を十字に絞ったことの寄与」として読める
class FixedMultiheadAttention2D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: int,
        n_head: int = 4,
        temperature: float = 1.0,
        seed: int = 0,
        attn_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        if units % n_head != 0:
            raise ValueError(f"units must be divisible by n_head, got units={units}, n_head={n_head}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        self.units = units

        g = torch.Generator().manual_seed(seed)

        # NOTE: nn.MultiheadAttention は出力次元が query の次元 (embed_dim) に固定されるため、
        #       units をパッチ次元と独立に選ぶには一度 units へ射影しておく必要がある
        #       (ViT の patch embedding に相当する)。分散を保つよう LeCun 一様で引く
        self.register_buffer("W_embed", lecun_uniform((input_dim, units), g))

        self.attention = nn.MultiheadAttention(units, n_head, bias=False, batch_first=True)

        # NOTE: temperature は Q 側の重みを 1/temperature 倍するだけで実現できる。
        #       nn.MultiheadAttention の 1/sqrt(d_head) は固定だが、
        #       softmax(qk^T/sqrt(d)) の q を定数倍することと温度を割ることは同じ操作のため、
        #       attention の実装には一切手を入れずに済む。
        #       大きくするほど注意が一様に近づき、attention は単なる空間平均へ退化する
        W_q = lecun_uniform((units, units), g) / float(temperature)
        W_k = lecun_uniform((units, units), g)

        # V/O は出力そのもののスケールを決めるので、他の classifier と揃えて Glorot にする
        W_v = glorot_uniform((units, units), g)
        W_o = glorot_uniform((units, units), g)

        # NOTE: nn.MultiheadAttention は x @ W.T の形で射影するため、転置して詰める
        with torch.no_grad():
            self.attention.in_proj_weight.copy_(torch.cat([W_q.T, W_k.T, W_v.T], dim=0))
            self.attention.out_proj.weight.copy_(W_o.T)

        # 固定重み (非訓練) として扱う
        self.attention.requires_grad_(False)

        if attn_mask is None:
            self.attn_mask = None
        else:
            self.register_buffer("attn_mask", attn_mask)

    def forward(self, inputs):
        B, N_h, N_w, _ = inputs.shape

        with torch.no_grad():
            x = (inputs @ self.W_embed).reshape(B, N_h * N_w, self.units)  # (B, T, units)

            out, _ = self.attention(x, x, x, need_weights=False, attn_mask=self.attn_mask)

            return out.reshape(B, N_h, N_w, self.units)

# 層を積むモデルのハイパーパラメータを層ごとに展開する
def per_layer_hparams(n_layer: int | None = None, **hparams) -> list[dict]:
    """ハイパーパラメータを層ごとの辞書のリストに展開する。

    リストで渡された値は「層ごとの値」、それ以外 (スカラーやタプル) は全層で共通の値として扱う。
    NOTE: タプルを層ごとの指定と解釈しないのは、patch_sizes=(4, 4) や padding=(1, 1) のようにタプル自体が 1 層分の値であることがあるため。

    Args:
        n_layer: 層数。None ならリストの長さから決まり、リストが一つも無ければ 1 層になる。

    Examples:
        >>> per_layer_hparams(None, units=[16, 32], leaky=0.9)
        [{'units': 16, 'leaky': 0.9}, {'units': 32, 'leaky': 0.9}]
        >>> per_layer_hparams(2, units=16)
        [{'units': 16}, {'units': 16}]
    """
    lengths = {name: len(value) for name, value in hparams.items() if isinstance(value, list)}

    if len(set(lengths.values())) > 1:
        raise ValueError(f"Per-layer lists must all have the same length, got {lengths}.")

    inferred = next(iter(lengths.values()), 1)
    if n_layer is None:
        n_layer = inferred
    elif lengths and n_layer != inferred:
        raise ValueError(f"n_layer ({n_layer}) does not match the per-layer lists {lengths}.")

    if n_layer < 1:
        raise ValueError(f"n_layer must be >= 1, got {n_layer}.")

    return [
        {name: (value[i] if isinstance(value, list) else value) for name, value in hparams.items()}
        for i in range(n_layer)
    ]


def per_layer_seeds(seed: int | list[int], consumed: list[int]) -> list[int]:
    """層ごとの seed を決める。

    スカラーを渡した場合は、各層が消費する seed の個数だけずらして、層同士が同じ重みになるのを防ぐ。
    リストを渡した場合はそのまま層ごとの seed として使う。

    Args:
        seed: 起点の seed、または層ごとの seed のリスト。
        consumed: 各層が消費する seed の個数 (Reservoir は 1、BiReservoir は 2、BiReservoir2D は 4)。
    """
    if isinstance(seed, list):
        if len(seed) != len(consumed):
            raise ValueError(f"seed must have {len(consumed)} entries (one per layer), got {len(seed)}.")

        return [int(s) for s in seed]

    seeds, next_seed = [], int(seed)
    for n in consumed:
        seeds.append(next_seed)
        next_seed += n

    return seeds


class Classifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.readout = LinearReadout(feature_dim, num_classes)

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        raise NotImplementedError

    # (B, C, H, W) -> (B, num_classes)
    def forward(self, images):
        return self.readout(self.features(images))


def get_activation(name: str) -> nn.Module:
    """活性化関数の名前から PyTorch のモジュールを返す。

    Args:
        name: 活性化関数の名前。'relu', 'tanh', 'sigmoid', 'gelu', 'swish', 'identity' のいずれか。
            'identity' は活性化を挟まない (非線形性の寄与を切り分ける対照実験用)。
    """
    name = name.lower()

    match name:
        case "relu":
            return nn.ReLU()
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "gelu":
            return nn.GELU()
        case "swish":
            return nn.SiLU()
        case "identity":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported activation function: {name}.")


if __name__ == "__main__":
    # quick smoke test
    torch.manual_seed(0)
    images = torch.randn(2, 1, 28, 28)  # e.g. MNIST, (B, C, H, W)

    p2v = Patchify(patch_sizes=(4, 4))
    patches = p2v(images)
    print("patches:", patches.shape)  # -> [2, 7, 7, 16]

    bi_reservoir2d = BiReservoir2D(input_dim=patches.shape[-1], output_dim=32, seed=(0, 1, 2, 3))
    out = bi_reservoir2d(patches)
    print("bi_reservoir2d output:", out.shape)  # -> [2, 7, 7, 32]
