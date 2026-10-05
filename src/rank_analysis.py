"""各条件が原理的に何次元の特徴を張れるか (アーキテクチャ由来のランク上限) を測る。

    python src/rank_analysis.py --dataset mnist
    python src/rank_analysis.py --dataset cifar_10 --out runs/exp1_rank_cifar_10.png

実データではなく一様乱数を流す。実データのランクは
    min(アーキテクチャの上限, データ自体の次元)
になるため、両者を分離できない (例: MNIST の conv2d は上限 9 に対し実データでは 4 しか出ない。
3x3 の局所窓の大半が背景の定数ゼロで、局所統計が縮退しているため)。入力側がフルランクな
一様乱数を流せば、残るのはアーキテクチャ側の制約だけになる。

理論上限の根拠: 特徴は features = GAP(phi(M(x))) の形で、空間混合 M が線形なら GAP と交換でき、
    GAP(M(x)) = (位置によらない線形写像) applied to (入力側の要約)
となって、入力側の部分空間の像に閉じ込められる。
    conv2d           : 重み共有なので GAP は「平均画像に W を掛ける」に潰れる -> C*K^2
    positionwise_fcl : GAP(W x_j) = W (平均パッチ)                            -> patch_dim
    attention        : GAP = W_vo (sum_j w_j(x) x_j)。softmax の重み w_j(x) は
                       着地点を変えるが部分空間は広げない                      -> patch_dim + pos
    fcl              : 空間平均を取らない (画像全体を 1 本に潰す)              -> 画像次元
リザバー系は位置ごとに異なる線形写像 A^k B がかかるため、この形の上限を持たない。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import networks  # noqa: E402

# 活性化を差し替えられる条件 (リザバー系は _scan が tanh 固定なので identity にできない)
LINEARIZABLE = {"fcl", "positionwise_fcl", "conv2d", "criss_cross_attention", "self_attention"}
MODELS = ["fcl", "positionwise_fcl", "conv2d", "reservoir_conv2d",
          "esn", "bi_esn", "bi_esn2d", "criss_cross_attention", "self_attention"]
SHAPES = {"mnist": (1, 28, 28), "fashion_mnist": (1, 28, 28), "cifar_10": (3, 32, 32)}
RESERVOIR_HP = dict(connectivity=0.26, leaky=0.85, spectral_radius=0.88)
EPS = 1e-10  # 数値ランクの閾値 (float32 の相対精度 1e-7 -> 固有値では 1e-14。その上に取る)


def ceiling(mt: str, shape: tuple[int, int, int], patch: int = 4, pos_dim: int = 16) -> int | None:
    """線形経路が張れる次元の上限。None は「この形の上限を持たない」。"""
    C, H, W = shape
    if mt == "fcl":
        return C * H * W
    if mt == "conv2d":
        return C * 3 * 3
    if mt == "positionwise_fcl":
        return patch * patch * C
    if mt in ("criss_cross_attention", "self_attention"):
        return patch * patch * C + pos_dim
    return None


def build_kwargs(mt: str, D: int, act: str) -> dict:
    if mt == "fcl":
        return {"units": D, "activations": act}
    if mt == "conv2d":
        return {"filters": D, "kernel_size": 3, "activations": act}
    if mt == "reservoir_conv2d":
        return {"num_reservoirs": 2, "units": D // 4, "kernel_size": 3,
                "connectivity": RESERVOIR_HP["connectivity"],
                "spectral_radius": RESERVOIR_HP["spectral_radius"]}
    if mt in ("esn", "bi_esn", "bi_esn2d"):
        return {"units": D, "patch_sizes": (4, 4), **RESERVOIR_HP}
    return {"units": D, "patch_sizes": (4, 4), "activations": act}


@torch.no_grad()
def numerical_rank(mt: str, D: int, X: torch.Tensor, shape, act: str, device) -> int:
    m = networks.build_classifier(mt, input_shape=shape, num_classes=10, n_layer=1, seed=0,
                                  **build_kwargs(mt, D, act)).to(device).eval()
    Z = torch.cat([m.features(X[i:i + 256].to(device)).cpu()
                   for i in range(0, len(X), 256)]).numpy().astype(np.float64)
    del m
    torch.cuda.empty_cache()

    Zc = Z - Z.mean(0)
    ev = np.linalg.eigvalsh(Zc.T @ Zc)[::-1].clip(min=0)
    return int((ev > ev[0] * EPS).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist", choices=sorted(SHAPES))
    p.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--n", type=int, default=3000, help="サンプル数。ランクは min(N-1, D) で頭打ちになる。")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    shape = SHAPES[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.n <= max(args.dims):
        raise SystemExit(f"--n ({args.n}) must exceed max D ({max(args.dims)}) or rank is capped by N.")

    # 入力側がフルランクな一様乱数。実データと違い、どの方向も等しく励起される
    X = torch.rand((args.n,) + shape, generator=torch.Generator().manual_seed(0))

    rows = {}
    print(f"=== {args.dataset} {shape} / 一様乱数 N={args.n} / 数値ランク (eps={EPS:.0e}) ===")
    hdr = f"{'条件':<24}{'活性化':>10}" + "".join(f"{'D=' + str(d):>8}" for d in args.dims) + f"{'理論上限':>12}"
    print(hdr)
    for mt in MODELS:
        acts = ["identity", "tanh"] if mt in LINEARIZABLE else ["tanh"]
        for act in acts:
            ranks = [numerical_rank(mt, D, X, shape, act, device) for D in args.dims]
            rows[(mt, act)] = ranks
            lim = ceiling(mt, shape)
            lim_s = str(lim) if (lim is not None and act == "identity") else "-"
            print(f"{mt:<24}{act:>10}" + "".join(f"{r:>8}" for r in ranks) + f"{lim_s:>12}", flush=True)

    if args.out:
        plot(rows, args.dims, args.dataset, shape, Path(args.out))


def plot(rows: dict, dims: list[int], dataset: str, shape, out: Path) -> None:
    import japanize_matplotlib  # noqa: F401
    import matplotlib.pyplot as plt

    # 検証済みカテゴリカルパレット (slot 1-8, light mode)
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

    # tanh の行だけを描く (identity は表で示す。線が 14 本になると読めない)
    keys = [k for k in rows if k[1] == "tanh"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # 参照線: ランク = D (フルランク)
    ax.plot(dims, dims, color=INK_MUTED, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.annotate("ランク = $D$ (フルランク)", xy=(dims[-1], dims[-1]), xytext=(-4, 6),
                textcoords="offset points", color=INK_MUTED, fontsize=8.5, ha="right")

    for (mt, _), color in zip(keys, SERIES):
        ax.plot(dims, rows[(mt, "tanh")], color=color, lw=2.0, marker="o", ms=7,
                mfc=SURFACE, mec=color, mew=2.0, label=mt, zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("ノード数 $D$", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("数値ランク", color=INK_MUTED, fontsize=10)
    ax.set_title(f"一様乱数入力で測ったランク上限（{dataset}）", color=INK, fontsize=12, pad=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"\n[INFO] wrote {out}")


if __name__ == "__main__":
    main()
