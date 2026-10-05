"""特徴行列 Z の固有値スペクトルを描き、実効ランクの根拠を示す。

    python src/plot_exp1_spectrum.py --dataset mnist --units 512

手順:
  1. 訓練画像 N 枚を各条件に通し、特徴行列 Z (N x D) を得る
  2. 列ごとに平均を引く (Zc)。共分散の固有値を見たいので中心化する
  3. Zc^T Zc の固有値を降順に並べる (Z の特異値の 2 乗に等しい)
  4. 最大固有値で正規化し、どこで落ちるかを見る

数値ランクは「lambda_i > lambda_max * eps を満たす i の個数」。eps は下の THRESHOLDS で
感度を確認できる。特徴を float32 で計算しているため相対精度は 1e-7 程度で、
固有値では 1e-14 に相当する。eps=1e-10 はその上にあり、雑音を数えない。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets as tvd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import networks  # noqa: E402

# 検証済みカテゴリカルパレット (slot 1-6, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
THRESHOLDS = [1e-6, 1e-8, 1e-10, 1e-12]

MODELS = ["fcl", "bi_esn2d", "positionwise_fcl", "criss_cross_attention", "self_attention", "conv2d"]
HP = dict(connectivity=0.26, leaky=0.85, spectral_radius=0.88)


def load(dataset: str, n: int):
    if dataset in ("mnist", "fashion_mnist"):
        cls = tvd.MNIST if dataset == "mnist" else tvd.FashionMNIST
        return cls("/dataset/torchvision", train=True).data.unsqueeze(1)[:n], (1, 28, 28)
    tr = tvd.CIFAR10("/dataset/torchvision", train=True)
    return torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()[:n], (3, 32, 32)


def kwargs_for(mt: str, D: int) -> dict:
    if mt == "fcl":
        return {"units": D}
    if mt == "conv2d":
        return {"filters": D, "kernel_size": 3}
    if mt in ("esn", "bi_esn", "bi_esn2d"):
        return {"units": D, "patch_sizes": (4, 4), **HP}
    return {"units": D, "patch_sizes": (4, 4)}


@torch.no_grad()
def eigenvalues(mt: str, D: int, X: torch.Tensor, shape, device) -> np.ndarray:
    m = networks.build_classifier(mt, input_shape=shape, num_classes=10, n_layer=1, seed=0,
                                  **kwargs_for(mt, D)).to(device).eval()
    Z = torch.cat([m.features(X[i:i + 256].to(device).float().div_(255.0)).cpu()
                   for i in range(0, len(X), 256)]).numpy().astype(np.float64)
    del m
    torch.cuda.empty_cache()

    Zc = Z - Z.mean(0)                              # 列ごとに中心化
    ev = np.linalg.eigvalsh(Zc.T @ Zc)[::-1]        # 降順の固有値 = 特異値の 2 乗
    return ev.clip(min=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--units", type=int, default=512)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, shape = load(args.dataset, args.n)

    spectra = {mt: eigenvalues(mt, args.units, X, shape, device) for mt in MODELS}

    print(f"=== {args.dataset} / D={args.units} / N={args.n} — 閾値ごとの数値ランク ===")
    print(f"{'条件':<24}" + "".join(f"{'eps=' + f'{t:.0e}':>12}" for t in THRESHOLDS))
    for mt, ev in spectra.items():
        counts = [int((ev > ev[0] * t).sum()) for t in THRESHOLDS]
        print(f"{mt:<24}" + "".join(f"{c:>12}" for c in counts))

    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # NOTE: 6 系列あるので直接ラベルは重なる。凡例で識別する (曲線の終端順に並べる)
    for (mt, ev), color in zip(spectra.items(), SERIES):
        norm = ev / ev[0]
        k = int((norm > 1e-10).sum())
        ax.plot(np.arange(1, len(norm) + 1), np.maximum(norm, 1e-18), color=color, lw=2.0,
                label=f"{mt} (rank {k})")

    ax.axhline(1e-10, color=INK_MUTED, lw=1.2, ls=(0, (5, 3)))
    ax.annotate("数値ランクの閾値 (eps=1e-10)", xy=(1, 1e-10), xytext=(3, -6),
                textcoords="offset points", color=INK_MUTED, fontsize=8.5, va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 3)
    ax.set_xlabel("固有値の順位 $i$", color=INK_MUTED, fontsize=10)
    ax.set_ylabel(r"$\lambda_i / \lambda_1$", color=INK_MUTED, fontsize=10)
    ax.set_title(f"特徴行列の固有値スペクトル（{args.dataset}, D={args.units}）",
                 color=INK, fontsize=12, pad=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

    leg = ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK,
                    handlelength=1.6, borderaxespad=0.4)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    out = Path(args.out or f"runs/exp1_spectrum_{args.dataset}_D{args.units}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"\n[INFO] wrote {out}")


if __name__ == "__main__":
    main()
