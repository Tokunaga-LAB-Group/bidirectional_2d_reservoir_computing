"""実験1で選ばれたハイパーパラメータを、ノード数 D に対してプロットする。

    python src/plot_exp1_hparams.py --param connectivity
    python src/plot_exp1_hparams.py --param spectral_radius --model bi_esn2d

各 fold (3 seeds x 5 folds = 15) で選ばれた値を点で、その中央値を線で描く。
探索は log 一様分布からの 10 試行なので、値が探索分布の中央値の周りに散らばったままなら
「そのパラメータは選ばれていない (= 性能に効いていない)」と読める。比較のためその中央値を破線で引く。
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import japanize_matplotlib  # noqa: F401  (matplotlib に日本語フォントを登録する)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = ["mnist", "fashion_mnist", "cifar_10"]
DATASET_LABEL = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "cifar_10": "CIFAR-10"}

# 検証済みカテゴリカルパレット (slot 1-3, light mode)
SERIES = {"mnist": "#2a78d6", "fashion_mnist": "#eb6834", "cifar_10": "#1baf7a"}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# 探索範囲と分布 (src/classification.py の suggest_params と一致させる)
SEARCH = {
    "connectivity": (0.05, 0.9, "log", "結合率 (connectivity)"),
    "spectral_radius": (0.5, 0.99, "uniform", "スペクトル半径"),
    "leaky": (0.5, 1.0, "uniform", "リーク率 (leaky)"),
    "beta": (1e-6, 1e3, "log", "リッジ係数 β"),
}


def collect_best(run_root: Path, model: str) -> pd.DataFrame:
    """各 fold で選ばれた最良 trial のパラメータを集める。"""
    rows = []
    for p in glob.glob(str(run_root / f"*/{model}/D*_L1/cv-*_seed-*/best_param.json")):
        info = json.loads(Path(p).with_name("info.json").read_text(encoding="utf-8"))
        params = json.loads(Path(p).read_text(encoding="utf-8"))["best_trial_params"]
        rows.append({"dataset": info["dataset"], "D": info["feature_dim"], **params})
    if not rows:
        raise SystemExit(f"No best_param.json found for {model} under {run_root}.")
    return pd.DataFrame(rows)


def prior_median(lo: float, hi: float, scale: str) -> float:
    return float(np.sqrt(lo * hi)) if scale == "log" else (lo + hi) / 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=str, default="runs/exp1")
    p.add_argument("--model", type=str, default="bi_esn2d")
    p.add_argument("--param", type=str, default="connectivity", choices=sorted(SEARCH))
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    lo, hi, scale, ylabel = SEARCH[args.param]
    df = collect_best(Path(args.run_root), args.model)
    dims = sorted(df["D"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True, facecolor=SURFACE)
    rng = np.random.default_rng(0)

    for ax, ds in zip(axes, DATASETS):
        sub = df[df.dataset == ds]
        color = SERIES[ds]
        ax.set_facecolor(SURFACE)

        # 探索分布の中央値。選ばれた値がここに張り付くなら「効いていない」
        ax.axhline(prior_median(lo, hi, scale), color=INK_MUTED, lw=1.2, ls=(0, (5, 3)), zorder=1)

        # 各 fold で選ばれた値。D ごとに 15 点あるので、重なりを避けて横に散らす
        for D in dims:
            v = sub[sub.D == D][args.param].to_numpy()
            if not len(v):
                continue
            jitter = D * (2 ** (rng.uniform(-0.11, 0.11, len(v))))
            ax.scatter(jitter, v, s=16, color=color, alpha=0.28, linewidths=0, zorder=2)

        med = sub.groupby("D")[args.param].median().reindex(dims)
        ax.plot(dims, med.to_numpy(), color=color, lw=2.0, marker="o", ms=7,
                mfc=SURFACE, mec=color, mew=2.0, zorder=3)

        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        if scale == "log":
            ax.set_yscale("log")
        ax.set_ylim(lo * 0.85, hi * 1.15)
        ax.set_title(DATASET_LABEL[ds], color=INK, fontsize=11, pad=8)
        ax.set_xlabel("ノード数 $D$", color=INK_MUTED, fontsize=10)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    axes[0].set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    # 破線の意味は 1 枚だけ直接ラベルする (凡例箱は作らない)。
    # 中央値の線と重ならないよう、破線の下側・右端に置く
    axes[-1].annotate("探索分布の中央値", xy=(dims[-1], prior_median(lo, hi, scale)),
                      xytext=(-2, -6), textcoords="offset points",
                      color=INK_MUTED, fontsize=8.5, va="top", ha="right")

    fig.suptitle(f"{args.model}: 各 fold で選ばれた{ylabel}（点 = 15 runs、線 = 中央値）",
                 color=INK, fontsize=12.5, y=1.02)
    fig.tight_layout()

    out = Path(args.out or f"runs/exp1_{args.model}_{args.param}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"[INFO] wrote {out} and {out.with_suffix('.pdf')}")

    # 図の数値を表としても出す (色だけに依存させないため)
    print(f"\n{args.model} / {args.param} — 15 runs の中央値")
    print(df.pivot_table(index="dataset", columns="D", values=args.param, aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
