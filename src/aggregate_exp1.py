"""実験1 (分類性能比較) の結果を集計する。

    python src/aggregate_exp1.py                      # 主表と D 走査を標準出力へ
    python src/aggregate_exp1.py --csv runs/exp1.csv  # 生の 15 runs も CSV に出す

runs/exp1/{dataset}/{model_type}/D{units}_L{n_layer}/cv-*_seed-*/metrics.csv を glob で集める。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 表の並び順。設計書 §1.1 の条件番号に対応する
MODEL_ORDER = [
    # NOTE: raw は特徴抽出を持たない下限。feature_dim は入力次元で固定なので D 走査には参加せず、
    #       表では水平な基準線になる
    "raw",
    "fcl",
    "positionwise_fcl",
    "conv2d",
    "reservoir_conv2d",
    "esn",
    "bi_esn",
    "bi_esn2d",
    "criss_cross_attention",
    "self_attention",
]
DATASET_ORDER = ["mnist", "fashion_mnist", "cifar_10"]


def collect(run_root: Path) -> pd.DataFrame:
    """全 fold の metrics.csv と info.json を 1 つの DataFrame にまとめる。"""
    rows = []
    for metrics_path in sorted(run_root.glob("*/*/D*_L*/cv-*_seed-*/metrics.csv")):
        fold_dir = metrics_path.parent
        info = json.loads((fold_dir / "info.json").read_text(encoding="utf-8"))

        row = pd.read_csv(metrics_path).iloc[0].to_dict()
        # NOTE: 条件はパスではなく info.json から取る。reservoir_conv2d は --units が
        #       リザバー 1 本あたりの値なので、パス中の D と args["units"] が一致しない
        row.update(
            dataset=info["dataset"],
            model_type=info["model_type"],
            feature_dim=info["feature_dim"],
            n_layer=info["n_layer"],
            elapsed_sec=info["elapsed_sec"],
        )
        rows.append(row)

    if not rows:
        raise SystemExit(f"No metrics.csv found under {run_root}.")

    df = pd.DataFrame(rows)
    df["model_type"] = pd.Categorical(df["model_type"], MODEL_ORDER, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    return df.sort_values(["dataset", "model_type", "feature_dim", "seed", "cv_id"])


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(dataset, model_type, feature_dim) ごとに 15 runs の平均と標準偏差を取る。"""
    g = df.groupby(["dataset", "model_type", "feature_dim"], observed=True)[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n"})


def fmt(mean: float, std: float) -> str:
    return f"{100 * mean:5.2f}±{100 * std:4.2f}"


def print_sweep(summary: pd.DataFrame, metric: str) -> None:
    """D 走査の表。条件 x D で、各セルが 15 runs の平均±標準偏差。"""
    dims = sorted(summary["feature_dim"].unique())
    for dataset in summary["dataset"].cat.categories:
        sub = summary[summary["dataset"] == dataset]
        if sub.empty:
            continue
        print(f"\n### {dataset} — test {metric} (%), 15 runs の平均±標準偏差")
        print(f"{'条件':<24}" + "".join(f"{'D=' + str(d):>14}" for d in dims) + f"{'最良 D':>10}")
        for model in summary["model_type"].cat.categories:
            r = sub[sub["model_type"] == model].set_index("feature_dim")
            if r.empty:
                continue
            cells = []
            for d in dims:
                cells.append(fmt(r.loc[d, f"{metric}_mean"], r.loc[d, f"{metric}_std"]) if d in r.index else " " * 11)
            best_d = int(r[f"{metric}_mean"].idxmax())
            print(f"{model:<24}" + "".join(f"{c:>14}" for c in cells) + f"{best_d:>10}")


def print_best_table(summary: pd.DataFrame, metric: str) -> None:
    """主表。条件ごとに最良の D を選び、その値を並べる。"""
    best = summary.loc[summary.groupby(["dataset", "model_type"], observed=True)[f"{metric}_mean"].idxmax()]
    print(f"\n### 主表 — 各条件の最良 D における test {metric} (%)")
    print(f"{'条件':<24}" + "".join(f"{d:>22}" for d in summary['dataset'].cat.categories))
    for model in summary["model_type"].cat.categories:
        cells = []
        for dataset in summary["dataset"].cat.categories:
            r = best[(best["model_type"] == model) & (best["dataset"] == dataset)]
            cells.append(
                f"{fmt(r.iloc[0][f'{metric}_mean'], r.iloc[0][f'{metric}_std'])} (D={int(r.iloc[0]['feature_dim'])})"
                if len(r)
                else ""
            )
        print(f"{model:<24}" + "".join(f"{c:>22}" for c in cells))


def print_paired(df: pd.DataFrame, metric: str, reference: str) -> None:
    """reference 条件との対応比較。

    NOTE: 15 runs は fold 同士が訓練データを共有していて独立ではないため、平均の差に
          通常の t 検定は使えない。同一 (dataset, D, seed, cv_id) での差を取り、
          差が正になった run 数を報告する形にする。
    """
    key = ["dataset", "feature_dim", "seed", "cv_id"]
    ref = df[df["model_type"] == reference][key + [metric]].rename(columns={metric: "ref"})
    merged = df.merge(ref, on=key, how="inner")
    merged["diff"] = merged[metric] - merged["ref"]

    print(f"\n### {reference} との差 (test {metric}, %) — 同一 (dataset, D, seed, fold) で対応を取る")
    print(f"{'条件':<24}{'平均差':>10}{'勝率':>12}{'最良 D での差':>16}")
    for model in df["model_type"].cat.categories:
        m = merged[merged["model_type"] == model]
        if m.empty or model == reference:
            continue
        win = float((m["diff"] > 0).mean())
        by_d = m.groupby("feature_dim", observed=True)["diff"].mean()
        print(f"{model:<24}{100 * m['diff'].mean():>10.2f}{win:>11.1%}{100 * by_d.max():>16.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate experiment 1 results.")
    p.add_argument("--run_root", type=str, default="runs/exp1")
    p.add_argument("--metric", type=str, default="test_acc", choices=["test_acc", "test_macro_f1", "test_top5"])
    p.add_argument("--reference", type=str, default="bi_esn2d", help="対応比較の基準にする条件。")
    p.add_argument("--csv", type=str, default="", help="生の 15 runs を書き出す先 (任意)。")
    args = p.parse_args()

    df = collect(Path(args.run_root))
    summary = summarize(df, args.metric)

    print(f"[INFO] {len(df)} runs / {df['model_type'].nunique()} conditions / {df['dataset'].nunique()} datasets")
    print_best_table(summary, args.metric)
    print_sweep(summary, args.metric)
    print_paired(df, args.metric, args.reference)

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
        print(f"\n[INFO] wrote {out} and {out.with_name(out.stem + '_summary.csv')}")


if __name__ == "__main__":
    main()
