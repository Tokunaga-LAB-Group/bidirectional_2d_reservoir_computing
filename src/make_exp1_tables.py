"""実験1の結果を「データセットごと・行=条件・列=D」の表にする。

    python src/make_exp1_tables.py                       # 標準出力
    python src/make_exp1_tables.py --out runs/exp1_tables.md   # Markdown で保存 (CSV も同時に出る)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aggregate_exp1 import DATASET_ORDER, MODEL_ORDER, collect

# 表に出す条件名 (設計書 §1.1 の並び)
DATASET_LABEL = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "cifar_10": "CIFAR-10"}

LABEL = {
    "raw": "raw (特徴抽出なし)",
    "fcl": "fcl",
    "positionwise_fcl": "positionwise_fcl",
    "conv2d": "conv2d",
    "reservoir_conv2d": "reservoir_conv2d",
    "esn": "esn",
    "bi_esn": "bi_esn",
    "bi_esn2d": "bi_esn2d",
    "criss_cross_attention": "criss_cross_attention",
    "self_attention": "self_attention",
}


def build_tables(df: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame]:
    """dataset -> (行=model_type, 列=D) の表。セルは 15 runs の 平均±標準偏差。"""
    g = df.groupby(["dataset", "model_type", "feature_dim"], observed=True)[metric]
    mean, std, n = g.mean() * 100, g.std() * 100, g.count()

    tables = {}
    for dataset in DATASET_ORDER:
        if dataset not in df["dataset"].unique():
            continue
        rows = {}
        for model in MODEL_ORDER:
            key = (dataset, model)
            if key not in mean.index.droplevel(2).unique():
                continue
            m, s, c = mean.loc[key], std.loc[key], n.loc[key]
            # NOTE: 未実行の (条件, D) は空欄にする。0 を入れると平均や最良 D の判断を誤らせる
            cells = {int(d): (f"{m[d]:.2f}±{s[d]:.2f}" if c[d] > 0 else "") for d in m.index}
            cells["best D"] = int(m.idxmax())
            cells["n"] = int(c.iloc[0])
            rows[LABEL[model]] = cells
        tables[dataset] = pd.DataFrame(rows).T
    return tables


def best_table(df: pd.DataFrame, metric: str) -> str:
    """条件ごとに最良 D を選び、3 データセットを 1 枚に並べる。

    NOTE: raw (特徴抽出なし) を基準にした差も併記する。各条件の値そのものより
          「読み出しだけで到達できる精度から何ポイント足せたか」の方が解釈しやすいため。
    """
    g = df.groupby(["dataset", "model_type", "feature_dim"], observed=True)[metric]
    mean, std = g.mean() * 100, g.std() * 100

    baseline = {ds: mean.loc[(ds, "raw")].iloc[0] for ds in DATASET_ORDER if (ds, "raw") in mean.droplevel(2).index}

    out = [f"\n# 最良 D での比較 — test {metric} (%), 15 runs の平均±標準偏差\n"]
    header = ["条件"] + [f"{DATASET_LABEL[d]}" for d in DATASET_ORDER]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for model in MODEL_ORDER:
        cells = []
        for ds in DATASET_ORDER:
            key = (ds, model)
            if key not in mean.droplevel(2).index:
                cells.append("")
                continue
            m, s_ = mean.loc[key], std.loc[key]
            bd = m.idxmax()
            delta = m[bd] - baseline.get(ds, float("nan"))
            tag = "入力次元" if model == "raw" else f"D={int(bd)}"
            cells.append(f"{m[bd]:.2f}±{s_[bd]:.2f} ({tag}, {delta:+.2f})")
        out.append("| " + " | ".join([LABEL[model]] + cells) + " |")
    out.append("\n括弧内は (最良 D, raw との差)。raw = 特徴抽出を置かず生画素をリッジ回帰に入れた下限。")
    return "\n".join(out) + "\n"


def to_markdown(tables: dict[str, pd.DataFrame], metric: str) -> str:
    out = [f"# 実験1 — test {metric} (%), 15 runs の平均±標準偏差\n"]
    for dataset, t in tables.items():
        out.append(f"\n## {dataset}\n")
        header = ["条件"] + [f"D={c}" if isinstance(c, int) else str(c) for c in t.columns]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for name, row in t.iterrows():
            out.append("| " + " | ".join([name] + [str(v) for v in row]) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=str, default="runs/exp1")
    p.add_argument("--metric", type=str, default="test_acc", choices=["test_acc", "test_macro_f1", "test_top5"])
    p.add_argument("--out", type=str, default="", help="Markdown の保存先。CSV も同じ場所に出す。")
    args = p.parse_args()

    df = collect(Path(args.run_root))
    tables = build_tables(df, args.metric)
    md = best_table(df, args.metric) + to_markdown(tables, args.metric)
    print(md)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        for dataset, t in tables.items():
            t.to_csv(out.with_name(f"{out.stem}_{dataset}.csv"))
        print(f"[INFO] wrote {out} and per-dataset CSVs")


if __name__ == "__main__":
    main()
