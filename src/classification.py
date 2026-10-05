from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

# NOTE: シェルスクリプトから任意のカレントディレクトリで起動されるため、リポジトリのルートを
#       __file__ から解決して sys.path に入れる。os.getcwd() 頼みだと 'import networks' が失敗する
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets


# -------------------------
# Reproducibility helpers
# -------------------------
def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Data
# -------------------------
def load_dataset(
    name: str,
    data_root: str,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray, int]:
    """Return x_train, y_train_int, x_test, y_test_int, num_classes.

    x は (N, C, H, W) の uint8 テンソル。バッチに切り出す時点で float [0, 1] に直す
    (STL-10 を float32 で全部持つと 1.4 GB 程度になるため)。
    """
    name = name.lower()
    root = os.path.expanduser(data_root)

    if name in ("mnist", "fashion_mnist"):
        # 両者は .data / .targets の構造が同じなので、データセットクラスだけ差し替える
        dataset_cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
        train = dataset_cls(root, train=True, download=True)
        test = dataset_cls(root, train=False, download=True)

        # (N, H, W) -> (N, 1, H, W)
        x_train, x_test = train.data.unsqueeze(1), test.data.unsqueeze(1)
        y_train, y_test = train.targets, test.targets
        num_classes = 10
    elif name == "cifar_10":
        train = datasets.CIFAR10(root, train=True, download=True)
        test = datasets.CIFAR10(root, train=False, download=True)

        # (N, H, W, C) -> (N, C, H, W)
        x_train = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()
        x_test = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous()
        y_train, y_test = torch.tensor(train.targets), torch.tensor(test.targets)
        num_classes = 10
    elif name == "stl_10":
        train = datasets.STL10(root, split="train", download=True)
        test = datasets.STL10(root, split="test", download=True)

        # STL-10 は最初から (N, C, H, W)
        x_train, x_test = torch.from_numpy(train.data), torch.from_numpy(test.data)
        y_train, y_test = torch.from_numpy(train.labels), torch.from_numpy(test.labels)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {name}. Use mnist, fashion_mnist, cifar_10 or stl_10.")

    y_train = y_train.to(torch.int64).reshape(-1).numpy()
    y_test = y_test.to(torch.int64).reshape(-1).numpy()

    return x_train, y_train, x_test, y_test, num_classes


def iter_batches(x: torch.Tensor, batch_size: int, device: torch.device):
    """uint8 の画像をバッチごとに device へ載せ、float [0, 1] に正規化して流す。"""
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size].to(device).float().div_(255.0)


# -------------------------
# Model
# -------------------------
# 各 model_type が実際に受け取るハイパーパラメータ。--tune の指定ミスを起動時に落とすために使う。
# NOTE: これは「モデルがどの引数を持つか」の定義であって、実験ごとの探索空間ではない。
#       どれを探索するかはシェルスクリプト側 (--tune) で決める
# NOTE: beta はここに含めない。Z^T Z を 1 回作れば格子上の総当たりがほぼ無料なので、
#       Optuna の探索対象にせず select_beta() で検証データから選ぶ
TUNABLE_PARAMS: dict[str, set[str]] = {
    # NOTE: raw は特徴抽出を持たず feature_dim = C*H*W で固定なので、探索するものが無い
    "raw": set(),
    "fcl": {"units", "input_scaling"},
    "positionwise_fcl": {"units", "patch", "input_scaling"},
    # NOTE: kernel_size は探索しない。reservoir_conv2d の原典が K=3 固定であり、
    #       CNN でも stem を除けば K=3 が標準のため。K を振ると受容野が変わり、
    #       実験3 (層数) で受容野の増加が層数の効果と交絡する
    "conv2d": {"units", "input_scaling"},
    # NOTE: input_scaling は標準的な ESN の 3 大ハイパーパラメータの 1 つ
    #       (input scaling / spectral radius / leaking rate)。spectral_radius は W_rec しか
    #       正規化しないので、入力側のスケールは独立に調整する必要がある
    "reservoir_conv2d": {"units", "connectivity", "spectral_radius", "input_scaling"},
    "esn": {"units", "patch", "connectivity", "leaky", "spectral_radius", "input_scaling"},
    "bi_esn": {"units", "patch", "connectivity", "leaky", "spectral_radius", "input_scaling"},
    "bi_esn2d": {"units", "patch", "connectivity", "leaky", "spectral_radius", "input_scaling"},
    # NOTE: attention の softmax 温度は 1/sqrt(d_head) が実装から自動的に決まるため探索しない。
    #       ヘッド数も CCNet 原典に合わせて 1 に固定し、探索対象から外す
    # NOTE: hybrid の深さと構成は --blocks で決まるので探索対象にしない
    "hybrid": {"units", "patch", "connectivity", "leaky", "spectral_radius"},
    "criss_cross_attention": {"units", "patch", "input_scaling"},
    "self_attention": {"units", "patch", "input_scaling"},
}


def validate_tune(model_type: str, tune: list[str]) -> None:
    """--tune に渡された名前が model_type に適用できるかを検証する。

    NOTE: 例えば conv2d に leaky を渡しても classifier_kwargs が拾わないため、検証しないと
          「探索したつもりで既定値のまま走る」という、ログ上は正常に見える失敗になる。
    """
    if model_type not in TUNABLE_PARAMS:
        raise ValueError(f"Unknown model_type: {model_type}. Available: {sorted(TUNABLE_PARAMS)}")

    invalid = sorted(set(tune) - TUNABLE_PARAMS[model_type])
    if invalid:
        raise ValueError(
            f"{model_type} does not take {invalid}. Tunable: {sorted(TUNABLE_PARAMS[model_type])}."
        )


def build_grid(args) -> dict[str, list[float]]:
    """--sampler grid のときの探索格子。param_distributions の範囲をそのまま等分する。

    探索次元が 1 (input_scaling だけ) の条件では、格子を総当たりする方が TPE より優れる。
    決定的で再現可能、かつ「全点を評価した」と言い切れるので探索予算の議論が要らない。
    実測でも 5 点格子と 30 試行 TPE の差は 0.13 pt 以内だった (fcl / conv2d / attention 2 種 /
    positionwise_fcl の MNIST D=512 で確認)。
    """
    import numpy as _np

    grid: dict[str, list[float]] = {}
    for name, dist in param_distributions(args).items():
        lo, hi = dist.low, dist.high
        if getattr(dist, "log", False):
            vals = _np.logspace(_np.log10(lo), _np.log10(hi), args.grid_points)
        else:
            vals = _np.linspace(lo, hi, args.grid_points)
        grid[name] = [float(v) for v in vals]
    return grid


def add_trials_from_csv(study, csv_path, args) -> int:
    """過去の trials.csv を読み、COMPLETE な試行を study に登録して探索を引き継ぐ。

    Optuna の study は in-memory で作っているため、試行回数を増やすと本来は最初からやり直しに
    なる。params と value は trials.csv に残っているので、同じ分布定義で FrozenTrial を組み直せば
    TPE は前回の 30 試行を踏まえて 31 試行目から探索を続けられる。

    NOTE: sampler の RNG 状態までは復元しないので、最初から n_trials で回した場合と同一の系列には
          ならない。「同じ予算で探索した」ことは保たれるが、系列の完全再現には trials.csv が要る。
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"warm start 元が見つからない: {csv_path}")

    dists = param_distributions(args)
    added = 0
    with open(csv_path, newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("state") != "COMPLETE" or not row.get("value"):
                continue
            params, ok = {}, True
            for name in dists:
                v = row.get(f"params_{name}")
                if not v:
                    ok = False
                    break
                params[name] = float(v)
            if not ok:
                continue
            ua = {}
            if row.get("user_attrs_beta"):
                ua["beta"] = float(row["user_attrs_beta"])
            study.add_trial(
                optuna.trial.create_trial(
                    params=params, distributions=dists, value=float(row["value"]), user_attrs=ua
                )
            )
            added += 1
    return added


def classifier_kwargs(model_type: str, hp: dict[str, Any]) -> dict[str, Any]:
    """共通のハイパーパラメータを、モデルごとの引数名へ振り分ける。

    NOTE: Optuna の探索範囲はリザバー系 (esn / bi_esn / bi_esn2d) に合わせてある。
          reservoir_conv2d の units はリザバー 1 本あたりの値で、読み出しに渡る次元は
          feature_dim = 2 * num_reservoirs * units になる。他の条件と D を揃えるには
          --units を D / (2 * num_reservoirs) に設定すること (D=512, num_reservoirs=2 なら 128)。
          --expect_feature_dim で起動時に検証できる。
    """
    if model_type in ("esn", "bi_esn", "bi_esn2d"):
        return {
            "patch_sizes": (hp["patch_h"], hp["patch_w"]),
            "units": hp["units"],
            "connectivity": hp["connectivity"],
            "leaky": hp["leaky"],
            "spectral_radius": hp["spectral_radius"],
            "input_scaling": hp["input_scaling"],
        }
    if model_type == "conv2d":
        return {
            "filters": hp["units"],
            "kernel_size": hp["kernel_size"],
            "input_scaling": hp["input_scaling"],
            "activations": hp["activation"],
        }
    if model_type == "reservoir_conv2d":
        return {
            "num_reservoirs": hp["num_reservoirs"],
            "units": hp["units"],
            "kernel_size": hp["kernel_size"],
            "connectivity": hp["connectivity"],
            "spectral_radius": hp["spectral_radius"],
            "input_scaling": hp["input_scaling"],
        }
    if model_type == "self_attention":
        return {
            "patch_sizes": (hp["patch_h"], hp["patch_w"]),
            "units": hp["units"],
            "input_scaling": hp["input_scaling"],
            "n_head": hp["n_head"],
            "activations": hp["activation"],
            "pos_encoding": hp["pos_encoding"],
        }
    if model_type == "criss_cross_attention":
        # NOTE: CCNet 原典は単一ヘッドで、softmax の温度に相当する量も持たない
        return {
            "patch_sizes": (hp["patch_h"], hp["patch_w"]),
            "units": hp["units"],
            "input_scaling": hp["input_scaling"],
            "activations": hp["activation"],
            "pos_encoding": hp["pos_encoding"],
        }
    if model_type == "positionwise_fcl":
        return {
            "patch_sizes": (hp["patch_h"], hp["patch_w"]),
            "units": hp["units"],
            "input_scaling": hp["input_scaling"],
            "activations": hp["activation"],
        }
    if model_type == "fcl":
        # fcl は画像全体を 1 本に潰すのでパッチ化しない
        return {"units": hp["units"], "input_scaling": hp["input_scaling"], "activations": hp["activation"]}
    if model_type == "hybrid":
        return {
            "blocks": hp["blocks"],
            "patch_sizes": (hp["patch_h"], hp["patch_w"]),
            "units": hp["units"],
            "kernel_size": hp["kernel_size"],
            "connectivity": hp["connectivity"],
            "leaky": hp["leaky"],
            "spectral_radius": hp["spectral_radius"],
            "activations": hp["activation"],
        }
    if model_type == "raw":
        # 特徴抽出器を持たないので渡す引数が無い
        return {}

    raise ValueError(f"Unknown model_type: {model_type}.")


# -------------------------
# Metrics
# -------------------------
def topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    k = int(k)
    if k <= 1:
        return float(np.mean(np.argmax(logits, axis=1) == y_true))
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([int(y_true[i]) in topk[i] for i in range(len(y_true))]))


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: np.ndarray,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    """Return (metrics, logits).

    NOTE: logits は実験5 (誤分類の重なり・混同行列) で使うため呼び出し側へ返す。float16 に落とすと
          top1 と top2 の差が小さいサンプルで argmax が反転する (CIFAR-10 で 10,000 枚中 3 枚) ため、
          保存時も float32 のままにする。
    """
    logits = torch.cat([model(xb).cpu() for xb in iter_batches(x, batch_size, device)]).numpy()
    y_pred = np.argmax(logits, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    top5 = topk_accuracy(logits, y_true, k=min(5, num_classes))
    return {"acc": acc, "macro_f1": macro_f1, "top5": top5}, logits


# -------------------------
# Ridge readout
# -------------------------
@torch.no_grad()
def accumulate_normal_equations(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_onehot: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """リッジ回帰の正規方程式 (Z^T Z, Y^T Z) をバッチごとに溜める。

    NOTE: float64 で溜める。(D, D) は大きくても 2048^2 なのでコストは無視できる一方、
          float32 のまま全データを足し込むと ZTZ で桁落ちする。
    """
    D, K = int(model.feature_dim), int(y_onehot.shape[-1])
    ZTZ = torch.zeros(D, D, dtype=torch.float64, device=device)
    YTZ = torch.zeros(K, D, dtype=torch.float64, device=device)

    start = 0
    for xb in iter_batches(x, batch_size, device):
        yb = y_onehot[start : start + len(xb)].to(device).double()
        zb = model.features(xb).double()  # (B, D)
        ZTZ += zb.T @ zb
        YTZ += yb.T @ zb
        start += len(xb)

    return ZTZ, YTZ


@torch.no_grad()
def extract_features(
    model: torch.nn.Module, x: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    """(N, D) の特徴行列を float64 で返す。beta を振る間は使い回す。"""
    return torch.cat([model.features(xb).double() for xb in iter_batches(x, batch_size, device)])


def solve_ridge(ZTZ: torch.Tensor, YTZ: torch.Tensor, beta: float) -> torch.Tensor:
    """W = (Z^T Z + beta I)^-1 Z^T Y を解く。(D, K) を返す。"""
    I = torch.eye(ZTZ.shape[0], dtype=ZTZ.dtype, device=ZTZ.device)
    return torch.linalg.solve(ZTZ + beta * I, YTZ.T)


def select_beta(
    ZTZ: torch.Tensor,
    YTZ: torch.Tensor,
    z_val: torch.Tensor,
    y_val: np.ndarray,
    betas: np.ndarray,
    metric: str,
    num_classes: int,
) -> tuple[float, float, torch.Tensor]:
    """検証データで最良の beta を選ぶ。Returns (best_beta, best_score, W_T).

    NOTE: beta は探索対象にしない。Z^T Z を 1 回作れば beta を変えても (D, D) の線形解を
          解き直すだけで済み、特徴抽出をやり直す必要がないため、格子上の総当たりがほぼ無料になる。
          以前 beta を Optuna の探索対象にしていたときは、9 桁の範囲を 10 試行で引くため
          当たり外れが大きく、reservoir_conv2d では 15 runs の標準偏差が 4.01 に達していた
          (beta を固定すると 0.17 まで下がる)。総当たりにすればこの分散が消える。
    """
    best = (float(betas[0]), -1.0, None)
    for beta in betas:
        W_T = solve_ridge(ZTZ, YTZ, float(beta))
        logits = (z_val @ W_T).cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
        score = (
            float(accuracy_score(y_val, y_pred))
            if metric == "acc"
            else float(f1_score(y_val, y_pred, average="macro"))
        )
        if score > best[1]:
            best = (float(beta), score, W_T)

    return best


# -------------------------
# Saving
# -------------------------
def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    # stable column order
    cols = [
        "seed",
        "cv_id",
        "val_metric",
        "val_score",
        "test_acc",
        "test_macro_f1",
        "test_top5",
        "best_trial",
        "timestamp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def save_study_artifacts(study: optuna.Study, study_dir: Path) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    # trials.csv
    df = study.trials_dataframe()
    df.to_csv(study_dir / "trials.csv", index=False)
    # best_trial.json
    bt = study.best_trial
    write_json(
        study_dir / "best_trial.json",
        {
            "value": float(bt.value),
            "params": bt.params,
            "number": int(bt.number),
        },
    )


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    # NOTE: --model_type の選択肢をレジストリから引くため、ここで import する
    import networks

    p = argparse.ArgumentParser(description="All-in-one ESN Optuna + CV + seed runner.")

    # Core experiment
    p.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    p.add_argument(
        "--dataset",
        type=str,
        default="cifar_10",
        choices=["mnist", "fashion_mnist", "cifar_10", "stl_10"],
    )
    p.add_argument("--data_root", type=str, default="~/torchvision_datasets", help="Dataset download directory.")
    p.add_argument("--model_type", type=str, default="esn", choices=networks.list_classifiers())
    p.add_argument("--N_cv", type=int, default=5, help="Number of stratified folds.")
    p.add_argument("--N_seed", type=int, default=5, help="Number of reservoir seeds.")
    p.add_argument("--n_trials", type=int, default=30, help="Optuna trials per (seed, fold).")

    # Fixed/default hyperparameters (can be tuned if flags enabled)
    p.add_argument("--patch_h", type=int, default=4)
    p.add_argument("--patch_w", type=int, default=4)
    p.add_argument("--units", type=int, default=512)
    p.add_argument("--connectivity", type=float, default=0.1)
    p.add_argument("--leaky", type=float, default=0.9)
    p.add_argument("--spectral_radius", type=float, default=0.95)
    p.add_argument("--input_scaling", type=float, default=1.0, help="W_in ~ U(-s, s) の s (標準的な ESN の入力スケーリング)。")
    p.add_argument(
        "--beta_grid",
        type=str,
        default="1e-6,1e3,19",
        help="リッジ係数を総当たりする対数格子 'min,max,num'。検証データで最良の値を選ぶ。",
    )
    p.add_argument("--n_layer", type=int, default=1, help="Number of stacked layers.")
    p.add_argument(
        "--expect_feature_dim",
        type=int,
        default=0,
        help="0 以外なら、構築したモデルの feature_dim がこの値と一致するか起動時に検証する。"
        " reservoir_conv2d は feature_dim = 2 * num_reservoirs * units なので --units とずれる。",
    )

    # conv2d / reservoir_conv2d のみで使う
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--num_reservoirs", type=int, default=5)
    p.add_argument("--activation", type=str, default="tanh")

    # self_attention / criss_cross_attention のみで使う
    p.add_argument(
        "--blocks",
        type=str,
        default="bi-fc",
        help="hybrid のブロック構成。'bi-fc-bi-fc' のようにハイフン区切り "
        "(bi=空間混合, fc=チャネル混合, conv=局所混合)。--n_layer と長さを一致させること。",
    )
    p.add_argument("--n_head", type=int, default=1, help="Number of attention heads (self_attention のみ。criss_cross は原典通り単一ヘッド)。")
    p.add_argument(
        "--pos_encoding",
        type=str,
        default="sincos",
        choices=["sincos", "none"],
        help="Positional encoding concatenated to the patches. 'none' は置換不変な対照条件。",
    )

    # Training/eval mechanics
    p.add_argument("--batch_size", type=int, default=256)

    # Optuna objective metric
    p.add_argument(
        "--optuna_metric",
        type=str,
        default="acc",
        choices=["acc", "macro_f1"],
        help="Metric to maximize on validation set within each fold.",
    )

    # Limits for quick tests
    p.add_argument(
        "--limit_train",
        type=int,
        default=0,
        help="Use only first N train samples (debug).",
    )

    # Saving
    p.add_argument("--save_name", type=str, default="runs/exp_allinone")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing fold/seed dirs.")

    # Optional optuna storage (sqlite etc.)
    p.add_argument(
        "--warm_start_from",
        type=str,
        default="",
        help="既存の実行結果ディレクトリ (RUN_ROOT/dataset/model/tag)。各 cv-*_seed-* の"
        " study/trials.csv を読み、その試行を新しい study に登録してから探索を再開する。"
        " Optuna の study は in-memory なので、試行回数を増やすときに前回分を捨てずに済む。"
        " n_trials は「合計」として解釈され、登録済みの数だけ追加試行が減る。",
    )
    p.add_argument(
        "--study_storage",
        type=str,
        default="",
        help="Optuna storage URL e.g. sqlite:///study.db (optional).",
    )
    p.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random", "grid"])
    p.add_argument(
        "--grid_points",
        type=int,
        default=12,
        help="--sampler grid のときの 1 パラメータあたりの格子点数 (対数一様のものは対数等間隔)。"
        " n_trials はこの点数から自動で決まる。",
    )
    p.add_argument("--pruner", type=str, default="median", choices=["none", "median"])

    # What to tune
    # NOTE: 条件ごとに探索対象が違う (設計書 §3.4) ため、boolean フラグを並べるのではなく
    #       名前のリストで受け取る。model_type に存在しないパラメータを渡した場合は
    #       黙って無視せず起動時に落とす (TUNABLE_PARAMS の検証を参照)
    p.add_argument(
        "--tune",
        type=str,
        nargs="*",
        default=[],
        metavar="PARAM",
        help="探索するパラメータ名。例: --tune beta connectivity leaky spectral_radius",
    )

    # Base seed for determinism of non-reservoir randomness (data shuffling etc.)
    p.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base seed (used to derive per-reservoir seed = base_seed + s).",
    )
    p.add_argument(
        "--tpe_seed",
        type=int,
        default=0,
        help="Base seed of the Optuna sampler. fold ごとに tpe_seed + s * N_cv + cv_id を使う。",
    )

    # project root for imports
    p.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Project root to add to sys.path so 'import networks' works.",
    )

    return p.parse_args()


def param_distributions(args: argparse.Namespace) -> dict[str, Any]:
    """--tune で探索されるパラメータの Optuna 分布。suggest_params と同じ定義を持つ。

    warm start (add_trials_from_csv) で FrozenTrial を組み直すのに必要。suggest_params が
    trial.suggest_* を呼ぶ形なので、分布だけを取り出せるようここに二重定義している。
    変更するときは suggest_params 側と必ず揃えること (下の検証が食い違いを検出する)。
    """
    from optuna.distributions import FloatDistribution, IntDistribution

    tune = set(args.tune)
    d: dict[str, Any] = {}
    if "units" in tune:
        d["units"] = IntDistribution(128, 2048, log=True)
    if "connectivity" in tune:
        d["connectivity"] = FloatDistribution(0.01, 1.0, log=True)
    if "leaky" in tune:
        d["leaky"] = FloatDistribution(0.01, 1.0)
    if "spectral_radius" in tune:
        d["spectral_radius"] = FloatDistribution(0.0, 1.0)
    if "input_scaling" in tune:
        d["input_scaling"] = FloatDistribution(1e-2, 1e2, log=True)
    if "patch" in tune:
        raise ValueError("patch は categorical なので warm start 未対応。P は外側のグリッドで扱う")
    return d


def suggest_params(args: argparse.Namespace, trial: optuna.Trial | None = None) -> dict[str, Any]:
    """Return a dict of hyperparameters (merging fixed defaults + tuned values).

    Args:
        trial: None なら探索を一切行わず、すべて args の既定値を返す。
            起動時に feature_dim を確定させるためのモデル構築で使う。
    """
    # trial が無いときは何も探索しない
    tune: set[str] = set(args.tune) if trial is not None else set()
    hp: dict[str, Any] = {"model_type": args.model_type}

    # patch
    if "patch" in tune:
        # 28x28 (MNIST 系) は 7 で、32x32 (CIFAR-10) は 8 で割り切れる
        candidates = [(2, 2), (4, 4), (7, 7)] if args.dataset in ("mnist", "fashion_mnist") else [(2, 2), (4, 4), (8, 8)]
        ph, pw = trial.suggest_categorical("patch", candidates)
        hp["patch_h"], hp["patch_w"] = int(ph), int(pw)
    else:
        hp["patch_h"], hp["patch_w"] = int(args.patch_h), int(args.patch_w)

    # units
    hp["units"] = int(trial.suggest_int("units", 128, 2048, log=True)) if "units" in tune else int(args.units)

    # connectivity
    hp["connectivity"] = (
        float(trial.suggest_float("connectivity", 0.01, 1.0, log=True))
        if "connectivity" in tune
        else float(args.connectivity)
    )

    # leaky
    # NOTE: leaky=0 は h <- (1-0)h + 0*h_tilde で状態が初期値から一切動かず、特徴が全ゼロに縮退する
    #       (実測: std 0.0、実効ランク 0.0)。下限は 0.01 に置く
    hp["leaky"] = float(trial.suggest_float("leaky", 0.01, 1.0)) if "leaky" in tune else float(args.leaky)

    # spectral radius
    # NOTE: 上限 1.0 はエコー状態性の条件 rho(W) < 1 に由来する理論的な制約で、恣意的な打ち切りではない。
    #       rho > 1 まで広げた掃引では 4 ケース中 3 ケースで悪化した (MNIST esn は rho=1.4 で 5.7 pt 崩壊)。
    #       下限 0 は W_rec=0 すなわち「再帰なし」で、ゼロ除算のガードがあり正常に動く
    #       (CIFAR-10 の esn は rho=0.5 が最良だったので、下を開けることの方が効く)
    hp["spectral_radius"] = (
        float(trial.suggest_float("spectral_radius", 0.0, 1.0))
        if "spectral_radius" in tune
        else float(args.spectral_radius)
    )

    # 入力スケーリング
    # NOTE: 標準的な ESN では W_in ~ U(-s, s) の s を調整する。参考実装でもタスクにより
    #       0.1 から 1e4 まで 5 桁の幅がある。
    #
    # NOTE: 上限は当初 1e1 だったが、位置符号を外して以降 attention 系と positionwise_fcl の
    #       最良値が上限近傍に張り付いた (criss_cross 69%, self_attention 57%, positionwise_fcl 45%)
    #       ため 1e2 に広げた。リザバー系は 1e1 でも上下限への張り付きが 0-1% で最適域が内点に
    #       あったので、この拡張は結果に影響しない (runs/exp1_wide は 1e1 で実行済み)
    hp["input_scaling"] = (
        float(trial.suggest_float("input_scaling", 1e-2, 1e2, log=True))
        if "input_scaling" in tune
        else float(args.input_scaling)
    )

    # conv 系のカーネルサイズ
    # NOTE: padding="same" なので、K を変えても特徴マップの解像度は変わらない
    hp["kernel_size"] = (
        int(trial.suggest_categorical("kernel_size", [3, 5, 7])) if "kernel_size" in tune else int(args.kernel_size)
    )

    # 探索対象にしていない、モデル固有のパラメータ
    hp["n_layer"] = int(args.n_layer)
    hp["n_head"] = int(args.n_head)
    hp["blocks"] = args.blocks
    hp["num_reservoirs"] = int(args.num_reservoirs)
    hp["activation"] = args.activation
    hp["pos_encoding"] = args.pos_encoding

    return hp


def main() -> None:
    args = parse_args()

    # 探索対象の指定ミスは、データを読む前に落とす
    validate_tune(args.model_type, args.tune)

    # NOTE: torch は最初に CUDA へ触れた時点で可視デバイスが決まるので、モデル構築より前に設定する
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Environment / imports
    sys.path.append(os.path.abspath(args.project_root))
    sys.path.append(os.getcwd())
    import networks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    x_train, y_train_int, x_test, y_test_int, num_classes = load_dataset(args.dataset, args.data_root)
    if args.limit_train and args.limit_train > 0:
        x_train = x_train[: args.limit_train]
        y_train_int = y_train_int[: args.limit_train]

    C, H, W = x_train.shape[1:]

    save_root = Path(args.save_name)
    save_root.mkdir(parents=True, exist_ok=True)

    # 読み出し入力次元 D を条件間で揃えるための確認。
    # NOTE: reservoir_conv2d だけ feature_dim = 2 * num_reservoirs * units で --units と一致しないため、
    #       ここで実際にモデルを 1 個組んで確かめる (最も遅い esn でも 0.3 秒程度)。
    #       これを省くと --num_reservoirs の指定漏れに気付かないまま数時間走ることになる
    default_hp = suggest_params(args)
    probe = networks.build_classifier(
        args.model_type,
        input_shape=(C, H, W),
        num_classes=num_classes,
        n_layer=default_hp["n_layer"],
        seed=args.base_seed,
        **classifier_kwargs(args.model_type, default_hp),
    )
    feature_dim = int(probe.feature_dim)
    del probe

    if args.expect_feature_dim > 0:
        if "units" in args.tune:
            print(
                f"[WARN] --tune に units が含まれるため feature_dim は trial ごとに変わる。"
                f" --expect_feature_dim の検証を飛ばす。",
                flush=True,
            )
        elif feature_dim != args.expect_feature_dim:
            raise ValueError(
                f"feature_dim={feature_dim} but --expect_feature_dim={args.expect_feature_dim}. "
                f"model_type={args.model_type} units={args.units} num_reservoirs={args.num_reservoirs}."
            )

    print(
        f"[INFO] dataset={args.dataset} train={len(x_train)} test={len(x_test)} classes={num_classes}",
        flush=True,
    )
    print(
        f"[INFO] model_type={args.model_type} feature_dim={feature_dim} n_layer={args.n_layer} "
        f"N_seed={args.N_seed} N_cv={args.N_cv} trials={args.n_trials}",
        flush=True,
    )
    print(f"[INFO] tune={sorted(args.tune) or '(none)'}", flush=True)
    print(f"[INFO] device={device} input_shape={(C, H, W)}", flush=True)
    print(f"[INFO] save_root={save_root.resolve()}", flush=True)

    # beta の格子を作る。Z^T Z を作り直さずに解き直すだけなので総当たりで良い
    lo, hi, num = args.beta_grid.split(",")
    betas = np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(num))

    # NOTE: 探索対象が空の条件 (raw / fcl / conv2d / attention 系など、beta 以外に振る
    #       ハイパーパラメータを持たないもの) では、何回試行しても同じ設定が繰り返されるだけ。
    #       1 試行で探索空間を尽くしているので、その場合は n_trials を 1 に落とす
    n_trials = args.n_trials if args.tune else 1
    if n_trials != args.n_trials:
        print(f"[INFO] --tune が空のため n_trials を {args.n_trials} -> 1 に変更", flush=True)

    print(f"[INFO] beta grid: {betas[0]:.1e} .. {betas[-1]:.1e} ({len(betas)} 点、検証データで選択)", flush=True)

    pruner = None
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=0)

    # Outer loops
    for s in range(args.N_seed):
        reservoir_seed = int(args.base_seed + s)
        set_global_determinism(reservoir_seed)

        skf = StratifiedKFold(n_splits=args.N_cv, shuffle=True, random_state=reservoir_seed)

        print(
            f"\n[SEED] s={s}/{args.N_seed-1} reservoir_seed={reservoir_seed}",
            flush=True,
        )

        # NOTE: 分割はラベルだけで決まるので、画像本体はダミーを渡して不要なコピーを避ける
        for cv_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_train_int)), y_train_int)):
            fold_dir = save_root / f"cv-{cv_id}_seed-{reservoir_seed}"
            if fold_dir.exists() and args.overwrite:
                # remove minimal files only (keep safety)
                for fn in [
                    "info.json",
                    "best_param.json",
                    "metrics.csv",
                    "model_weights.npz",
                    "logits_test.npy",
                ]:
                    p = fold_dir / fn
                    if p.exists():
                        p.unlink()
            fold_dir.mkdir(parents=True, exist_ok=True)

            x_tr, y_tr = x_train[tr_idx], y_train_int[tr_idx]
            x_va, y_va = x_train[va_idx], y_train_int[va_idx]

            y_tr_oh = F.one_hot(torch.from_numpy(y_tr), num_classes).float()

            print(
                f"\n=== START seed={reservoir_seed} cv={cv_id}/{args.N_cv-1} | train={len(x_tr)} val={len(x_va)} ===",
                flush=True,
            )

            # Build function
            def build_classifier(hp: dict[str, Any]) -> torch.nn.Module:
                set_global_determinism(reservoir_seed)

                model = networks.build_classifier(
                    hp["model_type"],
                    input_shape=(C, H, W),
                    num_classes=num_classes,
                    n_layer=hp["n_layer"],
                    seed=reservoir_seed,
                    **classifier_kwargs(hp["model_type"], hp),
                )

                return model.to(device).eval()

            # Study name per (condition, seed, cv)
            # NOTE: model_type / feature_dim / n_layer を含めないと、sqlite storage +
            #       load_if_exists=True のときに別条件の study を読み込んで trial が混線する
            study_name = (
                f"{args.dataset}_{args.model_type}_D{feature_dim}_L{args.n_layer}"
                f"_seed{reservoir_seed}_cv{cv_id}"
            )
            storage = args.study_storage.strip() or None

            # NOTE: sampler は fold ごとに作り直す。TPESampler は内部に RandomState を持つため、
            #       1 個を使い回すと fold 2 の乱数列が fold 1 の続きから始まり、
            #       特定の fold だけ再実行したときに通しで走らせた場合と別の trial 列になる
            tpe_seed = int(args.tpe_seed + s * args.N_cv + cv_id)
            if args.sampler == "grid":
                # NOTE: GridSampler は全点を評価しきると以降の suggest で警告を出すだけなので、
                #       n_trials を格子の総数ちょうどに合わせる (下の n_trials 上書きを参照)
                sampler = optuna.samplers.GridSampler(build_grid(args), seed=tpe_seed)
            elif args.sampler == "tpe":
                sampler = optuna.samplers.TPESampler(seed=tpe_seed)
            else:
                sampler = optuna.samplers.RandomSampler(seed=tpe_seed)

            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=bool(storage),
            )

            # 過去の試行を登録して探索を引き継ぐ (--warm_start_from)
            n_warm = 0
            if args.warm_start_from:
                prev_csv = (
                    Path(args.warm_start_from) / f"cv-{cv_id}_seed-{reservoir_seed}" / "study" / "trials.csv"
                )
                n_warm = add_trials_from_csv(study, prev_csv, args)
                print(
                    f"[WARM] seed={reservoir_seed} cv={cv_id} {prev_csv} から {n_warm} 試行を登録",
                    flush=True,
                )

            # Objective
            def objective(trial: optuna.Trial) -> float:
                hp = suggest_params(args, trial)

                model = build_classifier(hp)
                # NOTE: 特徴抽出は 1 回だけ行い、beta は格子上で総当たりする
                ZTZ, YTZ = accumulate_normal_equations(model, x_tr, y_tr_oh, args.batch_size, device)
                z_va = extract_features(model, x_va, args.batch_size, device)
                beta, score, _ = select_beta(
                    ZTZ, YTZ, z_va, y_va, betas, args.optuna_metric, num_classes
                )
                trial.set_user_attr("beta", beta)

                print(
                    "seed={} cv={} trial={} beta={:.3e} score={:.4f}".format(
                        reservoir_seed, cv_id, trial.number, beta, score
                    ),
                    flush=True,
                )
                # 次の trial の前にリザバーの重みを解放する
                del model, ZTZ, YTZ, z_va
                torch.cuda.empty_cache()

                # report for pruner
                trial.report(score, step=trial.number)
                return score

            start_t = dt.datetime.now()
            # NOTE: n_trials は合計。warm start で登録済みの分だけ追加試行を減らす
            if args.sampler == "grid":
                # 格子の総数 (全パラメータの直積) を試行数にする。--n_trials は無視される
                n_total = 1
                for vals in build_grid(args).values():
                    n_total *= len(vals)
                n_trials = n_total
            remaining = max(0, n_trials - n_warm)
            if remaining:
                study.optimize(objective, n_trials=remaining)

            best_trial = study.best_trial
            best_hp = suggest_params(args)  # 既定値で埋めてから、探索された値で上書きする
            best_hp["beta"] = float(best_trial.user_attrs.get("beta", betas[0]))
            # Override with actual best params (they are a subset)
            for k, v in best_trial.params.items():
                if k == "patch":
                    best_hp["patch_h"], best_hp["patch_w"] = int(v[0]), int(v[1])
                else:
                    best_hp[k] = v

            print(
                f"[seed={reservoir_seed} cv={cv_id}] BEST val_score={best_trial.value:.4f} params={best_trial.params}",
                flush=True,
            )

            # Refit best on fold-train, evaluate on test
            best_model = build_classifier(best_hp)
            ZTZ, YTZ = accumulate_normal_equations(best_model, x_tr, y_tr_oh, args.batch_size, device)
            z_va = extract_features(best_model, x_va, args.batch_size, device)
            best_beta, _, W_T64 = select_beta(ZTZ, YTZ, z_va, y_va, betas, args.optuna_metric, num_classes)
            best_hp["beta"] = best_beta
            W_T = W_T64.float().cpu().numpy()
            best_model.readout.set_kernel(W_T64.float())
            del ZTZ, YTZ, z_va

            test_metrics, test_logits = evaluate_model(
                best_model,
                x_test,
                y_test_int,
                num_classes=num_classes,
                batch_size=args.batch_size,
                device=device,
            )

            end_t = dt.datetime.now()
            elapsed = (end_t - start_t).total_seconds()

            print(
                f"[seed={reservoir_seed} cv={cv_id}] TEST acc={test_metrics['acc']:.4f} "
                + f"f1={test_metrics['macro_f1']:.4f} top5={test_metrics['top5']:.4f}",
                flush=True,
            )

            # Save artifacts
            timestamp = end_t.isoformat()
            info = {
                "dataset": args.dataset,
                "model_type": args.model_type,
                "feature_dim": feature_dim,
                "n_layer": int(args.n_layer),
                "tune": sorted(args.tune),
                "seed": reservoir_seed,
                "seed_index": s,
                "tpe_seed": tpe_seed,
                "study_name": study_name,
                "cv_id": cv_id,
                "N_cv": args.N_cv,
                "N_seed": args.N_seed,
                "train_size": int(len(x_tr)),
                "val_size": int(len(x_va)),
                "test_size": int(len(x_test)),
                "num_classes": int(num_classes),
                "optuna_metric": args.optuna_metric,
                "n_trials": int(n_trials),
                "beta_grid": args.beta_grid,
                "timestamp": timestamp,
                "elapsed_sec": float(elapsed),
                # NOTE: 解決後の引数を丸ごと残す。設定ファイルを別に持って保存するより、
                #       CLI での上書きまで反映された「実際に使われた値」の方が確実な記録になる。
                #       モデルへ渡された引数は classifier_kwargs(model_type, best_param.json の
                #       resolved_hyperparams) で再現できる
                "args": vars(args),
            }
            write_json(fold_dir / "info.json", info)

            # Save best parameters (best_hp + fixed)
            best_param = {
                "best_value": float(best_trial.value),
                "best_trial_number": int(best_trial.number),
                "best_trial_params": best_trial.params,
                "resolved_hyperparams": best_hp,
                "fixed_defaults": {
                    "patch_h": int(args.patch_h),
                    "patch_w": int(args.patch_w),
                    "units": int(args.units),
                    "connectivity": float(args.connectivity),
                    "leaky": float(args.leaky),
                    "spectral_radius": float(args.spectral_radius),
                    "beta_grid": args.beta_grid,
                    "model_type": args.model_type,
                },
            }
            write_json(fold_dir / "best_param.json", best_param)

            # metrics.csv
            write_metrics_csv(
                fold_dir / "metrics.csv",
                {
                    "seed": reservoir_seed,
                    "cv_id": cv_id,
                    "val_metric": args.optuna_metric,
                    "val_score": float(best_trial.value),
                    "test_acc": test_metrics["acc"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_top5": test_metrics["top5"],
                    "best_trial": int(best_trial.number),
                    "timestamp": timestamp,
                },
            )

            # model_weights.npz (readout only)
            np.savez(fold_dir / "model_weights.npz", W_T=W_T)

            # logits_test.npy — 実験5 (誤分類の重なり・混同行列) で使う
            np.save(fold_dir / "logits_test.npy", test_logits.astype(np.float32))

            # study artifacts
            save_study_artifacts(study, fold_dir / "study")

            print(f"[seed={reservoir_seed} cv={cv_id}] Saved to {fold_dir}", flush=True)
            print(f"=== END seed={reservoir_seed} cv={cv_id} ===", flush=True)

            del best_model
            torch.cuda.empty_cache()

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
