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

sys.path.append(os.getcwd())
sys.path.append("..")
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

    if name == "mnist":
        train = datasets.MNIST(root, train=True, download=True)
        test = datasets.MNIST(root, train=False, download=True)

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
        raise ValueError(f"Unknown dataset: {name}. Use mnist, cifar_10 or stl_10.")

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
def classifier_kwargs(model_type: str, hp: dict[str, Any]) -> dict[str, Any]:
    """共通のハイパーパラメータを、モデルごとの引数名へ振り分ける。

    NOTE: Optuna の探索範囲はリザバー系 (esn / bi_esn / bi_esn2d) に合わせてある。
          reservoir_conv2d の units はリザバー 1 本あたりの値なので --tune_units の範囲
          (128-2048) は大きすぎる。このモデルでは --units を固定して使うこと。
    """
    if model_type in ("esn", "bi_esn", "bi_esn2d"):
        return {
            "patch_sizes": (hp["patch_h"], hp["patch_w"]),
            "units": hp["units"],
            "connectivity": hp["connectivity"],
            "leaky": hp["leaky"],
            "spectral_radius": hp["spectral_radius"],
        }
    if model_type == "conv2d":
        return {"filters": hp["units"], "kernel_size": hp["kernel_size"], "activations": hp["activation"]}
    if model_type == "reservoir_conv2d":
        return {
            "num_reservoirs": hp["num_reservoirs"],
            "units": hp["units"],
            "kernel_size": hp["kernel_size"],
            "connectivity": hp["connectivity"],
            "spectral_radius": hp["spectral_radius"],
        }
    if model_type in ("self_attention", "criss_cross_attention"):
        return {
            "patch_sizes": (hp["patch_h"], hp["patch_w"]),
            "units": hp["units"],
            "n_head": hp["n_head"],
            "temperature": hp["temperature"],
            "activations": hp["activation"],
            "pos_encoding": hp["pos_encoding"],
        }
    if model_type == "positionwise_fcl":
        return {
            "patch_sizes": (hp["patch_h"], hp["patch_w"]),
            "units": hp["units"],
            "activations": hp["activation"],
        }
    if model_type == "fcl":
        # fcl は画像全体を 1 本に潰すのでパッチ化しない
        return {"units": hp["units"], "activations": hp["activation"]}

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
) -> dict[str, float]:
    logits = torch.cat([model(xb).cpu() for xb in iter_batches(x, batch_size, device)]).numpy()
    y_pred = np.argmax(logits, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    top5 = topk_accuracy(logits, y_true, k=min(5, num_classes))
    return {"acc": acc, "macro_f1": macro_f1, "top5": top5}


# -------------------------
# Ridge readout
# -------------------------
@torch.no_grad()
def fit_ridge_readout(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_onehot: torch.Tensor,
    beta: float,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.nn.Module, np.ndarray]:
    """Fit ridge readout and set the readout kernel. Returns (model, W_T).

    NOTE: 正規方程式は float64 で溜める。(D, D) は大きくても 2048^2 なのでコストは無視できる一方、
          float32 のまま全データを足し込むと ZTZ で桁落ちする。
    """
    D = int(model.feature_dim)
    K = int(y_onehot.shape[-1])

    ZTZ = torch.zeros(D, D, dtype=torch.float64, device=device)
    YTZ = torch.zeros(K, D, dtype=torch.float64, device=device)

    start = 0
    for xb in iter_batches(x, batch_size, device):
        yb = y_onehot[start : start + len(xb)].to(device).double()
        zb = model.features(xb).double()  # (B, D)

        ZTZ += zb.T @ zb  # (D, D)
        YTZ += yb.T @ zb  # (K, D)
        start += len(xb)

    reg = beta * torch.eye(D, dtype=torch.float64, device=device)
    W_T = torch.linalg.solve(ZTZ + reg, YTZ.T).float()  # (D, K)

    # 読み出しは LinearReadout (バイアスなし・非訓練) なので kernel だけを差し込む
    model.readout.set_kernel(W_T)

    return model, W_T.cpu().numpy()


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
        choices=["mnist", "cifar_10", "stl_10"],
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
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--n_layer", type=int, default=1, help="Number of stacked layers.")

    # conv2d / reservoir_conv2d のみで使う
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--num_reservoirs", type=int, default=5)
    p.add_argument("--activation", type=str, default="tanh")

    # self_attention / criss_cross_attention のみで使う
    p.add_argument("--n_head", type=int, default=4, help="Number of attention heads (units must be divisible by it).")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Softmax temperature of the scaled dot-product. 大きいほど注意が一様 (= 空間平均) に近づく。",
    )
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
        "--study_storage",
        type=str,
        default="",
        help="Optuna storage URL e.g. sqlite:///study.db (optional).",
    )
    p.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])
    p.add_argument("--pruner", type=str, default="median", choices=["none", "median"])

    # What to tune
    p.add_argument("--tune_model_type", action="store_true")
    p.add_argument("--tune_patch", action="store_true")
    p.add_argument("--tune_units", action="store_true")
    p.add_argument("--tune_connectivity", action="store_true")
    p.add_argument("--tune_leaky", action="store_true")
    p.add_argument("--tune_spectral_radius", action="store_true")
    p.add_argument("--tune_beta", action="store_true")
    p.add_argument("--tune_temperature", action="store_true")

    # Base seed for determinism of non-reservoir randomness (data shuffling etc.)
    p.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base seed (used to derive per-reservoir seed = base_seed + s).",
    )

    # project root for imports
    p.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Project root to add to sys.path so 'import networks' works.",
    )

    return p.parse_args()


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    """Return a dict of hyperparameters for this trial (merging fixed + tuned)."""
    hp: dict[str, Any] = {}

    # model type
    hp["model_type"] = (
        trial.suggest_categorical("model_type", ["esn", "bi_esn", "bi_esn2d"])
        if args.tune_model_type
        else args.model_type
    )

    # patch
    if args.tune_patch:
        if args.dataset == "mnist":
            ph, pw = trial.suggest_categorical("patch", [(2, 2), (4, 4), (7, 7)])
        else:
            ph, pw = trial.suggest_categorical("patch", [(2, 2), (4, 4), (8, 8)])
        hp["patch_h"], hp["patch_w"] = int(ph), int(pw)
    else:
        hp["patch_h"], hp["patch_w"] = int(args.patch_h), int(args.patch_w)

    # units
    hp["units"] = int(trial.suggest_int("units", 128, 2048, log=True)) if args.tune_units else int(args.units)

    # connectivity
    hp["connectivity"] = (
        float(trial.suggest_float("connectivity", 0.05, 0.9, log=True))
        if args.tune_connectivity
        else float(args.connectivity)
    )

    # leaky
    hp["leaky"] = float(trial.suggest_float("leaky", 0.5, 1.0)) if args.tune_leaky else float(args.leaky)

    # spectral radius
    hp["spectral_radius"] = (
        float(trial.suggest_float("spectral_radius", 0.5, 0.99))
        if args.tune_spectral_radius
        else float(args.spectral_radius)
    )

    # ridge beta
    hp["beta"] = float(trial.suggest_float("beta", 1e-5, 1e-3, log=True)) if args.tune_beta else float(args.beta)

    # attention の温度
    # NOTE: リザバー系の spectral_radius / leaky と同じく「どこまで混ぜるか」を決める量なので、
    #       探索対象としての位置づけも同じ
    hp["temperature"] = (
        float(trial.suggest_float("temperature", 0.05, 5.0, log=True))
        if args.tune_temperature
        else float(args.temperature)
    )

    # 探索対象にしていない、モデル固有のパラメータ
    hp["n_layer"] = int(args.n_layer)
    hp["kernel_size"] = int(args.kernel_size)
    hp["num_reservoirs"] = int(args.num_reservoirs)
    hp["activation"] = args.activation
    hp["n_head"] = int(args.n_head)
    hp["pos_encoding"] = args.pos_encoding

    return hp


def main() -> None:
    args = parse_args()

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

    print(
        f"[INFO] dataset={args.dataset} train={len(x_train)} test={len(x_test)} classes={num_classes}",
        flush=True,
    )
    print(
        f"[INFO] model_type(default)={args.model_type} N_seed={args.N_seed} N_cv={args.N_cv} trials={args.n_trials}",
        flush=True,
    )
    print(f"[INFO] device={device} input_shape={(C, H, W)}", flush=True)
    print(f"[INFO] save_root={save_root.resolve()}", flush=True)

    # Optuna common objects
    sampler = (
        optuna.samplers.TPESampler(seed=args.base_seed)
        if args.sampler == "tpe"
        else optuna.samplers.RandomSampler(seed=args.base_seed)
    )
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

            # Study name per (seed, cv)
            study_name = f"{args.dataset}_seed{reservoir_seed}_cv{cv_id}"
            storage = args.study_storage.strip() or None

            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=bool(storage),
            )

            # Objective
            def objective(trial: optuna.Trial) -> float:
                hp = suggest_params(trial, args)

                model = build_classifier(hp)
                model, _ = fit_ridge_readout(
                    model, x_tr, y_tr_oh, beta=hp["beta"], batch_size=args.batch_size, device=device
                )

                val_metrics = evaluate_model(
                    model,
                    x_va,
                    y_va,
                    num_classes=num_classes,
                    batch_size=args.batch_size,
                    device=device,
                )
                score = float(val_metrics[args.optuna_metric])

                print(
                    "seed={} cv={} trial={} val_acc={:.4f} val_f1={:.4f} score={:.4f}".format(
                        reservoir_seed,
                        cv_id,
                        trial.number,
                        val_metrics["acc"],
                        val_metrics["macro_f1"],
                        score,
                    ),
                    flush=True,
                )
                # 次の trial の前にリザバーの重みを解放する
                del model
                torch.cuda.empty_cache()

                # report for pruner
                trial.report(score, step=trial.number)
                return score

            start_t = dt.datetime.now()
            study.optimize(objective, n_trials=args.n_trials)

            best_trial = study.best_trial
            best_hp = suggest_params(best_trial, args)  # will use best params where tuned
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
            best_model, W_T = fit_ridge_readout(
                best_model,
                x_tr,
                y_tr_oh,
                beta=float(best_hp["beta"]),
                batch_size=args.batch_size,
                device=device,
            )

            test_metrics = evaluate_model(
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
                "seed": reservoir_seed,
                "seed_index": s,
                "cv_id": cv_id,
                "N_cv": args.N_cv,
                "N_seed": args.N_seed,
                "train_size": int(len(x_tr)),
                "val_size": int(len(x_va)),
                "test_size": int(len(x_test)),
                "num_classes": int(num_classes),
                "optuna_metric": args.optuna_metric,
                "n_trials": int(args.n_trials),
                "timestamp": timestamp,
                "elapsed_sec": float(elapsed),
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
                    "beta": float(args.beta),
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

            # study artifacts
            save_study_artifacts(study, fold_dir / "study")

            print(f"[seed={reservoir_seed} cv={cv_id}] Saved to {fold_dir}", flush=True)
            print(f"=== END seed={reservoir_seed} cv={cv_id} ===", flush=True)

            del best_model
            torch.cuda.empty_cache()

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
