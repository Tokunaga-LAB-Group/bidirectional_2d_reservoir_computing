#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from typing import Any, Dict, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.getcwd())
sys.path.append("..")
warnings.filterwarnings("ignore")

import numpy as np
import optuna
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras


# -------------------------
# Reproducibility helpers
# -------------------------
def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -------------------------
# Data
# -------------------------
def load_dataset(
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return x_train, y_train_int, x_test, y_test_int, num_classes."""
    name = name.lower()
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train[..., None]
        x_test = x_test[..., None]
        num_classes = 10
    elif name == "cifar_10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        num_classes = 10
    elif name == "stl_10":
        ds_train = tfds.load("stl10", split="train", as_supervised=True, batch_size=-1)
        ds_test = tfds.load("stl10", split="test", as_supervised=True, batch_size=-1)

        (x_train, y_train) = tfds.as_numpy(ds_train)
        (x_test, y_test) = tfds.as_numpy(ds_test)

        # 念のため dtype を揃える（STL-10 は uint8 画像・int64 ラベルになりがち）
        x_train = x_train.astype(np.uint8)
        x_test = x_test.astype(np.uint8)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {name}. Use mnist or cifar10.")

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.astype("int64").reshape(-1)
    y_test = y_test.astype("int64").reshape(-1)
    return x_train, y_train, x_test, y_test, num_classes


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


def evaluate_model(
    model: keras.Model,
    x: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
    batch_size: int,
) -> Dict[str, float]:
    logits = model.predict(x, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(logits, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    top5 = topk_accuracy(logits, y_true, k=min(5, num_classes))
    return {"acc": acc, "macro_f1": macro_f1, "top5": top5}


# -------------------------
# Ridge readout
# -------------------------
def fit_ridge_readout(
    model: keras.Model,
    x: np.ndarray,
    y_onehot: np.ndarray,
    beta: float,
    batch_size: int,
) -> Tuple[keras.Model, np.ndarray]:
    """Fit ridge readout and set the last Dense kernel. Returns (model, W_T)."""
    # Ensure model is built
    _ = model(np.empty((1,) + x.shape[1:], dtype=np.float32), training=False)

    # Feature extractor: penultimate layer
    fe = keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # Determine dimensions
    z0 = fe(tf.convert_to_tensor(x[:1], dtype=tf.float32), training=False)
    D = int(z0.shape[-1])
    K = int(y_onehot.shape[-1])

    ZTZ = tf.zeros((D, D), dtype=tf.float32)
    YTZ = tf.zeros((K, D), dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((x, y_onehot)).batch(batch_size)
    for xb, yb in ds:
        xb = tf.cast(xb, tf.float32)
        yb = tf.cast(yb, tf.float32)
        zb = fe(xb, training=False)  # (B, D)
        ZTZ += tf.matmul(zb, zb, transpose_a=True)  # (D, D)
        YTZ += tf.matmul(yb, zb, transpose_a=True)  # (K, D)

    reg = beta * tf.eye(D, dtype=tf.float32)
    W_T = tf.linalg.solve(ZTZ + reg, tf.transpose(YTZ))  # (D, K)

    # Last layer is Dense(use_bias=False) => only kernel
    model.layers[-1].set_weights([W_T.numpy()])
    return model, W_T.numpy()


# -------------------------
# Saving
# -------------------------
def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_metrics_csv(path: Path, row: Dict[str, Any]) -> None:
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
    p = argparse.ArgumentParser(description="All-in-one ESN Optuna + CV + seed runner.")

    # Core experiment
    p.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    p.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar_10", "stl_10"],
    )
    p.add_argument(
        "--model_type", type=str, default="esn", choices=["esn", "bi_esn", "bi_esn2d"]
    )
    p.add_argument("--N_cv", type=int, default=5, help="Number of stratified folds.")
    p.add_argument("--N_seed", type=int, default=5, help="Number of reservoir seeds.")
    p.add_argument(
        "--n_trials", type=int, default=30, help="Optuna trials per (seed, fold)."
    )

    # Fixed/default hyperparameters (can be tuned if flags enabled)
    p.add_argument("--patch_h", type=int, default=4)
    p.add_argument("--patch_w", type=int, default=4)
    p.add_argument("--units", type=int, default=512)
    p.add_argument("--connectivity", type=float, default=0.1)
    p.add_argument("--leaky", type=float, default=0.9)
    p.add_argument("--spectral_radius", type=float, default=0.95)
    p.add_argument("--beta", type=float, default=1e-3)

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
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing fold/seed dirs."
    )

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
        help="Project root to add to sys.path so 'import models' works.",
    )

    return p.parse_args()


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, Any]:
    """Return a dict of hyperparameters for this trial (merging fixed + tuned)."""
    hp: Dict[str, Any] = {}

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
    hp["units"] = (
        int(trial.suggest_int("units", 128, 2048, log=True))
        if args.tune_units
        else int(args.units)
    )

    # connectivity
    hp["connectivity"] = (
        float(trial.suggest_float("connectivity", 0.05, 0.9, log=True))
        if args.tune_connectivity
        else float(args.connectivity)
    )

    # leaky
    hp["leaky"] = (
        float(trial.suggest_float("leaky", 0.5, 1.0))
        if args.tune_leaky
        else float(args.leaky)
    )

    # spectral radius
    hp["spectral_radius"] = (
        float(trial.suggest_float("spectral_radius", 0.5, 0.99))
        if args.tune_spectral_radius
        else float(args.spectral_radius)
    )

    # ridge beta
    hp["beta"] = (
        float(trial.suggest_float("beta", 1e-5, 1e-3, log=True))
        if args.tune_beta
        else float(args.beta)
    )

    return hp


def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Environment / imports
    sys.path.append(os.path.abspath(args.project_root))
    sys.path.append(os.getcwd())
    import models  # noqa: F401

    # Data
    x_train, y_train_int, x_test, y_test_int, num_classes = load_dataset(args.dataset)
    if args.limit_train and args.limit_train > 0:
        x_train = x_train[: args.limit_train]
        y_train_int = y_train_int[: args.limit_train]

    H, W, C = x_train.shape[1:]

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

        skf = StratifiedKFold(
            n_splits=args.N_cv, shuffle=True, random_state=reservoir_seed
        )

        print(
            f"\n[SEED] s={s}/{args.N_seed-1} reservoir_seed={reservoir_seed}",
            flush=True,
        )

        for cv_id, (tr_idx, va_idx) in enumerate(skf.split(x_train, y_train_int)):
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

            y_tr_oh = keras.utils.to_categorical(y_tr, num_classes).astype("float32")

            print(
                f"\n=== START seed={reservoir_seed} cv={cv_id}/{args.N_cv-1} | train={len(x_tr)} val={len(x_va)} ===",
                flush=True,
            )

            # Build function
            def build_classifier(hp: Dict[str, Any]) -> keras.Model:
                set_global_determinism(reservoir_seed)

                return models.model.get_classifier(
                    input_shape=(H, W, C),
                    num_classes=num_classes,
                    patch_sizes=(hp["patch_h"], hp["patch_w"]),
                    model_type=hp["model_type"],
                    units=hp["units"],
                    connectivity=hp["connectivity"],
                    leaky=hp["leaky"],
                    spectral_radius=hp["spectral_radius"],
                    seed=reservoir_seed,
                )

            # Study name per (seed, cv)
            study_name = f"{args.dataset}_seed{reservoir_seed}_cv{cv_id}"
            storage = args.study_storage.strip() or None
            if storage is not None and "study_name" in storage:
                # ignore; users should set storage URL only
                pass

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
                    model, x_tr, y_tr_oh, beta=hp["beta"], batch_size=args.batch_size
                )

                val_metrics = evaluate_model(
                    model,
                    x_va,
                    y_va,
                    num_classes=num_classes,
                    batch_size=args.batch_size,
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
                # clear session to reduce memory growth
                keras.backend.clear_session()

                # report for pruner
                trial.report(score, step=trial.number)
                return score

            start_t = dt.datetime.now()
            study.optimize(objective, n_trials=args.n_trials)

            best_trial = study.best_trial
            best_hp = suggest_params(
                best_trial, args
            )  # will use best params where tuned
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
            )

            test_metrics = evaluate_model(
                best_model,
                x_test,
                y_test_int,
                num_classes=num_classes,
                batch_size=args.batch_size,
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

            # clear session after fold
            keras.backend.clear_session()

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
