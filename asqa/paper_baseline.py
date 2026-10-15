"""Vaizman et al. (2017) baseline, reproduced on our users, folds and windows.

Their method, as described in the paper's supplement:
  - the published per-sensor features (data/*.features_labels.csv);
  - per-sensor standardisation with training statistics;
  - one binary logistic regression per label, balanced class weights, C chosen
    from {0.001 .. 100} by F1 on a random stratified third of the training data,
    then refitted on all of it;
  - fusion: EF (concatenate), LFA (average sensor probabilities),
    LFL (second logistic regression over sensor probabilities).

What is ours, for a like-for-like comparison: the same 35 users, the same 5
user-disjoint folds (train + validation users train, test users test), the same
windows our pipeline keeps, and the same 6 raw labels. A window's predicted
class is the argmax of the per-label probabilities. Labels are not cleaned:
the paper trained on the published labels.

Missing sensors (the watch is absent for 35% of windows): standardised values
are imputed with 0 (the training mean) for EF; a single-sensor model trains only
on rows where its sensor exists; LFA averages the sensors present; LFL feeds a
neutral 0.5 for an absent sensor.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from asqa import config
from asqa.recognise import build_features
from asqa.splits import load_folds

SENSORS: dict[str, tuple[str, ...]] = {
    "Acc": ("raw_acc:",),
    "Gyro": ("proc_gyro:",),
    "WAcc": ("watch_acceleration:",),
    "Loc": ("location:", "location_quick_features:"),
    "Aud": ("audio_naive:",),
    "PS": ("discrete:",),
}
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SENSOR_SETS = {"phone_acc_gyro": ("Acc", "Gyro"), "all_six": tuple(SENSORS)}


def paper_features(csv_path, epoch_ts: np.ndarray) -> dict[str, np.ndarray]:
    """The published per-sensor features for the given windows."""
    frame = pd.read_csv(csv_path).set_index("timestamp")
    rows = frame.loc[epoch_ts]
    return {
        sensor: rows[[c for c in frame.columns if c.startswith(prefixes)]].to_numpy(dtype=np.float64)
        for sensor, prefixes in SENSORS.items()
    }


def load_user(user_id: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Paper features for exactly the windows our pipeline keeps, plus our labels."""
    _, labels, epoch_ts = build_features(user_id)
    return paper_features(config.DATA_DIR / f"{user_id}{config.LABEL_SUFFIX}", epoch_ts), labels


def load_users(user_ids: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    loaded = [load_user(u) for u in user_ids]
    blocks = {s: np.vstack([b[s] for b, _ in loaded]) for s in SENSORS}
    return blocks, np.concatenate([y for _, y in loaded])


def standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return (
        np.nan_to_num((train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num((test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _logreg(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=1000)


def fit_binary(X: np.ndarray, y: np.ndarray, seed: int = 42) -> LogisticRegression:
    """The paper's per-label procedure: grid-search C on an internal third, then refit."""
    X_in, X_val, y_in, y_val = train_test_split(X, y, test_size=1 / 3, stratify=y, random_state=seed)
    best = max(C_GRID, key=lambda C: f1_score(y_val, _logreg(C).fit(X_in, y_in).predict(X_val), zero_division=0))
    return _logreg(best).fit(X, y)


def per_label_probabilities(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, classes: list[str]):
    """(train, test) probability matrices, one column per label, from independent binary models."""
    models = Parallel(n_jobs=-1)(delayed(fit_binary)(X_train, (y_train == c).astype(int)) for c in classes)
    return (
        np.column_stack([m.predict_proba(X_train)[:, 1] for m in models]),
        np.column_stack([m.predict_proba(X_test)[:, 1] for m in models]),
    )


def run_fold(fold_index: int, folds: dict) -> dict:
    split = folds["splits"][str(fold_index)]
    train_blocks, y_train = load_users(split["train"] + split["validation"])
    test_blocks, y_test = load_users(split["test"])
    probabilities = fit_and_predict(train_blocks, y_train, test_blocks, tag=f"fold {fold_index}")
    classes = list(config.ACTIVITIES)
    return {"fold": fold_index, "results": {name: score(p, y_test, classes) for name, p in probabilities.items()}}


def fit_and_predict(
    train_blocks: dict[str, np.ndarray], y_train: np.ndarray, test_blocks: dict[str, np.ndarray], tag: str = ""
) -> dict[str, np.ndarray]:
    """Every single-sensor and fused system's (N_test, 6) label probabilities."""
    classes = list(config.ACTIVITIES)
    n_test = len(next(iter(test_blocks.values())))

    present_train = {s: ~np.isnan(train_blocks[s]).all(axis=1) for s in SENSORS}
    present_test = {s: ~np.isnan(test_blocks[s]).all(axis=1) for s in SENSORS}
    scaled = {s: standardise(train_blocks[s], test_blocks[s]) for s in SENSORS}

    # Single-sensor models, each trained only where its sensor exists.
    single: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sensor in SENSORS:
        Xtr, Xte = scaled[sensor]
        rows = present_train[sensor]
        p_rows, p_test = per_label_probabilities(Xtr[rows], y_train[rows], Xte, classes)
        p_train = np.full((len(y_train), len(classes)), 0.5)
        p_train[rows] = p_rows
        p_test[~present_test[sensor]] = 0.5
        single[sensor] = (p_train, p_test)
        print(f"    {tag}: {sensor} done", flush=True)

    probabilities: dict[str, np.ndarray] = {s: single[s][1] for s in SENSORS}
    for set_name, sensors in SENSOR_SETS.items():
        Xtr = np.hstack([scaled[s][0] for s in sensors])
        Xte = np.hstack([scaled[s][1] for s in sensors])
        probabilities[f"EF_{set_name}"] = per_label_probabilities(Xtr, y_train, Xte, classes)[1]

        stacked = np.stack([single[s][1] for s in sensors])
        weights = np.stack([present_test[s] for s in sensors]).astype(float)[:, :, None]
        probabilities[f"LFA_{set_name}"] = (stacked * weights).sum(0) / np.maximum(weights.sum(0), 1)

        lfl = np.zeros((n_test, len(classes)))
        for k, c in enumerate(classes):
            inputs_train = np.column_stack([single[s][0][:, k] for s in sensors])
            inputs_test = np.column_stack([single[s][1][:, k] for s in sensors])
            lfl[:, k] = _logreg(1.0).fit(inputs_train, (y_train == c).astype(int)).predict_proba(inputs_test)[:, 1]
        probabilities[f"LFL_{set_name}"] = lfl
    return probabilities


def score(probabilities: np.ndarray, y_true: np.ndarray, classes: list[str]) -> dict:
    y_pred = np.asarray(classes)[probabilities.argmax(axis=1)]
    # The paper's own metric: each label's binary decision (P > 0.5) scored by BA.
    binary_ba = {
        c: float(balanced_accuracy_score(y_true == c, probabilities[:, k] > 0.5))
        for k, c in enumerate(classes)
        if (y_true == c).any()
    }
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "binary_ba_per_label": binary_ba,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "labels": classes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, help="Run one fold only.")
    args = parser.parse_args()

    folds = load_folds()
    indices = [args.fold] if args.fold is not None else range(folds["n_folds"])
    results = []
    for index in indices:
        print(f"Fold {index}", flush=True)
        results.append(run_fold(index, folds))

    print(f"\n{'system':<24}{'accuracy':>16}{'balanced acc':>16}{'macro-F1':>16}")
    for name in results[0]["results"]:
        cells = ""
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = [r["results"][name][key] for r in results]
            cells += f"{np.mean(values):>9.3f} ± {np.std(values):.3f}"
        print(f"{name:<24}{cells}")

    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output = config.EVALUATION_DIR / ("paper_baseline_cv.json" if args.fold is None else f"paper_baseline_fold{args.fold}.json")
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
