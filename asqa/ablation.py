"""Ablations of the recogniser on the same 5 user-disjoint folds.

Each row changes one thing relative to the deployed per-window model
(hierarchical XGBoost, early fusion of acc + gyro, temporal context, time of
day, off-body label cleaning). Every row is scored on raw test labels.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asqa import config, context, features as feat
from asqa import recognise as R
from asqa.splits import load_folds

NAMES = list(context.FEATURE_NAMES_FULL)
SHARED = np.array([n.startswith(("ctx_gap", "tod_")) or "neighbours" in n for n in NAMES])
CROSS = np.array(["acc_gyro" in n for n in NAMES])
GYRO = np.array(["gyro" in n for n in NAMES]) & ~CROSS
ACC_COLUMNS = ~GYRO & ~CROSS
GYRO_COLUMNS = (GYRO | SHARED) & ~CROSS
NO_TOD = ~np.array([n.startswith("tod_") for n in NAMES])
BASE = np.arange(len(NAMES)) < feat.N_FEATURES
CLASSES = list(config.ACTIVITIES)


def _full(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    out = np.zeros((len(proba), len(CLASSES)))
    for i, c in enumerate(classes):
        out[:, CLASSES.index(c)] = proba[:, i]
    return out


def flat_xgb(Xt, yt, Xv, yv):
    classes = sorted(set(yt), key=CLASSES.index)
    model = R._xgb(len(classes), 42, 6, 400)
    keep = np.isin(yv, classes)
    model.fit(Xt, R._encode(yt, classes), sample_weight=R._weights(yt),
              eval_set=[(Xv[keep], R._encode(yv[keep], classes))], verbose=False)
    return lambda X: _full(model.predict_proba(X), classes)


def logreg(Xt, yt):
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(Xt, yt)
    return lambda X: _full(model.predict_proba(X), list(model.classes_))


def hierarchical(Xt, yt, Xv, yv):
    return R.train(Xt, yt, Xv, yv).predict_proba


def metrics(proba: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(CLASSES)[proba.argmax(1)]
    return (
        float((pred == y).mean()),
        float(balanced_accuracy_score(y, pred)),
        float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
    )


def run_fold(fold: int, folds: dict) -> dict[str, tuple[float, float, float]]:
    split = folds["splits"][str(fold)]
    Xt_raw, yt_raw, _, _ = R.load_users(split["train"], True)
    Xv_raw, yv_raw, _, _ = R.load_users(split["validation"], True)
    Xs, ys, _, _ = R.load_users(split["test"], True)

    threshold = R.off_body_threshold(Xt_raw, yt_raw)
    kt, kv = R.clean_mask(Xt_raw, yt_raw, threshold), R.clean_mask(Xv_raw, yv_raw, threshold)
    Xt, yt, Xv, yv = Xt_raw[kt], yt_raw[kt], Xv_raw[kv], yv_raw[kv]

    out = {}
    out["deployed: hierarchical XGBoost"] = metrics(hierarchical(Xt, yt, Xv, yv)(Xs), ys)
    out["flat XGBoost"] = metrics(flat_xgb(Xt, yt, Xv, yv)(Xs), ys)
    out["logistic regression (our features)"] = metrics(logreg(Xt, yt)(Xs), ys)
    out["no label cleaning"] = metrics(hierarchical(Xt_raw, yt_raw, Xv_raw, yv_raw)(Xs), ys)
    out["no time of day"] = metrics(hierarchical(Xt[:, NO_TOD], yt, Xv[:, NO_TOD], yv)(Xs[:, NO_TOD]), ys)
    out["no context, no time of day"] = metrics(hierarchical(Xt[:, BASE], yt, Xv[:, BASE], yv)(Xs[:, BASE]), ys)

    acc = flat_xgb(Xt[:, ACC_COLUMNS], yt, Xv[:, ACC_COLUMNS], yv)
    gyro = flat_xgb(Xt[:, GYRO_COLUMNS], yt, Xv[:, GYRO_COLUMNS], yv)
    pa, pg = acc(Xs[:, ACC_COLUMNS]), gyro(Xs[:, GYRO_COLUMNS])
    out["accelerometer only"] = metrics(pa, ys)
    out["gyroscope only"] = metrics(pg, ys)
    out["late fusion, average (LFA)"] = metrics((pa + pg) / 2, ys)
    stack = LogisticRegression(class_weight="balanced", max_iter=2000).fit(
        np.hstack([acc(Xv[:, ACC_COLUMNS]), gyro(Xv[:, GYRO_COLUMNS])]), yv
    )
    out["late fusion, learned (LFL)"] = metrics(_full(stack.predict_proba(np.hstack([pa, pg])), list(stack.classes_)), ys)
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    folds = load_folds()
    per_fold = []
    for fold in range(folds["n_folds"]):
        print(f"Fold {fold}", flush=True)
        per_fold.append(run_fold(fold, folds))

    print(f"\n{'variant':<38}{'accuracy':>16}{'balanced acc':>16}{'macro-F1':>16}")
    summary = {}
    for name in per_fold[0]:
        values = np.array([f[name] for f in per_fold])
        summary[name] = {"mean": values.mean(0).tolist(), "std": values.std(0).tolist()}
        print(f"{name:<38}" + "".join(f"{m:>9.3f} ± {s:.3f}" for m, s in zip(values.mean(0), values.std(0))))

    output = config.EVALUATION_DIR / "ablation_cv.json"
    output.write_text(json.dumps({"metrics": ["accuracy", "balanced_accuracy", "macro_f1"],
                                  "summary": summary, "per_fold": per_fold}, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
