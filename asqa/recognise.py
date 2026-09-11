"""L2 hierarchical activity recognition with XGBoost."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier

from asqa import config, context, features as feat
from asqa.preprocess import cached_users, load_cached
from asqa.splits import load_folds


def feature_cache_path(user_id: str) -> Path:
    return config.CACHE_DIR / "features" / f"{user_id}.npz"


def build_features(user_id: str, overwrite: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, coarse_labels, epoch_ts)`` for one user, caching to disk."""
    path = feature_cache_path(user_id)
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as data:
            return data["X"], data["labels"], data["epoch_ts"]

    cache = load_cached(user_id)
    X = feat.extract_batch(cache.windows)
    labels = np.asarray(cache.labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, labels=labels, epoch_ts=cache.epoch_ts)
    return X, labels, cache.epoch_ts


def build_context_features(user_id: str, overwrite: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Context-augmented features for one user, cached to disk."""
    path = config.CACHE_DIR / "context" / f"{user_id}.npz"
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as data:
            return data["X"], data["labels"], data["epoch_ts"]

    X, labels, epoch_ts = build_features(user_id)
    augmented = context.augment(X, epoch_ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=augmented, labels=labels, epoch_ts=epoch_ts)
    return augmented, labels, epoch_ts


def load_users(
    user_ids: list[str], with_context: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate features for several users, tracking who each row came from."""
    Xs, ys, ts, owners = [], [], [], []
    for user_id in user_ids:
        if with_context:
            X, labels, epoch_ts = build_context_features(user_id)
        else:
            X, labels, epoch_ts = build_features(user_id)
        Xs.append(X)
        ys.append(labels)
        ts.append(epoch_ts)
        owners.append(np.full(len(X), user_id))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(owners)


@dataclass
class StandingSplit:
    """Threshold separating standing in place from standing and moving."""

    threshold: float
    n_fitted: int

    def apply(self, X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expand coarse `standing` into the two challenge classes."""
        energy = X[:, feat.BODY_RMS_INDEX]
        out = labels.astype(object).copy()
        provenance = np.full(len(labels), config.PROVENANCE_ANNOTATED, dtype=object)
        standing = labels == "standing"
        out[standing] = np.where(
            energy[standing] <= self.threshold, "standing_in_place", "standing_and_moving"
        )
        provenance[standing] = config.PROVENANCE_DERIVED
        return out.astype(str), provenance.astype(str)


def fit_standing_split(X: np.ndarray, labels: np.ndarray, seed: int = 42) -> StandingSplit:
    energy = X[labels == "standing", feat.BODY_RMS_INDEX].reshape(-1, 1)
    if len(energy) < 20:
        return StandingSplit(threshold=float(np.median(energy)) if len(energy) else 0.0, n_fitted=len(energy))

    log_energy = np.log10(energy + 1e-6)
    mixture = GaussianMixture(n_components=2, random_state=seed, n_init=3).fit(log_energy)
    order = np.argsort(mixture.means_.ravel())
    low, high = mixture.means_.ravel()[order]
    threshold = float(10 ** ((low + high) / 2) - 1e-6)
    return StandingSplit(threshold=threshold, n_fitted=len(energy))


def _xgb(n_classes: int, seed: int, depth: int, estimators: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=estimators,
        max_depth=depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        tree_method="hist",
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        eval_metric="logloss" if n_classes == 2 else "mlogloss",
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
    )


def _weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights, so rare classes are not ignored."""
    classes, counts = np.unique(y, return_counts=True)
    weight = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.asarray([weight[label] for label in y])


@dataclass
class Recogniser:
    stage1: XGBClassifier
    stage2a: XGBClassifier
    stage2b: XGBClassifier
    static_classes: list[str]
    dynamic_classes: list[str]
    standing_split: StandingSplit
    temperature: float = 1.0


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated `P(activity | window)` as (N, 7) in ACTIVITIES order."""
        p_dynamic = self.stage1.predict_proba(X)[:, 1]
        p_static_leaf = self.stage2a.predict_proba(X)
        p_dynamic_leaf = self.stage2b.predict_proba(X)

        out = np.zeros((len(X), len(config.ACTIVITIES)))
        for index, activity in enumerate(self.static_classes):
            out[:, config.ACTIVITY_INDEX[activity]] = (1.0 - p_dynamic) * p_static_leaf[:, index]
        for index, activity in enumerate(self.dynamic_classes):
            out[:, config.ACTIVITY_INDEX[activity]] = p_dynamic * p_dynamic_leaf[:, index]

        total = out.sum(axis=1, keepdims=True)
        out = np.divide(out, total, where=total > 0, out=np.full_like(out, 1.0 / len(config.ACTIVITIES)))
        return _apply_temperature(out, self.temperature)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(config.ACTIVITIES)[np.argmax(self.predict_proba(X), axis=1)]


def train(
    X_train: np.ndarray,
    y_train_coarse: np.ndarray,
    X_val: np.ndarray,
    y_val_coarse: np.ndarray,
    seed: int = 42,
    depth: int = 6,
    estimators: int = 400,
) -> Recogniser:
    """Fit the three nodes plus the derived standing split and calibration."""
    standing_split = fit_standing_split(X_train, y_train_coarse, seed)
    y_train, _ = standing_split.apply(X_train, y_train_coarse)
    y_val, _ = standing_split.apply(X_val, y_val_coarse)

    static_set, dynamic_set = set(config.STATIC_ACTIVITIES), set(config.DYNAMIC_ACTIVITIES)

    b_train = np.isin(y_train, list(dynamic_set)).astype(int)
    b_val = np.isin(y_val, list(dynamic_set)).astype(int)
    stage1 = _xgb(2, seed, depth, estimators)
    stage1.fit(X_train, b_train, sample_weight=_weights(b_train), eval_set=[(X_val, b_val)], verbose=False)

    static_mask = np.isin(y_train, list(static_set))
    static_val_mask = np.isin(y_val, list(static_set))
    static_classes = sorted(set(y_train[static_mask]), key=lambda a: config.ACTIVITY_INDEX[a])
    stage2a = _xgb(len(static_classes), seed, depth, estimators)
    stage2a.fit(
        X_train[static_mask],
        _encode(y_train[static_mask], static_classes),
        sample_weight=_weights(y_train[static_mask]),
        eval_set=[(X_val[static_val_mask], _encode(y_val[static_val_mask], static_classes))],
        verbose=False,
    )

    dynamic_mask = np.isin(y_train, list(dynamic_set))
    dynamic_val_mask = np.isin(y_val, list(dynamic_set))
    dynamic_classes = sorted(set(y_train[dynamic_mask]), key=lambda a: config.ACTIVITY_INDEX[a])
    stage2b = _xgb(len(dynamic_classes), seed, depth, estimators)
    stage2b.fit(
        X_train[dynamic_mask],
        _encode(y_train[dynamic_mask], dynamic_classes),
        sample_weight=_weights(y_train[dynamic_mask]),
        eval_set=[(X_val[dynamic_val_mask], _encode(y_val[dynamic_val_mask], dynamic_classes))],
        verbose=False,
    )

    model = Recogniser(stage1, stage2a, stage2b, static_classes, dynamic_classes, standing_split)
    model.temperature = _fit_temperature(model, X_val, y_val)
    return model


def _encode(y: np.ndarray, classes: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(classes)}
    return np.asarray([lookup[label] for label in y])


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Sharpen or soften a probability vector without changing its ranking."""
    if abs(temperature - 1.0) < 1e-6:
        return probabilities
    scaled = np.exp(np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _fit_temperature(model: Recogniser, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit one scalar temperature on held-out users by minimising NLL."""
    raw = model.predict_proba(X_val)
    truth = np.array([config.ACTIVITY_INDEX.get(label, -1) for label in y_val])
    valid = truth >= 0
    raw, truth = raw[valid], truth[valid]
    if len(truth) == 0:
        return 1.0

    def negative_log_likelihood(temperature: float) -> float:
        if temperature <= 0:
            return np.inf
        scaled = _apply_temperature(raw, temperature)
        return float(-np.mean(np.log(np.clip(scaled[np.arange(len(truth)), truth], 1e-12, 1.0))))

    result = minimize_scalar(negative_log_likelihood, bounds=(0.25, 5.0), method="bounded")
    return float(result.x) if result.success else 1.0


def evaluate(model: Recogniser, X: np.ndarray, y_coarse: np.ndarray) -> dict:
    y_true, provenance = model.standing_split.apply(X, y_coarse)
    y_pred = model.predict(X)
    present = [a for a in config.ACTIVITIES if a in set(y_true) or a in set(y_pred)]
    annotated = provenance == config.PROVENANCE_ANNOTATED
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "accuracy_annotated_only": float((y_pred[annotated] == y_true[annotated]).mean()),
        "report": classification_report(y_true, y_pred, labels=present, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(config.ACTIVITIES)).tolist(),
        "labels": list(config.ACTIVITIES),
    }


def run_fold(fold_index: int, folds: dict, seed: int = 42, save: bool = True) -> dict:
    """Train the context-free and context-aware recognisers for one fold."""
    split = folds["splits"][str(fold_index)]
    metrics: dict = {"fold": fold_index, "split": split}
    trained: dict[str, Recogniser] = {}

    for variant, with_context in (("context_free", False), ("context_aware", True)):
        X_train, y_train, _, _ = load_users(split["train"], with_context)
        X_val, y_val, _, _ = load_users(split["validation"], with_context)
        X_test, y_test, _, _ = load_users(split["test"], with_context)

        if variant == "context_free":
            print(
                f"  fold {fold_index}: train {len(X_train)}, val {len(X_val)}, test {len(X_test)} windows"
            )
        model = train(X_train, y_train, X_val, y_val, seed=seed)
        result = evaluate(model, X_test, y_test)
        result["standing_threshold"] = model.standing_split.threshold
        result["temperature"] = model.temperature
        result["n_features"] = int(X_train.shape[1])
        result["n_train"] = len(X_train)
        result["n_test"] = len(X_test)
        metrics[variant] = result
        trained[variant] = model
        print(
            f"    {variant:<14} {X_train.shape[1]:>3} features  "
            f"accuracy {result['accuracy']:.3f}  macro-F1 {result['macro_f1']:.3f}"
        )

    X_test_plain, y_test_plain, _, _ = load_users(split["test"], with_context=False)
    isolated = context.augment_isolated(X_test_plain)
    truth, _ = trained["context_aware"].standing_split.apply(isolated, y_test_plain)
    predicted = trained["context_aware"].predict(isolated)
    metrics["context_aware_on_isolated_windows"] = float((predicted == truth).mean())
    print(
        f"    context_aware on isolated windows: "
        f"{metrics['context_aware_on_isolated_windows']:.3f} "
        f"(vs context_free {metrics['context_free']['accuracy']:.3f})"
    )

    metrics.update({k: v for k, v in metrics["context_free"].items() if k != "split"})

    if save:
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "context_free": trained["context_free"],
                "context_aware": trained["context_aware"],
                "model": trained["context_free"],
                "feature_names": feat.FEATURE_NAMES,
                "feature_names_context": context.FEATURE_NAMES_FULL,
                "activities": config.ACTIVITIES,
            },
            config.MODEL_DIR / f"recogniser_fold{fold_index}.joblib",
        )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, help="Run one fold only.")
    parser.add_argument("--cv", action="store_true", help="Run every fold.")
    parser.add_argument("--build-features", action="store_true", help="Extract and cache features, then exit.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.build_features:
        for user_id in cached_users():
            X, labels, _ = build_features(user_id, overwrite=True)
            print(f"{user_id[:8]}  {X.shape[0]:>6} windows x {X.shape[1]} features")
        return 0

    folds = load_folds()
    indices = range(folds["n_folds"]) if args.cv else [args.fold if args.fold is not None else 0]

    results = []
    for index in indices:
        print(f"\nFold {index}")
        results.append(run_fold(index, folds, seed=args.seed))

    if len(results) > 1:
        print(f"\n{'='*72}")
        print(f"{'variant':<16}{'accuracy':>18}{'macro-F1':>18}")
        for variant in ("context_free", "context_aware"):
            accuracies = [r[variant]["accuracy"] for r in results]
            f1s = [r[variant]["macro_f1"] for r in results]
            print(
                f"{variant:<16}{np.mean(accuracies):>11.3f} +/- {np.std(accuracies):.3f}"
                f"{np.mean(f1s):>11.3f} +/- {np.std(f1s):.3f}"
            )
        isolated = [r["context_aware_on_isolated_windows"] for r in results]
        print(f"\ncontext_aware on isolated windows: {np.mean(isolated):.3f} (Task 1 guard)")

        for variant in ("context_free", "context_aware"):
            per_class: dict[str, list[float]] = {a: [] for a in config.ACTIVITIES}
            for result in results:
                for activity in config.ACTIVITIES:
                    if activity in result[variant]["report"]:
                        per_class[activity].append(result[variant]["report"][activity]["f1-score"])
            print(f"\n{variant} per-class F1")
            print(f"{'class':<22}{'mean F1':>9}{'std':>8}{'folds':>7}")
            for activity, scores in per_class.items():
                if scores:
                    print(f"{activity:<22}{np.mean(scores):>9.3f}{np.std(scores):>8.3f}{len(scores):>7}")

    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output = config.EVALUATION_DIR / ("recognition_cv.json" if len(results) > 1 else f"recognition_fold{indices[0]}.json")
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    from asqa.recognise import main as _main

    raise SystemExit(_main())
