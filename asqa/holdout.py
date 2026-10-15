"""Held-out evaluation on the 5 users that no fold ever trained or tuned on.

Each fold's saved model is scored on the 5 new users, and so is the paper's
method (asqa.paper_baseline) trained on that same fold's train + validation
users, so both sides see identical training data. Results are mean +/- std over
the 5 fold models.

Only labelled windows are kept, and temporal context is built over them alone,
exactly as for the cached training users, so the feature distribution matches.
Labels are raw (uncleaned).
"""

from __future__ import annotations

import json

import joblib
import numpy as np

from asqa import config, context, features as feat
from asqa.decode import decode_labels
from asqa.paper_baseline import fit_and_predict, load_users as load_paper_users, paper_features, score
from asqa.pipeline import Pipeline, _pair_raw_directory
from asqa.preprocess import load_labels
from asqa.splits import load_folds

NEW_USERS_DIR = config.DATA_DIR / "5 new users"
CACHE = config.CACHE_DIR / "holdout"


def new_users() -> list[str]:
    return sorted(p.name.replace(config.LABEL_SUFFIX, "") for p in NEW_USERS_DIR.glob(f"*{config.LABEL_SUFFIX}"))


def load_new_user(user_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X_context, labels, epoch_ts)`` for one new user's labelled windows, cached."""
    path = CACHE / f"{user_id}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return data["X"], data["labels"], data["epoch_ts"]

    labels = load_labels(user_id, NEW_USERS_DIR)
    recording = _pair_raw_directory(NEW_USERS_DIR / "acc" / user_id, NEW_USERS_DIR / "gyro" / user_id)
    keep = np.array([int(t) in labels for t in recording.epoch_ts])
    epoch_ts = recording.epoch_ts[keep]
    y = np.array([labels[int(t)] for t in epoch_ts])
    X = context.augment(feat.extract_batch(recording.windows[keep]), epoch_ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, labels=y, epoch_ts=epoch_ts)
    return X, y, epoch_ts


def main() -> int:
    folds = load_folds()
    classes = list(config.ACTIVITIES)
    users = new_users()
    data = {u: load_new_user(u) for u in users}
    for u, (_, y, _) in data.items():
        print(f"{u[:8]}  {len(y):>5} labelled windows")
    y_all = np.concatenate([data[u][1] for u in users])

    runs = []
    for fold in range(folds["n_folds"]):
        print(f"Fold {fold} model", flush=True)
        model = joblib.load(config.MODEL_DIR / f"recogniser_fold{fold}.joblib")["context_aware"]
        transitions = Pipeline(fold=fold).transitions()
        probabilities = {u: model.predict_proba(data[u][0]) for u in users}

        def as_scores(predicted: np.ndarray) -> dict:
            onehot = (predicted[:, None] == np.asarray(classes)[None, :]).astype(float)
            return score(onehot, y_all, classes)

        results = {
            "ours_window": score(np.vstack([probabilities[u] for u in users]), y_all, classes),
            "ours_viterbi": as_scores(np.concatenate(
                [decode_labels(probabilities[u], data[u][2], transitions, "viterbi") for u in users])),
            "ours_posterior": as_scores(np.concatenate(
                [decode_labels(probabilities[u], data[u][2], transitions, "posterior") for u in users])),
        }

        split = folds["splits"][str(fold)]
        train_blocks, y_train = load_paper_users(split["train"] + split["validation"])
        per_user = [paper_features(NEW_USERS_DIR / f"{u}{config.LABEL_SUFFIX}", data[u][2]) for u in users]
        test_blocks = {s: np.vstack([b[s] for b in per_user]) for s in per_user[0]}
        paper = fit_and_predict(train_blocks, y_train, test_blocks, tag=f"paper fold {fold}")
        for name in ("EF_phone_acc_gyro", "LFA_phone_acc_gyro", "LFL_phone_acc_gyro", "EF_all_six", "LFA_all_six", "LFL_all_six"):
            results[f"paper_{name}"] = score(paper[name], y_all, classes)
        runs.append({"fold": fold, "results": results})

    print(f"\n{'system':<28}{'accuracy':>16}{'balanced acc':>16}{'macro-F1':>16}")
    for name in runs[0]["results"]:
        cells = ""
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = [r["results"][name][key] for r in runs]
            cells += f"{np.mean(values):>9.3f} ± {np.std(values):.3f}"
        print(f"{name:<28}{cells}")

    counts = {c: int((y_all == c).sum()) for c in classes}
    output = config.EVALUATION_DIR / "holdout.json"
    output.write_text(json.dumps({"users": users, "class_counts": counts, "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(f"\nclass counts: {counts}\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
