#!/usr/bin/env python3
"""L2c -- temporal decoding: a time-aware HMM over the window sequence.

The recogniser classifies each window in isolation, but human activity is
overwhelmingly persistent.  Measured across the corpus, the probability that a
minute carries the same label as the minute before it is 0.979; `lying_down`
repeats with probability 0.996.  A per-window classifier throws all of that
away, which is why an undecoded timeline answers "how long was the user
walking?" with hundreds of disconnected 20-second fragments rather than a few
intervals.

This module keeps the discriminative classifier and adds the temporal prior on
top -- the standard hybrid.  A purely generative HMM used *as* the classifier
would have to model `P(features | activity)`, which cannot be estimated from 122
running windows; the hybrid only needs the transition structure, which is
estimated from label sequences and is well determined.

Uneven sampling is handled explicitly.  ExtraSensory samples roughly once a
minute, but the gaps are ragged (jitter of a few seconds, and frequent dropouts
of minutes to hours).  Asserting a one-minute transition across a two-hour gap
would claim the user kept doing the same thing throughout.  Instead the
per-minute matrix is raised to the power `dt/60` -- the continuous-time Markov
reading -- so a long gap relaxes smoothly toward the stationary distribution and
the decoder stops pretending it knows.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.linalg import eig, expm, logm

from asqa import config

MINUTE_S = 60.0

# Beyond this the two windows are treated as independent observations: the
# transition prior has decayed to the stationary distribution anyway, and the
# matrix power becomes numerically pointless.
MAX_BRIDGE_S = 3600.0


def estimate_transitions(
    sequences: list[tuple[np.ndarray, np.ndarray]],
    smoothing: float = 1.0,
) -> np.ndarray:
    """Estimate the per-minute transition matrix from labelled sequences.

    ``sequences`` is a list of ``(epoch_ts, labels)`` per user.  Only adjacent
    pairs roughly one minute apart contribute, so the matrix has a well-defined
    time unit; wider gaps are handled at decode time by the matrix power.
    """
    n = len(config.ACTIVITIES)
    counts = np.full((n, n), smoothing)  # Laplace smoothing: no impossible moves

    for epoch_ts, labels in sequences:
        order = np.argsort(epoch_ts)
        epoch_ts, labels = np.asarray(epoch_ts)[order], np.asarray(labels)[order]
        gaps = np.diff(epoch_ts)
        adjacent = (gaps >= 55) & (gaps <= 65)
        for index in np.flatnonzero(adjacent):
            a, b = labels[index], labels[index + 1]
            if a in config.ACTIVITY_INDEX and b in config.ACTIVITY_INDEX:
                counts[config.ACTIVITY_INDEX[a], config.ACTIVITY_INDEX[b]] += 1

    return counts / counts.sum(axis=1, keepdims=True)


def stationary_distribution(transitions: np.ndarray) -> np.ndarray:
    """Left eigenvector of the transition matrix for eigenvalue 1."""
    values, vectors = eig(transitions.T)
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real(vectors[:, index])
    vector = np.abs(vector)
    return vector / vector.sum()


def _generator(transitions: np.ndarray) -> np.ndarray:
    """Continuous-time generator Q with ``expm(Q) == transitions``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        generator = np.real(logm(transitions))
    return np.nan_to_num(generator, nan=0.0, posinf=0.0, neginf=0.0)


class TimeAwareTransitions:
    """Transition matrices for arbitrary elapsed times, cached by rounded gap."""

    def __init__(self, per_minute: np.ndarray):
        self.per_minute = per_minute
        self.stationary = stationary_distribution(per_minute)
        self._generator = _generator(per_minute)
        self._cache: dict[int, np.ndarray] = {}

    def for_gap(self, gap_s: float) -> np.ndarray:
        """``P(next | current)`` after ``gap_s`` seconds have elapsed."""
        if gap_s >= MAX_BRIDGE_S:
            # Fully relaxed: the next window tells us nothing about this one.
            return np.tile(self.stationary, (len(self.stationary), 1))

        key = int(round(gap_s / 5.0))  # 5-second resolution is ample
        if key in self._cache:
            return self._cache[key]

        steps = max(gap_s, 1.0) / MINUTE_S
        matrix = np.real(expm(self._generator * steps))
        matrix = np.clip(matrix, 1e-12, None)
        matrix /= matrix.sum(axis=1, keepdims=True)
        self._cache[key] = matrix
        return matrix


def viterbi(
    probabilities: np.ndarray,
    epoch_ts: np.ndarray,
    transitions: TimeAwareTransitions,
    priors: np.ndarray | None = None,
) -> np.ndarray:
    """Most likely activity path through one user's window sequence.

    ``probabilities`` are the classifier's posteriors `P(activity | window)`.
    An HMM wants emission *likelihoods*, so a hybrid ordinarily divides the
    posteriors by the class priors implied by the classifier's training.

    Here that division is a no-op, and doing it wrong is catastrophic.  The
    recogniser is trained with inverse-frequency sample weights, so the prior it
    implies is already close to uniform -- not the corpus prior.  Dividing by the
    *corpus* prior therefore boosts the rare classes a second time, and because
    the transition matrix strongly favours staying put, Viterbi then latches onto
    an inflated rare class and holds it for the entire sequence.  Measured, that
    mistake drove accuracy from 0.444 down to 0.120.  The corpus prior belongs in
    the transition matrix, which is where it now lives, and nowhere else.
    """
    n_windows, n_states = probabilities.shape
    if n_windows == 0:
        return np.empty(0, dtype=int)

    if priors is None:
        priors = np.full(n_states, 1.0 / n_states)  # matches balanced training
    priors = np.clip(priors, 1e-9, None)

    emissions = np.log(np.clip(probabilities, 1e-12, 1.0)) - np.log(priors)

    order = np.argsort(epoch_ts)
    epoch_ts = np.asarray(epoch_ts)[order]
    emissions = emissions[order]

    scores = np.log(np.clip(transitions.stationary, 1e-12, None)) + emissions[0]
    backpointers = np.zeros((n_windows, n_states), dtype=np.int16)

    for index in range(1, n_windows):
        log_transition = np.log(transitions.for_gap(float(epoch_ts[index] - epoch_ts[index - 1])))
        candidates = scores[:, None] + log_transition  # (from, to)
        backpointers[index] = np.argmax(candidates, axis=0)
        scores = candidates[backpointers[index], np.arange(n_states)] + emissions[index]

    path = np.zeros(n_windows, dtype=int)
    path[-1] = int(np.argmax(scores))
    for index in range(n_windows - 1, 0, -1):
        path[index - 1] = backpointers[index, path[index]]

    # Undo the sort so the caller gets the path in its original order.
    result = np.empty_like(path)
    result[order] = path
    return result


def decode_labels(
    probabilities: np.ndarray,
    epoch_ts: np.ndarray,
    transitions: TimeAwareTransitions,
) -> np.ndarray:
    """Viterbi path as activity name strings."""
    return np.asarray(config.ACTIVITIES)[viterbi(probabilities, epoch_ts, transitions)]


# ── Evaluation ───────────────────────────────────────────────────────────────


def main() -> int:
    import joblib
    from sklearn.metrics import f1_score

    from asqa.recognise import build_features, load_users
    from asqa.splits import load_folds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--cv", action="store_true")
    args = parser.parse_args()

    folds = load_folds()
    indices = range(folds["n_folds"]) if args.cv else [args.fold if args.fold is not None else 0]

    results = []
    for fold_index in indices:
        split = folds["splits"][str(fold_index)]
        saved = joblib.load(config.MODEL_DIR / f"recogniser_fold{fold_index}.joblib")
        model = saved["model"]

        # Transitions come from the training users only.
        sequences = []
        for user_id in split["train"]:
            X, coarse, epoch_ts = build_features(user_id)
            labels, _ = model.standing_split.apply(X, coarse)
            sequences.append((epoch_ts, labels))
        per_minute = estimate_transitions(sequences)
        transitions = TimeAwareTransitions(per_minute)

        before_all, after_all, truth_all = [], [], []
        for user_id in split["test"]:
            X, coarse, epoch_ts = build_features(user_id)
            truth, _ = model.standing_split.apply(X, coarse)
            probabilities = model.predict_proba(X)
            before = np.asarray(config.ACTIVITIES)[np.argmax(probabilities, axis=1)]
            after = decode_labels(probabilities, epoch_ts, transitions)
            before_all.append(before)
            after_all.append(after)
            truth_all.append(truth)

        before = np.concatenate(before_all)
        after = np.concatenate(after_all)
        truth = np.concatenate(truth_all)
        present = sorted(set(truth) | set(after) | set(before))

        record = {
            "fold": fold_index,
            "accuracy_before": float((before == truth).mean()),
            "accuracy_after": float((after == truth).mean()),
            "macro_f1_before": float(f1_score(truth, before, labels=present, average="macro", zero_division=0)),
            "macro_f1_after": float(f1_score(truth, after, labels=present, average="macro", zero_division=0)),
            "self_transition_rate": float(np.mean(np.diag(per_minute))),
            "transition_matrix": per_minute.tolist(),
            "activities": list(config.ACTIVITIES),
        }
        results.append(record)
        print(
            f"fold {fold_index}: accuracy {record['accuracy_before']:.3f} -> {record['accuracy_after']:.3f} | "
            f"macro-F1 {record['macro_f1_before']:.3f} -> {record['macro_f1_after']:.3f}"
        )

    if len(results) > 1:
        print(f"\n{'='*64}")
        print(
            f"mean accuracy {np.mean([r['accuracy_before'] for r in results]):.3f} -> "
            f"{np.mean([r['accuracy_after'] for r in results]):.3f}"
        )
        print(
            f"mean macro-F1 {np.mean([r['macro_f1_before'] for r in results]):.3f} -> "
            f"{np.mean([r['macro_f1_after'] for r in results]):.3f}"
        )

    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output = config.EVALUATION_DIR / ("decode_cv.json" if len(results) > 1 else f"decode_fold{indices[0]}.json")
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    # Import through the package rather than calling the local `main`, so any
    # object pickled here records its class as `asqa.decode.X` and not
    # `__main__.X` -- the latter cannot be unpickled by any other entry point.
    from asqa.decode import main as _main

    raise SystemExit(_main())
