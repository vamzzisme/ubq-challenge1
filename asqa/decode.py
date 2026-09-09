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


def forward_backward(
    probabilities: np.ndarray,
    epoch_ts: np.ndarray,
    transitions: TimeAwareTransitions,
    priors: np.ndarray | None = None,
) -> np.ndarray:
    """Per-window posterior marginals `P(activity_t | all windows)`.

    Viterbi returns the single most likely *path*, which is the right object when
    the whole sequence must be self-consistent.  Marginal decoding instead asks
    what each window is, given everything observed before and after it, and so
    usually scores better per window.  Both are offered because the aggregation
    layer wants coherent intervals (Viterbi) while per-window accuracy reporting
    wants marginals.

    Returned in the caller's original row order.
    """
    n_windows, n_states = probabilities.shape
    if n_windows == 0:
        return np.empty((0, n_states))

    if priors is None:
        priors = np.full(n_states, 1.0 / n_states)  # see `viterbi` for why
    emissions = np.clip(probabilities, 1e-12, 1.0) / np.clip(priors, 1e-9, None)

    order = np.argsort(epoch_ts)
    epoch_ts = np.asarray(epoch_ts)[order]
    emissions = emissions[order]
    gaps = [float(epoch_ts[i] - epoch_ts[i - 1]) for i in range(1, n_windows)]

    # Scaled forward-backward: renormalising each step keeps long sequences from
    # underflowing without needing logs.
    alpha = np.zeros((n_windows, n_states))
    alpha[0] = transitions.stationary * emissions[0]
    alpha[0] /= alpha[0].sum() or 1.0
    for index in range(1, n_windows):
        alpha[index] = (alpha[index - 1] @ transitions.for_gap(gaps[index - 1])) * emissions[index]
        total = alpha[index].sum()
        alpha[index] /= total or 1.0

    beta = np.zeros((n_windows, n_states))
    beta[-1] = 1.0
    for index in range(n_windows - 2, -1, -1):
        beta[index] = transitions.for_gap(gaps[index]) @ (emissions[index + 1] * beta[index + 1])
        total = beta[index].sum()
        beta[index] /= total or 1.0

    posterior = alpha * beta
    totals = posterior.sum(axis=1, keepdims=True)
    posterior = np.divide(
        posterior, totals, where=totals > 0, out=np.full_like(posterior, 1.0 / n_states)
    )

    result = np.empty_like(posterior)
    result[order] = posterior
    return result


def decode_labels(
    probabilities: np.ndarray,
    epoch_ts: np.ndarray,
    transitions: TimeAwareTransitions,
    method: str = "viterbi",
) -> np.ndarray:
    """Decoded activity names, by Viterbi path or posterior marginals."""
    if method == "viterbi":
        return np.asarray(config.ACTIVITIES)[viterbi(probabilities, epoch_ts, transitions)]
    if method == "posterior":
        posterior = forward_backward(probabilities, epoch_ts, transitions)
        return np.asarray(config.ACTIVITIES)[np.argmax(posterior, axis=1)]
    raise ValueError(f"unknown decoding method: {method!r}")


# ── Evaluation ───────────────────────────────────────────────────────────────


def main() -> int:
    import joblib
    from sklearn.metrics import f1_score

    from asqa.recognise import build_context_features, build_features
    from asqa.splits import load_folds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--cv", action="store_true")
    args = parser.parse_args()

    folds = load_folds()
    indices = range(folds["n_folds"]) if args.cv else [args.fold if args.fold is not None else 0]

    from asqa import context as ctx

    results = []
    for fold_index in indices:
        split = folds["splits"][str(fold_index)]
        saved = joblib.load(config.MODEL_DIR / f"recogniser_fold{fold_index}.joblib")

        record: dict = {"fold": fold_index}
        for variant in ("context_free", "context_aware"):
            model = saved[variant]
            with_context = variant == "context_aware"

            def matrices(user_id: str, _ctx: bool = with_context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                return build_context_features(user_id) if _ctx else build_features(user_id)

            # Transitions come from the training users only.
            sequences = []
            for user_id in split["train"]:
                X, coarse, epoch_ts = matrices(user_id)
                labels, _ = model.standing_split.apply(X, coarse)
                sequences.append((epoch_ts, labels))
            per_minute = estimate_transitions(sequences)
            transitions = TimeAwareTransitions(per_minute)

            collected: dict[str, list[np.ndarray]] = {"truth": [], "window": [], "viterbi": [], "posterior": []}
            for user_id in split["test"]:
                X, coarse, epoch_ts = matrices(user_id)
                truth, _ = model.standing_split.apply(X, coarse)
                probabilities = model.predict_proba(X)
                collected["truth"].append(truth)
                collected["window"].append(np.asarray(config.ACTIVITIES)[np.argmax(probabilities, axis=1)])
                collected["viterbi"].append(decode_labels(probabilities, epoch_ts, transitions, "viterbi"))
                collected["posterior"].append(decode_labels(probabilities, epoch_ts, transitions, "posterior"))

            joined = {key: np.concatenate(value) for key, value in collected.items()}
            truth = joined["truth"]
            present = sorted(set(truth) | set(joined["viterbi"]) | set(joined["window"]))

            entry = {"self_transition_rate": float(np.mean(np.diag(per_minute)))}
            for method in ("window", "viterbi", "posterior"):
                predicted = joined[method]
                entry[f"accuracy_{method}"] = float((predicted == truth).mean())
                entry[f"macro_f1_{method}"] = float(
                    f1_score(truth, predicted, labels=present, average="macro", zero_division=0)
                )
            if variant == "context_aware":
                entry["transition_matrix"] = per_minute.tolist()
                entry["activities"] = list(config.ACTIVITIES)
            record[variant] = entry
            print(
                f"fold {fold_index} {variant:<14} window {entry['accuracy_window']:.3f} | "
                f"viterbi {entry['accuracy_viterbi']:.3f} (F1 {entry['macro_f1_viterbi']:.3f}) | "
                f"posterior {entry['accuracy_posterior']:.3f} (F1 {entry['macro_f1_posterior']:.3f})"
            )

        # Legacy flat keys, so earlier comparisons stay readable.
        record["accuracy_before"] = record["context_free"]["accuracy_window"]
        record["accuracy_after"] = record["context_free"]["accuracy_viterbi"]
        record["macro_f1_before"] = record["context_free"]["macro_f1_window"]
        record["macro_f1_after"] = record["context_free"]["macro_f1_viterbi"]
        record["self_transition_rate"] = record["context_aware"]["self_transition_rate"]
        record["transition_matrix"] = record["context_aware"]["transition_matrix"]
        record["activities"] = record["context_aware"]["activities"]
        results.append(record)

    if len(results) > 1:
        print(f"\n{'=' * 78}")
        print(f"{'':<16}{'':<12}{'accuracy':>20}{'macro-F1':>20}")
        best = (None, None, -1.0)
        for variant in ("context_free", "context_aware"):
            for method in ("window", "viterbi", "posterior"):
                accuracies = [r[variant][f"accuracy_{method}"] for r in results]
                f1s = [r[variant][f"macro_f1_{method}"] for r in results]
                mean_accuracy, mean_f1 = float(np.mean(accuracies)), float(np.mean(f1s))
                print(
                    f"{variant:<16}{method:<12}{mean_accuracy:>13.3f} +/- {np.std(accuracies):.3f}"
                    f"{mean_f1:>13.3f} +/- {np.std(f1s):.3f}"
                )
                if mean_f1 > best[2]:
                    best = (variant, method, mean_f1)
        print(f"\nbest by macro-F1: {best[0]} + {best[1]} ({best[2]:.3f})")

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
