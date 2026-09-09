#!/usr/bin/env python3
"""L2b -- user-disjoint fold assignment, aware of the rare classes.

Two constraints shape this layer.

First, generalisation is measured across *people*: no user may appear in both
the training and the evaluation portion of a fold, or the reported accuracy is
measuring memorisation of one person's gait.

Second, the rare classes are concentrated in a handful of users.  Running
accounts for 0.24% of labelled windows and lives in five users, one of whom
holds 40% of it.  A naive random fold assignment can therefore produce a fold
whose training portion contains almost no running at all, and the resulting
per-class F1 swings wildly between folds for reasons that have nothing to do
with the model.  Folds are therefore built rarest-class-first: the users who
carry running are spread across folds before anyone else is placed, so every
fold's training portion retains most of the rare-class evidence.

The assignment is written to ``outputs/folds.json`` and committed, so the
teaching team reruns exactly the split behind the reported numbers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from asqa import config
from asqa.preprocess import cached_users, load_cached

N_FOLDS = 5

# Held out for the qualitative timeline and QA demonstrations, and for the
# head-to-head against the previous implementation's recorded 0.532 accuracy.
# Keeping it fixed means the report's figures do not move between runs.
DEMO_USER = "0A986513-7828-4D53-AA1F-E02D6DF9561B"

# Rarest first: these drive the assignment order.
RARITY_ORDER = ("running", "bicycling", "walking", "standing")


def user_class_counts() -> dict[str, Counter]:
    """Coarse (pre-standing-split) class counts per cached user."""
    counts: dict[str, Counter] = {}
    for user_id in cached_users():
        cache = load_cached(user_id)
        counts[user_id] = Counter(cache.labels)
    return counts


def assign_folds(counts: dict[str, Counter], n_folds: int = N_FOLDS) -> dict[str, int]:
    """Spread rare-class carriers across folds, then balance by volume."""
    folds: dict[str, int] = {}
    fold_totals: Counter = Counter({index: 0 for index in range(n_folds)})
    fold_sizes: Counter = Counter({index: 0 for index in range(n_folds)})

    def place(user_id: str, fold: int) -> None:
        folds[user_id] = fold
        fold_totals[fold] += sum(counts[user_id].values())
        fold_sizes[fold] += 1

    # Pass 1: for each rare class in turn, deal its carriers round-robin so no
    # single fold holds out a disproportionate share of that class.
    for activity in RARITY_ORDER:
        carriers = sorted(
            (u for u in counts if u not in folds and counts[u][activity] > 0),
            key=lambda u: counts[u][activity],
            reverse=True,
        )
        for offset, user_id in enumerate(carriers):
            # Prefer the fold that currently holds least of this class.
            held = {
                index: sum(counts[u][activity] for u, f in folds.items() if f == index)
                for index in range(n_folds)
            }
            fold = min(range(n_folds), key=lambda i: (held[i], fold_sizes[i], fold_totals[i]))
            place(user_id, fold)

    # Pass 2: everyone else goes wherever balances total window count best.
    for user_id in sorted(counts, key=lambda u: sum(counts[u].values()), reverse=True):
        if user_id in folds:
            continue
        fold = min(range(n_folds), key=lambda i: (fold_sizes[i], fold_totals[i]))
        place(user_id, fold)

    return folds


def build_split(folds: dict[str, int], fold_index: int, n_validation: int = 2) -> dict[str, list[str]]:
    """Turn a fold assignment into train / validation / test user lists.

    Validation users are drawn from the *training* side (never the test side) so
    that early stopping and probability calibration never see the test people.
    """
    test = sorted(u for u, f in folds.items() if f == fold_index)
    remaining = sorted(u for u, f in folds.items() if f != fold_index)
    # Validation comes from the next fold round-robin, keeping it deterministic
    # and user-disjoint from both train and test.
    validation_fold = (fold_index + 1) % (max(folds.values()) + 1)
    validation_pool = [u for u in remaining if folds[u] == validation_fold]
    validation = validation_pool[:n_validation]
    train = [u for u in remaining if u not in validation]
    return {"train": train, "validation": validation, "test": test}


def summarise(counts: dict[str, Counter], folds: dict[str, int], n_folds: int = N_FOLDS) -> str:
    lines = []
    activities = ["sitting", "lying_down", "standing", "walking", "bicycling", "running"]
    header = f"{'fold':>4} {'users':>6} {'windows':>9}" + "".join(f"{a[:9]:>10}" for a in activities)
    lines.append(header)
    lines.append("-" * len(header))
    for index in range(n_folds):
        members = [u for u, f in folds.items() if f == index]
        totals = Counter()
        for user_id in members:
            totals.update(counts[user_id])
        lines.append(
            f"{index:>4} {len(members):>6} {sum(totals.values()):>9}"
            + "".join(f"{totals[a]:>10}" for a in activities)
        )
    return "\n".join(lines)


def load_folds(path: Path = config.FOLDS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or config.FOLDS_PATH

    counts = user_class_counts()
    if not counts:
        raise FileNotFoundError("No preprocessed users found; run `python -m asqa.preprocess` first.")

    folds = assign_folds(counts, args.folds)
    print(f"Assigned {len(folds)} users to {args.folds} user-disjoint folds\n")
    print(summarise(counts, folds, args.folds))

    # The demo user must actually be evaluated somewhere for the head-to-head
    # comparison to be meaningful.
    demo_fold = folds.get(DEMO_USER)
    print(f"\nDemo/held-out user {DEMO_USER[:8]} is in fold {demo_fold} (evaluated as test there)")

    payload = {
        "n_folds": args.folds,
        "demo_user": DEMO_USER,
        "demo_fold": demo_fold,
        "assignment": folds,
        "splits": {str(i): build_split(folds, i) for i in range(args.folds)},
        "class_counts": {u: dict(c) for u, c in counts.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    # Import through the package rather than calling the local `main`, so any
    # object pickled here records its class as `asqa.splits.X` and not
    # `__main__.X` -- the latter cannot be unpickled by any other entry point.
    from asqa.splits import main as _main

    raise SystemExit(_main())
