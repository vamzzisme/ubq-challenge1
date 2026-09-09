#!/usr/bin/env python3
"""Which `--fold` model is safe to use for a given user?

Every fold's model was trained on some users and held others out.  Asking
questions about a user with a model that trained on them measures memorisation,
not generalisation, and quietly inflates the answer.  This prints, per fold,
whether the user was held out (safe), used for validation, or trained on.

    python tools/which_fold.py <USER-ID>
    python tools/which_fold.py --list

A user who is not in the corpus at all is safe with any fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asqa.splits import load_folds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("user", nargs="?", help="ExtraSensory user id (full or unique prefix).")
    parser.add_argument("--list", action="store_true", help="List every user and its fold.")
    args = parser.parse_args()

    folds = load_folds()
    assignment = folds["assignment"]

    if args.list or not args.user:
        print(f"{'user':<40}{'fold':>6}   held out as test in")
        for user_id in sorted(assignment, key=lambda u: assignment[u]):
            print(f"{user_id:<40}{assignment[user_id]:>6}   --fold {assignment[user_id]}")
        print(f"\nDemo user: {folds['demo_user']} (fold {folds['demo_fold']})")
        return 0

    matches = [u for u in assignment if u == args.user or u.startswith(args.user)]
    if not matches:
        print(f"{args.user} is not in the training corpus.")
        print("Any --fold is safe; use --fold 0.")
        return 0
    if len(matches) > 1:
        print(f"Ambiguous prefix {args.user!r} matches: {', '.join(m[:8] for m in matches)}")
        return 1

    user_id = matches[0]
    print(f"{user_id}\n")
    for index in range(folds["n_folds"]):
        split = folds["splits"][str(index)]
        if user_id in split["test"]:
            verdict = "TEST      <- safe: this model never saw them"
        elif user_id in split["validation"]:
            verdict = "validation   (used for early stopping; avoid)"
        else:
            verdict = "TRAIN        (leaks: results will be optimistic)"
        print(f"  --fold {index}   {verdict}")

    safe = [i for i in range(folds["n_folds"]) if user_id in folds["splits"][str(i)]["test"]]
    print(f"\nUse: --fold {safe[0]}" if safe else "\nNo fold holds this user out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
