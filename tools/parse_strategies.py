#!/usr/bin/env python3
"""Compare three ways of getting a class name out of a 0.5B model.

The question this answers: why not just let the language model read an unusual
word and name the activity from the list?  That is what it is asked to do, and
on this model it mostly does not comply.  Run this to reproduce the numbers.

    python tools/parse_strategies.py

  generate            free generation, prompt names the closed vocabulary
  choose              the classes as lettered options; one token scored, so a
                      word outside the list cannot be produced at all
  generate+resolve    free generation, then stems and prefixes map the word
                      back onto the vocabulary (what the pipeline ships)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asqa import config  # noqa: E402
from asqa.answer import GROUP_MEMBERS, resolve_activity_word, resolve_group_word  # noqa: E402

# Deliberately unusual phrasings: the case where the rules have already failed
# and the model is the only thing left. Everyday wording never gets this far.
CASES = [
    ("did the user wander after 84h?", "walking"),
    ("when did she first start to jog?", "running"),
    ("did the bloke ever get on his pushbike?", "bicycling"),
    ("was he having a kip?", "lying_down"),
    ("how long did he amble about?", "walking"),
    ("did she ever leg it?", "running"),
    ("was the user having a lie-in?", "lying_down"),
    ("did he do any pedalling?", "bicycling"),
    ("how often did he perch somewhere?", "sitting"),
    ("was he stationary upright?", "standing_in_place"),
]


def ask(question: str, mode: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "asqa.slm_worker"],
        input=json.dumps({"question": question, "mode": mode, "max_new_tokens": 100}),
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(result.stdout).get("payload") or {}
    except json.JSONDecodeError:
        return {}


def main() -> None:
    tally = {"generate": 0, "choose": 0, "generate+resolve": 0}
    in_vocab = 0

    for question, want in CASES:
        parsed = ask(question, "parse")
        raw = (parsed.get("activities") or [None])[0]
        raw = str(raw) if raw else None

        # 1. free generation, taken literally
        if raw in config.ACTIVITY_INDEX:
            in_vocab += 1
            tally["generate"] += raw == want

        # 2. forced choice over lettered options
        chosen = ask(question, "choose").get("activity")
        tally["choose"] += chosen == want

        # 3. generation, then mapped back onto the vocabulary
        resolved = resolve_activity_word(raw) if raw else None
        if not resolved and not parsed.get("activities"):
            group = resolve_group_word(str(parsed.get("group") or ""))
            members = GROUP_MEMBERS.get(group or "", [])
            resolved = members[0] if len(members) == 1 else None
        tally["generate+resolve"] += resolved == want

        print(f"{question:<42} wrote={str(raw):<18} chose={str(chosen):<20} resolved={resolved}")

    total = len(CASES)
    print(f"\n{'strategy':<20} correct   note")
    print("-" * 62)
    print(f"{'generate':<20} {tally['generate']}/{total}      stayed in vocabulary {in_vocab}/{total}")
    print(f"{'choose':<20} {tally['choose']}/{total}      always in vocabulary, letter-biased")
    print(f"{'generate+resolve':<20} {tally['generate+resolve']}/{total}      shipped")


if __name__ == "__main__":
    main()
