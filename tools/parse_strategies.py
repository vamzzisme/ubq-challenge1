"""Compare three ways of getting a class name out of a 0.5B model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asqa import config
from asqa.answer import GROUP_MEMBERS, resolve_activity_word, resolve_group_word

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

        if raw in config.ACTIVITY_INDEX:
            in_vocab += 1
            tally["generate"] += raw == want

        chosen = ask(question, "choose").get("activity")
        tally["choose"] += chosen == want

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
