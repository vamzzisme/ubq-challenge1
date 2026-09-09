#!/usr/bin/env python3
"""Ask the Sensors -- answer questions about a wearable recording.

This is the runnable system the challenge asks for: given a recording and a set
of questions, it emits one structured answer per question in the required
format.

    python run.py --recording <path|user-id> --questions questions.txt
    python run.py --recording <path|user-id> --question "How long was the user walking?"

`--recording` accepts any of:
    * a directory of raw `<timestamp>.m_raw_acc.dat` files (the gyroscope
      partner directory is located automatically)
    * a single `.m_raw_acc.dat` file, for a one-window question
    * an ExtraSensory user id already in the preprocessing cache

Timestamps in every answer are **seconds from the start of the recording**.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from asqa import config
from asqa.answer import answer_question
from asqa.pipeline import Pipeline


def read_questions(args: argparse.Namespace) -> list[str]:
    if args.question:
        return list(args.question)
    if args.questions:
        lines = Path(args.questions).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    raise SystemExit("Provide --question or --questions.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recording", required=True, help="Recording directory, .dat file, or cached user id.")
    parser.add_argument("--question", action="append", help="A question (repeatable).")
    parser.add_argument("--questions", help="File with one question per line.")
    parser.add_argument("--output", type=Path, help="Write answers here instead of stdout.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the required text format.")
    parser.add_argument("--fold", type=int, default=0, help="Which trained fold's model to use.")
    parser.add_argument("--no-slm", action="store_true", help="Disable the language model for open-world questions.")
    parser.add_argument("--save-timeline", type=Path, help="Also write the intermediate activity timeline.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    args = parser.parse_args()

    questions = read_questions(args)

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr)

    log(f"Loading recording: {args.recording}")
    started = time.perf_counter()
    pipeline = Pipeline(fold=args.fold)
    timeline = pipeline.run(args.recording)
    elapsed = time.perf_counter() - started

    log(
        f"Analysed {len(timeline.windows)} windows -> {len(timeline.intervals)} activity intervals "
        f"in {elapsed:.1f}s "
        f"({'context-aware' if timeline.used_context_model else 'single-window'} model)"
    )
    if args.save_timeline:
        timeline.save(args.save_timeline)
        log(f"Wrote timeline to {args.save_timeline}")

    answers = [answer_question(question, timeline, use_slm=not args.no_slm) for question in questions]

    if args.json:
        rendered = json.dumps(
            {
                "recording": str(args.recording),
                "time_base": config.TIME_BASE,
                "answers": [
                    {"question": q, **a.to_dict()} for q, a in zip(questions, answers)
                ],
            },
            indent=2,
        )
    else:
        blocks = [f"Time base: all timestamps are {config.TIME_BASE}.", ""]
        for question, answer in zip(questions, answers):
            blocks.append(f"Query: \"{question}\"")
            blocks.append(answer.render())
            blocks.append("")
        rendered = "\n".join(blocks)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        log(f"Wrote {len(answers)} answers to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
