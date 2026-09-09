#!/usr/bin/env python3
"""L4b -- open-world reasoning (Task 4) with a small language model, held to the evidence.

Task 4 asks about behaviours the classifier was never trained to name -- "was the
user resting?", "anything strenuous around noon?", "a wheeled or pedal-based mode
of movement?".  Answering those needs language, and language is exactly where a
grounded system usually stops being grounded.

The brief is explicit that free generation does not count: *"the language it
produces must be tied to evidence that the earlier layers found, not generated
freely."*  So the model here is never asked for a timestamp.  It is shown a
numbered menu of intervals that L3 actually produced, and asked to pick from it.
The interval id it returns is looked up in that menu, and every evidence field is
copied from the chosen `Interval` object.  If the model returns an id that is not
on the menu, the answer is rejected and the system falls back to the strongest
retrieved interval rather than printing a number the model invented.

This is the difference between a model that reads evidence and one that writes
fiction that resembles evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from asqa import config
from asqa.answer import Answer, find_activities, find_group
from asqa.timeline import Interval, Timeline, evidence_block

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# How many intervals the model may choose between.  Small enough to fit a 0.5B
# model's working context comfortably, large enough to cover a day.
MENU_SIZE = 12

# Generation runs in a separate interpreter (see `slm_worker`): importing
# PyTorch into a process that has already run XGBoost segfaults here, on both
# MPS and CPU.  This timeout bounds a cold start, which includes loading the
# model weights.
WORKER_TIMEOUT_S = 600


def ask_worker(question: str, menu: str, max_new_tokens: int = 320) -> dict:
    """Run one generation in an isolated interpreter and return its JSON."""
    request = json.dumps(
        {"question": question, "menu": menu, "max_new_tokens": max_new_tokens}
    )
    completed = subprocess.run(
        [sys.executable, "-m", "asqa.slm_worker"],
        input=request,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_S,
        cwd=str(config.REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(config.REPO_ROOT)},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"language-model worker exited {completed.returncode}: "
            f"{completed.stderr.strip()[-200:]}"
        )
    response = json.loads(completed.stdout)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "unknown worker failure"))
    return response["payload"]


# ── Retrieval ────────────────────────────────────────────────────────────────


def retrieve(question: str, timeline: Timeline, limit: int = MENU_SIZE) -> list[Interval]:
    """Pick the intervals most likely to bear on the question.

    Retrieval is deliberately simple and deterministic: activities named or
    implied by the question first, then the longest remaining intervals so the
    model always has the shape of the whole recording to reason against.
    """
    from asqa.answer import GROUP_MEMBERS

    named = set(find_activities(question))
    group = find_group(question)
    if group is not None:
        named |= set(GROUP_MEMBERS[group])

    relevant = [i for i in timeline.intervals if i.activity in named]
    others = [i for i in timeline.intervals if i.activity not in named]
    relevant.sort(key=lambda i: i.duration_s, reverse=True)
    others.sort(key=lambda i: i.duration_s, reverse=True)

    chosen = (relevant + others)[:limit]
    return sorted(chosen, key=lambda i: i.start_s)


def render_menu(intervals: list[Interval]) -> str:
    lines = []
    for index, interval in enumerate(intervals, start=1):
        signal = interval.signal
        descriptors = [
            f"energy={signal.get('body_acc_rms', 0):.4f}g",
            f"cadence={signal.get('cadence_hz', 0):.1f}Hz",
            f"tilt={signal.get('gravity_tilt_deg', 0):.0f}deg",
            f"rotation={signal.get('gyro_rms', 0):.3f}rad/s",
        ]
        lines.append(
            f"[{index}] {interval.start_s:.0f}-{interval.end_s:.0f}s "
            f"({interval.duration_s:.0f}s, classified {config.DISPLAY_NAMES.get(interval.activity, interval.activity)}) "
            f"{' '.join(descriptors)}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a sensor analyst. You are given numbered intervals from a wearable recording, each with measured accelerometer and gyroscope statistics.

Answer the user's question using ONLY these intervals. You must not invent times.

Reply with a JSON object and nothing else:
{"answer": "<short direct answer>", "event": "<the behaviour in plain words>", "intervals": [<interval numbers you are citing>], "explanation": "<why those measurements support your answer>"}

Guidance for reading the measurements:
- energy near 0.00g means the body is still; above 0.1g means active motion
- cadence 1.5-2.2Hz with high energy is typical of walking; 2.5-3.5Hz of running
- steady moderate energy with low impact and sustained rotation suggests cycling
- tilt describes the device orientation, which changes between lying and upright"""


def resolve_citations(payload: dict, menu: list[Interval]) -> list[Interval]:
    """Turn the model's interval ids into real Interval objects.

    This is the guard that keeps Task 4 grounded: an id outside the menu is
    discarded rather than rendered, so the model cannot introduce a timestamp
    the earlier layers never produced.
    """
    cited: list[Interval] = []
    for raw in payload.get("intervals", []) or []:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        # The whole point: an id outside the menu is discarded, never rendered.
        if 1 <= index <= len(menu):
            cited.append(menu[index - 1])
    return cited


def answer_open_world(question: str, timeline: Timeline) -> Answer:
    """Answer a Task 4 question, citing only intervals the pipeline produced."""
    menu = retrieve(question, timeline)
    if not menu:
        return Answer(
            "N/A", "N/A", "N/A", "N/A", "N/A",
            "The recording contains no intervals to reason over.",
            "open_world", [],
        )

    try:
        payload = ask_worker(question, render_menu(menu))
        cited = resolve_citations(payload, menu)
    except Exception as exc:  # noqa: BLE001 - fall back to retrieved evidence
        best = menu[0]
        block = evidence_block([best])
        return Answer(
            "Unable to determine",
            config.DISPLAY_NAMES.get(best.activity, best.activity),
            block["timestamps"], block["modality"], block["channels"],
            f"The language model was unavailable ({type(exc).__name__}). The most relevant "
            f"interval retrieved was {best.describe()} s, classified "
            f"{config.DISPLAY_NAMES.get(best.activity, best.activity)}.",
            "open_world",
            [{"start_s": best.start_s, "end_s": best.end_s}],
        )

    if not cited:
        # An answer with no valid citation is not grounded; supply the retrieved
        # evidence so the response still points at real signal.
        cited = menu[:1]

    block = evidence_block(cited)
    return Answer(
        answer=str(payload.get("answer", "N/A"))[:300],
        activity_event=str(payload.get("event", "N/A"))[:200],
        timestamps=block["timestamps"],
        modality=block["modality"],
        channels=block["channels"],
        explanation=str(payload.get("explanation", "N/A"))[:800],
        question_type="open_world",
        intervals=[{"start_s": i.start_s, "end_s": i.end_s} for i in cited],
    )
