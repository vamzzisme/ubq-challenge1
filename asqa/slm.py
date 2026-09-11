"""L4b open-world reasoning (Task 4) with a small language model, held to the evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from asqa import config
from asqa.answer import Answer, find_activities, find_group
from asqa.timeline import Interval, Timeline, evidence_block

MENU_SIZE = 12

WORKER_TIMEOUT_S = 600


def ask_worker(question: str, menu: str = "", max_new_tokens: int = 320, mode: str = "answer") -> dict:
    """Run one generation in an isolated interpreter and return its JSON."""
    request = json.dumps(
        {"question": question, "menu": menu, "max_new_tokens": max_new_tokens, "mode": mode}
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


def parse_intent(question: str, rules):
    """Fill in the activities a question refers to, when the rules could not."""
    from asqa.intent import OPERATIONS, Intent

    try:
        payload = ask_worker(question, mode="parse", max_new_tokens=100)
    except Exception:
        return None

    from asqa.answer import GROUP_MEMBERS, resolve_activity_word, resolve_group_word

    proposed = payload.get("activities") or []
    activities: list[str] = []
    for raw in proposed:
        name = resolve_activity_word(str(raw))
        if name and name not in activities:
            activities.append(name)

    group = payload.get("group")
    group = resolve_group_word(str(group)) if group else None
    if not activities and group and not proposed:
        activities = list(GROUP_MEMBERS[group])

    if not activities:
        return None

    operation = rules.operation
    if operation not in OPERATIONS:
        proposed = str(payload.get("operation", "")).strip().lower().replace(" ", "_")
        if proposed not in OPERATIONS:
            return None
        operation = proposed

    return Intent(
        operation=operation,
        activities=activities,
        window=rules.window,
        at_time_s=rules.at_time_s,
        group=group,
        source="language-model",
    )


def retrieve(question: str, timeline: Timeline, limit: int = MENU_SIZE) -> list[Interval]:
    """Pick the intervals most likely to bear on the question."""
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
    """Turn the model's interval ids into real Interval objects."""
    cited: list[Interval] = []
    for raw in payload.get("intervals", []) or []:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
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
    except Exception as exc:
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
