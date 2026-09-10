#!/usr/bin/env python3
"""L4 -- interface: map a natural-language question onto the structured timeline.

The brief constrains this layer more tightly than the others: *"the language it
produces must be tied to evidence that the earlier layers found, not generated
freely."*  Every field emitted here therefore comes from a `timeline.Interval`
object.  The question selects which operation to run and which intervals to
read; it never supplies a number, a timestamp, or a claim of its own.

Question types handled, matching the four tiers:

    identification   what is the user doing (optionally at a time)   Task 1
    verification     is/was/did the user <activity>                  Task 1
    duration         how long                                        Task 2
    count            how many times / how often                      Task 2
    comparison       more time X or Y                                Task 2
    temporal         when did X begin / start                        Task 2, 3
    open_world       anything else -- routed to the SLM              Task 4

Anything the router cannot confidently place goes to the small language model,
which is given a menu of real intervals and may only cite from it (see `slm.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from asqa import config
from asqa.intent import Intent, TimeWindow, clip, find_window
from asqa.timeline import Interval, Timeline, cite, evidence_block

# Phrases that name each activity.  Ordered longest-first when matching so that
# "standing and moving" is not swallowed by "standing".
ACTIVITY_PHRASES: dict[str, tuple[str, ...]] = {
    "lying_down": ("lying down", "lie down", "lying", "laying down", "lay down", "lied down"),
    "sitting": ("sitting", "seated", "sit down", "sits", "sat", "sit"),
    "standing_in_place": ("standing in place", "standing still", "stood still", "standing in one place"),
    "standing_and_moving": ("standing and moving", "moving about", "shuffling", "standing while moving"),
    "walking": (
        "walking", "walk", "walked", "strolling", "stroll", "on foot",
        "wandering", "wander", "wandered", "ambling", "amble", "roaming", "roam",
        "hiking", "hike", "pacing", "trekking", "on the move by foot",
    ),
    "running": ("running", "run", "ran", "jogging", "jog", "sprinting", "sprint"),
    "bicycling": ("bicycling", "cycling", "bicycle", "biking", "bike", "cycle", "pedalling", "pedaling"),
}

# Loose language that maps onto a group of classes rather than one.  Task 4 asks
# about behaviour rather than class names ("resting", "strenuous", "a wheeled or
# pedal-based mode of movement"), but most such phrasings do resolve onto the
# seven classes.  Resolving them here rather than in the language model keeps the
# verdict tied to the classifier: a 0.5B model asked to judge "wheeled movement"
# was measured answering "Pedal-based mode" while citing a *sitting* interval.
# The model is better used for prose than for verdicts.
GROUP_PHRASES: dict[str, tuple[str, ...]] = {
    "resting": ("resting", "rest", "inactive", "idle", "sedentary", "still", "sleeping", "asleep"),
    "active": (
        "active", "exercising", "exercise", "strenuous", "vigorous", "exerting",
        "physical activity", "working out", "energetic", "tiring", "exhausting",
        "demanding", "effortful", "workout", "moving around",
    ),
    "wheeled": ("wheeled", "pedal-based", "pedal based", "pedalling", "pedaling", "two-wheeler", "on wheels"),
    "standing": ("standing", "stood", "stand"),
}

GROUP_MEMBERS: dict[str, list[str]] = {
    "resting": ["lying_down", "sitting"],
    "active": ["walking", "running", "bicycling"],
    "wheeled": ["bicycling"],
    "standing": ["standing_in_place", "standing_and_moving"],
}

VERIFICATION_STARTS = ("is ", "was ", "did ", "has ", "have ", "were ", "does ", "do ")


@dataclass
class Answer:
    """One response in the exact shape the brief specifies."""

    answer: str
    activity_event: str
    timestamps: str
    modality: str
    channels: str
    explanation: str
    question_type: str = "unknown"
    intervals: list[dict[str, float]] | None = None

    def render(self) -> str:
        return "\n".join(
            [
                f"Answer: {self.answer}",
                f"Activity/Event: {self.activity_event}",
                "Evidence:",
                f"    Timestamp(s): {self.timestamps}",
                f"    Sensor Modality: {self.modality}",
                f"    Sensor Channel(s): {self.channels}",
                f"Explanation: {self.explanation}",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "activity_event": self.activity_event,
            "evidence": {
                "timestamps": self.timestamps,
                "modality": self.modality,
                "channels": self.channels,
                "intervals": self.intervals or [],
            },
            "explanation": self.explanation,
            "question_type": self.question_type,
        }


def _from(intervals: list[Interval], answer: str, event: str, explanation: str, kind: str) -> Answer:
    block = evidence_block(intervals)
    return Answer(
        answer=answer,
        activity_event=event,
        timestamps=block["timestamps"],
        modality=block["modality"],
        channels=block["channels"],
        explanation=explanation,
        question_type=kind,
        intervals=[{"start_s": i.start_s, "end_s": i.end_s} for i in intervals],
    )


def _none(explanation: str, kind: str, event: str = "N/A") -> Answer:
    return Answer("N/A", event, "N/A", "N/A", "N/A", explanation, kind, [])


# ── Question parsing ─────────────────────────────────────────────────────────


def find_activities(question: str) -> list[str]:
    """Activities named in the question, in the order they are mentioned."""
    lower = question.lower()
    hits: list[tuple[int, str]] = []
    claimed: list[tuple[int, int]] = []

    ordered = sorted(
        ((activity, phrase) for activity, phrases in ACTIVITY_PHRASES.items() for phrase in phrases),
        key=lambda pair: len(pair[1]),
        reverse=True,
    )
    for activity, phrase in ordered:
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", lower):
            span = match.span()
            # A longer phrase already covering this text wins.
            if any(start <= span[0] < end for start, end in claimed):
                continue
            claimed.append(span)
            if activity not in [a for _, a in hits]:
                hits.append((span[0], activity))
    return [activity for _, activity in sorted(hits)]


def _stems(word: str) -> list[str]:
    """The word and its plausible stems, longest first."""
    forms = [word]
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            stem = word[: -len(suffix)]
            forms.append(stem)
            if suffix == "ing" and len(stem) >= 4:
                forms.append(stem + "e")  # amble -> ambling
    return forms


def _shared_prefix(left: str, right: str) -> str:
    out = []
    for a, b in zip(left, right):
        if a != b:
            break
        out.append(a)
    return "".join(out)


def resolve_activity_word(word: str) -> str | None:
    """Map one word onto a class, tolerating the forms a model actually emits.

    Only ever applied to the language model's output, never to the user's
    question -- the strict phrase matcher stays strict for the rules, because a
    loose match there would misread questions the rules currently get right.

    The model reaches for morphological variants of the vocabulary it was given:
    "wandering" for walk, "jogging" for run. Discarding those threw away parses
    that were semantically correct, so the word and its stems are matched
    against the phrase table, then against phrase prefixes.
    """
    word = word.strip().lower()
    if word.replace(" ", "_").replace("-", "_") in config.ACTIVITY_INDEX:
        return word.replace(" ", "_").replace("-", "_")

    for form in _stems(word):
        if hits := find_activities(form):
            return hits[0]

    # Last resort: a stem that prefixes a known phrase ("wander" -> "wandering").
    for form in _stems(word):
        if len(form) < 4:
            continue
        for activity, phrases in ACTIVITY_PHRASES.items():
            if any(phrase.startswith(form) or form.startswith(phrase) for phrase in phrases):
                return activity
    return None


def resolve_group_word(word: str) -> str | None:
    """Map one word onto a behaviour group, tolerating near-misses.

    The model returns "activity" where the vocabulary says "active", and
    "rest"/"restful" for resting. These are the same intent, so match on stems
    and prefixes rather than requiring the exact token.
    """
    word = word.strip().lower()
    if word in GROUP_MEMBERS:
        return word

    for form in _stems(word):
        if len(form) < 3:
            continue
        for group in GROUP_MEMBERS:
            if group.startswith(form) or form.startswith(group):
                return group
            # "activity" and "active" share a stem but neither prefixes the
            # other, so compare on the shared opening instead.
            if len(_shared_prefix(form, group)) >= 5:
                return group
        for group, phrases in GROUP_PHRASES.items():
            if any(p.startswith(form) or form.startswith(p) for p in phrases):
                return group
    return None


def find_group(question: str) -> str | None:
    lower = question.lower()
    for group, phrases in GROUP_PHRASES.items():
        if any(re.search(rf"\b{re.escape(p)}\b", lower) for p in phrases):
            return group
    return None


def find_time(question: str) -> float | None:
    """A time reference in seconds from the start, if the question gives one."""
    lower = question.lower()
    # Units must allow the plural: `(?:second|s)\b` never matches "25500 seconds",
    # because \b cannot sit between the "d" of "second" and the following "s".
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", lower)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", lower)
    if match:
        return float(match.group(1)) * 60
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", lower)
    if match:
        return float(match.group(1)) * 3600
    match = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", lower)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)
        return float(h * 3600 + m * 60 + s)
    return None


def classify(question: str) -> str:
    """Which operation the question is asking for."""
    lower = question.lower().strip()
    if any(
        p in lower
        for p in (
            "how long",
            "how much time",
            "how much of",  # "how much of the afternoon did she spend resting"
            "total time",
            "duration",
            "how many hours",
            "how many minutes",
            "how many seconds",
            "time did the user spend",
            "time was the user",
        )
    ):
        return "duration"
    if any(p in lower for p in ("how many times", "how often", "how many bouts", "number of times", "how many separate")):
        return "count"
    if any(p in lower for p in ("more time", "longer", "compare", "or ", "most time", "which activity")):
        if len(find_activities(question)) >= 2 or "most time" in lower or "which activity" in lower:
            return "comparison"
    if any(p in lower for p in ("when did", "when was", "at what time", "begin", "began", "start", "onset", "first")):
        return "temporal"
    if lower.startswith(VERIFICATION_STARTS) or "whether" in lower:
        return "verification"
    if any(p in lower for p in ("what activity", "what is the user", "what was the user", "what are they", "doing")):
        return "identification"
    return "open_world"


def parse_question(question: str) -> Intent:
    """Reduce a question to an operation, the classes it names, and a time window."""
    activities = find_activities(question)
    group = find_group(question)
    if not activities and group:
        activities = list(GROUP_MEMBERS[group])
    return Intent(
        operation=classify(question),
        activities=activities,
        window=find_window(question),
        at_time_s=find_time(question),
        group=group,
        source="rules",
    )


def _select(timeline: Timeline, activities: list[str], window: TimeWindow | None) -> list[Interval]:
    """Intervals for these activities, trimmed to the window if there is one."""
    chosen = [i for a in activities for i in timeline.by_activity(a)]
    chosen.sort(key=lambda i: i.start_s)
    return clip(chosen, window, timeline.duration_s)


def _window_note(window: TimeWindow | None, timeline: Timeline) -> str:
    """A clause naming the restriction, so an answer never hides one."""
    if window is None:
        return ""
    return f" {window.resolve(timeline.duration_s).describe()}"


# ── Explanation helpers ──────────────────────────────────────────────────────


def _describe_signal(intervals: list[Interval]) -> str:
    """A short, quantitative phrase drawn from the cited intervals themselves."""
    if not intervals:
        return ""
    longest = max(intervals, key=lambda i: i.duration_s)
    signal = longest.signal
    if not signal:
        return ""
    parts = []
    if "body_acc_rms" in signal:
        parts.append(f"body-acceleration energy {signal['body_acc_rms']:.3f} g")
    if "cadence_hz" in signal and signal.get("body_acc_rms", 0) > 0.05:
        parts.append(f"a {signal['cadence_hz']:.1f} Hz cadence")
    if "gravity_tilt_deg" in signal:
        parts.append(f"a gravity vector {signal['gravity_tilt_deg']:.0f} deg from the device axis")
    if "gyro_rms" in signal:
        parts.append(f"gyroscope energy {signal['gyro_rms']:.3f} rad/s")
    return ", ".join(parts[:3])


def _sampling_note(timeline: Timeline, intervals: list[Interval]) -> str:
    observed = sum(i.observed_s for i in intervals)
    spanned = sum(i.duration_s for i in intervals)
    if spanned <= 0:
        return ""
    return (
        f" The recording samples about {timeline.observed_s / timeline.duration_s * 100:.0f}% of "
        f"wall-clock time, so this span of {spanned:.0f} s rests on {observed:.0f} s of "
        f"directly observed sensor data."
    )


def _label(activity: str) -> str:
    return config.DISPLAY_NAMES.get(activity, activity)


# ── The operations ───────────────────────────────────────────────────────────


def answer_identification(question: str, timeline: Timeline, intent: Intent | None = None) -> Answer:
    intent = intent or parse_question(question)
    if intent.window is not None:
        present = timeline.present_activities()
        totals = {a: sum(i.duration_s for i in _select(timeline, [a], intent.window)) for a in present}
        totals = {a: v for a, v in totals.items() if v > 0}
        if not totals:
            return _none(
                f"No activity was detected{_window_note(intent.window, timeline)}.", "identification"
            )
        dominant = max(totals, key=lambda a: totals[a])
        intervals = _select(timeline, [dominant], intent.window)
        return _from(
            intervals, _label(dominant), _label(dominant),
            f"{_label(dominant).capitalize()} accounts for the greatest share"
            f"{_window_note(intent.window, timeline)} ({totals[dominant] / 60:.0f} min across "
            f"{len(intervals)} interval{'s' if len(intervals) != 1 else ''}), shown by "
            f"{_describe_signal(intervals) or 'the observed signal'}.",
            "identification",
        )

    time_s = intent.at_time_s
    if time_s is not None:
        window = timeline.at_time(time_s)
        if window is not None:
            interval = timeline.interval_at(time_s)
            intervals = [interval] if interval else []
            return _from(
                intervals,
                _label(window.activity),
                _label(window.activity),
                f"The sensor window covering {time_s:.0f} s was classified as "
                f"{_label(window.activity)} (confidence {window.confidence:.2f}), on "
                f"{_describe_signal(intervals) or 'the observed signal'}.",
                "identification",
            )

        # The instant was not directly sampled -- ExtraSensory records about 15 s
        # in every 60 -- but it may still fall inside a bout bracketed by windows
        # on both sides. Answering from the enclosing bout is better supported
        # than refusing, provided the gap is stated rather than hidden.
        interval = timeline.interval_at(time_s)
        if interval is not None:
            return _from(
                [interval],
                _label(interval.activity),
                _label(interval.activity),
                f"No window was sampled exactly at {time_s:.0f} s, but that moment lies inside a "
                f"{_label(interval.activity)} bout running from {interval.start_s:.0f} to "
                f"{interval.end_s:.0f} s, supported by {interval.n_windows} sampled windows showing "
                f"{_describe_signal([interval]) or 'a consistent signal'}.",
                "identification",
            )

        return _none(
            f"{time_s:.0f} s falls outside every sampled window and outside every detected "
            f"activity bout, so no activity can be attributed to that moment.",
            "identification",
        )

    dominant = timeline.dominant()
    if dominant is None:
        return _none("The recording contains no usable sensor windows.", "identification")
    intervals = timeline.by_activity(dominant)

    # A single short recording is one classification, not a share of a day.
    # Phrasing it as "the greatest share of the recording (0 min across 1 bouts)"
    # is both odd and misleading for the Task 1 single-window case.
    if len(timeline.windows) == 1:
        window = timeline.windows[0]
        return _from(
            intervals,
            _label(window.activity),
            _label(window.activity),
            f"This {timeline.duration_s:.0f}-second recording was classified as "
            f"{_label(window.activity)} (confidence {window.confidence:.2f}), on "
            f"{_describe_signal(intervals) or 'the observed signal'}.",
            "identification",
        )

    return _from(
        intervals,
        _label(dominant),
        _label(dominant),
        f"{_label(dominant).capitalize()} accounts for the greatest share of the recording "
        f"({timeline.total_duration(dominant) / 60:.0f} min across {len(intervals)} "
        f"bout{'s' if len(intervals) != 1 else ''}), shown by "
        f"{_describe_signal(intervals) or 'the observed signal'}.",
        "identification",
    )


def answer_verification(question: str, timeline: Timeline, intent: Intent | None = None) -> Answer:
    intent = intent or parse_question(question)
    if not intent.activities:
        return _none("The question does not name an activity this system recognises.", "verification")

    intervals = _select(timeline, intent.activities, intent.window)
    if intent.at_time_s is not None and intent.window is None:
        intervals = [i for i in intervals if i.start_s <= intent.at_time_s <= i.end_s]

    name = " or ".join(_label(a) for a in intent.activities)
    where = f" at {intent.at_time_s:.0f} s" if (intent.at_time_s is not None and intent.window is None) else ""
    where += _window_note(intent.window, timeline)

    if not intervals:
        detected = ", ".join(_label(a) for a in timeline.present_activities()) or "none"
        return Answer(
            "No", name, "N/A", "N/A", "N/A",
            f"No window was classified as {name}{where}. "
            f"The activities detected in the recording were: {detected}.",
            "verification", [],
        )

    total = sum(i.duration_s for i in intervals)
    return _from(
        intervals, "Yes", name,
        f"{name.capitalize()} was detected{where} across {len(intervals)} interval"
        f"{'s' if len(intervals) > 1 else ''} totalling {total:.0f} s, identified from "
        f"{_describe_signal(intervals) or 'the observed signal'}.",
        "verification",
    )


def answer_duration(question: str, timeline: Timeline, intent: Intent | None = None) -> Answer:
    intent = intent or parse_question(question)
    if not intent.activities:
        return _none("The question does not name an activity this system recognises.", "duration")

    intervals = _select(timeline, intent.activities, intent.window)
    name = " and ".join(_label(a) for a in intent.activities)
    where = _window_note(intent.window, timeline)

    if not intervals:
        return Answer(
            "0 seconds", name, "N/A", "N/A", "N/A",
            f"No window was classified as {name}{where}.", "duration", [],
        )

    total = sum(i.duration_s for i in intervals)
    return _from(
        intervals, f"{total:.0f} seconds", name,
        f"{name.capitalize()} was detected{where} in {len(intervals)} interval"
        f"{'s' if len(intervals) > 1 else ''} spanning {total:.0f} s in total"
        f" ({total / 60:.0f} min)." + _sampling_note(timeline, intervals),
        "duration",
    )


def answer_count(question: str, timeline: Timeline, intent: Intent | None = None) -> Answer:
    intent = intent or parse_question(question)
    if not intent.activities:
        return _none("The question does not name an activity this system recognises.", "count")

    activity = intent.activities[0]
    intervals = _select(timeline, [activity], intent.window)
    name = _label(activity)
    where = _window_note(intent.window, timeline)

    if not intervals:
        return Answer("0", name, "N/A", "N/A", "N/A", f"No {name} bout was detected{where}.", "count", [])
    return _from(
        intervals, str(len(intervals)), f"{name} bouts",
        f"{len(intervals)} separate {name} bout{'s' if len(intervals) > 1 else ''} were detected{where}. "
        f"A bout is a run of consecutive windows of the same activity separated by no more than "
        f"180 s; the longest lasted {max(i.duration_s for i in intervals):.0f} s.",
        "count",
    )


def answer_comparison(question: str, timeline: Timeline, intent: Intent | None = None) -> Answer:
    intent = intent or parse_question(question)
    where = _window_note(intent.window, timeline)

    def total(activity: str) -> float:
        return sum(i.duration_s for i in _select(timeline, [activity], intent.window))

    if len(intent.activities) < 2:
        present = timeline.present_activities()
        if not present:
            return _none("The recording contains no usable sensor windows.", "comparison")
        ranked = sorted(present, key=total, reverse=True)
        winner = ranked[0]
        intervals = _select(timeline, [winner], intent.window)
        return _from(
            intervals, _label(winner), ", ".join(_label(a) for a in ranked[:3]),
            f"{_label(winner).capitalize()} occupies the most time{where} "
            f"({total(winner) / 60:.0f} min), ahead of "
            + ", ".join(f"{_label(a)} ({total(a) / 60:.0f} min)" for a in ranked[1:3]) + ".",
            "comparison",
        )

    first, second = intent.activities[0], intent.activities[1]
    first_total, second_total = total(first), total(second)
    if first_total > second_total:
        verdict = _label(first)
    elif second_total > first_total:
        verdict = _label(second)
    else:
        verdict = "Equal"

    intervals = _select(timeline, [first, second], intent.window)
    return _from(
        intervals, verdict, f"{_label(first)}, {_label(second)}",
        f"{_label(first).capitalize()} totals {first_total:.0f} s and {_label(second)} totals "
        f"{second_total:.0f} s{where}, so "
        f"{verdict.lower() + ' occupies more of the recording' if verdict != 'Equal' else 'the two are equal'}.",
        "comparison",
    )


def answer_temporal(question: str, timeline: Timeline, intent: Intent | None = None) -> Answer:
    intent = intent or parse_question(question)
    if not intent.activities:
        return _none("The question does not name an activity this system recognises.", "temporal")

    activity = intent.activities[0]
    name = _label(activity)
    intervals = _select(timeline, [activity], intent.window)
    where = _window_note(intent.window, timeline)

    if not intervals:
        return Answer(
            "No", name, "N/A", "N/A", "N/A",
            f"No window was classified as {name}{where}, so it has no onset there.",
            "temporal", [],
        )

    first = intervals[0]
    return _from(
        [first], f"Yes, {name} began at {first.start_s:.0f} seconds", f"Onset of {name}",
        f"The first window classified as {name}{where} begins at {first.start_s:.0f} s and the "
        f"bout continues to {first.end_s:.0f} s, marked by "
        f"{_describe_signal([first]) or 'the observed signal'}.",
        "temporal",
    )


# ── Router ───────────────────────────────────────────────────────────────────

HANDLERS = {
    "identification": answer_identification,
    "verification": answer_verification,
    "duration": answer_duration,
    "count": answer_count,
    "comparison": answer_comparison,
    "temporal": answer_temporal,
}


def answer_question(question: str, timeline: Timeline, use_slm: bool = True) -> Answer:
    """Answer one question against one timeline, in the brief's output format.

    Three stages, cheapest first:

    1. **Rules.** A keyword router reduces the question to an `Intent` and the
       matching handler computes the answer. This covers the great majority of
       questions and costs well under a millisecond.
    2. **The model as a parser.** If the rules cannot place the question -- an
       unfamiliar synonym, a typo, an unusual phrasing -- the language model is
       asked what the question is *asking for*, not what the answer is. Its
       reply is validated against a closed vocabulary and handed to the same
       handler, so the answer is still computed from the timeline.
    3. **The model as an answerer.** Only if parsing also fails does the model
       answer directly, choosing from a menu of real intervals (see `slm.py`).

    Putting the parser ahead of the answerer matters: a 0.5B model is reliable
    at "which operation is this?" and unreliable at "what did this person do?".
    """
    intent = parse_question(question)

    if intent.operation in HANDLERS:
        result = HANDLERS[intent.operation](question, timeline, intent)
        if result.answer != "N/A" or not use_slm:
            return result

    if not use_slm:
        return _none("This question falls outside the system's structured operations.", "open_world")

    try:
        from asqa.slm import answer_open_world, parse_intent

        parsed = parse_intent(question, intent)
        if parsed is not None and parsed.operation in HANDLERS:
            result = HANDLERS[parsed.operation](question, timeline, parsed)
            if result.answer != "N/A":
                result.explanation += (
                    " (The wording was unfamiliar, so a language model mapped it to "
                    f"{', '.join(_label(a) for a in parsed.activities)}; the answer itself "
                    "is computed from the sensor timeline as usual.)"
                )
                return result

        return answer_open_world(question, timeline)
    except Exception as exc:  # noqa: BLE001 - the SLM is optional at runtime
        return _none(
            f"This question falls outside the system's structured operations and the "
            f"language model could not be used ({type(exc).__name__}: {exc}).",
            "open_world",
        )
