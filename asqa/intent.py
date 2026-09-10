#!/usr/bin/env python3
"""What a question is asking for, separated from how it was worded.

Splitting intent out from the handlers buys two things.

The first is **time windows**.  "How long was the user walking *after 84 hours*"
was previously parsed as far as the number 302400 and then the constraint was
silently dropped, so the answer covered the whole recording.  A window is now a
first-class part of the request and every handler applies it.

The second is **a target for the language model**.  A 0.5B model is poor at
judging what a person did, but good at mapping "anything tiring?" onto
`operation=verification, group=active`.  Producing an `Intent` is a job it can
do safely: the intent names an operation and a class, never a number or a
timestamp, so the answer is still computed from the timeline.  An intent that
names something outside these vocabularies is rejected rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

from asqa import config
from asqa.timeline import Interval


@dataclass(frozen=True)
class TimeWindow:
    """A span of the recording, in seconds from its start.

    Either bound may be None, meaning "open in that direction".  `from_end`
    marks a window expressed relative to the end ("the last two hours"), which
    can only be resolved once the recording's duration is known.
    """

    start_s: float | None = None
    end_s: float | None = None
    from_end: bool = False
    phrase: str = ""

    def resolve(self, duration_s: float) -> "TimeWindow":
        if not self.from_end:
            return self
        span = self.start_s or 0.0
        return TimeWindow(max(0.0, duration_s - span), None, False, self.phrase)

    def describe(self) -> str:
        def hhmm(value: float) -> str:
            return f"{value / 3600:.1f} h" if value >= 3600 else f"{value:.0f} s"

        if self.from_end:
            return f"the last {hhmm(self.start_s or 0.0)}"
        if self.start_s is not None and self.end_s is not None:
            return f"between {hhmm(self.start_s)} and {hhmm(self.end_s)}"
        if self.start_s is not None:
            return f"after {hhmm(self.start_s)}"
        if self.end_s is not None:
            return f"before {hhmm(self.end_s)}"
        return "the whole recording"


@dataclass
class Intent:
    """A question reduced to an operation over the timeline."""

    operation: str
    activities: list[str]
    window: TimeWindow | None = None
    at_time_s: float | None = None
    group: str | None = None
    source: str = "rules"  # "rules" or "language-model"


# ── Units ────────────────────────────────────────────────────────────────────

_UNITS = (
    (r"hours?|hrs?\b|h\b", 3600.0),
    (r"minutes?|mins?\b", 60.0),
    (r"seconds?|secs?\b|s\b", 1.0),
)


def _quantity(text: str) -> float | None:
    """The first duration in a fragment, in seconds."""
    for pattern, scale in _UNITS:
        match = re.search(rf"(\d+(?:\.\d+)?)\s*(?:{pattern})", text)
        if match:
            return float(match.group(1)) * scale
    match = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", text)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)
        return float(h * 3600 + m * 60 + s)
    return None


def find_window(question: str) -> TimeWindow | None:
    """Parse a range qualifier, if the question carries one.

    Handles: after/past/beyond X, before/within/under X, between X and Y,
    the first X, the last X.
    """
    lower = question.lower()

    match = re.search(
        r"between\s+(.{1,24}?)\s+and\s+(.{1,24}?)(?:[,.?]|$)", lower
    )
    if match:
        low, high = _quantity(match.group(1)), _quantity(match.group(2))
        # "between 10 and 20 hours" leaves the unit off the first number.
        if low is None and high is not None:
            number = re.search(r"(\d+(?:\.\d+)?)", match.group(1))
            unit = high / float(re.search(r"(\d+(?:\.\d+)?)", match.group(2)).group(1))
            low = float(number.group(1)) * unit if number else None
        if low is not None and high is not None:
            return TimeWindow(min(low, high), max(low, high), phrase=match.group(0).strip())

    match = re.search(r"\b(?:the\s+)?first\s+(.{1,24}?)(?:[,.?]|$| of)", lower)
    if match and (value := _quantity(match.group(1))) is not None:
        return TimeWindow(0.0, value, phrase=f"the first {match.group(1).strip()}")

    match = re.search(r"\b(?:the\s+)?last\s+(.{1,24}?)(?:[,.?]|$| of)", lower)
    if match and (value := _quantity(match.group(1))) is not None:
        return TimeWindow(value, None, from_end=True, phrase=f"the last {match.group(1).strip()}")

    match = re.search(r"\b(?:after|past|beyond|since|from)\s+(.{1,24}?)(?:[,.?]|$)", lower)
    if match and (value := _quantity(match.group(1))) is not None:
        return TimeWindow(value, None, phrase=f"after {match.group(1).strip()}")

    match = re.search(r"\b(?:before|under|within|up to|prior to)\s+(.{1,24}?)(?:[,.?]|$)", lower)
    if match and (value := _quantity(match.group(1))) is not None:
        return TimeWindow(None, value, phrase=f"before {match.group(1).strip()}")

    return None


# ── Applying a window to evidence ────────────────────────────────────────────


def clip(intervals: Iterable[Interval], window: TimeWindow | None, duration_s: float) -> list[Interval]:
    """Restrict intervals to a window, trimming those that straddle its edge.

    Trimming rather than filtering matters: a walk from 80 h to 90 h contributes
    only its post-84 h portion to "how long after 84 hours", and citing the
    whole bout as evidence would overstate what the window contains.  Observed
    time is scaled by the same proportion, since sampling is uniform within a
    bout.
    """
    if window is None:
        return list(intervals)

    resolved = window.resolve(duration_s)
    low = resolved.start_s if resolved.start_s is not None else float("-inf")
    high = resolved.end_s if resolved.end_s is not None else float("inf")

    out: list[Interval] = []
    for interval in intervals:
        start, end = max(interval.start_s, low), min(interval.end_s, high)
        if end <= start:
            continue
        if start == interval.start_s and end == interval.end_s:
            out.append(interval)
            continue
        fraction = (end - start) / max(interval.duration_s, 1e-9)
        out.append(
            replace(
                interval,
                start_s=round(start, 2),
                end_s=round(end, 2),
                observed_s=round(interval.observed_s * fraction, 2),
                n_windows=max(1, int(round(interval.n_windows * fraction))),
            )
        )
    return out


# ── Validating a language model's proposal ───────────────────────────────────

OPERATIONS = ("identification", "verification", "duration", "count", "comparison", "temporal")


def from_payload(payload: dict, fallback_question: str = "") -> Intent | None:
    """Build an Intent from a model's JSON, or None if it is not usable.

    Every field is checked against a closed vocabulary.  The model may pick an
    operation and name activities; it may not invent either, and it is never
    trusted for times -- those are re-read from the question by the rules, which
    do not hallucinate.
    """
    operation = str(payload.get("operation", "")).strip().lower().replace(" ", "_")
    if operation not in OPERATIONS:
        return None

    activities: list[str] = []
    for raw in payload.get("activities", []) or []:
        name = str(raw).strip().lower().replace(" ", "_")
        if name in config.ACTIVITY_INDEX and name not in activities:
            activities.append(name)

    group = payload.get("group")
    group = str(group).strip().lower() if group else None
    if group not in (None, "", "null", "none"):
        from asqa.answer import GROUP_MEMBERS

        if group not in GROUP_MEMBERS:
            group = None
        elif not activities:
            activities = list(GROUP_MEMBERS[group])
    else:
        group = None

    if operation != "identification" and not activities:
        return None  # nothing to operate on

    return Intent(
        operation=operation,
        activities=activities,
        window=find_window(fallback_question),
        at_time_s=None,
        group=group,
        source="language-model",
    )
