#!/usr/bin/env python3
"""L3 -- aggregation: turn per-window evidence into intervals, durations and counts.

This is the layer the temporal questions are actually answered from.  It holds
two ideas that the earlier implementation got wrong and that the brief cares
about directly.

**Observed time is not spanned time.**  ExtraSensory observes 15 seconds out of
every 60.  A walking bout running from 900 s to 1200 s therefore *spans* 300
seconds but was only *observed* for about 75 of them.  Reporting the span as a
duration would overstate every answer by roughly 4x; reporting only the observed
seconds understates what the person actually did.  Both numbers are kept, the
distinction is stated in the answer, and `duration_s` reports the span, which is
what "how long was the user walking" means in ordinary language.

**Evidence must be citable.**  An answer resting on 256 windows cannot cite 256
timestamps -- the previous implementation emitted exactly that, an unreadable and
unscoreable wall of ranges.  Intervals are merged, then the largest few are cited
with the remainder summarised, so `Timestamp(s)` stays checkable by a human.

Modality and channel strings come from `config.evidence_source()`, the single
definition shared with the ground-truth builder.  When the two disagreed, every
grounded answer scored zero regardless of how good the model was.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from asqa import config, features as feat

# Windows of the same activity separated by no more than this are one bout.  At
# ExtraSensory's one-minute cadence this bridges a couple of missed samples
# without welding genuinely separate episodes together.
DEFAULT_MERGE_GAP_S = 180.0

# How many intervals an answer may cite before the rest are summarised.
MAX_CITED_INTERVALS = 6

# Features quoted in explanations, in the order they read most naturally.
EXPLANATION_FEATURES = (
    "body_acc_rms",
    "cadence_hz",
    "gravity_tilt_deg",
    "gyro_rms",
    "jerk_rms",
    "spectral_entropy",
)


@dataclass
class Window:
    """One analysed sensor window, in seconds from the start of the recording."""

    start_s: float
    end_s: float
    activity: str
    confidence: float
    epoch_s: int
    signal: dict[str, float] = field(default_factory=dict)

    def covers(self, time_s: float) -> bool:
        return self.start_s <= time_s <= self.end_s


@dataclass
class Interval:
    """A contiguous bout of one activity."""

    activity: str
    start_s: float
    end_s: float
    observed_s: float
    n_windows: int
    mean_confidence: float
    signal: dict[str, float] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Wall-clock span of the bout."""
        return self.end_s - self.start_s

    @property
    def unobserved_s(self) -> float:
        return max(0.0, self.duration_s - self.observed_s)

    def describe(self) -> str:
        return f"{self.start_s:.0f} to {self.end_s:.0f}"


@dataclass
class Timeline:
    """Everything the interface layer is allowed to say about a recording."""

    time_base: str
    recording_start_epoch: int
    duration_s: float
    windows: list[Window]
    intervals: list[Interval]
    observed_s: float
    used_context_model: bool = True

    # ── queries the interface layer builds answers from ──

    def by_activity(self, activity: str) -> list[Interval]:
        return [interval for interval in self.intervals if interval.activity == activity]

    def total_duration(self, activity: str) -> float:
        return sum(interval.duration_s for interval in self.by_activity(activity))

    def total_observed(self, activity: str) -> float:
        return sum(interval.observed_s for interval in self.by_activity(activity))

    def count(self, activity: str) -> int:
        return len(self.by_activity(activity))

    def first(self, activity: str) -> Interval | None:
        intervals = self.by_activity(activity)
        return intervals[0] if intervals else None

    def at_time(self, time_s: float) -> Window | None:
        """The observed window covering a time, or None if nothing was sampled."""
        for window in self.windows:
            if window.covers(time_s):
                return window
        return None

    def interval_at(self, time_s: float) -> Interval | None:
        for interval in self.intervals:
            if interval.start_s <= time_s <= interval.end_s:
                return interval
        return None

    def dominant(self) -> str | None:
        if not self.intervals:
            return None
        totals: dict[str, float] = {}
        for interval in self.intervals:
            totals[interval.activity] = totals.get(interval.activity, 0.0) + interval.duration_s
        return max(totals, key=lambda a: totals[a])

    def present_activities(self) -> list[str]:
        seen = {interval.activity for interval in self.intervals}
        return [a for a in config.ACTIVITIES if a in seen]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "time_base": self.time_base,
            "recording_start_epoch": self.recording_start_epoch,
            "duration_s": self.duration_s,
            "observed_s": self.observed_s,
            "used_context_model": self.used_context_model,
            "evidence_policy": (
                "observed_s counts only directly sampled sensor time; duration_s is the "
                "wall-clock span of the bout, which includes the unsampled gaps between "
                "windows. ExtraSensory samples about 15 s in every 60 s."
            ),
            "interval_count": len(self.intervals),
            "window_count": len(self.windows),
            "intervals": [
                {**asdict(interval), "duration_s": interval.duration_s, "unobserved_s": interval.unobserved_s}
                for interval in self.intervals
            ],
            "windows": [asdict(window) for window in self.windows],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


# ── Construction ─────────────────────────────────────────────────────────────


def _signal_summary(X_row: np.ndarray) -> dict[str, float]:
    """The handful of physical quantities an explanation may quote."""
    summary = {}
    for name in EXPLANATION_FEATURES:
        if name in feat.FEATURE_NAMES:
            summary[name] = round(float(X_row[feat.FEATURE_NAMES.index(name)]), 4)
    return summary


def build_timeline(
    activities: np.ndarray,
    epoch_ts: np.ndarray,
    X: np.ndarray,
    confidences: np.ndarray | None = None,
    window_duration_s: float = config.WINDOW_DURATION_S,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    used_context_model: bool = True,
) -> Timeline:
    """Assemble a timeline from decoded per-window activities.

    ``epoch_ts`` are the recording's window timestamps; everything downstream is
    expressed as seconds from the earliest of them, which is the convention the
    brief asks be stated and held to.
    """
    if len(activities) == 0:
        return Timeline(config.TIME_BASE, 0, 0.0, [], [], 0.0, used_context_model)

    order = np.argsort(epoch_ts)
    activities = np.asarray(activities)[order]
    epoch_ts = np.asarray(epoch_ts)[order]
    X = np.asarray(X)[order]
    confidences = (
        np.ones(len(activities)) if confidences is None else np.asarray(confidences)[order]
    )

    start_epoch = int(epoch_ts[0])
    windows = [
        Window(
            start_s=float(ts - start_epoch),
            end_s=float(ts - start_epoch + window_duration_s),
            activity=str(activity),
            confidence=round(float(confidence), 4),
            epoch_s=int(ts),
            signal=_signal_summary(row),
        )
        for ts, activity, confidence, row in zip(epoch_ts, activities, confidences, X)
    ]

    intervals: list[Interval] = []
    current: list[Window] = [windows[0]]
    for window in windows[1:]:
        previous = current[-1]
        same = window.activity == previous.activity
        # Gap measured from the end of the last observed window, so back-to-back
        # windows read as a gap of zero.
        if same and (window.start_s - previous.end_s) <= merge_gap_s:
            current.append(window)
        else:
            intervals.append(_close(current))
            current = [window]
    intervals.append(_close(current))

    observed = sum(window.end_s - window.start_s for window in windows)
    duration = windows[-1].end_s - windows[0].start_s
    return Timeline(
        time_base=config.TIME_BASE,
        recording_start_epoch=start_epoch,
        duration_s=round(duration, 2),
        windows=windows,
        intervals=intervals,
        observed_s=round(observed, 2),
        used_context_model=used_context_model,
    )


def _close(group: list[Window]) -> Interval:
    observed = sum(window.end_s - window.start_s for window in group)
    signal: dict[str, float] = {}
    for name in EXPLANATION_FEATURES:
        values = [window.signal[name] for window in group if name in window.signal]
        if values:
            signal[name] = round(float(np.median(values)), 4)
    return Interval(
        activity=group[0].activity,
        start_s=round(group[0].start_s, 2),
        end_s=round(group[-1].end_s, 2),
        observed_s=round(observed, 2),
        n_windows=len(group),
        mean_confidence=round(float(np.mean([w.confidence for w in group])), 4),
        signal=signal,
    )


# ── Evidence formatting ──────────────────────────────────────────────────────


def cite(intervals: list[Interval], limit: int = MAX_CITED_INTERVALS) -> str:
    """Render intervals as a `Timestamp(s)` field a person can actually check.

    The longest bouts are cited by name and the remainder summarised, because an
    answer that lists every one of 256 windows cannot be verified by a reader or
    scored by a grader.
    """
    if not intervals:
        return "N/A"

    ranked = sorted(intervals, key=lambda i: i.duration_s, reverse=True)
    shown = sorted(ranked[:limit], key=lambda i: i.start_s)
    rendered = ", ".join(interval.describe() for interval in shown)

    remainder = len(intervals) - len(shown)
    if remainder > 0:
        extra = sum(interval.duration_s for interval in ranked[limit:])
        rendered += f" (+{remainder} shorter interval{'s' if remainder > 1 else ''} totalling {extra:.0f} s)"
    return f"{rendered} (seconds from start)"


def evidence_block(intervals: list[Interval]) -> dict[str, str]:
    """The three Evidence fields the brief requires, for a set of intervals."""
    if not intervals:
        return {"timestamps": "N/A", "modality": "N/A", "channels": "N/A"}
    source = config.evidence_source()
    return {"timestamps": cite(intervals), "modality": source["modality"], "channels": source["channels"]}


def load_timeline(path: Path) -> Timeline:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    windows = [Window(**{k: v for k, v in w.items()}) for w in data["windows"]]
    intervals = [
        Interval(
            activity=i["activity"],
            start_s=i["start_s"],
            end_s=i["end_s"],
            observed_s=i["observed_s"],
            n_windows=i["n_windows"],
            mean_confidence=i["mean_confidence"],
            signal=i.get("signal", {}),
        )
        for i in data["intervals"]
    ]
    return Timeline(
        time_base=data["time_base"],
        recording_start_epoch=data["recording_start_epoch"],
        duration_s=data["duration_s"],
        windows=windows,
        intervals=intervals,
        observed_s=data["observed_s"],
        used_context_model=data.get("used_context_model", True),
    )
