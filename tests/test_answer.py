"""Tests for the interface layer: question parsing and the output contract."""

from __future__ import annotations

from asqa import config
from asqa.answer import (
    answer_question,
    classify,
    find_activities,
    find_group,
    find_time,
)
from asqa.timeline import Interval, Timeline, Window, cite


def _timeline() -> Timeline:
    """A small synthetic recording: sitting, a walk, a run, then lying down."""
    windows, intervals = [], []
    plan = [
        ("sitting", 0, 600),
        ("walking", 660, 900),
        ("running", 960, 1080),
        ("lying_down", 1140, 1800),
    ]
    signal = {
        "sitting": {"body_acc_rms": 0.01, "cadence_hz": 0.8, "gravity_tilt_deg": 30.0, "gyro_rms": 0.02},
        "walking": {"body_acc_rms": 0.35, "cadence_hz": 1.8, "gravity_tilt_deg": 70.0, "gyro_rms": 1.0},
        "running": {"body_acc_rms": 0.90, "cadence_hz": 2.9, "gravity_tilt_deg": 65.0, "gyro_rms": 1.8},
        "lying_down": {"body_acc_rms": 0.002, "cadence_hz": 0.5, "gravity_tilt_deg": 5.0, "gyro_rms": 0.001},
    }
    for activity, start, end in plan:
        n = 0
        for t in range(start, end, 60):
            windows.append(Window(float(t), float(t + 15), activity, 0.8, 1_400_000_000 + t, signal[activity]))
            n += 1
        intervals.append(
            Interval(activity, float(start), float(end - 60 + 15), float(n * 15), n, 0.8, signal[activity])
        )
    return Timeline(config.TIME_BASE, 1_400_000_000, 1800.0, windows, intervals, sum(15.0 for _ in windows))


def test_find_time_handles_plural_units() -> None:
    """`at 25500 seconds` must parse; a \\b after "second" cannot match a plural."""
    assert find_time("What was the user doing at 25500 seconds?") == 25500
    assert find_time("at 300 second") == 300
    assert find_time("at 5 minutes") == 300
    assert find_time("at 2 hours") == 7200
    assert find_time("around 10:30") == 37800
    assert find_time("What is the user doing?") is None


def test_find_activities_prefers_the_longer_phrase() -> None:
    assert find_activities("was the user standing in place?") == ["standing_in_place"]
    assert find_activities("standing and moving") == ["standing_and_moving"]
    assert find_activities("was she lying down") == ["lying_down"]
    assert find_activities("cycling or running") == ["bicycling", "running"]
    assert find_activities("running or cycling") == ["running", "bicycling"]


def test_find_group_covers_unlabelled_behaviour() -> None:
    assert find_group("was the user resting?") == "resting"
    assert find_group("anything strenuous?") == "active"


def test_classify_routes_each_tier() -> None:
    assert classify("What activity is the user performing?") == "identification"
    assert classify("Is the user running?") == "verification"
    assert classify("How long was the user walking?") == "duration"
    assert classify("How many times did the user walk?") == "count"
    assert classify("Did the user spend more time walking or running?") == "comparison"
    assert classify("When did the user begin running?") == "temporal"


def test_output_has_the_required_fields_in_order() -> None:
    rendered = answer_question("How long was the user walking?", _timeline(), use_slm=False).render()
    lines = [line.rstrip() for line in rendered.splitlines()]
    assert lines[0].startswith("Answer: ")
    assert lines[1].startswith("Activity/Event: ")
    assert lines[2] == "Evidence:"
    assert lines[3].strip().startswith("Timestamp(s): ")
    assert lines[4].strip().startswith("Sensor Modality: ")
    assert lines[5].strip().startswith("Sensor Channel(s): ")
    assert lines[6].startswith("Explanation: ")


def test_duration_reports_plain_seconds() -> None:
    """The brief's example reads `700 seconds`, not `700 observed seconds`."""
    answer = answer_question("How long was the user walking?", _timeline(), use_slm=False)
    assert answer.answer.endswith("seconds")
    assert "observed seconds" not in answer.answer


def test_pinpoint_question_answers_that_moment() -> None:
    answer = answer_question("What was the user doing at 700 seconds?", _timeline(), use_slm=False)
    assert answer.answer == "walking", f"expected walking at 700 s, got {answer.answer}"


def test_verification_is_negative_when_absent() -> None:
    answer = answer_question("Was the user bicycling?", _timeline(), use_slm=False)
    assert answer.answer == "No"
    assert answer.timestamps == "N/A", "a negative answer must not cite evidence for a non-event"


def test_verification_is_positive_with_evidence() -> None:
    answer = answer_question("Is the user running?", _timeline(), use_slm=False)
    assert answer.answer == "Yes"
    assert answer.timestamps != "N/A"
    assert answer.modality == config.SENSOR_MODALITY


def test_count_counts_bouts() -> None:
    assert answer_question("How many times did the user walk?", _timeline(), use_slm=False).answer == "1"


def test_comparison_picks_the_longer() -> None:
    answer = answer_question(
        "Did the user spend more time sitting or running?", _timeline(), use_slm=False
    )
    assert answer.answer == "sitting"


def test_temporal_reports_the_onset() -> None:
    answer = answer_question("When did the user begin running?", _timeline(), use_slm=False)
    assert "960" in answer.answer, f"expected onset at 960 s, got {answer.answer!r}"


def test_evidence_citation_is_bounded() -> None:
    """An answer must never dump hundreds of ranges into one field."""
    many = [Interval("walking", float(i * 100), float(i * 100 + 50), 15.0, 1, 0.9, {}) for i in range(200)]
    rendered = cite(many)
    assert rendered.count(" to ") <= 7, "too many intervals cited to be checkable"
    assert "shorter interval" in rendered, "the uncited remainder must still be accounted for"


def test_explanation_quotes_real_measurements() -> None:
    answer = answer_question("Is the user running?", _timeline(), use_slm=False)
    assert any(token in answer.explanation for token in ("g", "Hz", "rad/s")), answer.explanation


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
