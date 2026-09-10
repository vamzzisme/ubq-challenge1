#!/usr/bin/env python3
"""Tests for time windows and for validating what the language model proposes.

Two things are guarded here.

A window must actually *restrict* the answer. "How long was the user walking
after 84 hours" previously parsed the number and then dropped it, so the answer
silently covered the whole recording — a wrong answer that looked authoritative.

And the model's proposal must be validated against a closed vocabulary. The
0.5B model was measured returning `"jogging"`, a word that is not a class; the
validator must either canonicalise it or drop it, never pass it through.
"""

from __future__ import annotations

from asqa import config
from asqa.answer import answer_question, parse_question
from asqa.intent import TimeWindow, clip, find_window
from asqa.timeline import Interval, Timeline, Window

HOUR = 3600.0


def _timeline() -> Timeline:
    """Six hours: walking in hour 1, sitting in hour 3, walking again in hour 5."""
    plan = [("walking", 0, HOUR), ("sitting", 2 * HOUR, 3 * HOUR), ("walking", 4 * HOUR, 5 * HOUR)]
    windows, intervals = [], []
    for activity, start, end in plan:
        n = int((end - start) // 60)
        for i in range(n):
            t = start + i * 60
            windows.append(Window(t, t + 15, activity, 0.8, int(t), {"body_acc_rms": 0.3}))
        intervals.append(Interval(activity, start, end, n * 15.0, n, 0.8, {"body_acc_rms": 0.3}))
    return Timeline(config.TIME_BASE, 0, 6 * HOUR, windows, intervals, sum(15.0 for _ in windows))


# ── parsing ──


def test_after() -> None:
    w = find_window("How long was the user walking after 84 hours?")
    assert w is not None and w.start_s == 84 * HOUR and w.end_s is None


def test_before() -> None:
    w = find_window("Did she walk before 12 hours?")
    assert w is not None and w.end_s == 12 * HOUR and w.start_s is None


def test_between() -> None:
    w = find_window("Did she walk between 10 and 20 hours?")
    assert w is not None and (w.start_s, w.end_s) == (10 * HOUR, 20 * HOUR)


def test_first_and_last() -> None:
    first = find_window("How long walking in the first 2 hours?")
    assert first is not None and (first.start_s, first.end_s) == (0.0, 2 * HOUR)

    last = find_window("What was he doing in the last 3 hours?")
    assert last is not None and last.from_end
    # Only resolvable once the recording length is known.
    assert last.resolve(10 * HOUR).start_s == 7 * HOUR


def test_no_window_when_none_given() -> None:
    assert find_window("How long was the user walking?") is None


def test_minutes_and_seconds() -> None:
    assert find_window("after 90 minutes").start_s == 5400
    assert find_window("after 300 seconds").start_s == 300


# ── clipping ──


def test_clip_trims_a_straddling_interval() -> None:
    """A bout crossing the boundary contributes only its in-window part."""
    interval = Interval("walking", 0.0, 1000.0, 500.0, 10, 0.9, {})
    out = clip([interval], TimeWindow(start_s=600.0), duration_s=2000.0)
    assert len(out) == 1
    assert out[0].start_s == 600.0 and out[0].end_s == 1000.0
    assert out[0].duration_s == 400.0
    # Observed time scales with the kept fraction, not the whole bout.
    assert abs(out[0].observed_s - 200.0) < 1.0


def test_clip_drops_intervals_outside() -> None:
    interval = Interval("walking", 0.0, 500.0, 250.0, 5, 0.9, {})
    assert clip([interval], TimeWindow(start_s=600.0), 2000.0) == []


# ── end to end ──


def test_window_restricts_a_duration_answer() -> None:
    timeline = _timeline()
    whole = answer_question("How long was the user walking?", timeline, use_slm=False)
    after = answer_question("How long was the user walking after 3 hours?", timeline, use_slm=False)
    assert whole.answer == "7200 seconds", whole.answer
    assert after.answer == "3600 seconds", f"window ignored: {after.answer}"


def test_window_can_make_the_answer_negative() -> None:
    """The honest case: the activity happened, but not inside the window."""
    answer = answer_question("Did the user walk between 2 and 3 hours?", _timeline(), use_slm=False)
    assert answer.answer == "No", answer.answer
    assert answer.timestamps == "N/A"


def test_window_appears_in_the_explanation() -> None:
    answer = answer_question("How long was the user walking after 3 hours?", _timeline(), use_slm=False)
    assert "3.0 h" in answer.explanation, answer.explanation


def test_tiring_reaches_the_active_group_without_a_model() -> None:
    """A synonym the rules know must not need the language model."""
    intent = parse_question("was the user doing anything tiring?")
    assert intent.group == "active"
    assert set(intent.activities) == {"walking", "running", "bicycling"}


# ── validating the model's proposal ──


def test_model_words_are_canonicalised_or_dropped() -> None:
    from asqa.answer import find_activities

    # The 0.5B model really does return these instead of the class names.
    assert find_activities("jogging") == ["running"]
    assert find_activities("cycling") == ["bicycling"]
    # And a word that means nothing here resolves to nothing.
    assert find_activities("gardening") == []


def test_model_morphology_is_tolerated() -> None:
    """The model emits variants of its own vocabulary; those are not nonsense.

    Measured: asked about "did the user wander after 84h?" the 0.5B model
    returned activities=["wandering"] and group="activity". Both were correct in
    substance and both were being discarded, so a right answer became "N/A".
    """
    from asqa.answer import resolve_activity_word, resolve_group_word

    assert resolve_activity_word("wandering") == "walking"
    assert resolve_activity_word("jogging") == "running"
    assert resolve_activity_word("pedalling") == "bicycling"
    assert resolve_group_word("activity") == "active"
    assert resolve_group_word("activities") == "active"
    assert resolve_group_word("restful") == "resting"


def test_tolerance_still_rejects_nonsense() -> None:
    """Tolerance must not become a licence to accept anything."""
    from asqa.answer import resolve_activity_word, resolve_group_word

    for word in ("gardening", "swimming", "teleporting", ""):
        assert resolve_activity_word(word) is None, word
    for word in ("nonsense", "purple", ""):
        assert resolve_group_word(word) is None, word


def test_wander_needs_no_model() -> None:
    """A real walking synonym belongs in the rules, not in the model's job."""
    intent = parse_question("did the user wander after 84h?")
    assert intent.operation == "verification"
    assert intent.activities == ["walking"]
    assert intent.window is not None and intent.window.start_s == 84 * HOUR


def test_compact_hour_suffix() -> None:
    """"84h" must parse exactly as "84 hours" does."""
    assert find_window("after 84h?").start_s == 84 * HOUR
    assert find_window("after 90m?") is None or find_window("after 90 minutes").start_s == 5400


def test_invalid_operation_is_rejected() -> None:
    from asqa.intent import from_payload

    assert from_payload({"operation": "teleporting", "activities": ["walking"]}) is None
    assert from_payload({"operation": "duration", "activities": ["flying"]}) is None
    good = from_payload({"operation": "duration", "activities": ["walking"]})
    assert good is not None and good.activities == ["walking"]


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
