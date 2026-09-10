#!/usr/bin/env python3
"""Tests for the dashboard's API contract.

The important one is `test_ask_matches_the_cli_exactly`. A dashboard is only a
demonstration of the system if it *is* the system; the moment it starts shaping
its own answers it becomes a second implementation that drifts from the one being
reported on. That test pins the API's answer to what `answer_question()` returns
for the same input, so any divergence fails here rather than in a demo.

These exercise the `Backend` directly rather than over HTTP: the routing is a
dozen lines of stdlib, while the caching, model selection and payload shaping are
where mistakes would actually hide.
"""

from __future__ import annotations

import sys

from asqa import config
from asqa.answer import answer_question
from asqa.dashboard import ACTIVITY_COLOURS, Backend

REQUIRED_ANSWER_FIELDS = {"answer", "activity_event", "evidence", "explanation", "question_type"}
REQUIRED_EVIDENCE_FIELDS = {"timestamps", "modality", "channels", "intervals"}


def _backend() -> Backend | None:
    """A Backend, or None when the repo has no trained models yet."""
    if not sorted(config.MODEL_DIR.glob("recogniser_fold*.joblib")):
        return None
    try:
        return Backend()
    except FileNotFoundError:
        return None


def _first_user(backend: Backend) -> str | None:
    users = backend.users()
    return users[0]["id"] if users else None


def test_every_activity_has_a_colour() -> None:
    """The ribbon cannot render a class it has no colour for."""
    assert set(ACTIVITY_COLOURS) == set(config.ACTIVITIES), (
        f"colour map and ACTIVITIES disagree: "
        f"{set(ACTIVITY_COLOURS) ^ set(config.ACTIVITIES)}"
    )
    assert len(set(ACTIVITY_COLOURS.values())) == len(ACTIVITY_COLOURS), "two classes share a colour"


def test_ui_file_exists_and_is_self_contained() -> None:
    from asqa.dashboard import UI_PATH

    assert UI_PATH.exists(), f"missing {UI_PATH}"
    html = UI_PATH.read_text(encoding="utf-8")
    # Served from a local machine that may be offline; no external fetches.
    for forbidden in ("http://cdn", "https://cdn", "unpkg.com", "googleapis.com"):
        assert forbidden not in html, f"UI reaches out to {forbidden}; it must be self-contained"
    assert "/api/ask" in html and "/api/session" in html and "/api/users" in html


def test_fold_selection_never_leaks(backend: Backend) -> None:
    """A user must never be questioned with a model that trained on them."""
    for row in backend.users():
        if not row["in_corpus"]:
            continue
        fold = backend.fold_for(row["id"])
        split = backend.folds["splits"][str(fold)]
        assert row["id"] in split["test"], (
            f"{row['id'][:8]} would be answered by fold {fold}, which did not hold them out"
        )
        assert row["id"] not in split["train"], f"{row['id'][:8]} leaks into fold {fold} training"


def test_users_payload_shape(backend: Backend) -> None:
    users = backend.users()
    assert users, "no preprocessed users found"
    for row in users:
        assert {"id", "short", "fold", "windows", "dominant", "distribution"} <= set(row)
        assert row["short"] == row["id"][:8]
        assert 0 <= row["fold"] < backend.folds["n_folds"]


def test_session_omits_per_window_detail(backend: Backend, user: str) -> None:
    """The heavy per-window rows must not be shipped to the browser."""
    payload = backend.session(user)
    assert "windows" not in payload, "per-window detail leaked into the session payload (~1 MB)"
    assert payload["intervals"], "no intervals returned"
    assert payload["colours"] == ACTIVITY_COLOURS
    assert payload["duration_s"] > 0

    import json

    size_kb = len(json.dumps(payload)) / 1024
    assert size_kb < 400, f"session payload is {size_kb:.0f} KB; too heavy for an interactive UI"


def test_session_is_cached(backend: Backend, user: str) -> None:
    backend.session(user)
    import time

    started = time.perf_counter()
    backend.session(user)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"second session build took {elapsed:.1f}s; the cache is not working"


def test_ask_returns_the_required_fields(backend: Backend, user: str) -> None:
    payload = backend.ask(user, "How long was the user walking?", use_slm=False)
    assert REQUIRED_ANSWER_FIELDS <= set(payload)
    assert REQUIRED_EVIDENCE_FIELDS <= set(payload["evidence"])
    # `rendered` is what a grader reads; it must carry the brief's exact labels.
    for label in ("Answer:", "Activity/Event:", "Evidence:", "Timestamp(s):",
                  "Sensor Modality:", "Sensor Channel(s):", "Explanation:"):
        assert label in payload["rendered"], f"missing {label!r} in the rendered answer"


def test_ask_matches_the_cli_exactly(backend: Backend, user: str) -> None:
    """The dashboard must not reshape answers. Same input, same output as the CLI."""
    timeline = backend.timeline(user)
    for question in (
        "How long was the user walking?",
        "Is the user running?",
        "What activity is the user performing?",
        "How many times did the user walk?",
    ):
        direct = answer_question(question, timeline, use_slm=False).to_dict()
        via_api = backend.ask(user, question, use_slm=False)
        for field in REQUIRED_ANSWER_FIELDS:
            assert via_api[field] == direct[field], (
                f"dashboard and answer_question disagree on {field!r} for {question!r}:\n"
                f"  api    = {via_api[field]!r}\n  direct = {direct[field]!r}"
            )


def test_cited_intervals_exist_in_the_timeline(backend: Backend, user: str) -> None:
    """Grounding guard: the UI can only highlight intervals the pipeline produced."""
    timeline = backend.timeline(user)
    real = [(i.start_s, i.end_s) for i in timeline.intervals]
    payload = backend.ask(user, "How long was the user walking?", use_slm=False)
    for cited in payload["evidence"]["intervals"]:
        assert any(
            start <= cited["start_s"] and cited["end_s"] <= end
            for start, end in real
        ), f"cited interval {cited} does not correspond to any interval in the timeline"


if __name__ == "__main__":
    backend = _backend()
    user = _first_user(backend) if backend else None
    if backend is None:
        print("SKIP  no trained models; run `python -m asqa.recognise --cv` first")

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        needs = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        if ("backend" in needs or "user" in needs) and (backend is None or user is None):
            print(f"SKIP  {name}")
            continue
        args = []
        if "backend" in needs:
            args.append(backend)
        if "user" in needs:
            args.append(user)
        try:
            fn(*args)
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    sys.exit(1 if failures else 0)
