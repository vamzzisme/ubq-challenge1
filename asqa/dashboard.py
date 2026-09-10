#!/usr/bin/env python3
"""A local dashboard for exploring the system: pick a user, ask questions, see the evidence.

Run it, open the printed URL, choose one of the preprocessed users, and chat with
their recording. Every answer is clickable; clicking one highlights on a timeline
ribbon exactly the intervals that answer cited.

    python -m asqa.dashboard
    python -m asqa.dashboard --port 8080 --open

**This module contains no answering logic.** It is a thin transport over
`Pipeline` and `answer_question`, for the same reason `slm.py` may not invent a
timestamp: a dashboard that computed its own answers would stop demonstrating the
system it claims to demonstrate, and would drift from it silently. If you find
yourself wanting to reshape an answer here, change the interface layer instead.

Bound to 127.0.0.1 — this loads models and reads local data, and is not meant to
be exposed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from asqa import config
from asqa.answer import answer_question
from asqa.pipeline import Pipeline
from asqa.splits import load_folds
from asqa.timeline import Timeline, load_timeline

UI_PATH = Path(__file__).resolve().parent / "dashboard.html"

# Colours for the seven activities, taken from the validated categorical palette
# used by asqa/figures.py: fixed slot order, never cycled.
ACTIVITY_COLOURS = {
    "lying_down": "#4a3aa7",           # violet
    "sitting": "#2a78d6",              # blue
    "standing_in_place": "#1baf7a",    # aqua
    "standing_and_moving": "#eda100",  # yellow
    "walking": "#eb6834",              # orange
    "running": "#e34948",              # red
    "bicycling": "#e87ba4",            # magenta
}


class Backend:
    """Model and timeline caches shared by every request."""

    def __init__(self) -> None:
        self.folds = load_folds()
        self._pipelines: dict[int, Pipeline] = {}
        self._timelines: dict[str, Timeline] = {}
        self._lock = threading.Lock()

    # ── model selection ──

    def fold_for(self, user_id: str) -> int:
        """The fold that held this user out, so a model never sees its own training user.

        Questioning a user with a model that trained on them measures memorisation
        rather than generalisation. Choosing the fold here, rather than leaving it
        to whoever opens the page, makes that mistake impossible to make by
        accident. Same rule as `tools/which_fold.py`.
        """
        assigned = self.folds["assignment"].get(user_id)
        if assigned is not None:
            return int(assigned)
        return 0  # not in the corpus: no model trained on them, so any fold is safe

    def pipeline(self, fold: int) -> Pipeline:
        with self._lock:
            if fold not in self._pipelines:
                self._pipelines[fold] = Pipeline(fold=fold)
            return self._pipelines[fold]

    # ── timelines ──

    def _cache_path(self, user_id: str) -> Path:
        return config.TIMELINE_DIR / f"dash_{user_id}.json"

    def timeline(self, user_id: str) -> Timeline:
        if user_id in self._timelines:
            return self._timelines[user_id]

        path = self._cache_path(user_id)
        if path.exists():
            timeline = load_timeline(path)
        else:
            timeline = self.pipeline(self.fold_for(user_id)).run(user_id)
            timeline.save(path)
        self._timelines[user_id] = timeline
        return timeline

    # ── API payloads ──

    def users(self) -> list[dict[str, Any]]:
        """Every preprocessed user, with enough detail to choose between them."""
        from asqa.preprocess import cached_users

        counts = self.folds.get("class_counts", {})
        rows = []
        for user_id in cached_users():
            distribution = counts.get(user_id, {})
            total = sum(distribution.values()) or 1
            dominant = max(distribution, key=lambda k: distribution[k]) if distribution else "unknown"
            rows.append({
                "id": user_id,
                "short": user_id[:8],
                "fold": self.fold_for(user_id),
                "in_corpus": user_id in self.folds["assignment"],
                "windows": total,
                "dominant": dominant,
                "distribution": distribution,
                "cached": user_id in self._timelines or self._cache_path(user_id).exists(),
            })
        return rows

    def session(self, user_id: str) -> dict[str, Any]:
        """Timeline summary and intervals. Per-window detail is deliberately omitted.

        A full timeline serialises to about 1 MB, almost all of it per-window
        rows; the intervals alone are ~20 KB and already carry the signal values
        the ribbon's tooltips need.
        """
        started = time.perf_counter()
        timeline = self.timeline(user_id)
        payload = timeline.to_dict()
        payload.pop("windows", None)

        totals: dict[str, dict[str, float]] = {}
        for interval in timeline.intervals:
            entry = totals.setdefault(interval.activity, {"seconds": 0.0, "bouts": 0})
            entry["seconds"] += interval.duration_s
            entry["bouts"] += 1

        payload.update({
            "user": user_id,
            "fold": self.fold_for(user_id),
            "held_out": user_id in self.folds["assignment"],
            "totals": totals,
            "colours": ACTIVITY_COLOURS,
            "display_names": config.DISPLAY_NAMES,
            "elapsed_s": round(time.perf_counter() - started, 3),
        })
        return payload

    def ask(self, user_id: str, question: str, use_slm: bool) -> dict[str, Any]:
        timeline = self.timeline(user_id)
        started = time.perf_counter()
        answer = answer_question(question, timeline, use_slm=use_slm)
        payload = answer.to_dict()
        payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        payload["rendered"] = answer.render()
        return payload


class Handler(BaseHTTPRequestHandler):
    backend: Backend

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        if "/api/" in str(args[0] if args else ""):
            sys.stderr.write(f"  {args[0]}\n")

    # ── helpers ──

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    # ── routes ──

    @property
    def route(self) -> str:
        """The path with any query string removed.

        `self.path` includes the query, so routing on it directly 404s a
        perfectly ordinary deep link such as `/?user=0A986513`.
        """
        return urlparse(self.path).path

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        if self.route in ("/", "/index.html"):
            try:
                self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self._json({"error": f"UI not found at {UI_PATH}"}, 500)
        elif self.route == "/api/users":
            self._json({"users": self.backend.users()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
        except json.JSONDecodeError:
            self._json({"error": "malformed JSON"}, 400)
            return

        try:
            if self.route == "/api/session":
                user = body.get("user")
                if not user:
                    self._json({"error": "no user given"}, 400)
                    return
                self._json(self.backend.session(user))
            elif self.route == "/api/ask":
                user, question = body.get("user"), (body.get("question") or "").strip()
                if not user or not question:
                    self._json({"error": "user and question are both required"}, 400)
                    return
                self._json(self.backend.ask(user, question, bool(body.get("use_slm"))))
            else:
                self._json({"error": "not found"}, 404)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, 404)
        except Exception as exc:  # noqa: BLE001 - surface the failure in the UI
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--open", action="store_true", help="Open a browser once the server is up.")
    args = parser.parse_args()

    models = sorted(config.MODEL_DIR.glob("recogniser_fold*.joblib"))
    if not models:
        print(
            f"No trained models in {config.MODEL_DIR}.\n"
            f"Run `python -m asqa.recognise --cv` first.",
            file=sys.stderr,
        )
        return 1

    Handler.backend = Backend()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Ask the Sensors dashboard -> {url}")
    print(f"  {len(models)} fold models available, loaded on demand")
    print("  Ctrl-C to stop")

    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    from asqa.dashboard import main as _main

    raise SystemExit(_main())
