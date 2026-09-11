"""A local dashboard for exploring the system: pick a user, ask questions, see the evidence."""

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

ACTIVITY_COLOURS = {
    "lying_down": "#4a3aa7",
    "sitting": "#2a78d6",
    "standing_in_place": "#1baf7a",
    "standing_and_moving": "#eda100",
    "walking": "#eb6834",
    "running": "#e34948",
    "bicycling": "#e87ba4",
}


DEFAULT_CORPUS = config.DATA_DIR / "5 new users"

DEFAULT_FOLD = 3


class Backend:
    """Model and timeline caches shared by every request."""

    def __init__(self, corpus: Path | None = None, fold: int = DEFAULT_FOLD) -> None:
        self.folds = load_folds()
        self.corpus = corpus if corpus is not None else DEFAULT_CORPUS
        self.default_fold = fold
        self._pipelines: dict[int, Pipeline] = {}
        self._timelines: dict[str, Timeline] = {}
        self._lock = threading.Lock()

    @property
    def uses_corpus(self) -> bool:
        """True when serving a held-out directory rather than the training cache."""
        return (self.corpus / "acc").is_dir()

    def corpus_users(self) -> list[str]:
        return sorted(p.name for p in (self.corpus / "acc").iterdir() if p.is_dir())

    def recording_path(self, user_id: str) -> Path | str:
        """Where to read this user's raw signal from."""
        if self.uses_corpus and (self.corpus / "acc" / user_id).is_dir():
            return self.corpus / "acc" / user_id
        return user_id


    def fold_for(self, user_id: str) -> int:
        """The fold that held this user out, so a model never sees its own training user."""
        assigned = self.folds["assignment"].get(user_id)
        if assigned is not None:
            return int(assigned)
        return self.default_fold

    def pipeline(self, fold: int) -> Pipeline:
        with self._lock:
            if fold not in self._pipelines:
                self._pipelines[fold] = Pipeline(fold=fold)
            return self._pipelines[fold]


    def _cache_path(self, user_id: str) -> Path:
        return config.TIMELINE_DIR / f"dash_{user_id}.json"

    def timeline(self, user_id: str) -> Timeline:
        if user_id in self._timelines:
            return self._timelines[user_id]

        path = self._cache_path(user_id)
        if path.exists():
            timeline = load_timeline(path)
        else:
            source = self.recording_path(user_id)
            timeline = self.pipeline(self.fold_for(user_id)).run(source)
            timeline.save(path)
        self._timelines[user_id] = timeline
        return timeline


    def users(self) -> list[dict[str, Any]]:
        """The selectable recordings."""
        from asqa.preprocess import cached_users

        counts = self.folds.get("class_counts", {})
        ids = self.corpus_users() if self.uses_corpus else cached_users()

        rows = []
        for user_id in ids:
            trained_on = user_id in self.folds["assignment"]
            cached = user_id in self._timelines or self._cache_path(user_id).exists()

            if cached:
                timeline = self.timeline(user_id)
                windows = len(timeline.windows)
                totals: dict[str, float] = {}
                for interval in timeline.intervals:
                    totals[interval.activity] = totals.get(interval.activity, 0.0) + interval.duration_s
                dominant = max(totals, key=lambda k: totals[k]) if totals else "unknown"
                duration = timeline.duration_s
            else:
                distribution = counts.get(user_id, {})
                windows = sum(distribution.values()) or self._raw_window_estimate(user_id)
                dominant = max(distribution, key=lambda k: distribution[k]) if distribution else None
                duration = 0.0

            rows.append({
                "id": user_id,
                "short": user_id[:8],
                "fold": self.fold_for(user_id),
                "trained_on": trained_on,
                "windows": windows,
                "duration_s": duration,
                "dominant": dominant,
                "cached": cached,
            })
        return rows

    def _raw_window_estimate(self, user_id: str) -> int:
        """Number of raw captures, for a user not yet analysed."""
        directory = self.corpus / "acc" / user_id
        return len(list(directory.glob("*.dat"))) if directory.is_dir() else 0

    def session(self, user_id: str) -> dict[str, Any]:
        """Timeline summary and intervals. Per-window detail is deliberately omitted."""
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
            "trained_on": user_id in self.folds["assignment"],
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

    def log_message(self, fmt: str, *args: Any) -> None:
        if "/api/" in str(args[0] if args else ""):
            sys.stderr.write(f"  {args[0]}\n")


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


    @property
    def route(self) -> str:
        """The path with any query string removed."""
        return urlparse(self.path).path

    def do_GET(self) -> None:
        if self.route in ("/", "/index.html"):
            try:
                self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self._json({"error": f"UI not found at {UI_PATH}"}, 500)
        elif self.route == "/api/users":
            self._json({"users": self.backend.users()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
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
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--open", action="store_true", help="Open a browser once the server is up.")
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help="Directory of held-out recordings (expects acc/<user>/ and gyro/<user>/).",
    )
    parser.add_argument(
        "--training-users", action="store_true",
        help="Serve the preprocessed training users instead of the held-out corpus.",
    )
    parser.add_argument(
        "--fold", type=int, default=DEFAULT_FOLD,
        help=f"Which fold's model answers never-seen users (default {DEFAULT_FOLD}).",
    )
    args = parser.parse_args()

    models = sorted(config.MODEL_DIR.glob("recogniser_fold*.joblib"))
    if not models:
        print(
            f"No trained models in {config.MODEL_DIR}.\n"
            f"Run `python -m asqa.recognise --cv` first.",
            file=sys.stderr,
        )
        return 1

    corpus = None if args.training_users else args.corpus
    backend = Backend(corpus=corpus, fold=args.fold)
    Handler.backend = backend

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Ask the Sensors dashboard -> {url}")
    if backend.uses_corpus:
        n = len(backend.corpus_users())
        print(f"  serving {n} held-out recordings from {backend.corpus}")
        print(f"  none of them appears in any model's training set; answered by fold {args.fold}")
    else:
        print(f"  serving preprocessed training users (--training-users)")
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
