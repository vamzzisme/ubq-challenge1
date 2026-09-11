"""Resource cost, and the accuracy-versus-overhead trade-off."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from asqa import config, context, features as feat


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def target_description() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "note": "Laptop-class CPU; single process, no GPU used for the classifier.",
    }


def count_parameters(model) -> int:
    """Total tree nodes across the three stages -- the tree analogue of parameters."""
    total = 0
    for stage in (model.stage1, model.stage2a, model.stage2b):
        try:
            frame = stage.get_booster().trees_to_dataframe()
            total += len(frame)
        except Exception:
            total += int(getattr(stage, "n_estimators", 0)) * 2 ** int(getattr(stage, "max_depth", 0) or 6)
    return total


def measure_latency(model, X: np.ndarray, repeats: int = 30) -> dict[str, float]:
    """Latency for one window's worth of classification, in milliseconds."""
    single = X[:1]
    model.predict_proba(single)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_proba(single)
        timings.append((time.perf_counter() - started) * 1000)
    array = np.asarray(timings)
    return {
        "mean_ms": round(float(array.mean()), 3),
        "median_ms": round(float(np.median(array)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
    }


def measure_feature_cost(windows: np.ndarray, repeats: int = 10) -> dict[str, float]:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        feat.extract(windows[0])
        timings.append((time.perf_counter() - started) * 1000)
    return {"feature_extraction_ms": round(float(np.median(timings)), 3)}


def disk_size_mb(model, tag: str) -> float:
    path = config.MODEL_DIR / f"_bench_{tag}.joblib"
    joblib.dump(model, path)
    size = path.stat().st_size / 1e6
    path.unlink(missing_ok=True)
    return round(size, 3)


def benchmark_slm(question: str, menu: str) -> dict:
    """Cost of one open-world query, measured in the worker's own process."""
    script = (
        "import json,sys,time,resource,subprocess\n"
        "start=time.perf_counter()\n"
        f"req={json.dumps(json.dumps({'question': question, 'menu': menu, 'max_new_tokens': 200}))}\n"
        "p=subprocess.run([sys.executable,'-m','asqa.slm_worker'],input=req,capture_output=True,text=True)\n"
        "elapsed=time.perf_counter()-start\n"
        "rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss\n"
        "print(json.dumps({'latency_s':elapsed,'child_peak_rss_mb':rss/(1024*1024),'ok':p.returncode==0}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(config.REPO_ROOT),
        env={"PYTHONPATH": str(config.REPO_ROOT), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        timeout=900,
    )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": completed.stderr[-300:] or "worker produced no output"}
    from asqa.slm import ask_worker

    # Ask the worker which model it loaded rather than naming one here: the default
    # is resolved from ASQA_SLM_MODEL at the worker's import time, so a constant in
    # this file would silently mislabel the very numbers measured above.
    try:
        meta = ask_worker("", mode="meta", max_new_tokens=1)
    except Exception as exc:
        meta = {"model": "unknown", "error": f"{type(exc).__name__}: {exc}"}

    return {
        "model": meta.get("model", "unknown"),
        "parameters": meta.get("parameters"),
        "dtype": meta.get("dtype"),
        "device": meta.get("device"),
        "latency_s_including_cold_start": round(payload["latency_s"], 2),
        "peak_rss_mb": round(payload["child_peak_rss_mb"], 1),
        "note": (
            "Runs in an isolated process (PyTorch and XGBoost cannot share one "
            "interpreter here). Latency includes loading the weights, which "
            "dominates a single query; a served deployment would amortise it. "
            "Set ASQA_SLM_MODEL to measure a different operating point."
        ),
    }


def qa_accuracy(model, users: list[str], reference, transitions, keep=None,
                with_context: bool = True) -> dict[str, float]:
    """Overall QA accuracy for one operating point.

    The brief puts question-answering accuracy on the y-axis of the
    accuracy-versus-overhead figure, not per-window recognition accuracy, and the
    two differ: a configuration can lose windows scattered through a bout and
    barely move a duration answer, or lose a whole bout and change several. The
    reference labels come from the deployed model's standing split in every case,
    so only the configuration under test varies.
    """
    from asqa.evaluate import evaluate_recording, summarise
    from asqa.pipeline import timeline_from_model
    from asqa.recognise import build_context_features

    rows = []
    for user_id in users:
        Xc, coarse, epoch_ts = build_context_features(user_id)
        truth, _ = reference.standing_split.apply(Xc, coarse)
        # context.augment appends its columns, so the first N_FEATURES are the base.
        base = Xc[:, : feat.N_FEATURES]
        X = Xc if with_context else base
        if keep is not None:
            X = X[:, keep]
        timeline = timeline_from_model(
            model, X, base, epoch_ts, transitions, used_context=with_context
        )
        rows.extend(evaluate_recording(timeline, truth, epoch_ts))
    summary = summarise(rows)
    return {
        "qa_accuracy_macro": round(summary["overall_qa_accuracy_macro"], 4),
        "qa_accuracy_micro": round(summary["overall_qa_accuracy_micro"], 4),
        "qa_grounded_accuracy_macro": round(summary["overall_grounded_accuracy_macro"], 4),
        "qa_cases": summary["case_count"],
    }


def main() -> int:
    from asqa.pipeline import Pipeline
    from asqa.recognise import evaluate, load_users, train
    from asqa.splits import load_folds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--skip-slm", action="store_true")
    args = parser.parse_args()

    folds = load_folds()
    split = folds["splits"][str(args.fold)]

    print("Loading fold data...")
    Xc_train, y_train, _, _ = load_users(split["train"], with_context=True)
    Xc_val, y_val, _, _ = load_users(split["validation"], with_context=True)
    Xc_test, y_test, _, _ = load_users(split["test"], with_context=True)
    X_train, _, _, _ = load_users(split["train"], with_context=False)
    X_val, _, _, _ = load_users(split["validation"], with_context=False)
    X_test, _, _, _ = load_users(split["test"], with_context=False)

    saved = joblib.load(config.MODEL_DIR / f"recogniser_fold{args.fold}.joblib")
    transitions = Pipeline(fold=args.fold).transitions()
    points = []

    def record(tag: str, model, X_eval, note: str, keep=None, with_context: bool = True) -> None:
        metrics = evaluate(model, X_eval, y_test)
        entry = {
            "configuration": tag,
            "note": note,
            "window_accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "n_features": int(X_eval.shape[1]),
            "tree_nodes": count_parameters(model),
            "size_mb": disk_size_mb(model, tag),
            **measure_latency(model, X_eval),
        }
        # The figure the brief asks for plots question-answering accuracy, so each
        # operating point is carried all the way through decoding and aggregation.
        entry.update(
            qa_accuracy(
                model, split["test"], saved["context_aware"], transitions,
                keep=keep, with_context=with_context,
            )
        )
        points.append(entry)
        print(
            f"  {tag:<14} window acc {entry['window_accuracy']:.3f}  "
            f"QA acc {entry['qa_accuracy_macro']:.3f}  F1 {entry['macro_f1']:.3f}  "
            f"{entry['size_mb']:>7.2f} MB  {entry['median_ms']:>6.2f} ms  "
            f"{entry['tree_nodes']:>8} nodes"
        )

    print("\nOperating points:")
    record("full", saved["context_aware"], Xc_test, "deployed model: hierarchical XGBoost, temporal context")
    record("context-free", saved["context_free"], X_test, "no temporal context: 40 features",
           with_context=False)

    print("  training shallow variant...")
    shallow = train(Xc_train, y_train, Xc_val, y_val, depth=3, estimators=100)
    record("shallow", shallow, Xc_test, "depth 3, 100 trees: the edge candidate")

    print("  training pruned variant...")
    importance = np.zeros(Xc_train.shape[1])
    for stage in (saved["context_aware"].stage1, saved["context_aware"].stage2a, saved["context_aware"].stage2b):
        scores = getattr(stage, "feature_importances_", None)
        if scores is not None and len(scores) == len(importance):
            importance += scores
    keep = importance >= np.percentile(importance, 40)
    pruned = train(Xc_train[:, keep], y_train, Xc_val[:, keep], y_val)
    record("pruned", pruned, Xc_test[:, keep], f"{int(keep.sum())} of {len(keep)} features retained",
           keep=keep)

    payload = {
        "target": target_description(),
        "fold": args.fold,
        "test_windows": int(len(y_test)),
        "peak_rss_mb_process": round(peak_rss_mb(), 1),
        **measure_feature_cost(np.zeros((1, config.WINDOW_SAMPLES, 6), dtype=np.float32)),
        "operating_points": points,
    }

    if not args.skip_slm:
        print("\nMeasuring the language model (isolated process)...")
        payload["language_model"] = benchmark_slm(
            "Was the user resting?",
            "[1] 0-600s (600s, classified sitting) energy=0.0100g cadence=0.8Hz tilt=30deg rotation=0.020rad/s",
        )
        print(f"  {payload['language_model']}")

    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output = config.EVALUATION_DIR / "benchmark.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    from asqa.benchmark import main as _main

    raise SystemExit(_main())
