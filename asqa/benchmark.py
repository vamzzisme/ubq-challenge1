#!/usr/bin/env python3
"""Resource cost, and the accuracy-versus-overhead trade-off.

The brief asks for model size in parameters and on disk, peak memory during
inference, and time to answer a single query, measured on a stated target.  It
also offers extra credit for showing the *shape* of the trade-off across at
least two operating points rather than asserting that a model is small.

Operating points measured here, all sharing the same feature and decoding
layers so that only the recogniser changes:

    full        the deployed hierarchical XGBoost, context-aware
    shallow     the same design at depth 3 with a quarter of the trees
    pruned      full, with features below an importance floor removed
    context-free  no temporal context: 40 features instead of 103
    slm         the Qwen2.5-0.5B open-world component, measured separately
                because it dominates and folding it into an average would hide
                that a single Task 4 query costs more than the entire
                classification pipeline

Target hardware is recorded in the output so the numbers are interpretable.
"""

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
    # Linux reports kilobytes, macOS bytes.
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
        except Exception:  # noqa: BLE001 - fall back to a coarse estimate
            total += int(getattr(stage, "n_estimators", 0)) * 2 ** int(getattr(stage, "max_depth", 0) or 6)
    return total


def measure_latency(model, X: np.ndarray, repeats: int = 30) -> dict[str, float]:
    """Latency for one window's worth of classification, in milliseconds."""
    single = X[:1]
    model.predict_proba(single)  # warm up
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
    return {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "parameters": 494_032_768,
        "latency_s_including_cold_start": round(payload["latency_s"], 2),
        "peak_rss_mb": round(payload["child_peak_rss_mb"], 1),
        "note": (
            "Runs in an isolated process (PyTorch and XGBoost cannot share one "
            "interpreter here). Latency includes loading the weights, which "
            "dominates a single query; a served deployment would amortise it."
        ),
    }


def main() -> int:
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
    points = []

    def record(tag: str, model, X_eval, note: str) -> None:
        metrics = evaluate(model, X_eval, y_test)
        entry = {
            "configuration": tag,
            "note": note,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "n_features": int(X_eval.shape[1]),
            "tree_nodes": count_parameters(model),
            "size_mb": disk_size_mb(model, tag),
            **measure_latency(model, X_eval),
        }
        points.append(entry)
        print(
            f"  {tag:<14} acc {entry['accuracy']:.3f}  F1 {entry['macro_f1']:.3f}  "
            f"{entry['size_mb']:>7.2f} MB  {entry['median_ms']:>6.2f} ms  "
            f"{entry['tree_nodes']:>8} nodes"
        )

    print("\nOperating points:")
    record("full", saved["context_aware"], Xc_test, "deployed model: hierarchical XGBoost, temporal context")
    record("context-free", saved["context_free"], X_test, "no temporal context: 40 features")

    print("  training shallow variant...")
    shallow = train(Xc_train, y_train, Xc_val, y_val, depth=3, estimators=100)
    record("shallow", shallow, Xc_test, "depth 3, 100 trees: the edge candidate")

    print("  training pruned variant...")
    importance = np.zeros(Xc_train.shape[1])
    for stage in (saved["context_aware"].stage1, saved["context_aware"].stage2a, saved["context_aware"].stage2b):
        scores = getattr(stage, "feature_importances_", None)
        if scores is not None and len(scores) == len(importance):
            importance += scores
    keep = importance >= np.percentile(importance, 40)  # drop the least useful 40%
    pruned = train(Xc_train[:, keep], y_train, Xc_val[:, keep], y_val)
    record("pruned", pruned, Xc_test[:, keep], f"{int(keep.sum())} of {len(keep)} features retained")

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
