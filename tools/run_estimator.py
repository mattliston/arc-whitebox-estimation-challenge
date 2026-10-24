#!/usr/bin/env python3
"""Run one real width-1024/depth-16 inference on a freshly sampled network.

This is an in-process numerical smoke test, not hosted grading or an accuracy
measurement. No Monte Carlo target or training data is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WIDTH, DEPTH, BUDGET = 1024, 16, 2**41


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", choices=("337619", "337641"), default="337641")
    parser.add_argument("--estimator", type=Path, help="alternative estimator path, e.g. a rebuilt source")
    parser.add_argument("--seed", type=int, default=20261022, help="fresh-network and estimator seed")
    parser.add_argument("--threads", type=int, default=1, help="BLAS threads (default: 1)")
    parser.add_argument("--wall-time-limit", type=float, default=600,
                        help="smoke-test predict cap in seconds; hosted cap is 120 (default: 600)")
    parser.add_argument("--output", type=Path, default=None, help="new result directory")
    args = parser.parse_args()
    if args.threads < 1 or args.wall_time_limit <= 0:
        parser.error("--threads and --wall-time-limit must be positive")
    source = (args.estimator or ROOT / f"estimator/submission_{args.submission}.py").resolve()
    output = (args.output or ROOT / f"_build/inference-{args.submission}").resolve()
    if output.exists():
        parser.error(f"output already exists; choose a new directory: {output}")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_limits, threadpool_info
    from whestbench.generation import sample_mlp
    from whestbench.sdk import SetupContext
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "flopscope", "whestbench")}
    expected = {"numpy": "2.2.6", "flopscope": "0.12.3", "whestbench": "0.16.2"}
    if versions != expected:
        parser.error(f"install tools/requirements-reproduce.txt; found {versions}")
    output.mkdir(parents=True)
    report = {"passed": False, "kind": "fresh-network in-process inference",
              "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
              "width": WIDTH, "depth": DEPTH, "flop_budget": BUDGET, "seed": args.seed,
              "threads": args.threads, "predict_wall_limit_s": args.wall_time_limit,
              "hosted_timing_enforced": False, "accuracy_evaluated": False}
    estimator = None
    try:
        with threadpool_limits(limits=args.threads):
            report["blas"] = threadpool_info()
            print("Generating a fresh He-initialized network...", flush=True)
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as generation_budget:
                mlp = sample_mlp(WIDTH, DEPTH, rng=fnp.random.default_rng(args.seed), seed=args.seed)
            report["generation"] = generation_budget.summary_dict()
            report["weights_sha256"] = hashlib.sha256(b"".join(
                np.asarray(weight, dtype="<f4").tobytes() for weight in mlp.weights)).hexdigest()
            print(f"Loading {source.name} and running setup...", flush=True)
            start = time.perf_counter()
            spec = importlib.util.spec_from_file_location("release_estimator", source)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load {source}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            estimator = module.Estimator()
            context = SetupContext(width=WIDTH, depth=DEPTH, flop_budget=BUDGET,
                                   api_version="1.0", seed=args.seed,
                                   scratch_dir=str(output), submission_dir=str(source.parent))
            estimator.setup(context)
            report["load_and_setup_wall_s"] = time.perf_counter() - start
            print(f"Predicting at budget {BUDGET}...", flush=True)
            with flops.BudgetContext(flop_budget=BUDGET, quiet=True,
                                     wall_time_limit_s=args.wall_time_limit) as prediction_budget:
                prediction = estimator.predict(mlp, BUDGET)
            report["prediction"] = prediction_budget.summary_dict()
            values = np.asarray(prediction)
            if values.shape != (DEPTH, WIDTH) or values.dtype != np.dtype("float32"):
                raise RuntimeError(f"Unexpected prediction shape/dtype: {values.shape}, {values.dtype}")
            if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
                raise RuntimeError("Prediction is non-finite, negative, or entirely zero")
            # An exact Gaussian marginal is available for the first layer.
            first_weights = np.asarray(mlp.weights[0], dtype="float64")
            first_exact = np.sqrt(np.sum(first_weights * first_weights, axis=0) / (2 * np.pi))
            first_error = float(np.max(np.abs(values[0] - first_exact)))
            if first_error > 1e-5:
                raise RuntimeError(f"First-layer Gaussian mean check failed: {first_error}")
            np.save(output / "prediction.npy", values, allow_pickle=False)
            report["result"] = {"shape": list(values.shape), "dtype": str(values.dtype),
                                "min": float(values.min()), "max": float(values.max()),
                                "mean": float(values.mean()), "positive_count": int((values > 0).sum()),
                                "first_layer_gaussian_max_abs_error": first_error,
                                "raw_float32_sha256": hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()}
            report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if estimator is not None:
            estimator.teardown()
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(output), "result": report["result"],
                      "flops_used": report["prediction"]["flops_used"],
                      "predict_wall_s": report["prediction"]["wall_time_s"]}, indent=2))


if __name__ == "__main__":
    main()
