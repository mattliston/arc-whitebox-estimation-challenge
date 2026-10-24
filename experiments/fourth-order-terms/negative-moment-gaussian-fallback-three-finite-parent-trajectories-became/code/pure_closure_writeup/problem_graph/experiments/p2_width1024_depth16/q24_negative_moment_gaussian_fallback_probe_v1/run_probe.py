#!/usr/bin/env python3
"""Target-free probe of negative-moment Gaussian fallback."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "fullbudget_k21_k4_closure_q24_manual_rho_v1" / "estimator.py"
PARENT_SHA256 = "7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00"
FIXTURE_HELPER = HERE.parent / "q24_two_point_moment_repair_probe_v1" / "run_probe.py"
FIXTURE_HELPER_SHA256 = "2c37c363155ac2b709de00a0d7a949a33a43106592e0df88daa7509bf2468b2d"
PREREGISTER = HERE / "PREREGISTER.md"
OUTPUT = HERE / "runs" / "probe_v1" / "results.json"
PROBE_BUDGET = 10**15


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fallback(label: str, weights: list[np.ndarray], helper) -> dict:
    module = load(f"q24_gaussian_fallback_{label}", PARENT)
    parent_fn = module._univariate_edgeworth
    counts: list[tuple[object, object, object]] = []

    def fallback(*args, **kwargs):
        r1, r2, r3, r4 = parent_fn(*args, **kwargs)
        gaussian = args[1]
        g1, g2, g3, g4 = gaussian[1], gaussian[2], gaussian[3], gaussian[4]
        invalid = fnp.bitwise_or(r3 < 0.0, r4 < 0.0)
        o1 = fnp.where(invalid, g1, r1)
        o2 = fnp.where(invalid, g2, r2)
        o3 = fnp.where(invalid, g3, r3)
        o4 = fnp.where(invalid, g4, r4)
        counts.append(
            (
                fnp.count_nonzero(invalid),
                fnp.count_nonzero(o3 < 0.0),
                fnp.count_nonzero(o4 < 0.0),
            )
        )
        return o1, o2, o3, o4

    module._univariate_edgeworth = fallback
    width = int(weights[0].shape[0])
    mlp = MLP(width=width, depth=16, weights=weights, seed=0, name=label)
    try:
        with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
            output = module._predict(mlp)
        array = np.asarray(output)
        summary = budget.summary_dict(by_namespace=False)
        result = {
            "outcome": "FINITE" if bool(np.isfinite(array).all()) else "NONFINITE",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "output_sha256": helper.digest_array(array),
            "flops": int(budget.flops_used),
            "calls": sum(int(v.get("calls", 0))
                         for v in summary.get("operations", {}).values()),
        }
    except Exception as exc:
        array = None
        result = {
            "outcome": "EXCEPTION",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    result["activation_counts_by_reset"] = [int(np.asarray(x[0])) for x in counts]
    result["negative_r3_counts_by_reset"] = [int(np.asarray(x[1])) for x in counts]
    result["negative_r4_counts_by_reset"] = [int(np.asarray(x[2])) for x in counts]
    return result | {"_array": array}


def main() -> int:
    if digest(PARENT) != PARENT_SHA256:
        raise SystemExit("parent hash mismatch")
    if digest(FIXTURE_HELPER) != FIXTURE_HELPER_SHA256:
        raise SystemExit("fixture helper hash mismatch")
    if OUTPUT.exists() or OUTPUT.parent.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT.parent}")
    helper = load("q24_frozen_fixture_helper", FIXTURE_HELPER)
    rows = []
    for label, weights in helper.fixture_rows():
        parent = helper.run_arm(label, weights, repaired=False)
        fallback = run_fallback(label, weights, helper)
        pa = parent.pop("_array")
        fa = fallback.pop("_array")
        comparison = None
        if pa is not None and fa is not None:
            delta = fa.astype(np.float64) - pa.astype(np.float64)
            comparison = {
                "all_layer_mse": float(np.mean(delta * delta)),
                "final_layer_mse": float(np.mean(delta[-1] * delta[-1])),
                "maximum_absolute_difference": float(np.max(np.abs(delta))),
            }
        rows.append({"fixture": label, "parent": parent, "fallback": fallback,
                     "finite_pair_disagreement": comparison})
        print(label, parent["outcome"], fallback["outcome"], flush=True)

    no_regression = all(
        not (row["parent"]["outcome"] == "FINITE"
             and row["fallback"]["outcome"] != "FINITE")
        for row in rows
    )
    activation_count = sum(sum(row["fallback"]["activation_counts_by_reset"])
                           for row in rows)
    post_nonnegative = all(
        sum(row["fallback"]["negative_r3_counts_by_reset"]) == 0
        and sum(row["fallback"]["negative_r4_counts_by_reset"]) == 0
        for row in rows
    )
    recovered = sum(
        row["parent"]["outcome"] != "FINITE"
        and row["fallback"]["outcome"] == "FINITE"
        for row in rows
    )
    passed = bool(no_regression and post_nonnegative and activation_count > 0)
    receipt = {
        "schema": "q24-negative-moment-gaussian-fallback-probe-v1",
        "status": "PASS_CONSTRUCTION_PROBE__SOURCE_SCIENCE_HOLD"
                  if passed else "REJECT_CONSTRUCTION_PROBE",
        "bindings": {
            "parent_estimator_sha256": PARENT_SHA256,
            "preregister_sha256": digest(PREREGISTER),
            "runner_sha256": digest(Path(__file__)),
            "fixture_helper_sha256": FIXTURE_HELPER_SHA256,
        },
        "scope": {"targets": 0, "truth": 0, "dataset_rows": 0,
                  "full_width_1024_runs": 0, "packages": 0,
                  "submissions": 0},
        "gates": {
            "no_parent_finite_regression": no_regression,
            "all_returned_r3_r4_represented_nonnegative": post_nonnegative,
            "activation_count": activation_count,
            "parent_failure_recovered_count": recovered,
            "construction_pass": passed,
        },
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(receipt["gates"], sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
