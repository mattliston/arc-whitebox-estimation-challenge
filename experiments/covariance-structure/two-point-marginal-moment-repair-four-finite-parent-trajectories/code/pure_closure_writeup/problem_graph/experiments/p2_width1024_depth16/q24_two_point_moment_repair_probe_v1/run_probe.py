#!/usr/bin/env python3
"""Target-free preregistered two-point marginal-moment repair probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PARENT = HERE.parent / "fullbudget_k21_k4_closure_q24_manual_rho_v1" / "estimator.py"
PARENT_SHA256 = "7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00"
PREREGISTER = HERE / "PREREGISTER.md"
OUTPUT = HERE / "runs" / "probe_v1" / "results.json"
DEPTH = 16
PROBE_BUDGET = 10**15


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(tag: str):
    if sha256(PARENT) != PARENT_SHA256:
        raise RuntimeError("frozen parent hash mismatch")
    spec = importlib.util.spec_from_file_location(tag, PARENT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen parent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(array.dtype.str.encode("ascii"))
    h.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def fixture_rows():
    eye = np.eye(8, dtype=np.float32)
    yield "identity_width8_float32", [eye.copy() for _ in range(DEPTH)]

    rng = np.random.default_rng(20260823)
    near = [
        (np.eye(8) + 0.02 * rng.normal(size=(8, 8))).astype(np.float32)
        for _ in range(DEPTH)
    ]
    yield "near_identity_seed20260823_scale0p02_float32", near

    rng = np.random.default_rng(20260823)
    qr = []
    for _ in range(DEPTH):
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        qr.append((0.35 * q).astype(np.float64))
    yield "qr_seed20260823_scale0p35_float64", qr

    rng = np.random.default_rng(17)
    normal = [rng.normal(size=(8, 8)).astype(np.float32) * np.float32(0.12)
              for _ in range(DEPTH)]
    yield "pure_normal_seed17_scale0p12_float32", normal

    for width in (32, 64, 128):
        for seed in (20260825, 20260826):
            rng = np.random.default_rng(seed)
            scale = np.sqrt(2.0 / width)
            weights = [
                (rng.standard_normal((width, width)) * scale).astype(np.float32)
                for _ in range(DEPTH)
            ]
            yield f"official_he_width{width}_seed{seed}", weights


def run_arm(label: str, weights: list[np.ndarray], repaired: bool) -> dict:
    module = load_module(f"q24_two_point_probe_{label}_{int(repaired)}")
    counts: list[tuple[object, object, object]] = []
    if repaired:
        parent_fn = module._univariate_edgeworth

        def repaired_fn(*args, **kwargs):
            r1, r2, r3, r4 = parent_fn(*args, **kwargs)
            r1sq = r1 * r1
            r2sq = r2 * r2
            variance = r2 - r1sq
            margin_a = r1 * r3 - r2sq
            centered = r3 - r1 * r2
            margin_b = (r4 - r2sq) * variance - centered * centered
            positive = r1 > 0.0
            invalid = fnp.bitwise_or(
                fnp.bitwise_or(margin_a < 0.0, margin_b < 0.0),
                fnp.invert(positive),
            )
            safe_mean = fnp.where(positive, r1, 1.0)
            atom = r2 / safe_mean
            boundary3 = r2 * atom
            boundary4 = boundary3 * atom
            zero = fnp.zeros_like(r2)
            repaired2 = fnp.where(positive, r2, zero)
            repaired3 = fnp.where(
                invalid, fnp.where(positive, boundary3, zero), r3
            )
            repaired4 = fnp.where(
                invalid, fnp.where(positive, boundary4, zero), r4
            )
            rr1sq = r1 * r1
            rr2sq = repaired2 * repaired2
            repaired_a = r1 * repaired3 - rr2sq
            repaired_v = repaired2 - rr1sq
            repaired_c = repaired3 - r1 * repaired2
            repaired_b = (
                (repaired4 - rr2sq) * repaired_v
                - repaired_c * repaired_c
            )
            counts.append(
                (
                    fnp.count_nonzero(invalid),
                    fnp.count_nonzero(repaired_a < 0.0),
                    fnp.count_nonzero(repaired_b < 0.0),
                )
            )
            return r1, repaired2, repaired3, repaired4

        module._univariate_edgeworth = repaired_fn

    width = int(weights[0].shape[0])
    mlp = MLP(width=width, depth=DEPTH, weights=weights, seed=0, name=label)
    try:
        with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
            output = module._predict(mlp)
        array = np.asarray(output)
        summary = budget.summary_dict(by_namespace=False)
        calls = sum(
            int(operation.get("calls", 0))
            for operation in summary.get("operations", {}).values()
        )
        result = {
            "outcome": "FINITE" if bool(np.isfinite(array).all()) else "NONFINITE",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "output_sha256": digest_array(array),
            "flops": int(budget.flops_used),
            "calls": calls,
        }
    except Exception as exc:  # registered outcome, never rescue/retry
        result = {
            "outcome": "EXCEPTION",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
        array = None
    if repaired:
        result["repair_counts_by_reset"] = [int(np.asarray(x[0])) for x in counts]
        result["post_negative_A_counts_by_reset"] = [int(np.asarray(x[1])) for x in counts]
        result["post_negative_B_counts_by_reset"] = [int(np.asarray(x[2])) for x in counts]
    return result | {"_array": array}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    if OUTPUT.exists() or OUTPUT.parent.exists():
        raise SystemExit(f"refusing to overwrite existing run namespace: {OUTPUT.parent}")
    rows = []
    for label, weights in fixture_rows():
        base = run_arm(label, weights, repaired=False)
        repair = run_arm(label, weights, repaired=True)
        base_array = base.pop("_array")
        repair_array = repair.pop("_array")
        comparison = None
        if base_array is not None and repair_array is not None:
            delta = repair_array.astype(np.float64) - base_array.astype(np.float64)
            comparison = {
                "all_layer_mse": float(np.mean(delta * delta)),
                "final_layer_mse": float(np.mean(delta[-1] * delta[-1])),
                "maximum_absolute_difference": float(np.max(np.abs(delta))),
            }
        rows.append(
            {"fixture": label, "parent": base, "repair": repair,
             "finite_pair_disagreement": comparison}
        )
        print(label, base["outcome"], repair["outcome"], flush=True)

    no_regression = all(
        not (row["parent"]["outcome"] == "FINITE"
             and row["repair"]["outcome"] != "FINITE")
        for row in rows
    )
    post_signs = all(
        all(value == 0 for value in row["repair"].get("post_negative_A_counts_by_reset", []))
        and all(value == 0 for value in row["repair"].get("post_negative_B_counts_by_reset", []))
        for row in rows
    )
    activations = sum(
        sum(row["repair"].get("repair_counts_by_reset", [])) for row in rows
    )
    recovered = sum(
        row["parent"]["outcome"] != "FINITE"
        and row["repair"]["outcome"] == "FINITE"
        for row in rows
    )
    mechanism_pass = bool(no_regression and post_signs and activations > 0)
    receipt = {
        "schema": "q24-two-point-moment-repair-target-free-probe-v1",
        "status": "PASS_MECHANISM__SOURCE_AND_SCIENCE_HOLD" if mechanism_pass
                  else "REJECT_MECHANISM_PROBE",
        "bindings": {
            "parent_estimator_sha256": PARENT_SHA256,
            "preregister_sha256": sha256(PREREGISTER),
            "runner_sha256": sha256(Path(__file__)),
        },
        "scope": {
            "targets": 0, "truth": 0, "dataset_rows": 0,
            "full_width_1024_runs": 0, "packages": 0, "submissions": 0,
        },
        "gates": {
            "no_parent_finite_to_repair_nonfinite_regression": no_regression,
            "all_repaired_post_A_and_B_represented_nonnegative": post_signs,
            "repair_activation_count": activations,
            "parent_failure_recovered_count": recovered,
            "mechanism_pass": mechanism_pass,
        },
        "rows": rows,
    }
    atomic_json(OUTPUT, receipt)
    print(json.dumps(receipt["gates"], sort_keys=True))
    return 0 if mechanism_pass else 2


if __name__ == "__main__":
    sys.exit(main())
