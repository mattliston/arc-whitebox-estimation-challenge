#!/usr/bin/env python3
"""One-use target-free CS3 design-risk microprobe; no estimator source."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREGISTER_RISK_AUDIT_V1.json"
OUTPUT = HERE / "RISK_AUDIT_RESULT_V1.json"
WIDTHS_COST = (4, 5, 8, 16, 32)
WIDTHS_BRANCH = (4, 8, 16, 32)
WIDTHS_PROPAGATION = (32, 64, 128, 256)
WIDTHS_BACKEND = (128, 256, 512)
SEED = 20260824


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calls(budget) -> int:
    return int(sum(row["calls"] for row in budget.summary_dict()["operations"].values()))


def operations(budget) -> dict[str, dict[str, int]]:
    return {
        name: {"calls": int(row["calls"]), "flops": int(row["flop_cost"])}
        for name, row in sorted(budget.summary_dict()["operations"].items())
    }


def arrays(width: int):
    covariance64 = fnp.asarray(np.eye(width, dtype=np.float64), dtype=fnp.float64)
    valid = np.ones(width, dtype=bool)
    valid[width // 3] = False
    valid_plane = fnp.asarray(valid[:, None] & valid[None, :], dtype=fnp.bool_)
    weight32 = fnp.asarray(np.eye(width, dtype=np.float32), dtype=fnp.float32)
    weight64 = fnp.asarray(np.eye(width, dtype=np.float64), dtype=fnp.float64)
    weight32_t = fnp.transpose(weight32)
    weight64_t = fnp.transpose(weight64)
    return covariance64, valid_plane, weight32, weight32_t, weight64, weight64_t


def localized_fp32_ops(covariance64, valid_plane, weight32, weight32_t):
    stored32 = fnp.where(
        valid_plane,
        fnp.asarray(covariance64, dtype=fnp.float32),
        0.0,
    )
    stored64 = fnp.asarray(stored32, dtype=fnp.float64)
    _stored_variance = fnp.diag(stored64)
    pushed32 = (weight32_t @ stored32) @ weight32
    pushed64 = fnp.asarray(pushed32, dtype=fnp.float64)
    _pushed_variance = fnp.diag(pushed64)
    return pushed32


def localized_fp64_ops(covariance64, valid_plane, weight64, weight64_t):
    stored64 = fnp.where(valid_plane, covariance64, 0.0)
    _stored_variance = fnp.diag(stored64)
    pushed64 = (weight64_t @ stored64) @ weight64
    _pushed_variance = fnp.diag(pushed64)
    return pushed64


def cost_probe() -> dict:
    rows = []
    for width in WIDTHS_COST:
        covariance64, valid_plane, weight32, weight32_t, weight64, weight64_t = arrays(width)
        with flops.BudgetContext(flop_budget=10**12, quiet=True) as budget32:
            localized_fp32_ops(covariance64, valid_plane, weight32, weight32_t)
        with flops.BudgetContext(flop_budget=10**12, quiet=True) as budget64:
            localized_fp64_ops(covariance64, valid_plane, weight64, weight64_t)
        expected32 = 4 * width**3 + 8 * width**2
        expected64 = 8 * width**3 + 4 * width**2
        rows.append({
            "width": width,
            "fp32": {
                "flops": int(budget32.flops_used),
                "expected": expected32,
                "exact_match": int(budget32.flops_used) == expected32,
                "calls": calls(budget32),
                "operations": operations(budget32),
            },
            "fp64": {
                "flops": int(budget64.flops_used),
                "expected": expected64,
                "exact_match": int(budget64.flops_used) == expected64,
                "calls": calls(budget64),
                "operations": operations(budget64),
            },
        })
    flag = fnp.asarray(True, dtype=fnp.bool_)
    alpha = fnp.asarray(1.0, dtype=fnp.float64)
    with flops.BudgetContext(flop_budget=1000, quiet=True) as scalar_budget:
        fnp.where(flag, alpha, 0.0)
    return {
        "rows": rows,
        "all_exact": all(row["fp32"]["exact_match"] and row["fp64"]["exact_match"] for row in rows),
        "removed_scalar_where": {
            "flops": int(scalar_budget.flops_used),
            "calls": calls(scalar_budget),
            "operations": operations(scalar_budget),
            "expected_flops": 8,
            "exact_match": int(scalar_budget.flops_used) == 8 and calls(scalar_budget) == 1,
        },
    }


def branch_probe() -> dict:
    rows = []
    for width in WIDTHS_BRANCH:
        covariance64, valid_plane, weight32, weight32_t, _weight64, _weight64_t = arrays(width)
        for case in ("all_true", "one_false"):
            mask = np.ones(width, dtype=bool)
            if case == "one_false":
                mask[width // 3] = False
            tracked_mask = fnp.asarray(mask, dtype=fnp.bool_)
            with flops.BudgetContext(flop_budget=10**12, quiet=True) as budget:
                all_valid_scalar = fnp.all(tracked_mask)
                before_bool_flops = int(budget.flops_used)
                before_bool_calls = calls(budget)
                native_bool = bool(all_valid_scalar)
                after_bool_flops = int(budget.flops_used)
                after_bool_calls = calls(budget)
                localized_executed = not native_bool
                if localized_executed:
                    localized_fp32_ops(covariance64, valid_plane, weight32, weight32_t)
            expected = width - 1
            if localized_executed:
                expected += 4 * width**3 + 8 * width**2
            rows.append({
                "width": width,
                "case": case,
                "tracked_scalar_type": type(all_valid_scalar).__name__,
                "tracked_scalar_shape": list(all_valid_scalar.shape),
                "native_bool": native_bool,
                "localized_executed": localized_executed,
                "before_bool_flops": before_bool_flops,
                "after_bool_flops": after_bool_flops,
                "before_bool_calls": before_bool_calls,
                "after_bool_calls": after_bool_calls,
                "bool_added_flops": after_bool_flops - before_bool_flops,
                "bool_added_calls": after_bool_calls - before_bool_calls,
                "total_flops": int(budget.flops_used),
                "expected_total_flops": expected,
                "exact_total": int(budget.flops_used) == expected,
                "operations": operations(budget),
            })
    return {
        "rows": rows,
        "all_reduction_exact": all(row["before_bool_flops"] == row["width"] - 1 for row in rows),
        "native_bool_zero_added_flops_and_calls": all(
            row["bool_added_flops"] == 0 and row["bool_added_calls"] == 0 for row in rows
        ),
        "branch_executes_expected_path_and_bills_it": all(row["exact_total"] for row in rows),
        "authority_boundary": "Pinned local runtime/billing behavior only; source validation and residual meter remain mandatory.",
    }


def norm_ratio(delta: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    numerator = float(np.linalg.norm(delta.ravel()))
    return numerator / denominator if denominator else (0.0 if numerator == 0.0 else math.inf)


def wick_planes(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagonal = np.diag(matrix)
    return (
        3.0 * diagonal * diagonal,
        3.0 * diagonal[:, None] * matrix,
        np.outer(diagonal, diagonal) + 2.0 * matrix * matrix,
    )


def propagation_probe() -> dict:
    rows = []
    u32 = 2.0**-24
    for width in WIDTHS_PROPAGATION:
        for case_index, case in enumerate(("one_invalid", "eighth_invalid")):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, width, case_index, 0]))
            a = rng.standard_normal((width, width), dtype=np.float64)
            covariance = (a @ a.T) / width + 0.25 * np.eye(width, dtype=np.float64)
            weight64 = rng.standard_normal((width, width), dtype=np.float64) * math.sqrt(2.0 / width)
            valid = np.ones(width, dtype=bool)
            if case == "one_invalid":
                valid[width // 3] = False
            else:
                valid[np.arange(3, width, 8)] = False
            valid_plane = valid[:, None] & valid[None, :]
            masked64 = np.where(valid_plane, covariance, 0.0)
            stored32 = np.where(valid_plane, covariance.astype(np.float32), np.float32(0.0))
            weight32 = weight64.astype(np.float32)
            first32 = weight32.T @ stored32
            pushed32 = first32 @ weight32
            pushed32_as64 = pushed32.astype(np.float64)
            pushed64 = (weight64.T @ masked64) @ weight64

            stored64 = stored32.astype(np.float64)
            weight32_as64 = weight32.astype(np.float64)
            ideal_stored = (weight32_as64.T @ stored64) @ weight32_as64
            observed_stored_error = np.abs(pushed32_as64 - ideal_stored)
            gamma = (width * u32) / (1.0 - width * u32)
            abs_a = np.abs(weight32_as64.T)
            abs_b = np.abs(stored64)
            abs_c = np.abs(weight32_as64)
            first_error_bound = gamma * (abs_a @ abs_b)
            push_error_bound = first_error_bound @ abs_c + gamma * (np.abs(first32.astype(np.float64)) @ abs_c)
            reference_slack = 64.0 * np.finfo(np.float64).eps * (
                (abs_a @ abs_b) @ abs_c + 1.0
            )
            bound_with_slack = push_error_bound + reference_slack
            bound_violation = observed_stored_error > bound_with_slack

            wick64 = wick_planes(pushed64)
            wick32 = wick_planes(pushed32_as64)
            rows.append({
                "width": width,
                "case": case,
                "invalid_count": int((~valid).sum()),
                "stored_invalid_nonzero_count": int(np.count_nonzero(stored32[~valid, :]))
                + int(np.count_nonzero(stored32[:, ~valid])),
                "stored_diagonal_invalid_nonzero_count": int(np.count_nonzero(np.diag(stored32)[~valid])),
                "pushed_s_relative_frobenius_vs_fp64": norm_ratio(pushed32_as64 - pushed64, pushed64),
                "pushed_s_maxabs_vs_fp64": float(np.max(np.abs(pushed32_as64 - pushed64))),
                "stored_arithmetic_relative_frobenius_error": norm_ratio(pushed32_as64 - ideal_stored, ideal_stored),
                "stored_arithmetic_maxabs_error": float(np.max(observed_stored_error)),
                "stored_arithmetic_max_bound": float(np.max(bound_with_slack)),
                "stored_arithmetic_bound_violation_count": int(np.count_nonzero(bound_violation)),
                "wick_relative_frobenius_vs_fp64": {
                    name: norm_ratio(candidate - reference, reference)
                    for name, candidate, reference in zip(("K4", "K31", "K22"), wick32, wick64)
                },
                "finite": bool(np.isfinite(pushed32_as64).all()) and all(bool(np.isfinite(x).all()) for x in wick32),
                "gamma_width": gamma,
            })
    return {
        "rows": rows,
        "all_finite": all(row["finite"] for row in rows),
        "all_stored_invalid_exact_zero": all(
            row["stored_invalid_nonzero_count"] == 0 and row["stored_diagonal_invalid_nonzero_count"] == 0
            for row in rows
        ),
        "all_stored_arithmetic_within_gamma_bound": all(
            row["stored_arithmetic_bound_violation_count"] == 0 for row in rows
        ),
        "maximum_pushed_s_relative_frobenius_vs_fp64": max(
            row["pushed_s_relative_frobenius_vs_fp64"] for row in rows
        ),
        "maximum_wick_relative_frobenius_vs_fp64": {
            name: max(row["wick_relative_frobenius_vs_fp64"][name] for row in rows)
            for name in ("K4", "K31", "K22")
        },
        "accuracy_boundary": "No downstream or raw-MSE implication; this only quantifies one-layer carrier rounding.",
    }


def backend_probe() -> dict:
    rows = []
    for width in WIDTHS_BACKEND:
        rng = np.random.default_rng(np.random.SeedSequence([SEED, width, 99]))
        weight32 = fnp.asarray(rng.standard_normal((width, width), dtype=np.float32), dtype=fnp.float32)
        stored32 = fnp.asarray(rng.standard_normal((width, width), dtype=np.float32), dtype=fnp.float32)
        weight64 = fnp.asarray(np.asarray(weight32, dtype=np.float64), dtype=fnp.float64)
        stored64 = fnp.asarray(np.asarray(stored32, dtype=np.float64), dtype=fnp.float64)
        for dtype_name, weight, stored in (
            ("float32", weight32, stored32),
            ("float64", weight64, stored64),
        ):
            weight_t = fnp.transpose(weight)
            with flops.BudgetContext(flop_budget=10**12, quiet=True):
                (weight_t @ stored) @ weight
            measurements = []
            for _ in range(3):
                started = time.perf_counter()
                with flops.BudgetContext(flop_budget=10**12, quiet=True) as budget:
                    (weight_t @ stored) @ weight
                measurements.append({
                    "flops": int(budget.flops_used),
                    "calls": calls(budget),
                    "backend_s": float(budget.flopscope_backend_time_s),
                    "wall_s": time.perf_counter() - started,
                })
            rows.append({
                "width": width,
                "dtype": dtype_name,
                "measurements": measurements,
                "median_backend_s": float(np.median([m["backend_s"] for m in measurements])),
                "median_wall_s": float(np.median([m["wall_s"] for m in measurements])),
            })
    ratios = []
    for width in WIDTHS_BACKEND:
        fp32 = next(row for row in rows if row["width"] == width and row["dtype"] == "float32")
        fp64 = next(row for row in rows if row["width"] == width and row["dtype"] == "float64")
        ratios.append({
            "width": width,
            "fp64_over_fp32_median_backend": fp64["median_backend_s"] / fp32["median_backend_s"],
            "fp64_minus_fp32_median_backend_s": fp64["median_backend_s"] - fp32["median_backend_s"],
        })
    return {
        "rows": rows,
        "ratios": ratios,
        "authority_boundary": "Local <=512 microbenchmark only; not a width-1024 estimator or board-wall claim.",
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable output already exists: {OUTPUT}")
    flops.configure(symmetry_warnings=False)
    result = {
        "schema": "q24-cs3-localized-pairing-design-risk-audit-result-v1",
        "status": "PASS_PINNED_BRANCH_AND_EXACT_COST_MICROPROBE__FP32_ROUNDING_QUANTIFIED__SOURCE_HOLD",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bindings": {
            "preregister": {"path": PREREG.name, "sha256": sha256(PREREG)},
            "runner": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        },
        "environment": {
            "flopscope": importlib.metadata.version("flopscope"),
            "whestbench": importlib.metadata.version("whestbench"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "branch_legality": branch_probe(),
        "exact_cost": cost_probe(),
        "propagation": propagation_probe(),
        "backend": backend_probe(),
        "access": {
            "candidate_source_files": 0,
            "fullwidth_width1024_runs": 0,
            "dataset_rows": 0,
            "targets_teachers_scores_or_board": 0,
            "packages": 0,
            "submissions": 0,
            "network_or_credentials": 0,
            "graph_edits": 0
        }
    }
    checks = [
        result["branch_legality"]["all_reduction_exact"],
        result["branch_legality"]["native_bool_zero_added_flops_and_calls"],
        result["branch_legality"]["branch_executes_expected_path_and_bills_it"],
        result["exact_cost"]["all_exact"],
        result["exact_cost"]["removed_scalar_where"]["exact_match"],
        result["propagation"]["all_finite"],
        result["propagation"]["all_stored_invalid_exact_zero"],
        result["propagation"]["all_stored_arithmetic_within_gamma_bound"],
    ]
    if not all(checks):
        result["status"] = "REJECT_PINNED_BRANCH_COST_OR_FP32_PROPAGATION_MICROPROBE"
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(HERE, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
