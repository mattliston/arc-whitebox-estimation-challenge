#!/usr/bin/env python3
"""Tiny exact parity plus polynomial static bill for the two D4 CSEs."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1" / "estimator.py"
CANDIDATE = HERE / "estimator.py"
RESULT = HERE / "CSE_STATIC_RECEIPT.json"
WIDTHS = (8, 16, 32)
PRODUCTION_WIDTH = 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(meter) -> dict[str, object]:
    summary = meter.summary_dict(by_namespace=False)
    return {
        "flops": int(summary["flops_used"]),
        "calls": {
            name: int(record["calls"])
            for name, record in sorted(summary["operations"].items())
        },
        "costs": {
            name: int(record["flop_cost"])
            for name, record in sorted(summary["operations"].items())
        },
    }


def exact_quadratic(points: list[tuple[int, int]]) -> tuple[Fraction, Fraction, Fraction]:
    matrix = [
        [Fraction(x * x), Fraction(x), Fraction(1), Fraction(y)]
        for x, y in points
    ]
    for pivot in range(3):
        scale = matrix[pivot][pivot]
        matrix[pivot] = [value / scale for value in matrix[pivot]]
        for row in range(3):
            if row == pivot:
                continue
            factor = matrix[row][pivot]
            matrix[row] = [
                left - factor * right
                for left, right in zip(matrix[row], matrix[pivot])
            ]
    return matrix[0][3], matrix[1][3], matrix[2][3]


def evaluate(coefficients: tuple[Fraction, Fraction, Fraction], width: int) -> int:
    a, b, c = coefficients
    value = a * width * width + b * width + c
    if value.denominator != 1:
        raise AssertionError(value)
    return value.numerator


def make_pair_inputs(module, width: int, seed: int):
    rng = np.random.default_rng(seed)
    coefficients = rng.standard_normal((3, 39, width)) * 0.1
    sigma = np.exp(rng.standard_normal(width) * 0.1)
    base = rng.standard_normal((width, width)) * 0.03
    covariance = 0.5 * (base + base.T)
    np.fill_diagonal(covariance, sigma * sigma)
    fusion = module._fusion_context(width)
    context = module._q24_context(
        fnp.asarray(coefficients, dtype=fnp.float64),
        fnp.asarray(sigma, dtype=fnp.float64),
        fnp.asarray(covariance, dtype=fnp.float64),
        fusion,
    )
    k3 = rng.standard_normal(width) * 0.01
    k21 = rng.standard_normal((width, width)) * 0.01
    k4 = rng.standard_normal(width) * 0.01
    k31 = rng.standard_normal((width, width)) * 0.01
    raw_k22 = rng.standard_normal((width, width)) * 0.01
    k22 = 0.5 * (raw_k22 + raw_k22.T)
    weights = module._edgeworth_weights(
        fnp.asarray(k3, dtype=fnp.float64),
        fnp.asarray(k21, dtype=fnp.float64),
        fnp.asarray(k4, dtype=fnp.float64),
        fnp.asarray(k31, dtype=fnp.float64),
        fnp.asarray(k22, dtype=fnp.float64),
        fusion,
    )
    return context, weights, fusion[12:14]


def pair_arm(module, width: int, seed: int):
    context, weights, scratch = make_pair_inputs(module, width, seed)
    with flops.BudgetContext(flop_budget=10_000_000_000, quiet=True) as meter:
        values = module._four_pair_edgeworth(context, weights, scratch)
    return [np.array(value, copy=True) for value in values], inventory(meter)


def reset_arm(module, width: int, seed: int):
    rng = np.random.default_rng(seed)
    k21_target = rng.standard_normal((width, width)) * 0.01
    cp_post = rng.standard_normal((width, width)) * 0.01
    k4 = rng.standard_normal(width) * 0.01
    k31 = rng.standard_normal((width, width)) * 0.01
    raw_k22 = rng.standard_normal((width, width)) * 0.01
    k22 = 0.5 * (raw_k22 + raw_k22.T)
    raw_covariance = rng.standard_normal((width, width)) * 0.01
    covariance = 0.5 * (raw_covariance + raw_covariance.T)
    variance = np.exp(rng.standard_normal(width) * 0.1)
    np.fill_diagonal(covariance, variance)
    covariance2 = covariance * covariance
    variance_outer = variance[:, None] * variance[None, :]
    diagonal = np.eye(width, dtype=bool)
    with flops.BudgetContext(flop_budget=10_000_000_000, quiet=True) as meter:
        values = module._reset_stars(
            fnp.asarray(k21_target, dtype=fnp.float64),
            fnp.asarray(cp_post, dtype=fnp.float64),
            fnp.asarray(k4, dtype=fnp.float64),
            fnp.asarray(k31, dtype=fnp.float64),
            fnp.asarray(k22, dtype=fnp.float64),
            fnp.asarray(covariance, dtype=fnp.float64),
            fnp.asarray(variance, dtype=fnp.float64),
            fnp.asarray(covariance2, dtype=fnp.float64),
            fnp.asarray(variance_outer, dtype=fnp.float64),
            None,
            fnp.asarray(diagonal, dtype=fnp.bool_),
        )
    arrays = [np.asarray(values[0]), np.asarray(values[1]), np.asarray(values[2])]
    return [np.array(value, copy=True) for value in arrays], inventory(meter)


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    parent = load("d4_parent_static", PARENT)
    candidate = load("d4_cse_static", CANDIDATE)
    records = []
    pair_old_points: list[tuple[int, int]] = []
    pair_new_points: list[tuple[int, int]] = []
    reset_old_points: list[tuple[int, int]] = []
    reset_new_points: list[tuple[int, int]] = []
    for width in WIDTHS:
        old_pair, old_pair_meter = pair_arm(parent, width, 608_300 + width)
        new_pair, new_pair_meter = pair_arm(candidate, width, 608_300 + width)
        old_reset, old_reset_meter = reset_arm(parent, width, 608_400 + width)
        new_reset, new_reset_meter = reset_arm(candidate, width, 608_400 + width)
        pair_byte = [np.array_equal(left, right) for left, right in zip(old_pair, new_pair)]
        pair_maximum = [
            float(np.max(np.abs(left - right)))
            for left, right in zip(old_pair, new_pair)
        ]
        pair_rms = [
            float(np.sqrt(np.mean((left - right) ** 2)))
            for left, right in zip(old_pair, new_pair)
        ]
        reset_byte = [np.array_equal(left, right) for left, right in zip(old_reset, new_reset)]
        # M21/M31 are byte-identical.  The symmetric M11/M22 reconstructions
        # are mathematically identical and may differ by one final fp64 ulp
        # because a missing derivative channel is read as a transpose view.
        pair_numeric = (
            pair_byte[1]
            and pair_byte[2]
            and pair_maximum[0] <= 1.0e-15
            and pair_maximum[3] <= 1.0e-15
        )
        if not pair_numeric or not all(reset_byte):
            raise AssertionError(
                (width, pair_byte, pair_maximum, reset_byte)
            )
        pair_old_points.append((width, old_pair_meter["flops"]))
        pair_new_points.append((width, new_pair_meter["flops"]))
        reset_old_points.append((width, old_reset_meter["flops"]))
        reset_new_points.append((width, new_reset_meter["flops"]))
        records.append({
            "width": width,
            "pair_byte_parity": pair_byte,
            "pair_maximum_absolute_delta": pair_maximum,
            "pair_rms_delta": pair_rms,
            "pair_numeric_parity": pair_numeric,
            "reset_byte_parity": reset_byte,
            "old_pair": old_pair_meter,
            "new_pair": new_pair_meter,
            "old_reset": old_reset_meter,
            "new_reset": new_reset_meter,
        })

    pair_old_poly = exact_quadratic(pair_old_points)
    pair_new_poly = exact_quadratic(pair_new_points)
    reset_old_poly = exact_quadratic(reset_old_points)
    reset_new_poly = exact_quadratic(reset_new_points)
    pair_old = evaluate(pair_old_poly, PRODUCTION_WIDTH)
    pair_new = evaluate(pair_new_poly, PRODUCTION_WIDTH)
    reset_old = evaluate(reset_old_poly, PRODUCTION_WIDTH)
    reset_new = evaluate(reset_new_poly, PRODUCTION_WIDTH)
    incremental_saving = 14 * (pair_old - pair_new) + 15 * (reset_old - reset_new)
    payload = {
        "schema": "p2r0c1-d4-nongemm-cse-static-v1",
        "status": "PASS_TINY_BYTE_PARITY_AND_EXACT_STATIC_BILL",
        "widths": list(WIDTHS),
        "records": records,
        "production_width": PRODUCTION_WIDTH,
        "production_full_pair_resets": 14,
        "production_star_resets": 15,
        "production_pair_old_flops_per_reset": pair_old,
        "production_pair_new_flops_per_reset": pair_new,
        "production_pair_saving_per_reset": pair_old - pair_new,
        "production_reset_old_flops_per_reset": reset_old,
        "production_reset_new_flops_per_reset": reset_new,
        "production_reset_saving_per_reset": reset_old - reset_new,
        "production_incremental_saving_flops": incremental_saving,
        "predicted_d4_cold_flops": 1_319_871_710_583 - incremental_saving,
        "predicted_d4_rank279_hot_flops": 1_356_595_288_584 - incremental_saving,
        "predicted_d4_rank35_hot_flops": 1_317_948_959_232 - incremental_saving,
        "parent_sha256": sha256(PARENT),
        "candidate_sha256": sha256(CANDIDATE),
        "runner_sha256": sha256(Path(__file__)),
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "status", "production_incremental_saving_flops",
        "predicted_d4_cold_flops", "predicted_d4_rank279_hot_flops",
        "predicted_d4_rank35_hot_flops",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
