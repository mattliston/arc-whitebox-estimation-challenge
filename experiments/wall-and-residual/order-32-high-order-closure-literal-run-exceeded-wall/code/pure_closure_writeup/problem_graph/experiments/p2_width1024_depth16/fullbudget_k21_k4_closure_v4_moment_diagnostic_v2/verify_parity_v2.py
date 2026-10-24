#!/usr/bin/env python3
"""V2 tiny target-free parity, normalized-channel, and exact-price gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import warnings

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "fullbudget_k21_k4_closure_v4" / "estimator.py"
DIAGNOSTIC = HERE / "estimator.py"
FLOP_BUDGET = 100_000_000_000
EXPECTED_FULL_INCREMENT = 5_902_272
TWO_U = 2.220446049250313e-16


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_mlp(width: int = 4, depth: int = 4) -> MLP:
    rng = np.random.default_rng(20260823)
    scale = np.sqrt(2.0 / width)
    weights = [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights, seed=20260823)


def _standalone_increment(module, *, width: int, depth: int) -> int:
    raw1 = fnp.asarray(np.linspace(0.4, 1.2, width), dtype=fnp.float64)
    raw2 = fnp.asarray(np.linspace(0.8, 2.4, width), dtype=fnp.float64)
    raw3 = fnp.asarray(np.linspace(1.8, 7.2, width), dtype=fnp.float64)
    raw4 = fnp.asarray(np.linspace(5.0, 25.0, width), dtype=fnp.float64)
    with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as budget:
        rows = [
            module._normalized_moment_worst_row(raw1, raw2, raw3, raw4)
            for _ in range(depth)
        ]
        _ = fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float64)
    return int(budget.flops_used)


def _up_np(value):
    return np.nextafter(value, np.inf)


def _add_sub_error_np(error_x, error_y, result):
    inherited = _up_np(error_x + error_y)
    local = _up_np(TWO_U * np.abs(result))
    return _up_np(inherited + local)


def _multiply_error_np(x, error_x, y, error_y, result):
    left = _up_np(np.abs(x) * error_y)
    right = _up_np(np.abs(y) * error_x)
    inherited = _up_np(left + right)
    cross = _up_np(error_x * error_y)
    inherited = _up_np(inherited + cross)
    local = _up_np(TWO_U * np.abs(result))
    return _up_np(inherited + local)


def _expected_certified_row(raw1, raw2, raw3, raw4):
    zero = np.zeros_like(raw1)
    raw1_squared = raw1 * raw1
    error_raw1_squared = _multiply_error_np(
        raw1, zero, raw1, zero, raw1_squared
    )
    variance = raw2 - raw1_squared
    error_variance = _add_sub_error_np(zero, error_raw1_squared, variance)
    raw2_squared = raw2 * raw2
    error_raw2_squared = _multiply_error_np(
        raw2, zero, raw2, zero, raw2_squared
    )
    a_left = raw1 * raw3
    error_a_left = _multiply_error_np(raw1, zero, raw3, zero, a_left)
    margin_a = a_left - raw2_squared
    error_margin_a = _add_sub_error_np(
        error_a_left, error_raw2_squared, margin_a
    )
    raw1_raw2 = raw1 * raw2
    error_raw1_raw2 = _multiply_error_np(
        raw1, zero, raw2, zero, raw1_raw2
    )
    centered = raw3 - raw1_raw2
    error_centered = _add_sub_error_np(zero, error_raw1_raw2, centered)
    raw4_minus_raw2_squared = raw4 - raw2_squared
    error_raw4_minus = _add_sub_error_np(
        zero, error_raw2_squared, raw4_minus_raw2_squared
    )
    b_left = raw4_minus_raw2_squared * variance
    error_b_left = _multiply_error_np(
        raw4_minus_raw2_squared,
        error_raw4_minus,
        variance,
        error_variance,
        b_left,
    )
    centered_squared = centered * centered
    error_centered_squared = _multiply_error_np(
        centered, error_centered, centered, error_centered, centered_squared
    )
    margin_b = b_left - centered_squared
    error_margin_b = _add_sub_error_np(
        error_b_left, error_centered_squared, margin_b
    )
    scale_a = np.abs(a_left) + np.abs(raw2_squared)
    scale_b = np.abs(b_left) + np.abs(centered_squared)
    positive_a = scale_a > 0.0
    positive_b = scale_b > 0.0
    relative_a = np.zeros_like(scale_a)
    relative_b = np.zeros_like(scale_b)
    relative_error_a = np.zeros_like(scale_a)
    relative_error_b = np.zeros_like(scale_b)
    np.divide(margin_a, scale_a, out=relative_a, where=positive_a)
    np.divide(margin_b, scale_b, out=relative_b, where=positive_b)
    np.divide(
        error_margin_a, scale_a, out=relative_error_a, where=positive_a
    )
    np.divide(
        error_margin_b, scale_b, out=relative_error_b, where=positive_b
    )
    negative_a = margin_a < -error_margin_a
    positive_sign_a = margin_a > error_margin_a
    negative_b = margin_b < -error_margin_b
    positive_sign_b = margin_b > error_margin_b
    index_a = np.argmin(np.where(positive_a, relative_a, np.inf))
    index_b = np.argmin(np.where(positive_b, relative_b, np.inf))
    return np.array(
        [
            relative_a[index_a], relative_error_a[index_a],
            margin_a[index_a], error_margin_a[index_a],
            index_a, scale_a[index_a],
            raw1[index_a], raw2[index_a], raw3[index_a], raw4[index_a],
            np.count_nonzero(negative_a),
            raw1.size - np.count_nonzero(negative_a) - np.count_nonzero(positive_sign_a),
            np.count_nonzero(positive_sign_a), np.count_nonzero(~positive_a),
            relative_b[index_b], relative_error_b[index_b],
            margin_b[index_b], error_margin_b[index_b],
            index_b, scale_b[index_b],
            raw1[index_b], raw2[index_b], raw3[index_b], raw4[index_b],
            np.count_nonzero(negative_b),
            raw1.size - np.count_nonzero(negative_b) - np.count_nonzero(positive_sign_b),
            np.count_nonzero(positive_sign_b), np.count_nonzero(~positive_b),
        ],
        dtype=np.float64,
    )


def verify() -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops.configure(symmetry_warnings=False)
    v4 = _load("fb_ec4_v4_moment_parent", V4)
    diagnostic = _load("fb_ec4_v4_normalized_moment_diagnostic_v2", DIAGNOSTIC)
    mlp = _tiny_mlp()
    with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as parent_budget:
        parent_output = v4._predict(mlp)
    with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as diagnostic_budget:
        replay_output, moment_rows = diagnostic._predict_with_diagnostics(mlp)

    parent_array = np.asarray(parent_output)
    replay_array = np.asarray(replay_output)
    diagnostic_array = np.asarray(moment_rows)
    assert np.array_equal(parent_array, replay_array)
    assert diagnostic_array.shape == (mlp.depth, 28)
    assert np.isfinite(diagnostic_array).all()

    raw1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    raw2 = np.array([2.0, 1.0, 2.0, 2.0], dtype=np.float64)
    raw3 = np.array([6.0, 1.0, 1.0, 6.0], dtype=np.float64)
    raw4 = np.array([24.0, 1.0, 1.0, 12.0], dtype=np.float64)
    expected = _expected_certified_row(raw1, raw2, raw3, raw4)
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        actual = diagnostic._normalized_moment_worst_row(
            fnp.asarray(raw1), fnp.asarray(raw2), fnp.asarray(raw3), fnp.asarray(raw4)
        )
    np.testing.assert_array_equal(np.asarray(actual), expected)

    standalone_cost = _standalone_increment(
        diagnostic, width=mlp.width, depth=mlp.depth
    )
    measured_increment = int(diagnostic_budget.flops_used - parent_budget.flops_used)
    assert measured_increment == standalone_cost
    width3_depth1 = _standalone_increment(diagnostic, width=3, depth=1)
    width5_depth1 = _standalone_increment(diagnostic, width=5, depth=1)
    width3_depth4 = _standalone_increment(diagnostic, width=3, depth=4)
    assert width3_depth1 == 1 * (360 * 3 + 252)
    assert width5_depth1 == 1 * (360 * 5 + 252)
    assert width3_depth4 == 4 * (360 * 3 + 252)
    assert 16 * (360 * 1024 + 252) == EXPECTED_FULL_INCREMENT
    return {
        "status": "PASS_V4_V2_CERTIFIED_NORMALIZED_TRAJECTORY_BIT_PARITY_AND_PRICE",
        "tiny_width": mlp.width,
        "tiny_depth": mlp.depth,
        "mean_output_bitwise_equal": True,
        "diagnostic_shape": list(diagnostic_array.shape),
        "parent_flops": int(parent_budget.flops_used),
        "diagnostic_flops": int(diagnostic_budget.flops_used),
        "measured_additive_flops": measured_increment,
        "standalone_additive_flops": standalone_cost,
        "exact_increment_formula": "depth * (360 * width + 252)",
        "expected_width1024_depth16_increment": EXPECTED_FULL_INCREMENT,
        "price_probes": {
            "width3_depth1": width3_depth1,
            "width5_depth1": width5_depth1,
            "width3_depth4": width3_depth4,
        },
    }


if __name__ == "__main__":
    print(verify())
