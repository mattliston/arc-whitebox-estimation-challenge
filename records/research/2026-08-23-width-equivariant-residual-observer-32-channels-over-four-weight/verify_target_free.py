#!/usr/bin/env python3
"""Target-free synthetic verification of fast residual ridge V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import ridge_science as science


HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rng = np.random.default_rng(20260823)
    raw_unstandardized = np.zeros((science.DEPTH, science.WIDTH, 20), dtype=np.float32)
    raw_unstandardized[..., 0] = np.asarray(
        [np.float32(2.0 * layer0 / 15.0 - 1.0) for layer0 in range(science.DEPTH)]
    )[:, None]
    raw_unstandardized[..., 1:] = rng.normal(
        0.25, 1.5, (science.DEPTH, science.WIDTH, 19)
    ).astype(np.float32)
    normalization = science.NormalizationStatistics.empty()
    normalization.add_raw_network(raw_unstandardized)
    centers, scales = normalization.finalize(1)
    raw = science.standardize_raw_network(raw_unstandardized, centers, scales)
    expected_depth = np.asarray(
        [np.float32(2.0 * layer0 / 15.0 - 1.0) for layer0 in range(science.DEPTH)]
    )
    if not np.array_equal(raw[:, 0, 0], expected_depth):
        raise AssertionError("depth bypass changed")
    host = np.abs(rng.standard_normal((science.DEPTH, science.WIDTH))).astype(np.float32)
    sigma = (0.25 + rng.random((science.DEPTH, science.WIDTH))).astype(np.float32)
    known = np.zeros((science.DEPTH, 77), dtype=np.float64)
    known[1:, :20] = rng.normal(0.0, 0.002, (science.DEPTH - 1, 20))
    target = science.predict_network(raw, host, sigma, "context77", known)

    stats = science.SufficientStatistics.empty("context77")
    stats.add_network(raw, host, sigma, target)
    solutions = science.solve_ladder(stats)
    if tuple(solutions) != tuple(f"lambda_{index}" for index in range(5)):
        raise AssertionError("ridge ladder order changed")
    for value in solutions.values():
        if value.shape != (16, 77) or value.dtype != np.float64 or np.any(value[0] != 0.0):
            raise AssertionError("synthetic ridge solution schema failed")
    recovered = science.predict_network(
        raw, host, sigma, "context77", solutions["lambda_0"]
    )
    baseline_error = science.ordered_layer_mse(host[None], target[None])[15]
    recovered_error = science.ordered_layer_mse(recovered[None], target[None])[15]
    if not recovered_error < np.float64(0.01) * baseline_error:
        raise AssertionError("synthetic weighted ridge did not recover the planted relation")

    unchanged = science.predict_network(
        raw, host, sigma, "linear20", np.zeros((16, 20), dtype=np.float64)
    )
    if not np.array_equal(unchanged, host):
        raise AssertionError("zero coefficients do not reproduce the host exactly")
    if not np.array_equal(target[0], host[0]):
        raise AssertionError("synthetic target violated layer-1 structural zero")

    networks = 2
    host_stack = np.stack((host, host)).astype(np.float32)
    target_stack = np.stack((target, target)).astype(np.float32)
    better = target_stack.copy()
    worse_l2 = target_stack.copy()
    worse_l2[:, 1] = host_stack[:, 1] + np.float32(1.0)
    selected = science.select_once(
        host_stack,
        (("better", better), ("worse_l2", worse_l2)),
        target_stack,
    )
    if selected["winner"] != "better":
        raise AssertionError("strict selection did not choose the exact synthetic target")
    rejected = next(row for row in selected["rows"] if row["name"] == "worse_l2")
    if rejected["layer2_non_regression"] is not False:
        raise AssertionError("layer-2 regression gate failed")

    spec = json.loads((HERE / "SCIENCE_SPEC.json").read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_TARGET_FREE_SCIENCE__HOLD_EXECUTION":
        raise AssertionError("science status changed")
    receipt = {
        "schema": "phase2-fast-residual-ridge-target-free-verification-v1",
        "status": "PASS_TARGET_FREE_RIDGE_SCIENCE__HOLD_ALL_DATA_AND_EXECUTION",
        "source_sha256": _sha256(HERE / "ridge_science.py"),
        "design_sha256": _sha256(HERE / "DESIGN.md"),
        "spec_sha256": _sha256(HERE / "SCIENCE_SPEC.json"),
        "basis_widths": science.BASIS_WIDTH,
        "ridge_ladder": [float(value) for value in science.RIDGE_LAMBDAS],
        "synthetic_layer1_host_exact": True,
        "synthetic_zero_coefficients_host_exact": True,
        "synthetic_fit_only_normalization_and_depth_bypass": True,
        "synthetic_l2_regression_rejected": True,
        "synthetic_planted_final_relation_recovered_below_one_percent_host_mse": True,
        "dataset_rows_opened": 0,
        "feature_rows_opened_or_hashed": 0,
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "fits": 0,
        "selection_metrics_computed_on_real_data": 0,
        "scores": 0,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
