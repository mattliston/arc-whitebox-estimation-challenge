#!/usr/bin/env python3
"""Reconstruct the target-free scalar readout-span diagnostic.

The public targets are not reopened.  For two already-scored predictions
``p0`` and ``p1``, write ``d = p1-p0`` and ``p(t)=p0+t*d``.  The persisted
MSEs at t=0 and t=1 plus E[d^2], computed from the sealed prediction arrays,
determine the entire quadratic MSE(t).  This script verifies the sealed input
hashes and reports the exact minimizing scalar at every layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
RECEIPTS = P2 / "data_gate" / "receipts"
RESULT = RECEIPTS / "joint_target_open" / (
    "0b3c04441938e2930f4d8fc92e9ca788d0dcfcc4f0107a8b152867df735d5a7e."
    "joint-score.json"
)
H4 = RECEIPTS / "analytic_ladder" / (
    "mini100-analytic-ladder-20260823T174059.488648Z-8bfdf89e2642"
) / "predictions_order04.npy"
CP1792 = RECEIPTS / "cp_ladder" / (
    "mini100-cp-ladder-20260823T180623.516178Z-78b82234e78d"
) / "predictions_cp3_cap1792.npy"
CP512 = RECEIPTS / "cp_ladder" / (
    "mini100-cp-ladder-20260823T180623.516178Z-78b82234e78d"
) / "predictions_cp3_cap512.npy"
SUMMARY = HERE / "SCALAR_SPAN_RESULT.json"

EXPECTED_SHA256 = {
    RESULT: "992f8e27d8b44c4ce236051022b4baf43e59b6f610ca57aedb6eb4315b73045c",
    H4: "ce11fe179e5ad458f1cdb125d55bca108d1e2c7ab5a3701dc3a141eca0c612df",
    CP1792: "cd0a12db8fe25b04929cb59e87dba0acec52390757bd61702eefd840799884fe",
    CP512: "504d8aae237b9869297a02e9062b3a30b67af205e5c35fd5cf59d36a07c9b26e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_projection(
    p0: np.ndarray,
    p1: np.ndarray,
    mse0: np.ndarray,
    mse1: np.ndarray,
) -> dict[str, list[float | None]]:
    delta2 = np.mean((p1 - p0) ** 2, axis=(0, 2), dtype=np.float64)
    cross = (mse1 - mse0 - delta2) / 2.0
    valid = delta2 > 0.0
    optimum = np.full(delta2.shape, np.nan, dtype=np.float64)
    optimum_mse = np.full(delta2.shape, np.nan, dtype=np.float64)
    optimum[valid] = -cross[valid] / delta2[valid]
    optimum_mse[valid] = mse0[valid] - cross[valid] ** 2 / delta2[valid]
    return {
        "delta_mean_square": [float(x) for x in delta2],
        "optimal_scalar": [None if not np.isfinite(x) else float(x) for x in optimum],
        "optimal_mse": [None if not np.isfinite(x) else float(x) for x in optimum_mse],
        "gain_fraction_vs_endpoint": [
            None
            if not np.isfinite(optimum_mse[index]) or mse1[index] == 0.0
            else float((mse1[index] - optimum_mse[index]) / mse1[index])
            for index in range(len(mse1))
        ],
    }


def main() -> None:
    observed = {str(path.relative_to(P2)): sha256(path) for path in EXPECTED_SHA256}
    for path, expected in EXPECTED_SHA256.items():
        if observed[str(path.relative_to(P2))] != expected:
            raise AssertionError(f"sealed input hash mismatch: {path}")

    result = json.loads(RESULT.read_text(encoding="utf-8"))
    h4 = np.load(H4).astype(np.float64)
    cp1792 = np.load(CP1792).astype(np.float64)
    cp512 = np.load(CP512).astype(np.float64)
    if h4.shape != (100, 16, 1024) or cp1792.shape != h4.shape or cp512.shape != h4.shape:
        raise AssertionError("unexpected prediction shape")

    metrics = result["candidate_metrics"]
    h4_mse = np.asarray(
        metrics["analytic_hermite_order_ladder/4"]["per_layer_raw_mse"],
        dtype=np.float64,
    )
    cp1792_mse = np.asarray(
        metrics["cp_cap_ladder/cp3_cap1792"]["per_layer_raw_mse"],
        dtype=np.float64,
    )
    cp512_mse = np.asarray(
        metrics["cp_cap_ladder/cp3_cap512"]["per_layer_raw_mse"],
        dtype=np.float64,
    )

    h4_span = scalar_projection(h4, cp1792, h4_mse, cp1792_mse)
    cap_span = scalar_projection(cp512, cp1792, cp512_mse, cp1792_mse)
    report = {
        "schema": "phase2-sealed-readout-span-diagnostic-v1",
        "status": "PASS_TARGET_FREE_RECONSTRUCTION__SCALAR_SPAN_INSUFFICIENT",
        "inputs_sha256": observed,
        "shape": list(h4.shape),
        "h4_to_cp1792": h4_span,
        "cp512_to_cp1792": cap_span,
        "layer16": {
            "cp1792_mse": float(cp1792_mse[-1]),
            "h4_to_cp1792_optimal_scalar": h4_span["optimal_scalar"][-1],
            "h4_to_cp1792_optimal_mse": h4_span["optimal_mse"][-1],
            "h4_to_cp1792_gain_fraction": h4_span["gain_fraction_vs_endpoint"][-1],
            "cp512_to_cp1792_optimal_scalar": cap_span["optimal_scalar"][-1],
            "cp512_to_cp1792_optimal_mse": cap_span["optimal_mse"][-1],
            "cp512_to_cp1792_gain_fraction": cap_span["gain_fraction_vs_endpoint"][-1],
        },
        "access": {
            "new_target_values_decoded": 0,
            "new_target_iterator_calls": 0,
            "new_score_computations": 0,
            "fits": 0,
        },
    }

    sealed_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    expected_summary = {
        "schema": "phase2-sealed-readout-span-summary-v1",
        "status": "PASS_TARGET_FREE_RECONSTRUCTION__SCALAR_SPAN_INSUFFICIENT",
        "inputs_sha256": observed,
        "layer2": {
            "cp1792_mse": float(cp1792_mse[1]),
            "h4_to_cp1792_optimal_scalar": h4_span["optimal_scalar"][1],
            "h4_to_cp1792_optimal_mse": h4_span["optimal_mse"][1],
            "h4_to_cp1792_gain_fraction": h4_span["gain_fraction_vs_endpoint"][1],
            "cp512_to_cp1792_optimal_scalar": cap_span["optimal_scalar"][1],
            "cp512_to_cp1792_optimal_mse": cap_span["optimal_mse"][1],
            "cp512_to_cp1792_gain_fraction": cap_span["gain_fraction_vs_endpoint"][1],
        },
        "layer16": report["layer16"],
        "conclusion": (
            "L2 has a small correctable residual; L16 cannot be solved by a "
            "one-dimensional CP-amplitude readout."
        ),
        "access": report["access"],
    }
    if sealed_summary != expected_summary:
        raise AssertionError("sealed scalar-span summary does not replay exactly")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
