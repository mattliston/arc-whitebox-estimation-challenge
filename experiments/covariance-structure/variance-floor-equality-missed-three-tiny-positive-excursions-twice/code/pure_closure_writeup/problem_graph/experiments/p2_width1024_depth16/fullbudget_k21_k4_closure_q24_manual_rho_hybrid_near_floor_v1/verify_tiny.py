#!/usr/bin/env python3
"""Static and tiny numerical gates for the near-floor repair trigger."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PARENT = P2 / "fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py"
NEAR = HERE / "estimator.py"
GLOBAL = P2 / "fullbudget_k21_k4_closure_q24_manual_rho_raw_diag_load_rescale_v1/estimator.py"
EXPECTED_SHA256 = {
    PARENT: "5e0fccfe8569b43aa85f77997c1dd890c5dd1173f114167e6fb50f7e1d85e5b5",
    NEAR: "96a19116bee4a85f14f826c0a1634fe93bf39a21d78100eb46cd7f31add1e52b",
    GLOBAL: "7b93659914b2a100efb824b2aefc7abaef259df602e60603403901264ec91554",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def outside_repair(text: str) -> tuple[str, str]:
    start = text.index("def _repair_covariance(")
    stop = text.index("\ndef ", start + 1)
    return text[:start], text[stop:]


def run_repair(module, raw: np.ndarray, variance: np.ndarray):
    width = variance.size
    fusion_context = [None] * 18
    with flops.BudgetContext(flop_budget=10**8, quiet=True) as budget:
        raw_f = fnp.asarray(raw, dtype=fnp.float64)
        variance_f = fnp.asarray(variance, dtype=fnp.float64)
        diagonal = fnp.asarray(np.eye(width, dtype=bool))
        fusion_context[17] = fnp.empty((8, width, width), dtype=fnp.float64)
        result = module._repair_covariance(
            raw_f, variance_f, diagonal, fusion_context
        )
    return np.asarray(result).copy(), int(budget.flops_used)


def main() -> None:
    for path, expected in EXPECTED_SHA256.items():
        assert sha256(path) == expected, path
    parent_text = PARENT.read_text()
    near_text = NEAR.read_text()
    expected_near_text = parent_text.replace(
        '"""Use raw-scale repair exactly when a marginal reached the hard floor."""',
        '"""Use raw-scale repair when a marginal is within 2x of the hard floor."""',
    ).replace(
        "if bool(fnp.any(variance == _VAR_FLOOR)):",
        "if bool(fnp.any(variance <= 2.0 * _VAR_FLOOR)):",
    )
    assert near_text == expected_near_text
    parent = load("near_floor_parent", PARENT)
    near = load("near_floor_candidate", NEAR)
    global_b = load("near_floor_global", GLOBAL)
    parent_outside = outside_repair(PARENT.read_text())
    assert outside_repair(NEAR.read_text()) == parent_outside
    assert outside_repair(GLOBAL.read_text()) == parent_outside

    base_variance = np.array([0.04, 1.0, 4.0], dtype=np.float64)
    indefinite = np.array(
        [[1.0, 0.95, 0.95], [0.95, 1.0, -0.95], [0.95, -0.95, 1.0]],
        dtype=np.float64,
    )
    rows = []
    for name, first_variance, expected in (
        ("healthy", 0.04, "parent"),
        ("exact_floor", 1.0e-30, "global"),
        ("rank55_one_ulp", 1.0000000000000003e-30, "global"),
        ("rank71_four_ulps", 1.0000000000000008e-30, "global"),
        ("two_x_floor", 2.0e-30, "global"),
        ("above_two_x", np.nextafter(2.0e-30, np.inf), "parent"),
    ):
        variance = base_variance.copy()
        variance[0] = first_variance
        scale = np.sqrt(variance)
        raw = indefinite * scale[:, None] * scale[None, :]
        parent_result, parent_flops = run_repair(parent, raw.copy(), variance.copy())
        near_result, near_flops = run_repair(near, raw.copy(), variance.copy())
        global_result, global_flops = run_repair(global_b, raw.copy(), variance.copy())
        expected_result = parent_result if expected == "parent" else global_result
        assert np.array_equal(near_result, expected_result)
        if expected == "global":
            assert np.array_equal(np.diag(near_result), variance)
        assert np.isfinite(near_result).all()
        assert float(np.linalg.eigvalsh(near_result)[0]) >= -2.0e-12
        rows.append({
            "case": name,
            "expected_path": expected,
            "candidate_flops": near_flops,
            "parent_flops": parent_flops,
            "global_flops": global_flops,
        })

    print(json.dumps({
        "status": "PASS",
        "trigger": "any(variance <= 2.0 * _VAR_FLOOR)",
        "source_sha256": sha256(NEAR),
        "source_diff_exactly_docstring_and_trigger": True,
        "outside_repair_parent_bytes_equal": True,
        "rows": rows,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
