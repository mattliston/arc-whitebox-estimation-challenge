#!/usr/bin/env python3
"""Tiny exact-price projection for the full-V4 Q24 source."""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import flopscope as flops
import numpy as np


HERE = Path(__file__).resolve().parent
PARENT_PATH = HERE.parent / "fullbudget_k21_k4_closure_v4" / "estimator.py"
CHILD_PATH = HERE / "estimator.py"
DEPTH = 16
PARENT_LITERAL_FLOPS = 2_179_742_954_469
PHASE2_BUDGET = 2**41


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run(module, width):
    mlp = type("SyntheticMLP", (), {})()
    mlp.width = width
    mlp.depth = DEPTH
    mlp.weights = [np.eye(width, dtype=np.float32) for _ in range(DEPTH)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**12, quiet=True) as budget:
            output = np.asarray(module._predict(mlp))
    operations = budget.summary_dict()["operations"]
    assert np.isfinite(output).all()
    return budget.flops_used, sum(item["calls"] for item in operations.values())


def main() -> None:
    parent = load("q24_resource_parent", PARENT_PATH)
    child = load("q24_resource_child", CHILD_PATH)
    rows = []
    for width in (4, 5, 6, 8, 12, 16):
        parent_flops, parent_calls = run(parent, width)
        child_flops, child_calls = run(child, width)
        rows.append(
            {
                "width": width,
                "parent_flops": parent_flops,
                "child_flops": child_flops,
                "delta_parent_minus_child": parent_flops - child_flops,
                "parent_calls": parent_calls,
                "child_calls": child_calls,
            }
        )

    d4, d5, d6 = (rows[index]["delta_parent_minus_child"] for index in range(3))
    coefficient_n2 = (d6 - 2 * d5 + d4) // 2
    coefficient_n1 = (d5 - d4) - 9 * coefficient_n2
    coefficient_n0 = d4 - 16 * coefficient_n2 - 4 * coefficient_n1
    for row in rows:
        width = row["width"]
        assert row["delta_parent_minus_child"] == (
            coefficient_n2 * width * width
            + coefficient_n1 * width
            + coefficient_n0
        )
        assert row["parent_calls"] == row["child_calls"]

    width = 1024
    projected_delta = (
        coefficient_n2 * width * width
        + coefficient_n1 * width
        + coefficient_n0
    )
    projected_child = PARENT_LITERAL_FLOPS - projected_delta
    assert projected_child < PHASE2_BUDGET
    print(
        json.dumps(
            {
                "status": "PASS_Q24_TINY_EXACT_PRICE_POLYNOMIAL_PROJECTION",
                "rows": rows,
                "delta_polynomial": {
                    "n_squared": coefficient_n2,
                    "n": coefficient_n1,
                    "constant": coefficient_n0,
                },
                "production_projection": {
                    "width": width,
                    "depth": DEPTH,
                    "parent_literal_flops": PARENT_LITERAL_FLOPS,
                    "projected_removed_flops": projected_delta,
                    "projected_child_flops": projected_child,
                    "headroom_below_2pow41": PHASE2_BUDGET - projected_child,
                    "calls_delta": 0,
                    "persistent_rho_arena_bytes_removed": 8 * width * width * 8,
                },
                "claim_boundary": "Literal whole-source meter remains authoritative.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
