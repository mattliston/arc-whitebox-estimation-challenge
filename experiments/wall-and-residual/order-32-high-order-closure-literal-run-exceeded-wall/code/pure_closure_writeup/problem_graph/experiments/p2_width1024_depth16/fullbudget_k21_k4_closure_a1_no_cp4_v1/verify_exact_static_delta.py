#!/usr/bin/env python3
"""Exact branch-matched FlopScope interpolation for the A1 CP4 deletion."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import flopscope as flops
import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "fullbudget_k21_k4_closure_v4" / "estimator.py"
CANDIDATE = HERE / "estimator.py"
PARENT_RECEIPT = (
    HERE.parent
    / "meter_protocol"
    / "runs"
    / "fb_ec4_v4_sparse_resetarena_cp3compact_n1_s8_seed20260824"
    / "receipt.json"
)
WIDTHS = (2, 4, 8, 16, 32, 64)
PRODUCTION_WIDTH = 1024
DEPTH = 16
PARENT_FLOPS = 2_179_742_954_469
PHASE2_BUDGET = 1 << 41
EXPECTED_CALL_DELETION = 925
EXPECTED_OPERATION_COEFFICIENTS = {
    # Basis order: n^3, n^2, n*log2(n), n, 1.
    "add": (0, 56, 0, 60, 0),
    "argsort": (0, 0, 156, 312, 0),
    "asarray": (0, 108, 0, 120, 0),
    "concatenate": (0, 42, 0, 42, 0),
    "einsum": (160, -54, 0, 0, 0),
    "getitem": (0, 156, 0, 156, 0),
    "matmul": (58, 29, 0, -15, 0),
    "multiply": (0, 404, 0, 216, 1),
    "negative": (0, 0, 0, 39, 0),
    "subtract": (0, 112, 0, 86, 0),
    "sum": (0, 39, 0, -39, 0),
    "where": (0, 880, 0, 0, 0),
}
EXPECTED_TOTAL_COEFFICIENTS = (218, 1772, 156, 977, 1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _basis(width: int) -> tuple[int, int, int, int, int]:
    assert width > 0 and width & (width - 1) == 0
    log2_width = width.bit_length() - 1
    return (
        width**3,
        width**2,
        width * log2_width,
        width,
        1,
    )


def _evaluate(coefficients, width: int) -> Fraction:
    return sum(
        Fraction(coefficient) * value
        for coefficient, value in zip(coefficients, _basis(width), strict=True)
    )


def _solve(rows, values) -> tuple[Fraction, ...]:
    matrix = [
        [Fraction(value) for value in row] + [Fraction(target)]
        for row, target in zip(rows, values, strict=True)
    ]
    count = len(matrix)
    assert all(len(row) == count + 1 for row in matrix)
    for column in range(count):
        pivot = next(
            row for row in range(column, count) if matrix[row][column]
        )
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        for row in range(count):
            if row == column:
                continue
            scale = matrix[row][column]
            if scale:
                matrix[row] = [
                    value - scale * pivot_value
                    for value, pivot_value in zip(
                        matrix[row], matrix[column], strict=True
                    )
                ]
    return tuple(matrix[index][-1] for index in range(count))


def _fixture(width: int) -> MLP:
    rng = np.random.default_rng(20260823)
    weights = [
        (
            np.eye(width)
            + 0.02 * rng.standard_normal((width, width)) / np.sqrt(width)
        ).astype(np.float32)
        for _ in range(DEPTH)
    ]
    return MLP(width=width, depth=DEPTH, weights=weights, seed=20260823)


def _measure(width: int) -> dict[str, object]:
    parent = _load(f"fb_ec4_v4_static_delta_n{width}", PARENT)
    candidate = _load(f"fb_ec4_a1_static_delta_n{width}", CANDIDATE)
    # Cap=2n reproduces production's exact rank and branch schedule:
    # n at W2, 2n at W3..W16, and thirteen candidate-rank-3n truncations.
    parent._CP4_CAP = 2 * width
    mlp = _fixture(width)
    with flops.BudgetContext(
        flop_budget=10_000_000_000_000, quiet=True
    ) as parent_budget:
        parent._predict(mlp)
    with flops.BudgetContext(
        flop_budget=10_000_000_000_000, quiet=True
    ) as candidate_budget:
        candidate._predict(mlp)
    parent_operations = parent_budget.summary_dict()["operations"]
    candidate_operations = candidate_budget.summary_dict()["operations"]
    operations = {}
    for name in sorted(set(parent_operations) | set(candidate_operations)):
        parent_item = parent_operations.get(name, {"calls": 0, "flop_cost": 0})
        candidate_item = candidate_operations.get(
            name, {"calls": 0, "flop_cost": 0}
        )
        delta = int(parent_item["flop_cost"] - candidate_item["flop_cost"])
        if delta:
            operations[name] = delta
    parent_calls = sum(item["calls"] for item in parent_operations.values())
    candidate_calls = sum(
        item["calls"] for item in candidate_operations.values()
    )
    delta = int(parent_budget.flops_used - candidate_budget.flops_used)
    assert delta == sum(operations.values())
    assert parent_calls - candidate_calls == EXPECTED_CALL_DELETION
    assert parent_operations["argsort"]["calls"] == 13
    return {
        "width": width,
        "parent_flops": int(parent_budget.flops_used),
        "candidate_flops": int(candidate_budget.flops_used),
        "deletion_flops": delta,
        "deletion_calls": parent_calls - candidate_calls,
        "operation_deletion_flops": operations,
    }


def verify() -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops.configure(symmetry_warnings=False)
    assert _sha256(PARENT) == (
        "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c"
    )
    assert _sha256(CANDIDATE) == (
        "5b222ef4f8120adb9443da5c2a1f89c79085a768ac2aef10c61c18413ef6a025"
    )
    measurements = [_measure(width) for width in WIDTHS]
    fit = measurements[:5]
    holdout = measurements[5]
    fit_rows = [_basis(item["width"]) for item in fit]

    operation_coefficients = {}
    production_operation_deletions = {}
    for operation, expected in EXPECTED_OPERATION_COEFFICIENTS.items():
        coefficients = _solve(
            fit_rows,
            [item["operation_deletion_flops"][operation] for item in fit],
        )
        assert coefficients == tuple(Fraction(value) for value in expected)
        assert _evaluate(coefficients, holdout["width"]) == (
            holdout["operation_deletion_flops"][operation]
        )
        production = _evaluate(coefficients, PRODUCTION_WIDTH)
        assert production.denominator == 1
        operation_coefficients[operation] = [int(value) for value in coefficients]
        production_operation_deletions[operation] = int(production)

    total_coefficients = _solve(
        fit_rows,
        [item["deletion_flops"] for item in fit],
    )
    assert total_coefficients == tuple(
        Fraction(value) for value in EXPECTED_TOTAL_COEFFICIENTS
    )
    assert _evaluate(total_coefficients, holdout["width"]) == holdout["deletion_flops"]
    production_deletion = _evaluate(total_coefficients, PRODUCTION_WIDTH)
    assert production_deletion.denominator == 1
    production_deletion = int(production_deletion)
    assert production_deletion == sum(production_operation_deletions.values())
    assert production_deletion == 235_936_392_193

    parent_receipt = json.loads(PARENT_RECEIPT.read_text())
    assert parent_receipt["measured"]["flops_used_per_mlp"] == [PARENT_FLOPS]
    parent_calls = sum(
        item["calls"]
        for item in parent_receipt["measured"]["operation_totals"].values()
    )
    assert parent_calls == 6_878
    candidate_flops = PARENT_FLOPS - production_deletion
    candidate_calls = parent_calls - EXPECTED_CALL_DELETION
    assert candidate_flops == 1_943_806_562_276
    assert candidate_calls == 5_953
    assert production_deletion / PARENT_FLOPS > 0.107
    return {
        "schema": "fb-ec4-32-a1-no-cp4-exact-static-delta-verification-v1",
        "status": "PASS_EXACT_BRANCH_MATCHED_FLOPSCOPE_DELTA__LITERAL_METER_STILL_REQUIRED",
        "candidate_sha256": _sha256(CANDIDATE),
        "parent_sha256": _sha256(PARENT),
        "basis": ["n^3", "n^2", "n*log2(n)", "n", "1"],
        "fit_widths": list(WIDTHS[:5]),
        "heldout_validation_width": WIDTHS[5],
        "measurements": measurements,
        "operation_coefficients": operation_coefficients,
        "total_coefficients": [int(value) for value in total_coefficients],
        "exact_total_formula": "218*n^3 + 1772*n^2 + 156*n*log2(n) + 977*n + 1",
        "production": {
            "width": PRODUCTION_WIDTH,
            "depth": DEPTH,
            "capped_resets": 13,
            "operation_deletion_flops": production_operation_deletions,
            "deletion_flops": production_deletion,
            "parent_literal_flops": PARENT_FLOPS,
            "candidate_projected_flops": candidate_flops,
            "deletion_fraction_of_parent": production_deletion / PARENT_FLOPS,
            "candidate_fraction_of_2pow41": candidate_flops / PHASE2_BUDGET,
            "candidate_headroom_flops": PHASE2_BUDGET - candidate_flops,
            "deletion_calls": EXPECTED_CALL_DELETION,
            "parent_literal_calls": parent_calls,
            "candidate_projected_calls": candidate_calls,
            "literal_meter_still_required": True,
        },
        "scope": {
            "full_shape_runs": 0,
            "dataset_members_opened": 0,
            "targets_opened": 0,
            "scores_computed": 0,
            "samples": 0,
            "fits": 0,
            "submissions": 0,
            "graph_edits": 0,
        },
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
