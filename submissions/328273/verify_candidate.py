#!/usr/bin/env python3
"""Static, tiny, and price gates for manual-rho plus exact-floor repair."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import flopscope as flops
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / "fullbudget_k21_k4_closure_q24_v1/estimator.py"
MANUAL = ROOT / "fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py"
HYBRID = (
    ROOT
    / "fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1"
    / "hybrid_exact_var_floor_trigger/estimator.py"
)
RAW_DIAG = (
    ROOT
    / "fullbudget_k21_k4_closure_q24_covariance_repair_variants_v1"
    / "raw_diag_load_rescale/estimator.py"
)
CANDIDATE = HERE / "estimator.py"
METER_ROOT = ROOT / "meter_protocol/runs"
PARENT_RECEIPT = (
    METER_ROOT
    / "fb_ec4_q24_full_v1_b38cd865_n1_s8_seed20260824/receipt.json"
)
MANUAL_RECEIPT = (
    METER_ROOT
    / "fb_ec4_q24_manual_rho_v1_7d0c9e30_n1_s8_seed20260824/receipt.json"
)
RAW_DIAG_RECEIPT = (
    METER_ROOT
    / "fb_ec4_q24_rawdiag_5399e914_n1_s8_seed20260824/receipt.json"
)
EXPECTED_HASHES = {
    PARENT: "b38cd865297c344626da3c4f414e1d4923a9efdc62147e3103f155b2034f7b52",
    MANUAL: "7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00",
    HYBRID: "aa87cee5e26ab25590a693d9ac08d5c2bc79cdd43b0c643ca9176df21919addb",
    RAW_DIAG: "5399e9140554fe676797d7bb6307bafbedf40a3ba9511b4eda8023d56732b160",
    CANDIDATE: "5e0fccfe8569b43aa85f77997c1dd890c5dd1173f114167e6fb50f7e1d85e5b5",
}
PHASE2_BUDGET = 1 << 41
DEPTH = 16
Q_TERMS = 25
VAR_FLOOR = 1.0e-30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def function_text(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1:node.end_lineno])


def function_map(source: str) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def operation_calls(budget) -> int:
    return int(sum(
        item["calls"]
        for item in budget.summary_dict()["operations"].values()
    ))


def static_gate() -> dict[str, object]:
    for path, expected in EXPECTED_HASHES.items():
        assert sha256(path) == expected, path
    parent = PARENT.read_text()
    manual = MANUAL.read_text()
    hybrid = HYBRID.read_text()
    candidate = CANDIDATE.read_text()
    parent_q = function_text(parent, "_q24_context")
    manual_q = function_text(manual, "_q24_context")
    parent_repair = function_text(parent, "_repair_covariance")
    hybrid_repair = function_text(hybrid, "_repair_covariance")
    assert candidate == hybrid.replace(parent_q, manual_q)
    assert candidate == manual.replace(parent_repair, hybrid_repair)
    pmap = function_map(parent)
    cmap = function_map(candidate)
    assert pmap.keys() == cmap.keys()
    changed = sorted(name for name in pmap if pmap[name] != cmap[name])
    assert changed == ["_q24_context", "_repair_covariance"]
    return {
        "all_source_hashes_exact": True,
        "candidate_equals_hybrid_with_manual_q24_only": True,
        "candidate_equals_manual_rho_with_hybrid_repair_only": True,
        "changed_executable_functions_vs_b38": changed,
    }


def context_case(module, covariance: np.ndarray, coefficients: np.ndarray):
    width = covariance.shape[0]
    with flops.BudgetContext(flop_budget=10**10, quiet=True) as budget:
        fusion = module._fusion_context(width)
        arena = np.asarray(fusion[14])
        arena.view(np.uint64)[...] = np.uint64(0x7FF0000000001234)
        poison_before = int(np.isnan(arena).sum())
        values = module._q24_context(
            module.fnp.asarray(coefficients, dtype=module.fnp.float64),
            module.fnp.asarray(np.ones(width), dtype=module.fnp.float64),
            module.fnp.asarray(covariance, dtype=module.fnp.float64),
            fusion,
        )
        copied = tuple(np.asarray(value).copy() for value in values)
    return copied, int(budget.flops_used), operation_calls(budget), poison_before


def context_gate(hybrid, candidate) -> dict[str, object]:
    width = 4
    coefficients = (
        np.arange(3 * 39 * width, dtype=np.float64).reshape(3, 39, width)
        / 257.0
        - 0.5
    )
    one_below = np.nextafter(np.float64(1.0), np.float64(0.0))
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    cases = {
        "signed_zero_near_one": np.asarray([
            [1.0, -0.0, 0.5, -0.75],
            [0.0, 1.0, one_below, -one_below],
            [0.5, one_below, 1.0, subnormal],
            [-0.75, -one_below, -subnormal, 1.0],
        ]),
        "clip_and_repeated_powers": np.asarray([
            [1.25, -1.25, 0.25, -0.25],
            [-1.25, 1.25, -0.5, 0.5],
            [0.25, -0.5, 1.25, -1.25],
            [-0.25, 0.5, -1.25, 1.25],
        ]),
    }
    rows = []
    for name, covariance in cases.items():
        old, old_flops, old_calls, old_poison = context_case(
            hybrid, covariance, coefficients
        )
        new, new_flops, new_calls, new_poison = context_case(
            candidate, covariance, coefficients
        )
        assert len(old) == len(new) == 5
        assert all(array_sha256(a) == array_sha256(b) for a, b in zip(old, new))
        assert old_poison == new_poison == Q_TERMS * width * width
        assert new_flops - old_flops == 2 * width * width
        assert new_calls - old_calls == 22
        rows.append({
            "case": name,
            "all_context_fields_byte_equal": True,
            "poison_cells_before": new_poison,
            "manual_delta_flops": new_flops - old_flops,
            "manual_delta_calls": new_calls - old_calls,
            "powers_sha256": array_sha256(new[3]),
        })
    return {"width": width, "rows": rows}


def run_repair(module, raw: np.ndarray, variance: np.ndarray):
    width = variance.size
    with flops.BudgetContext(flop_budget=10**8, quiet=True) as budget:
        fusion = [None] * 18
        fusion[17] = module.fnp.empty((8, width, width), dtype=module.fnp.float64)
        np.asarray(fusion[17]).view(np.uint64)[...] = np.uint64(0x7FF0000000001234)
        result = module._repair_covariance(
            module.fnp.asarray(raw, dtype=module.fnp.float64),
            module.fnp.asarray(variance, dtype=module.fnp.float64),
            module.fnp.asarray(np.eye(width, dtype=bool)),
            fusion,
        )
        copied = np.asarray(result).copy()
    return copied, int(budget.flops_used), operation_calls(budget)


def repair_gate(parent, raw_diag, hybrid, candidate) -> dict[str, object]:
    variance = np.asarray([0.04, 1.0, 4.0], dtype=np.float64)
    healthy = np.asarray([
        [1.0, 0.2, -0.1],
        [0.2, 1.0, 0.3],
        [-0.1, 0.3, 1.0],
    ])
    indefinite = np.asarray([
        [1.0, 0.95, 0.95],
        [0.95, 1.0, -0.95],
        [0.95, -0.95, 1.0],
    ])
    rows = []
    for name, first_variance, reference, correlation in (
        ("ordinary", variance[0], parent, healthy),
        ("nextafter_floor", np.nextafter(VAR_FLOOR, np.inf), parent, healthy),
        ("exact_floor", VAR_FLOOR, raw_diag, indefinite),
    ):
        requested = variance.copy()
        requested[0] = first_variance
        scale = np.sqrt(requested)
        raw = correlation * scale[:, None] * scale[None, :]
        expected, expected_flops, expected_calls = run_repair(
            reference, raw.copy(), requested.copy()
        )
        old, old_flops, old_calls = run_repair(
            hybrid, raw.copy(), requested.copy()
        )
        new, new_flops, new_calls = run_repair(
            candidate, raw.copy(), requested.copy()
        )
        assert array_sha256(old) == array_sha256(new)
        assert array_sha256(expected) == array_sha256(new)
        assert old_flops == new_flops and old_calls == new_calls
        if name == "exact_floor":
            assert np.array_equal(np.diag(new), requested)
        assert np.isfinite(new).all()
        rows.append({
            "case": name,
            "reference": "raw_diag" if name == "exact_floor" else "b38_parent",
            "reference_byte_equal": True,
            "aa87_hybrid_byte_equal": True,
            "repair_flops": new_flops,
            "repair_calls": new_calls,
            "branch_check_delta_flops": new_flops - expected_flops,
            "branch_check_delta_calls": new_calls - expected_calls,
            "output_sha256": array_sha256(new),
        })
    return {"rows": rows}


def synthetic_mlp(width: int, weights):
    mlp = type("SyntheticMLP", (), {})()
    mlp.width = width
    mlp.depth = DEPTH
    mlp.weights = weights
    return mlp


def run_predict(module, mlp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=PHASE2_BUDGET, quiet=True) as budget:
            output = np.asarray(module._predict(mlp)).copy()
    return output, int(budget.flops_used), operation_calls(budget)


def prediction_gate(hybrid, candidate) -> dict[str, object]:
    width = 8
    identity = [np.eye(width, dtype=np.float32) for _ in range(DEPTH)]
    signed_zero = []
    for layer in range(DEPTH):
        value = np.eye(width, dtype=np.float32)
        value[value == 0.0] = np.float32(-0.0 if layer % 2 else 0.0)
        signed_zero.append(value)
    rng = np.random.default_rng(20260823)
    near_identity = [
        (np.eye(width) + rng.normal(scale=0.02, size=(width, width))).astype(np.float32)
        for _ in range(DEPTH)
    ]
    rows = []
    for name, weights in (
        ("identity", identity),
        ("alternating_signed_zero_identity", signed_zero),
        ("near_identity_seed20260823", near_identity),
    ):
        mlp = synthetic_mlp(width, weights)
        old, old_flops, old_calls = run_predict(hybrid, mlp)
        new, new_flops, new_calls = run_predict(candidate, mlp)
        assert array_sha256(old) == array_sha256(new), name
        assert np.isfinite(new).all()
        assert new_flops - old_flops == 30 * width * width
        assert new_calls - old_calls == 330
        rows.append({
            "fixture": name,
            "output_byte_equal": True,
            "output_sha256": array_sha256(new),
            "flop_delta": new_flops - old_flops,
            "call_delta": new_calls - old_calls,
        })
    return {"width": width, "depth": DEPTH, "rows": rows}


def receipt_totals(path: Path) -> tuple[int, int]:
    receipt = json.loads(path.read_text())
    measured = receipt["measured"]
    flops_used = int(measured["flops_used_per_mlp"][0])
    calls = sum(int(item["calls"]) for item in measured["operation_totals"].values())
    return flops_used, calls


def price_gate(repair_rows: list[dict[str, object]]) -> dict[str, object]:
    parent_flops, parent_calls = receipt_totals(PARENT_RECEIPT)
    manual_flops, manual_calls = receipt_totals(MANUAL_RECEIPT)
    raw_flops, raw_calls = receipt_totals(RAW_DIAG_RECEIPT)
    width = 1024
    q_delta_flops = 30 * width * width
    q_delta_calls = 330
    assert manual_flops - parent_flops == q_delta_flops
    assert manual_calls - parent_calls == q_delta_calls
    repair_invocations = DEPTH - 2
    assert (raw_flops - parent_flops) % repair_invocations == 0
    assert (raw_calls - parent_calls) % repair_invocations == 0
    raw_path_delta_flops = (raw_flops - parent_flops) // repair_invocations
    raw_path_delta_calls = (raw_calls - parent_calls) // repair_invocations
    ordinary = next(row for row in repair_rows if row["case"] == "ordinary")
    exact = next(row for row in repair_rows if row["case"] == "exact_floor")
    assert ordinary["branch_check_delta_flops"] == exact["branch_check_delta_flops"]
    assert ordinary["branch_check_delta_calls"] == exact["branch_check_delta_calls"]
    # The direct width-3 gate prices scalar equality plus the metered any.
    assert ordinary["branch_check_delta_flops"] == 8
    assert ordinary["branch_check_delta_calls"] == 2
    check_flops_per_repair = 3 * width - 1
    check_calls_per_repair = 2
    fixed_flops = (
        parent_flops
        + q_delta_flops
        + repair_invocations * check_flops_per_repair
    )
    fixed_calls = (
        parent_calls
        + q_delta_calls
        + repair_invocations * check_calls_per_repair
    )
    projections = []
    for trigger_count in (0, repair_invocations):
        flops_used = fixed_flops + trigger_count * raw_path_delta_flops
        calls = fixed_calls + trigger_count * raw_path_delta_calls
        assert flops_used < PHASE2_BUDGET
        projections.append({
            "exact_floor_trigger_count": trigger_count,
            "projected_flops": flops_used,
            "projected_calls": calls,
            "budget_fraction": flops_used / PHASE2_BUDGET,
            "headroom_flops": PHASE2_BUDGET - flops_used,
        })
    return {
        "prior_literal_receipts": {
            "b38": {"flops": parent_flops, "calls": parent_calls},
            "manual_rho": {"flops": manual_flops, "calls": manual_calls},
            "global_raw_diag": {"flops": raw_flops, "calls": raw_calls},
        },
        "exact_formula": (
            f"flops={fixed_flops}+t*({raw_path_delta_flops}); "
            f"calls={fixed_calls}+t*({raw_path_delta_calls}); 0<=t<=14"
        ),
        "manual_rho_delta": {"flops": q_delta_flops, "calls": q_delta_calls},
        "branch_check_per_repair": {
            "flops": check_flops_per_repair,
            "calls": check_calls_per_repair,
        },
        "raw_path_delta_per_trigger": {
            "flops": raw_path_delta_flops,
            "calls": raw_path_delta_calls,
        },
        "endpoint_projections": projections,
        "literal_fullshape_run_performed": False,
    }


def main() -> None:
    if not sys.dont_write_bytecode:
        raise RuntimeError("run with python -B")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops.configure(symmetry_warnings=False)
    static = static_gate()
    parent = load("hybrid_rho_parent", PARENT)
    manual = load("hybrid_rho_manual", MANUAL)
    hybrid = load("hybrid_rho_aa87", HYBRID)
    raw_diag = load("hybrid_rho_raw_diag", RAW_DIAG)
    candidate = load("hybrid_rho_candidate", CANDIDATE)
    del manual
    context = context_gate(hybrid, candidate)
    repair = repair_gate(parent, raw_diag, hybrid, candidate)
    prediction = prediction_gate(hybrid, candidate)
    price = price_gate(repair["rows"])
    print(json.dumps({
        "schema": "fb-ec4-q24-manual-rho-hybrid-floor-verification-v1",
        "status": "PASS_STATIC_TINY_BYTE_PARITY_AND_EXACT_PRICE_FORMULA",
        "candidate": {"path": str(CANDIDATE), "sha256": sha256(CANDIDATE)},
        "static": static,
        "q24_context_signed_zero_poison": context,
        "covariance_repair_branch_parity": repair,
        "deterministic_depth16_parity_vs_aa87": prediction,
        "production_price_projection": price,
        "scope": {
            "width1024_runs": 0,
            "meter_runs": 0,
            "packages": 0,
            "submissions": 0,
            "graph_edits": 0,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
