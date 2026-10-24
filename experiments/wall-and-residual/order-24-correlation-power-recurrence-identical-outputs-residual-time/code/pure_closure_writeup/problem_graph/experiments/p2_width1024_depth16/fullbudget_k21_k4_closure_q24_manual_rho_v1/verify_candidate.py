#!/usr/bin/env python3
"""Target-free author verification for the additive Q24 manual-rho HOLD."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import warnings

import flopscope as flops
import numpy as np


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "fullbudget_k21_k4_closure_q24_v1" / "estimator.py"
CANDIDATE = HERE / "estimator.py"
PARENT_METER = (
    HERE.parent
    / "meter_protocol"
    / "runs"
    / "fb_ec4_q24_full_v1_b38cd865_n1_s8_seed20260824"
    / "receipt.json"
)
PARENT_SHA256 = "b38cd865297c344626da3c4f414e1d4923a9efdc62147e3103f155b2034f7b52"
CANDIDATE_SHA256 = "7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00"
PARENT_METER_SHA256 = "d69fd338972386af2c45f120d2983d5036a9a52ae57845a01977658e4c21fd11"
PARENT_LITERAL_FLOPS = 2_131_249_329_109
PARENT_LITERAL_CALLS = 6_878
PHASE2_BUDGET = 1 << 41
DEPTH = 16
Q_TERMS = 25
PRICE_WIDTHS = (4, 5, 6, 8, 12, 16)
HE_PANELS = (
    (32, 20260825),
    (32, 20260826),
    (64, 20260825),
    (64, 20260826),
    (128, 20260825),
    (128, 20260826),
)

OLD_BLOCK = """    fnp.cumprod(
        fnp.broadcast_to(
            correlation,
            (_Q24_TERMS - 1, width, width),
        ),
        axis=0,
        out=powers[1:],
    )"""

NEW_BLOCK = """    fnp.copyto(powers[1], correlation)
    for power in range(2, _Q24_TERMS):
        fnp.multiply(
            powers[power - 1],
            correlation,
            out=powers[power],
        )"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operation_calls(budget) -> int:
    return int(
        sum(item["calls"] for item in budget.summary_dict()["operations"].values())
    )


def _operation_field(budget, operation: str, field: str) -> int:
    if field == "flops":
        field = "flop_cost"
    return int(
        budget.summary_dict()["operations"].get(operation, {}).get(field, 0)
    )


def _imports(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return sorted(found)


def _function_map(tree: ast.Module) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _static_gate() -> dict[str, object]:
    parent_source = PARENT.read_text(encoding="utf-8")
    candidate_source = CANDIDATE.read_text(encoding="utf-8")
    parent_tree = ast.parse(parent_source)
    candidate_tree = ast.parse(candidate_source)
    expected_imports = sorted(
        [
            "flopscope",
            "flopscope.numpy",
            "whestbench.BaseEstimator",
            "whestbench.domain.MLP",
        ]
    )
    assert _sha256(PARENT) == PARENT_SHA256
    assert _sha256(CANDIDATE) == CANDIDATE_SHA256
    assert parent_source.count(OLD_BLOCK) == 1
    assert candidate_source == parent_source.replace(OLD_BLOCK, NEW_BLOCK)
    assert _imports(candidate_tree) == expected_imports

    parent_functions = _function_map(parent_tree)
    candidate_functions = _function_map(candidate_tree)
    assert parent_functions.keys() == candidate_functions.keys()
    changed_functions = sorted(
        name
        for name in parent_functions
        if parent_functions[name] != candidate_functions[name]
    )
    assert changed_functions == ["_q24_context"]

    q24 = next(
        node
        for node in candidate_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_q24_context"
    )
    call_names = []
    for node in ast.walk(q24):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                call_names.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                call_names.append(node.func.id)
    assert "cumprod" not in call_names
    assert call_names.count("copyto") == 2
    assert call_names.count("multiply") == 1
    loops = [node for node in ast.walk(q24) if isinstance(node, ast.For)]
    assert len(loops) == 1
    assert ast.unparse(loops[0].iter) == "range(2, _Q24_TERMS)"

    forbidden_imports = (
        "numpy",
        "scipy",
        "ctypes",
        "cffi",
        "asyncio",
        "threading",
        "multiprocessing",
        "subprocess",
    )
    assert all(
        not any(item == token or item.startswith(token + ".") for token in forbidden_imports)
        for item in _imports(candidate_tree)
    )
    forbidden_call_names = {"open", "eval", "exec", "compile", "__import__"}
    observed_forbidden_calls = sorted(
        node.func.id
        for node in ast.walk(candidate_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in forbidden_call_names
    )
    assert not observed_forbidden_calls
    assert "_Q24_TERMS = 25" in candidate_source
    assert "_PRODUCTION_WIDTH = 1024" in candidate_source
    assert "_PRODUCTION_DEPTH = 16" in candidate_source
    assert "_PHASE2_FLOP_BUDGET = 2_199_023_255_552" in candidate_source
    assert "or int(budget) < _PHASE2_FLOP_BUDGET" in candidate_source
    assert not any(HERE.rglob("__pycache__"))
    assert not any(HERE.rglob("*.pyc"))
    return {
        "parent_hash_exact": True,
        "candidate_hash_exact": True,
        "candidate_equals_one_exact_text_replacement": True,
        "changed_executable_functions": changed_functions,
        "imports_exact": expected_imports,
        "q24_context_call_names": sorted(call_names),
        "manual_loop": "range(2, _Q24_TERMS)",
        "forbidden_imports_absent": True,
        "forbidden_dynamic_or_file_calls": observed_forbidden_calls,
        "production_guard_unchanged": True,
        "pycache_absent_before_execution": True,
    }


def _context_case(module, covariance: np.ndarray, coefficients: np.ndarray):
    width = covariance.shape[0]
    sigma = np.ones(width, dtype=np.float64)
    with flops.BudgetContext(flop_budget=10_000_000_000, quiet=True) as budget:
        fusion = module._fusion_context(width)
        arena = np.asarray(fusion[14])
        arena.view(np.uint64)[...] = np.uint64(0x7FF0000000001234)
        poison_count_before = int(np.isnan(arena).sum())
        context = module._q24_context(
            module.fnp.asarray(coefficients, dtype=module.fnp.float64),
            module.fnp.asarray(sigma, dtype=module.fnp.float64),
            module.fnp.asarray(covariance, dtype=module.fnp.float64),
            fusion,
        )
        copied = tuple(np.asarray(value).copy() for value in context)
        shares = bool(np.shares_memory(np.asarray(context[3]), arena))
    return {
        "values": copied,
        "flops": int(budget.flops_used),
        "calls": _operation_calls(budget),
        "operations": {
            name: {
                "calls": _operation_field(budget, name, "calls"),
                "flops": _operation_field(budget, name, "flops"),
            }
            for name in ("copyto", "multiply", "broadcast_to", "cumprod")
        },
        "poison_count_before": poison_count_before,
        "shares_rho_arena": shares,
    }


def _tiny_context_gate(parent, candidate) -> dict[str, object]:
    width = 4
    coefficients = (
        np.arange(3 * 39 * width, dtype=np.float64).reshape(3, 39, width)
        / 257.0
        - 0.5
    )
    one_below = np.nextafter(np.float64(1.0), np.float64(0.0))
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    cases = (
        (
            "signed_zero_near_one",
            np.asarray(
                [
                    [1.0, -0.0, 0.5, -0.75],
                    [0.0, 1.0, one_below, -one_below],
                    [0.5, one_below, 1.0, subnormal],
                    [-0.75, -one_below, -subnormal, 1.0],
                ],
                dtype=np.float64,
            ),
        ),
        (
            "clip_and_repeated_powers",
            np.asarray(
                [
                    [1.25, -1.25, 0.25, -0.25],
                    [-1.25, 1.25, -0.5, 0.5],
                    [0.25, -0.5, 1.25, -1.25],
                    [-0.25, 0.5, -1.25, 1.25],
                ],
                dtype=np.float64,
            ),
        ),
    )
    rows = []
    for name, covariance in cases:
        parent_result = _context_case(parent, covariance, coefficients)
        candidate_result = _context_case(candidate, covariance, coefficients)
        parent_values = parent_result.pop("values")
        candidate_values = candidate_result.pop("values")
        assert len(parent_values) == len(candidate_values) == 5
        for old, new in zip(parent_values, candidate_values, strict=True):
            assert _array_sha256(old) == _array_sha256(new), name
        powers = candidate_values[3]
        clipped = np.clip(covariance, -1.0, 1.0)
        expected = np.empty_like(powers)
        expected[0] = 1.0
        expected[1] = clipped
        for power in range(2, Q_TERMS):
            np.multiply(expected[power - 1], clipped, out=expected[power])
        assert _array_sha256(powers) == _array_sha256(expected)
        assert np.isfinite(powers).all()
        assert parent_result["poison_count_before"] == Q_TERMS * width * width
        assert candidate_result["poison_count_before"] == Q_TERMS * width * width
        assert parent_result["shares_rho_arena"] is True
        assert candidate_result["shares_rho_arena"] is True
        assert candidate_result["flops"] - parent_result["flops"] == 2 * width * width
        assert candidate_result["calls"] - parent_result["calls"] == 22
        for operation, call_delta, flop_delta in (
            ("copyto", 1, 2 * width * width),
            ("multiply", 23, 46 * width * width),
            ("broadcast_to", -1, 0),
            ("cumprod", -1, -46 * width * width),
        ):
            assert (
                candidate_result["operations"][operation]["calls"]
                - parent_result["operations"][operation]["calls"]
                == call_delta
            )
            assert (
                candidate_result["operations"][operation]["flops"]
                - parent_result["operations"][operation]["flops"]
                == flop_delta
            )
        rows.append(
            {
                "case": name,
                "all_context_fields_bitwise_equal": True,
                "rho_arena_poison_overwritten": True,
                "rho_arena_alias": True,
                "powers_sha256": _array_sha256(powers),
                "parent_context_flops": parent_result["flops"],
                "candidate_context_flops": candidate_result["flops"],
                "flop_delta": candidate_result["flops"] - parent_result["flops"],
                "call_delta": candidate_result["calls"] - parent_result["calls"],
            }
        )
    return {"width": width, "case_count": len(rows), "rows": rows}


def _synthetic_mlp(width: int, weights):
    mlp = type("SyntheticMLP", (), {})()
    mlp.width = width
    mlp.depth = DEPTH
    mlp.weights = weights
    return mlp


def _run_predict(module, mlp, allow_failure: bool = False) -> dict[str, object]:
    budget = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with flops.BudgetContext(flop_budget=PHASE2_BUDGET, quiet=True) as budget:
                output = np.asarray(module._predict(mlp)).copy()
    except Exception as error:
        if not allow_failure:
            raise
        assert budget is not None
        return {
            "completed": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "flops_before_failure": int(budget.flops_used),
            "calls_before_failure": _operation_calls(budget),
        }
    assert budget is not None
    return {
        "completed": True,
        "finite": bool(np.isfinite(output).all()),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "output_sha256": _array_sha256(output),
        "output": output,
        "flops": int(budget.flops_used),
        "calls": _operation_calls(budget),
        "operation_totals": {
            name: {
                "calls": _operation_field(budget, name, "calls"),
                "flops": _operation_field(budget, name, "flops"),
            }
            for name in ("copyto", "multiply", "broadcast_to", "cumprod")
        },
    }


def _canonical_fixtures():
    width = 8
    identity = [np.eye(width, dtype=np.float32) for _ in range(DEPTH)]
    rng = np.random.default_rng(20260823)
    near_identity = [
        (np.eye(width) + rng.normal(scale=0.02, size=(width, width))).astype(np.float32)
        for _ in range(DEPTH)
    ]
    rng = np.random.default_rng(20260823)
    qr = []
    for _ in range(DEPTH):
        q, _ = np.linalg.qr(rng.normal(size=(width, width)))
        qr.append((0.35 * q).astype(np.float64))
    rng = np.random.default_rng(17)
    contractive = [
        rng.normal(scale=0.12, size=(width, width)).astype(np.float32)
        for _ in range(DEPTH)
    ]
    return (
        ("identity_width8_float32", identity),
        ("near_identity_seed20260823_scale0p02_float32", near_identity),
        ("qr_seed20260823_scale0p35_float64", qr),
        ("pure_normal_seed17_scale0p12_float32", contractive),
    )


def _canonical_gate(parent, candidate) -> dict[str, object]:
    rows = []
    for name, weights in _canonical_fixtures():
        mlp = _synthetic_mlp(8, weights)
        old = _run_predict(parent, mlp)
        new = _run_predict(candidate, mlp)
        old_output = old.pop("output")
        new_output = new.pop("output")
        assert old["completed"] is True and new["completed"] is True
        assert old["finite"] is True and new["finite"] is True
        assert _array_sha256(old_output) == _array_sha256(new_output), name
        assert old["output_sha256"] == new["output_sha256"]
        assert new["flops"] - old["flops"] == 30 * 8 * 8
        assert new["calls"] - old["calls"] == 330
        rows.append(
            {
                "fixture": name,
                "output_bitwise_equal": True,
                "output_finite": True,
                "output_sha256": new["output_sha256"],
                "parent_flops": old["flops"],
                "candidate_flops": new["flops"],
                "flop_delta": new["flops"] - old["flops"],
                "call_delta": new["calls"] - old["calls"],
            }
        )
    return {"fixture_count": len(rows), "width": 8, "depth": DEPTH, "rows": rows}


def _he_fixture(width: int, seed: int):
    rng = np.random.default_rng(seed)
    scale = math.sqrt(2.0 / width)
    weights = [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(DEPTH)
    ]
    digest = hashlib.sha256()
    for layer, weight in enumerate(weights):
        digest.update(layer.to_bytes(4, "big"))
        digest.update(memoryview(np.ascontiguousarray(weight)).cast("B"))
    return _synthetic_mlp(width, weights), digest.hexdigest()


def _he_gate(parent, candidate) -> dict[str, object]:
    rows = []
    for width, seed in HE_PANELS:
        mlp, weight_sha256 = _he_fixture(width, seed)
        old = _run_predict(parent, mlp, allow_failure=True)
        new = _run_predict(candidate, mlp, allow_failure=True)
        assert old["completed"] == new["completed"], (width, seed)
        if old["completed"]:
            old_output = old.pop("output")
            new_output = new.pop("output")
            assert old["finite"] is True and new["finite"] is True
            assert _array_sha256(old_output) == _array_sha256(new_output)
            outcome = "FINITE_BITWISE_EQUAL"
            output_sha256 = new["output_sha256"]
        else:
            assert old["exception_type"] == new["exception_type"]
            assert old["exception_message"] == new["exception_message"]
            outcome = "MATCHED_EXCEPTION"
            output_sha256 = None
        rows.append(
            {
                "width": width,
                "seed": seed,
                "weight_sha256": weight_sha256,
                "outcome": outcome,
                "output_sha256": output_sha256,
                "exception_type": None if old["completed"] else new["exception_type"],
                "exception_message": None if old["completed"] else new["exception_message"],
            }
        )
    return {
        "panel_count": len(rows),
        "fixture_law": "float32(default_rng(seed).N(0,1)*sqrt(2/width)), depth16",
        "rows": rows,
        "scope": "off-contract target-free outcome parity only",
    }


def _price_gate(parent, candidate) -> dict[str, object]:
    rows = []
    for width in PRICE_WIDTHS:
        mlp = _synthetic_mlp(
            width,
            [np.eye(width, dtype=np.float32) for _ in range(DEPTH)],
        )
        old = _run_predict(parent, mlp)
        new = _run_predict(candidate, mlp)
        old_output = old.pop("output")
        new_output = new.pop("output")
        assert _array_sha256(old_output) == _array_sha256(new_output)
        expected_flop_delta = 30 * width * width
        assert new["flops"] - old["flops"] == expected_flop_delta
        assert new["calls"] - old["calls"] == 330
        for operation, call_delta, flop_delta in (
            ("copyto", 15, 30 * width * width),
            ("multiply", 345, 690 * width * width),
            ("broadcast_to", -15, 0),
            ("cumprod", -15, -690 * width * width),
        ):
            assert (
                new["operation_totals"][operation]["calls"]
                - old["operation_totals"][operation]["calls"]
                == call_delta
            )
            assert (
                new["operation_totals"][operation]["flops"]
                - old["operation_totals"][operation]["flops"]
                == flop_delta
            )
        rows.append(
            {
                "width": width,
                "parent_flops": old["flops"],
                "candidate_flops": new["flops"],
                "flop_delta": new["flops"] - old["flops"],
                "expected_flop_delta_30w2": expected_flop_delta,
                "parent_calls": old["calls"],
                "candidate_calls": new["calls"],
                "call_delta": new["calls"] - old["calls"],
                "output_bitwise_equal": True,
            }
        )

    assert _sha256(PARENT_METER) == PARENT_METER_SHA256
    meter = json.loads(PARENT_METER.read_text(encoding="utf-8"))
    assert int(meter["measured"]["flops_used_per_mlp"][0]) == PARENT_LITERAL_FLOPS
    parent_meter_calls = sum(
        int(item["calls"])
        for item in meter["measured"]["operation_totals"].values()
    )
    assert parent_meter_calls == PARENT_LITERAL_CALLS
    production_width = 1024
    production_delta = 30 * production_width * production_width
    projected = PARENT_LITERAL_FLOPS + production_delta
    projected_calls = PARENT_LITERAL_CALLS + 330
    assert production_delta == 31_457_280
    assert projected == 2_131_280_786_389
    assert projected_calls == 7_208
    assert projected < PHASE2_BUDGET
    return {
        "rows": rows,
        "exact_delta_law": "+30*w^2 FLOPs and +330 calls at depth16",
        "operation_delta_law": {
            "copyto": {"calls": 15, "flops": "+30*w^2"},
            "multiply": {"calls": 345, "flops": "+690*w^2"},
            "broadcast_to": {"calls": -15, "flops": 0},
            "cumprod": {"calls": -15, "flops": "-690*w^2"},
        },
        "production_projection": {
            "width": production_width,
            "depth": DEPTH,
            "parent_literal_meter_flops": PARENT_LITERAL_FLOPS,
            "candidate_delta_flops": production_delta,
            "candidate_projected_flops": projected,
            "phase2_budget_flops": PHASE2_BUDGET,
            "headroom_flops": PHASE2_BUDGET - projected,
            "budget_fraction": projected / PHASE2_BUDGET,
            "budget_percent": 100.0 * projected / PHASE2_BUDGET,
            "parent_literal_calls": PARENT_LITERAL_CALLS,
            "candidate_projected_calls": projected_calls,
            "literal_fullshape_meter_run": False,
            "wall_residual_memory_authority": "UNMEASURED_HOLD",
        },
    }


def verify() -> dict[str, object]:
    if not __import__("sys").dont_write_bytecode:
        raise RuntimeError("run with Python -B / PYTHONDONTWRITEBYTECODE=1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops.configure(symmetry_warnings=False)
    static = _static_gate()
    parent = _load("q24_manual_rho_parent", PARENT)
    candidate = _load("q24_manual_rho_candidate", CANDIDATE)
    tiny = _tiny_context_gate(parent, candidate)
    canonical = _canonical_gate(parent, candidate)
    he = _he_gate(parent, candidate)
    price = _price_gate(parent, candidate)
    return {
        "schema": "fb-ec4-q24-manual-rho-author-verification-v1",
        "status": "PASS_TARGET_FREE_BITWISE_AND_EXACT_PRICE__IMMUTABLE_HOLD",
        "candidate": {
            "path": "estimator.py",
            "bytes": CANDIDATE.stat().st_size,
            "sha256": _sha256(CANDIDATE),
        },
        "parent": {
            "path": "../fullbudget_k21_k4_closure_q24_v1/estimator.py",
            "bytes": PARENT.stat().st_size,
            "sha256": _sha256(PARENT),
        },
        "static": static,
        "tiny_context_and_poison": tiny,
        "canonical_depth16": canonical,
        "official_he_small_panels": he,
        "exact_flopscope_price": price,
        "scope": {
            "dataset_members_opened": 0,
            "public_targets_opened": 0,
            "synthetic_targets_opened": 0,
            "truth_computed": False,
            "raw_mse_or_score_computed": False,
            "fits": 0,
            "samples": 0,
            "fullshape_width1024_runs": 0,
            "packages": 0,
            "submissions": 0,
            "graph_edits": 0,
            "strongest_authority": "METER_ONLY_PROJECTION_AND_TARGET_FREE_BITWISE_SOURCE_EQUIVALENCE",
            "wall_improvement_claimed": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
