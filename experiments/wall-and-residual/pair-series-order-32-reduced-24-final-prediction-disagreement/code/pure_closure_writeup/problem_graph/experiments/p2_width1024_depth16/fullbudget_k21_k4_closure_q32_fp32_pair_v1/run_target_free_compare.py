#!/usr/bin/env python3
"""Run the preregistered target-free Q32 fp32 pair comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import time
import traceback
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench.generation import sample_mlp


HERE = Path(__file__).resolve().parent
PARENT_PATH = HERE.parent / "fullbudget_k21_k4_closure_v4" / "estimator.py"
Q24_PATH = HERE.parent / "fullbudget_k21_k4_closure_q24_v1" / "estimator.py"
CANDIDATE_PATH = HERE / "estimator.py"
PREREGISTER_PATH = HERE / "PREREGISTER.md"
AUTHORIZATION_PATH = HERE / "authorization_manifest.json"
DEPTH = 16
PARENT_SHA256 = "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c"
Q24_SHA256 = "b38cd865297c344626da3c4f414e1d4923a9efdc62147e3103f155b2034f7b52"
CANDIDATE_SHA256 = "66dcadb6d3229e83fcd57405f0618a51fd6d8f224d1881b2a5bc215de218646d"
PRODUCTION_PARENT_FLOPS = 2_179_742_954_469
PHASE2_FLOP_BUDGET = 2_199_023_255_552
PRODUCTION_PARENT_BACKEND_S = 34.35804038029164
PRODUCTION_PARENT_WALL_S = 35.34110458288342
PRODUCTION_PARENT_RSS_BYTES = 4_618_567_680
PRODUCTION_PARENT_EINSUM_BACKEND_S = 26.004235


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_mlp(width: int, weights: list[np.ndarray]):
    value = type("TargetFreeSyntheticMLP", (), {})()
    value.width = width
    value.depth = DEPTH
    value.weights = weights
    return value


def weight_digest(weights: list[np.ndarray]) -> dict[str, Any]:
    digest = hashlib.sha256()
    layers = []
    for layer_index, weight in enumerate(weights):
        array = np.asarray(weight)
        layer_digest = hashlib.sha256()
        layer_digest.update(array.dtype.str.encode("ascii"))
        layer_digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        layer_digest.update(array.tobytes(order="C"))
        layer_sha = layer_digest.hexdigest()
        layers.append(
            {
                "layer": layer_index + 1,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": layer_sha,
            }
        )
        digest.update(f"{layer_index + 1}:{layer_sha}\n".encode("ascii"))
    return {"aggregate_sha256": digest.hexdigest(), "layers": layers}


def official_he_weights(width: int, seed: int) -> tuple[list[np.ndarray], bool]:
    rng = np.random.default_rng(seed)
    scale = float(np.sqrt(2.0 / width))
    weights = [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(DEPTH)
    ]

    # Independently execute the pinned official helper and require exact bytes.
    with flops.BudgetContext(flop_budget=10**10, quiet=True):
        official = sample_mlp(
            width,
            DEPTH,
            rng=fnp.random.default_rng(seed),
            seed=seed,
        )
    exact = all(
        np.array_equal(np.asarray(authority), local)
        for authority, local in zip(official.weights, weights)
    )
    return weights, exact


def fixture_roster() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width, seed in ((8, 2026082408), (32, 2026082432), (64, 2026082464)):
        weights, official_exact = official_he_weights(width, seed)
        rows.append(
            {
                "name": f"official_he_w{width}_seed{seed}",
                "family": "official_iid_gaussian_he",
                "width": width,
                "seed": seed,
                "official_helper_byte_equal": official_exact,
                "weights": weights,
            }
        )

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
    pure_normal = [
        rng.normal(scale=0.12, size=(width, width)).astype(np.float32)
        for _ in range(DEPTH)
    ]
    rng = np.random.default_rng(20260823)
    identity_plus_qr = []
    for _ in range(DEPTH):
        q, _ = np.linalg.qr(rng.standard_normal((width, width)))
        identity_plus_qr.append((np.eye(width) + 0.35 * q).astype(np.float64))
    rng = np.random.default_rng(17)
    alternate_contractive = [
        (
            0.88 * np.eye(width)
            + 0.12 * rng.standard_normal((width, width)) / np.sqrt(width)
        ).astype(np.float32)
        for _ in range(DEPTH)
    ]
    stress = (
        ("identity_float32", identity),
        ("near_identity_seed20260823_scale0p02_float32", near_identity),
        ("qr_seed20260823_scale0p35_float64", qr),
        ("canonical_pure_normal_seed17_scale0p12_float32", pure_normal),
        ("alternate_identity_plus_qr_seed20260823_scale0p35_float64", identity_plus_qr),
        (
            "alternate_0p88_identity_plus_normal_seed17_scale0p12_over_sqrt8_float32",
            alternate_contractive,
        ),
    )
    rows.extend(
        {
            "name": name,
            "family": "canonical_alternate_stress",
            "width": width,
            "seed": None,
            "official_helper_byte_equal": None,
            "weights": weights,
        }
        for name, weights in stress
    )
    return rows


def install_reference_seam(module) -> None:
    fnp_module = module.fnp

    def independent_g_four_all(context):
        _features, left64, right64, rho64, factorial64 = context
        left32 = fnp_module.asarray(left64, dtype=fnp_module.float32)
        right32 = fnp_module.asarray(right64, dtype=fnp_module.float32)
        rho32 = fnp_module.asarray(rho64, dtype=fnp_module.float32)
        factorial32 = fnp_module.asarray(factorial64, dtype=fnp_module.float32)
        result32 = fnp_module.einsum(
            "mdti,mdtj,tij,t->mdij",
            left32,
            right32,
            rho32,
            factorial32,
            optimize=[(0, 3), (0, 1, 2)],
        )
        return fnp_module.asarray(result32, dtype=fnp_module.float64)

    module._g_four_all = independent_g_four_all


def install_action_trace(module) -> None:
    module._mixed_pair_trace = None
    original_connected = module._connected_planes

    def connected_wrapper(*args, **kwargs):
        result = original_connected(*args, **kwargs)
        if module._mixed_pair_trace is not None:
            module._mixed_pair_trace.append(
                tuple(np.asarray(value).copy() for value in result[:4])
            )
        return result

    module._connected_planes = connected_wrapper


def operation_record(summary: dict[str, Any], name: str) -> dict[str, Any]:
    record = summary["operations"].get(name, {})
    return {
        "flop_cost": int(record.get("flop_cost", 0)),
        "calls": int(record.get("calls", 0)),
        "flopscope_backend_time_s": float(record.get("flopscope_backend_time_s", 0.0)),
        "flopscope_overhead_time_s": float(record.get("flopscope_overhead_time_s", 0.0)),
    }


def run(module, mlp) -> dict[str, Any]:
    module._mixed_pair_trace = []
    start = time.perf_counter()
    output = None
    error = None
    budget = flops.BudgetContext(flop_budget=10**15, quiet=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with budget:
                output = np.asarray(module._predict(mlp)).copy()
    except Exception as exception:  # registered numerical failures are evidence
        error = {
            "type": type(exception).__name__,
            "message": str(exception),
            "traceback": traceback.format_exc(),
        }
    harness_wall = time.perf_counter() - start
    trace = module._mixed_pair_trace
    module._mixed_pair_trace = None
    summary = budget.summary_dict()
    return {
        "output": output,
        "trace": trace,
        "success": output is not None and error is None,
        "error": error,
        "flops": int(budget.flops_used),
        "calls": int(sum(item["calls"] for item in summary["operations"].values())),
        "flopscope_backend_time_s": float(summary["flopscope_backend_time_s"]),
        "flopscope_overhead_time_s": float(summary["flopscope_overhead_time_s"]),
        "residual_wall_time_s": float(summary["residual_wall_time_s"]),
        "budget_wall_time_s": float(summary["wall_time_s"]),
        "harness_wall_time_s": float(harness_wall),
        "operation_focus": {
            name: operation_record(summary, name)
            for name in ("asarray", "einsum", "cumprod", "matmul", "linalg.eigh")
        },
    }


def metric(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    delta = left64 - right64
    square = delta * delta
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(square))),
        "mse": float(np.mean(square)),
        "sse": float(np.sum(square)),
        "changed": bool(np.any(delta != 0.0)),
        "finite": bool(np.isfinite(left64).all() and np.isfinite(right64).all()),
    }


def output_envelope(parent: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    assert parent.shape == child.shape
    per_layer = [metric(child[index], parent[index]) for index in range(parent.shape[0])]
    changed = [index + 1 for index, value in enumerate(per_layer) if value["changed"]]
    all_metric = metric(child, parent)
    return {
        "available": True,
        "first_changed_layer": changed[0] if changed else None,
        "final_layer": per_layer[-1],
        "all_layers": all_metric,
        "maximum_layer_mse": max(item["mse"] for item in per_layer),
        "maximum_layer_mse_layer": 1 + max(
            range(len(per_layer)), key=lambda index: per_layer[index]["mse"]
        ),
        "maximum_layer_max_abs": max(item["max_abs"] for item in per_layer),
        "maximum_layer_max_abs_layer": 1 + max(
            range(len(per_layer)), key=lambda index: per_layer[index]["max_abs"]
        ),
        "per_layer": [
            {"layer": index + 1, **value} for index, value in enumerate(per_layer)
        ],
    }


def action_envelope(parent_trace, child_trace) -> dict[str, Any]:
    complete = len(parent_trace) == len(child_trace)
    common_count = min(len(parent_trace), len(child_trace))
    names = ("K21", "K4", "K31", "K22")
    result: dict[str, Any] = {}
    for action_index, name in enumerate(names):
        per_reset = [
            metric(child[action_index], parent[action_index])
            for parent, child in zip(parent_trace[:common_count], child_trace[:common_count])
        ]
        changed = [index + 1 for index, value in enumerate(per_reset) if value["changed"]]
        result[name] = {
            "available": bool(per_reset),
            "complete": complete,
            "parent_reset_count": len(parent_trace),
            "child_reset_count": len(child_trace),
            "reset_count": len(per_reset),
            "first_changed_reset": changed[0] if changed else None,
            "maximum_reset_mse": max((item["mse"] for item in per_reset), default=None),
            "maximum_reset_mse_reset": None if not per_reset else 1 + max(
                range(len(per_reset)), key=lambda index: per_reset[index]["mse"]
            ),
            "maximum_reset_max_abs": max((item["max_abs"] for item in per_reset), default=None),
            "maximum_reset_max_abs_reset": None if not per_reset else 1 + max(
                range(len(per_reset)), key=lambda index: per_reset[index]["max_abs"]
            ),
            "per_reset": [
                {"reset": index + 1, **value} for index, value in enumerate(per_reset)
            ],
        }
    return result


def unavailable_output_envelope(run_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": False,
        "error": run_record["error"],
        "first_changed_layer": None,
        "final_layer": None,
        "all_layers": None,
        "maximum_layer_mse": None,
        "maximum_layer_mse_layer": None,
        "maximum_layer_max_abs": None,
        "maximum_layer_max_abs_layer": None,
        "per_layer": [],
    }


def isolated_kernel_probe(module, width: int) -> dict[str, Any]:
    pair_left = np.ones((4, 17, 33, width), dtype=np.float64)
    pair_right = np.ones((4, 17, 33, width), dtype=np.float64)
    rho_powers = np.ones((33, width, width), dtype=np.float64)
    inverse_factorial = np.asarray(module._INV_FACTORIAL, dtype=np.float64)
    context = (None, pair_left, pair_right, rho_powers, inverse_factorial)
    start = time.perf_counter()
    with flops.BudgetContext(flop_budget=10**13, quiet=True) as budget:
        output = np.asarray(module._g_four_all(context))
    wall = time.perf_counter() - start
    summary = budget.summary_dict()
    return {
        "flops": int(budget.flops_used),
        "calls": int(sum(item["calls"] for item in summary["operations"].values())),
        "backend_s": float(summary["flopscope_backend_time_s"]),
        "overhead_s": float(summary["flopscope_overhead_time_s"]),
        "wall_s": float(wall),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "finite": bool(np.isfinite(output).all()),
        "operations": {
            name: operation_record(summary, name) for name in ("asarray", "einsum")
        },
    }


def traces_bitwise_equal(left, right) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(left_value, right_value)
        for left_event, right_event in zip(left, right)
        for left_value, right_value in zip(left_event, right_event)
    )


def traces_finite(trace) -> bool:
    return all(np.isfinite(value).all() for event in trace for value in event)


def quadratic_through(points: list[tuple[int, int]]) -> tuple[Fraction, Fraction, Fraction]:
    matrix = [
        [Fraction(width * width), Fraction(width), Fraction(1), Fraction(value)]
        for width, value in points
    ]
    for pivot in range(3):
        pivot_row = next(row for row in range(pivot, 3) if matrix[row][pivot] != 0)
        matrix[pivot], matrix[pivot_row] = matrix[pivot_row], matrix[pivot]
        scale = matrix[pivot][pivot]
        matrix[pivot] = [value / scale for value in matrix[pivot]]
        for row in range(3):
            if row == pivot:
                continue
            factor = matrix[row][pivot]
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[row], matrix[pivot])
            ]
    return matrix[0][3], matrix[1][3], matrix[2][3]


def fraction_json(value: Fraction) -> int | str:
    return int(value) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def api_smoke(candidate) -> dict[str, Any]:
    estimator = candidate.Estimator()
    offshape = make_mlp(7, [np.eye(7, dtype=np.float32) for _ in range(DEPTH)])
    with flops.BudgetContext(flop_budget=10**7, quiet=True):
        offshape_output = np.asarray(
            estimator.predict(offshape, budget=PHASE2_FLOP_BUDGET)
        )

    underbudget = type("UnderBudgetProductionShape", (), {})()
    underbudget.width = 1024
    underbudget.depth = 16
    underbudget.weights = None
    with flops.BudgetContext(flop_budget=10**7, quiet=True):
        underbudget_output = np.asarray(
            estimator.predict(underbudget, budget=PHASE2_FLOP_BUDGET - 1)
        )
    return {
        "offshape": {
            "shape": list(offshape_output.shape),
            "dtype": str(offshape_output.dtype),
            "finite": bool(np.isfinite(offshape_output).all()),
            "all_zero": bool(np.all(offshape_output == 0.0)),
        },
        "underbudget_production_geometry": {
            "shape": list(underbudget_output.shape),
            "dtype": str(underbudget_output.dtype),
            "finite": bool(np.isfinite(underbudget_output).all()),
            "all_zero": bool(np.all(underbudget_output == 0.0)),
            "production_path_entered": False,
        },
    }


def main(output_path: Path) -> None:
    assert sha256(PARENT_PATH) == PARENT_SHA256
    assert sha256(Q24_PATH) == Q24_SHA256
    assert sha256(CANDIDATE_PATH) == CANDIDATE_SHA256

    parent = load("mixed_pair_parent_v4", PARENT_PATH)
    q24 = load("mixed_pair_q24", Q24_PATH)
    candidate = load("mixed_pair_candidate", CANDIDATE_PATH)
    reference = load("mixed_pair_independent_reference", PARENT_PATH)
    install_reference_seam(reference)
    for module in (parent, q24, candidate, reference):
        install_action_trace(module)

    rows: list[dict[str, Any]] = []
    for fixture in fixture_roster():
        weights = fixture.pop("weights")
        mlp = make_mlp(fixture["width"], weights)
        parent_run = run(parent, mlp)
        q24_run = run(q24, mlp)
        candidate_run = run(candidate, mlp)
        reference_run = run(reference, mlp)

        reference_output_equal = bool(
            candidate_run["success"]
            and reference_run["success"]
            and np.array_equal(candidate_run["output"], reference_run["output"])
        )
        reference_trace_equal = bool(
            candidate_run["success"]
            and reference_run["success"]
            and traces_bitwise_equal(candidate_run["trace"], reference_run["trace"])
        )
        all_finite = all(
            item["success"]
            and np.isfinite(item["output"]).all()
            and traces_finite(item["trace"])
            for item in (parent_run, q24_run, candidate_run, reference_run)
        )

        output_metrics = {
            "candidate_minus_v4": (
                output_envelope(parent_run["output"], candidate_run["output"])
                if parent_run["success"] and candidate_run["success"]
                else unavailable_output_envelope(candidate_run)
            ),
            "q24_minus_v4": (
                output_envelope(parent_run["output"], q24_run["output"])
                if parent_run["success"] and q24_run["success"]
                else unavailable_output_envelope(q24_run)
            ),
        }
        action_metrics = {
            "candidate_minus_v4": action_envelope(
                parent_run["trace"], candidate_run["trace"]
            ),
            "q24_minus_v4": action_envelope(parent_run["trace"], q24_run["trace"]),
        }

        run_receipts = {}
        for name, item in (
            ("v4", parent_run),
            ("q24", q24_run),
            ("candidate", candidate_run),
            ("independent_reference", reference_run),
        ):
            run_receipts[name] = {
                key: value
                for key, value in item.items()
                if key not in ("output", "trace")
            }
            run_receipts[name]["output_shape"] = (
                list(item["output"].shape) if item["output"] is not None else None
            )
            run_receipts[name]["output_dtype"] = (
                str(item["output"].dtype) if item["output"] is not None else None
            )
            run_receipts[name]["output_finite"] = bool(
                item["output"] is not None and np.isfinite(item["output"]).all()
            )
            run_receipts[name]["trace_events"] = len(item["trace"])

        rows.append(
            {
                **fixture,
                "weight_digest": weight_digest(weights),
                "reference_candidate_output_bitwise_equal": reference_output_equal,
                "reference_candidate_actions_bitwise_equal": reference_trace_equal,
                "all_outputs_and_actions_finite": all_finite,
                "output_metrics": output_metrics,
                "action_metrics": action_metrics,
                "runs": run_receipts,
            }
        )

    official_rows = [row for row in rows if row["family"] == "official_iid_gaussian_he"]
    stress_rows = [row for row in rows if row["family"] == "canonical_alternate_stress"]

    exact_seam = all(
        row["reference_candidate_output_bitwise_equal"]
        and row["reference_candidate_actions_bitwise_equal"]
        for row in rows
    )
    finite = all(row["all_outputs_and_actions_finite"] for row in rows)
    official_materiality = all(
        row["output_metrics"]["candidate_minus_v4"]["available"]
        and row["output_metrics"]["candidate_minus_v4"]["final_layer"]["mse"] <= 1.0e-10
        and row["output_metrics"]["candidate_minus_v4"]["maximum_layer_mse"] <= 1.0e-8
        and row["output_metrics"]["candidate_minus_v4"]["maximum_layer_max_abs"] <= 1.0e-3
        for row in official_rows
    )

    changed_stress = [
        row
        for row in stress_rows
        if row["output_metrics"]["q24_minus_v4"]["available"]
        and row["output_metrics"]["q24_minus_v4"]["all_layers"]["changed"]
    ]
    stress_closer = all(
        row["output_metrics"]["candidate_minus_v4"]["available"]
        and row["output_metrics"]["candidate_minus_v4"]["all_layers"]["mse"]
        < row["output_metrics"]["q24_minus_v4"]["all_layers"]["mse"]
        and row["output_metrics"]["candidate_minus_v4"]["all_layers"]["max_abs"]
        < row["output_metrics"]["q24_minus_v4"]["all_layers"]["max_abs"]
        for row in changed_stress
    )

    official_q24_changed = any(
        row["output_metrics"]["q24_minus_v4"]["available"]
        and row["output_metrics"]["q24_minus_v4"]["all_layers"]["changed"]
        for row in official_rows
    )
    candidate_official_available = all(
        row["output_metrics"]["candidate_minus_v4"]["available"]
        for row in official_rows
    )
    candidate_official_sse = (
        sum(
            row["output_metrics"]["candidate_minus_v4"]["all_layers"]["sse"]
            for row in official_rows
        )
        if candidate_official_available
        else None
    )
    q24_official_available = all(
        row["output_metrics"]["q24_minus_v4"]["available"]
        for row in official_rows
    )
    q24_official_sse = (
        sum(
            row["output_metrics"]["q24_minus_v4"]["all_layers"]["sse"]
            for row in official_rows
        )
        if q24_official_available
        else None
    )
    candidate_official_worst_abs = (
        max(
            row["output_metrics"]["candidate_minus_v4"]["all_layers"]["max_abs"]
            for row in official_rows
        )
        if candidate_official_available
        else None
    )
    q24_official_worst_abs = (
        max(
            row["output_metrics"]["q24_minus_v4"]["all_layers"]["max_abs"]
            for row in official_rows
        )
        if q24_official_available
        else None
    )
    if official_q24_changed:
        official_closer = bool(
            candidate_official_available
            and q24_official_available
            and candidate_official_sse < q24_official_sse
            and candidate_official_worst_abs < q24_official_worst_abs
        )
    else:
        official_closer = bool(
            q24_official_available
            and all(
                row["output_metrics"]["candidate_minus_v4"]["available"]
                and not row["output_metrics"]["candidate_minus_v4"]["all_layers"]["changed"]
                for row in official_rows
            )
        )

    action_gate_rows = []
    action_closer = True
    for row in rows:
        for action in ("K21", "K4", "K31", "K22"):
            candidate_action = row["action_metrics"]["candidate_minus_v4"][action]
            q24_action = row["action_metrics"]["q24_minus_v4"][action]
            if not candidate_action["available"] or not candidate_action["complete"]:
                passed = False
                comparator = "candidate_action_incomplete"
            elif not q24_action["available"] or not q24_action["complete"]:
                passed = False
                comparator = "q24_action_authority_incomplete"
            elif q24_action["maximum_reset_mse"] > 0.0 or q24_action["maximum_reset_max_abs"] > 0.0:
                passed = (
                    candidate_action["maximum_reset_mse"] < q24_action["maximum_reset_mse"]
                    and candidate_action["maximum_reset_max_abs"] < q24_action["maximum_reset_max_abs"]
                )
                comparator = "strictly_lower_than_nonzero_q24"
            else:
                passed = (
                    candidate_action["maximum_reset_mse"] <= 1.0e-8
                    and candidate_action["maximum_reset_max_abs"] <= 1.0e-3
                )
                comparator = "absolute_floating_gate_when_q24_zero"
            action_closer &= passed
            action_gate_rows.append(
                {
                    "fixture": row["name"],
                    "action": action,
                    "comparator": comparator,
                    "candidate_maximum_reset_mse": candidate_action["maximum_reset_mse"],
                    "q24_maximum_reset_mse": q24_action["maximum_reset_mse"],
                    "candidate_maximum_reset_max_abs": candidate_action["maximum_reset_max_abs"],
                    "q24_maximum_reset_max_abs": q24_action["maximum_reset_max_abs"],
                    "passed": passed,
                }
            )

    kernel_probes = []
    width_delta_points = []
    for width in (8, 32, 64):
        parent_probe = isolated_kernel_probe(parent, width)
        candidate_probe = isolated_kernel_probe(candidate, width)
        per_call_delta = candidate_probe["flops"] - parent_probe["flops"]
        width_delta_points.append((width, 14 * per_call_delta))
        kernel_probes.append(
            {
                "width": width,
                "v4": parent_probe,
                "candidate": candidate_probe,
                "per_call_candidate_minus_v4_flops": per_call_delta,
                "fourteen_call_candidate_minus_v4_flops": 14 * per_call_delta,
            }
        )
    coefficient_a, coefficient_b, coefficient_c = quadratic_through(width_delta_points)
    projected_delta = coefficient_a * 1024 * 1024 + coefficient_b * 1024 + coefficient_c
    projected_flops = Fraction(PRODUCTION_PARENT_FLOPS) + projected_delta
    cast_shape_flops = 28 * (101 * 1024 * 1024 + 4488 * 1024 + 33)

    parent_kernel_backend_small = sum(item["v4"]["backend_s"] for item in kernel_probes)
    candidate_kernel_backend_small = sum(item["candidate"]["backend_s"] for item in kernel_probes)
    empirical_einsum_ratio = candidate_kernel_backend_small / max(
        parent_kernel_backend_small, np.finfo(np.float64).tiny
    )
    central_backend_projection = (
        PRODUCTION_PARENT_BACKEND_S - PRODUCTION_PARENT_EINSUM_BACKEND_S
        + PRODUCTION_PARENT_EINSUM_BACKEND_S * empirical_einsum_ratio
    )
    # Small shapes are not production timing authority.  Keep a deliberately
    # broad range around the empirical mapping.
    projected_backend_range = [
        max(0.0, central_backend_projection * 0.80),
        central_backend_projection * 1.25,
    ]
    projected_wall_range = [
        value + (PRODUCTION_PARENT_WALL_S - PRODUCTION_PARENT_BACKEND_S)
        for value in projected_backend_range
    ]

    extra_live_fp32_elements = 101 * 1024 * 1024 + 4488 * 1024 + 33
    extra_live_bytes = extra_live_fp32_elements * 4
    projected_peak_rss = PRODUCTION_PARENT_RSS_BYTES + extra_live_bytes
    resource_pass = projected_flops.denominator == 1 and int(projected_flops) <= PHASE2_FLOP_BUDGET
    memory_projection_pass = projected_peak_rss < 8 * 1024**3

    smoke = api_smoke(candidate)
    api_pass = (
        smoke["offshape"]["shape"] == [16, 7]
        and smoke["offshape"]["dtype"] == "float32"
        and smoke["offshape"]["finite"]
        and smoke["offshape"]["all_zero"]
        and smoke["underbudget_production_geometry"]["shape"] == [16, 1024]
        and smoke["underbudget_production_geometry"]["dtype"] == "float32"
        and smoke["underbudget_production_geometry"]["finite"]
        and smoke["underbudget_production_geometry"]["all_zero"]
    )
    official_helper_pass = all(row["official_helper_byte_equal"] for row in official_rows)

    gates = {
        "independent_implementation_bitwise_parity": exact_seam,
        "all_outputs_and_actions_finite": finite,
        "official_helper_weight_bytes_exact": official_helper_pass,
        "official_law_scientific_materiality": official_materiality,
        "stress_output_closer_than_q24": stress_closer,
        "official_output_closer_than_q24": official_closer,
        "all4_action_envelopes_closer_than_q24": action_closer,
        "projected_flops_at_most_2pow41": resource_pass,
        "projected_static_peak_memory_below_8gib": memory_projection_pass,
        "pinned_api_smoke": api_pass,
    }
    passed = all(gates.values())

    result = {
        "schema": "fb-ec4-q32-fp32-pair-target-free-comparison-v1",
        "status": (
            "PASS_SOURCE_ONLY_INDEPENDENT_AUDIT_HOLD__PRODUCTION_METER_UNRUN"
            if passed
            else "CLOSED_TARGET_FREE_GATE_FAILURE__NO_AUDIT_HOLD"
        ),
        "scope": {
            "target_free": True,
            "dataset_rows_opened": 0,
            "targets_or_teachers_opened": 0,
            "scores_computed": 0,
            "fits": 0,
            "samples": 0,
            "production_shape_runs": 0,
            "submissions": 0,
            "canonical_graph_edits": 0,
        },
        "artifacts": {
            "parent": {"path": str(PARENT_PATH.relative_to(HERE.parent)), "sha256": sha256(PARENT_PATH)},
            "q24": {"path": str(Q24_PATH.relative_to(HERE.parent)), "sha256": sha256(Q24_PATH)},
            "candidate": {"path": "estimator.py", "sha256": sha256(CANDIDATE_PATH), "bytes": CANDIDATE_PATH.stat().st_size},
            "preregister": {"path": "PREREGISTER.md", "sha256": sha256(PREREGISTER_PATH)},
            "authorization": {"path": "authorization_manifest.json", "sha256": sha256(AUTHORIZATION_PATH)},
            "runner": {"path": "run_target_free_compare.py", "sha256": sha256(Path(__file__))},
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "flopscope": importlib.metadata.version("flopscope"),
            "whestbench": importlib.metadata.version("whestbench"),
            "numpy": np.__version__,
        },
        "gates": gates,
        "fixture_count": len(rows),
        "official_fixture_count": len(official_rows),
        "stress_fixture_count": len(stress_rows),
        "rows": rows,
        "closer_than_q24_summary": {
            "changed_stress_fixtures": [row["name"] for row in changed_stress],
            "official_q24_changed": official_q24_changed,
            "candidate_official_total_sse": candidate_official_sse,
            "q24_official_total_sse": q24_official_sse,
            "candidate_official_worst_abs": candidate_official_worst_abs,
            "q24_official_worst_abs": q24_official_worst_abs,
            "action_gate_rows": action_gate_rows,
        },
        "resource_projection": {
            "classification": "TARGET_FREE_SMALL_WIDTH_PROJECTION_ONLY__NOT_LITERAL_METER",
            "width_delta_points": [
                {"width": width, "candidate_minus_v4_flops": value}
                for width, value in width_delta_points
            ],
            "isolated_g_four_all_probes": kernel_probes,
            "quadratic_delta": {
                "a_n2": fraction_json(coefficient_a),
                "b_n": fraction_json(coefficient_b),
                "c": fraction_json(coefficient_c),
            },
            "production_parent_flops": PRODUCTION_PARENT_FLOPS,
            "projected_candidate_minus_v4_flops": fraction_json(projected_delta),
            "projected_candidate_flops": fraction_json(projected_flops),
            "phase2_flop_budget": PHASE2_FLOP_BUDGET,
            "projected_headroom_flops": fraction_json(Fraction(PHASE2_FLOP_BUDGET) - projected_flops),
            "cast_shape_billed_flops_at_width1024": cast_shape_flops,
            "empirical_small_width_einsum_backend_ratio": empirical_einsum_ratio,
            "projected_backend_s_range": projected_backend_range,
            "projected_wall_s_range": projected_wall_range,
            "wall_projection_warning": (
                "Widths 8/32/64 do not establish production-kernel timing or the 120-second board gate."
            ),
            "parent_sampled_descendant_peak_rss_bytes": PRODUCTION_PARENT_RSS_BYTES,
            "additional_live_fp32_cast_bytes_upper_bound": extra_live_bytes,
            "projected_peak_rss_bytes": projected_peak_rss,
            "memory_projection_warning": (
                "Static additive bound, not sampled width1024 RSS; production execution remains forbidden."
            ),
        },
        "api_smoke": smoke,
        "verdict": {
            "source_only_audit_hold": passed,
            "production_resource_qualification": "UNRUN_AND_UNAUTHORIZED",
            "raw_mse_or_accuracy": "UNSCORED",
            "board_promotion": "WITHHELD",
        },
    }

    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "gates": gates,
        "candidate_sha256": CANDIDATE_SHA256,
        "results_path": str(output_path),
        "results_sha256": sha256(output_path),
        "projected_candidate_flops": fraction_json(projected_flops),
        "projected_wall_s_range": projected_wall_range,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    arguments = parser.parse_args()
    main(arguments.output)
