#!/usr/bin/env python3
"""Target-free synthetic D1-vs-D1U Python-overhead discriminator."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
D1_SOURCE = HERE.parent / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1" / "estimator.py"
D1U_SOURCE = HERE / "estimator.py"
RESULT = HERE / "SYNTHETIC_MICRO_RESULT.json"
D1_SHA256 = "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e"
D1U_SHA256 = "42184107e4621b905066d638933b243c312e203cfb51632d04cab04c2780dc55"
KERNEL_CALLS = 58
KERNEL_REPETITIONS = 17
WEIGHT_LAYERS = 15
WEIGHT_REPETITIONS = 17
THREADS = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_node(path: Path, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError((path, name, len(matches)))
    return matches[0]


def compile_function(path: Path, name: str):
    node = function_node(path, name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"fnp": fnp}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def static_fnp_calls(path: Path, name: str) -> list[str]:
    calls: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "fnp"
            ):
                calls.append(node.func.attr)
            self.generic_visit(node)

    Visitor().visit(function_node(path, name))
    return calls


def arena_views(m: int, k: int, n: int):
    m2, k2, n2 = m // 2, k // 2, n // 2
    return (
        fnp.empty((7, m2, k2), dtype=fnp.float32),
        fnp.empty((7, k2, n2), dtype=fnp.float32),
        fnp.empty((4, m2, k2), dtype=fnp.float32),
        fnp.empty((4, k2, n2), dtype=fnp.float32),
        fnp.empty((7, m2, n2), dtype=fnp.float32),
        fnp.empty((3, m2, n2), dtype=fnp.float32),
        fnp.empty((m, n), dtype=fnp.float32),
        fnp.empty((m, n), dtype=fnp.float32),
    )


def inventory(meter) -> dict[str, object]:
    calls = Counter(record.op_name for record in meter.op_log)
    costs = Counter()
    ordered = []
    for record in meter.op_log:
        costs[record.op_name] += int(record.flop_cost)
        ordered.append((record.op_name, int(record.flop_cost)))
    return {
        "flops": int(meter.flops_used),
        "calls": dict(sorted(calls.items())),
        "costs": dict(sorted(costs.items())),
        "ordered": ordered,
    }


def measure(call, calls: int, budget: int) -> tuple[float, float, dict[str, object]]:
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=budget, quiet=True) as meter:
        for _ in range(calls):
            call()
    outer = time.perf_counter() - started
    residual = float(meter.summary_dict()["residual_wall_time_s"])
    return outer, residual, inventory(meter)


def main() -> None:
    if sha256(D1_SOURCE) != D1_SHA256 or sha256(D1U_SOURCE) != D1U_SHA256:
        raise AssertionError("source binding changed")
    old_name = "_cp3_winograd_arena_depth1"
    new_name = "_cp3_winograd_arena_depth1_unchecked"
    old_calls = static_fnp_calls(D1_SOURCE, old_name)
    new_calls = static_fnp_calls(D1U_SOURCE, new_name)
    expected_calls = (
        ["add", "subtract", "subtract", "subtract"]
        + ["copyto"] * 7
        + ["subtract"] * 4
        + ["copyto"] * 7
        + ["matmul"]
        + ["add", "add", "add", "add", "add", "subtract", "add"]
    )
    if old_calls != expected_calls or new_calls != expected_calls:
        raise AssertionError((old_calls, new_calls))

    old_kernel = compile_function(D1_SOURCE, old_name)
    new_kernel = compile_function(D1U_SOURCE, new_name)
    rng = np.random.default_rng(608_261)
    left = fnp.asarray(rng.standard_normal((64, 64)).astype(np.float32))
    right = fnp.asarray(rng.standard_normal((64, 64)).astype(np.float32))
    old_views = arena_views(64, 64, 64)
    new_views = arena_views(64, 64, 64)
    old_out, new_out = old_views[6], new_views[6]
    old_call = lambda: old_kernel(left, right, old_out, old_views)
    new_call = lambda: new_kernel(left, right, new_out, new_views)

    weight = fnp.asarray(rng.standard_normal((1024, 1024)).astype(np.float32))
    reused_weight2 = fnp.empty((1024, 1024), dtype=fnp.float32)
    reused_weight3 = fnp.empty((1024, 1024), dtype=fnp.float32)

    def allocated_weight_powers():
        weight2 = weight * weight
        weight3 = weight2 * weight
        return weight2, weight3

    def reused_weight_powers():
        fnp.multiply(weight, weight, out=reused_weight2)
        fnp.multiply(reused_weight2, weight, out=reused_weight3)
        return reused_weight2, reused_weight3

    with threadpool_limits(limits=THREADS):
        with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as old_one:
            returned_old = old_call()
        with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as new_one:
            returned_new = new_call()
        old_one_inventory = inventory(old_one)
        new_one_inventory = inventory(new_one)
        if old_one_inventory != new_one_inventory:
            raise AssertionError("kernel operation stream changed")
        if returned_old is not old_out or returned_new is not new_out:
            raise AssertionError("out identity changed")
        if not np.array_equal(np.asarray(old_out), np.asarray(new_out)):
            raise AssertionError("kernel byte parity changed")

        old_weights = allocated_weight_powers()
        new_weights = reused_weight_powers()
        if not all(
            np.array_equal(np.asarray(old), np.asarray(new))
            for old, new in zip(old_weights, new_weights)
        ):
            raise AssertionError("weight-power byte parity changed")

        old_kernel_outer = []
        new_kernel_outer = []
        old_kernel_residual = []
        new_kernel_residual = []
        old_weight_outer = []
        new_weight_outer = []
        old_weight_residual = []
        new_weight_residual = []
        kernel_inventories = []
        weight_inventories = []
        for repetition in range(max(KERNEL_REPETITIONS, WEIGHT_REPETITIONS)):
            kernel_arms = (
                ((old_call, old_kernel_outer, old_kernel_residual),
                 (new_call, new_kernel_outer, new_kernel_residual))
                if repetition % 2 == 0
                else ((new_call, new_kernel_outer, new_kernel_residual),
                      (old_call, old_kernel_outer, old_kernel_residual))
            )
            for call, outers, residuals in kernel_arms:
                outer, residual, observed = measure(call, KERNEL_CALLS, 2_000_000_000)
                outers.append(outer)
                residuals.append(residual)
                kernel_inventories.append(observed)

            weight_arms = (
                ((allocated_weight_powers, old_weight_outer, old_weight_residual),
                 (reused_weight_powers, new_weight_outer, new_weight_residual))
                if repetition % 2 == 0
                else ((reused_weight_powers, new_weight_outer, new_weight_residual),
                      (allocated_weight_powers, old_weight_outer, old_weight_residual))
            )
            for call, outers, residuals in weight_arms:
                outer, residual, observed = measure(call, WEIGHT_LAYERS, 100_000_000)
                outers.append(outer)
                residuals.append(residual)
                weight_inventories.append(observed)

    expected_kernel_inventory = {
        "add": 7 * KERNEL_CALLS,
        "copyto": 14 * KERNEL_CALLS,
        "matmul": KERNEL_CALLS,
        "subtract": 8 * KERNEL_CALLS,
    }
    expected_weight_inventory = {"multiply": 2 * WEIGHT_LAYERS}
    if not all(item["calls"] == expected_kernel_inventory for item in kernel_inventories):
        raise AssertionError("kernel roster drift")
    if not all(item["calls"] == expected_weight_inventory for item in weight_inventories):
        raise AssertionError("weight roster drift")
    old_kernel_flops = {item["flops"] for item in kernel_inventories[0::2]}
    new_kernel_flops = {item["flops"] for item in kernel_inventories[1::2]}
    old_weight_flops = {item["flops"] for item in weight_inventories[0::2]}
    new_weight_flops = {item["flops"] for item in weight_inventories[1::2]}
    if old_kernel_flops != new_kernel_flops or old_weight_flops != new_weight_flops:
        raise AssertionError("bill changed")

    result = {
        "schema": "p2r0c1-d1u-target-free-synthetic-micro-v1",
        "status": "PASS",
        "sources": {"d1": D1_SHA256, "d1u": D1U_SHA256},
        "threads": THREADS,
        "kernel": {
            "shape": [64, 64, 64],
            "calls_per_arm": KERNEL_CALLS,
            "repetitions": KERNEL_REPETITIONS,
            "single_call_ordered_inventory_equal": True,
            "single_call_byte_parity": True,
            "static_fnp_call_sequence_equal": True,
            "median_old_outer_seconds": statistics.median(old_kernel_outer),
            "median_new_outer_seconds": statistics.median(new_kernel_outer),
            "median_outer_saving_seconds": statistics.median(old_kernel_outer)
            - statistics.median(new_kernel_outer),
            "median_old_residual_seconds": statistics.median(old_kernel_residual),
            "median_new_residual_seconds": statistics.median(new_kernel_residual),
            "median_residual_saving_seconds": statistics.median(old_kernel_residual)
            - statistics.median(new_kernel_residual),
            "operation_calls": expected_kernel_inventory,
            "flops_per_arm": next(iter(old_kernel_flops)),
        },
        "weight_powers": {
            "shape": [1024, 1024],
            "layers_per_arm": WEIGHT_LAYERS,
            "repetitions": WEIGHT_REPETITIONS,
            "byte_parity": True,
            "median_old_outer_seconds": statistics.median(old_weight_outer),
            "median_new_outer_seconds": statistics.median(new_weight_outer),
            "median_outer_saving_seconds": statistics.median(old_weight_outer)
            - statistics.median(new_weight_outer),
            "median_old_residual_seconds": statistics.median(old_weight_residual),
            "median_new_residual_seconds": statistics.median(new_weight_residual),
            "median_residual_saving_seconds": statistics.median(old_weight_residual)
            - statistics.median(new_weight_residual),
            "operation_calls": expected_weight_inventory,
            "flops_per_arm": next(iter(old_weight_flops)),
            "persistent_bytes_added": 2 * 1024 * 1024 * 4,
        },
        "access": {
            "estimator_imported": False,
            "estimator_called": False,
            "fixtures_opened": [],
            "targets_opened": [],
            "network": False,
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
