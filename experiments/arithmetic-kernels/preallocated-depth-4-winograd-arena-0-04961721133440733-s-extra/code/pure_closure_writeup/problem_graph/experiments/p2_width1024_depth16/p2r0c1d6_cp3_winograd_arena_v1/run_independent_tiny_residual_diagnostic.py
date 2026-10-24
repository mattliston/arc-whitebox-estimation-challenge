#!/usr/bin/env python3
"""Independent tiny residual diagnostic for the frozen D6 kernel source.

This script parses and compiles only the three D6 kernel helpers.  It never
imports or calls the estimator, opens no fixture/target/score/name/board, and
never executes an exact production shape.  The fixed tiny arena is derived
independently from the sealed formula.  Its purpose is to test whether the
dimension-insensitive Python/view residual of a 58-contraction roster can meet
the preregistered D6-minus-direct threshold of at most 10 ms.
"""

from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "estimator.py"
RESULT = HERE / "INDEPENDENT_TINY_RESIDUAL_DIAGNOSTIC.json"
SOURCE_SHA256 = "e9ebe9abebe4d632e20d81396aea2bc64a511277d88bc6fd13fb6a80721ffd58"
LEVELS = 4
RADIX = 7
MAX_SHAPE = (64, 64, 64)
GAPPED_SHAPE = (64, 32, 48)
ROSTER_CALLS = 58
REPETITIONS = 8
THREADS = 2
RESIDUAL_DELTA_GATE_S = 0.010


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def exact(value: Fraction) -> int:
    if value.denominator != 1:
        raise AssertionError(value)
    return value.numerator


def tiny_specs(m: int, k: int, n: int) -> tuple[int, tuple[tuple, ...]]:
    """Derive the production 15-region/26-view layout at a tiny maximum."""
    q = Fraction(7, 4)
    a, b, c = m * k, k * n, m * n
    regions: dict[str, tuple[int, int]] = {}
    cursor = 0

    def region(name: str, size: int) -> None:
        nonlocal cursor
        regions[name] = (cursor, cursor + size)
        cursor += size

    for level in range(1, LEVELS + 1):
        region(f"a{level}", exact(q**level * a))
        region(f"b{level}", exact(q**level * b))
    region("dependency", exact(q**3 * max(a, b)))
    region("leaf", exact(q**4 * c))
    region("product0", exact(q**3 * c))
    region("product1", exact(q**3 * c))
    region("u", exact(q**3 * c))
    region("output0", c)
    region("output1", c)

    expected = sum(q**level * (a + b) for level in range(1, 5))
    expected += q**3 * max(a, b) + q**4 * c + 3 * q**3 * c + 2 * c
    assert cursor == exact(expected)
    specs: list[tuple[int, int, tuple[int, ...]]] = []

    def view(owner: str, shape: tuple[int, ...]) -> None:
        start, owner_stop = regions[owner]
        stop = start + product(shape)
        assert stop <= owner_stop
        specs.append((start, stop, shape))

    for level in range(1, LEVELS + 1):
        prefix = (RADIX,) * level
        view(f"a{level}", prefix + (m // 2**level, k // 2**level))
        view(f"b{level}", prefix + (k // 2**level, n // 2**level))
    for level in range(1, LEVELS + 1):
        prefix = (RADIX,) * (level - 1)
        view("dependency", (4,) + prefix + (m // 2**level, k // 2**level))
    for level in range(1, LEVELS + 1):
        prefix = (RADIX,) * (level - 1)
        view("dependency", (4,) + prefix + (k // 2**level, n // 2**level))
    view("leaf", (RADIX,) * 4 + (m // 16, n // 16))
    view("product0", (RADIX,) * 3 + (m // 8, n // 8))
    view("product1", (RADIX,) * 2 + (m // 4, n // 4))
    view("product0", (RADIX,) + (m // 2, n // 2))
    for level in range(4, 0, -1):
        prefix = (RADIX,) * (level - 1)
        view("u", (3,) + prefix + (m // 2**level, n // 2**level))
    view("output0", (m, n))
    view("output1", (m, n))
    assert len(regions) == 15 and len(specs) == 26
    return cursor, tuple(specs)


def load_kernel_functions() -> dict[str, object]:
    """Compile only the frozen helper definitions; never import estimator.py."""
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    required = {
        "_cp3_transform_level",
        "_cp3_reconstruct_level",
        "_cp3_winograd_arena",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required
    ]
    assert {node.name for node in nodes} == required
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "fnp": fnp,
        "_CP3_WINOGRAD_LEVELS": LEVELS,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


def bind_views(arena, specs):
    return tuple(
        fnp.reshape(arena[start:stop], shape)
        for start, stop, shape in specs
    )


def recursive_d5(a: np.ndarray, b: np.ndarray, levels: int) -> np.ndarray:
    if levels == 0:
        return np.matmul(a, b)
    m2, k2, n2 = a.shape[-2] // 2, a.shape[-1] // 2, b.shape[-1] // 2
    a11, a12 = a[..., :m2, :k2], a[..., :m2, k2:]
    a21, a22 = a[..., m2:, :k2], a[..., m2:, k2:]
    b11, b12 = b[..., :k2, :n2], b[..., :k2, n2:]
    b21, b22 = b[..., k2:, :n2], b[..., k2:, n2:]
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    p = recursive_d5(
        np.stack((a11, a12, s4, a22, s1, s2, s3), axis=0),
        np.stack((b11, b21, b22, t4, t1, t2, t3), axis=0),
        levels - 1,
    )
    u1 = p[0] + p[1]
    u2 = p[0] + p[5]
    u3 = u2 + p[6]
    u4 = u2 + p[4]
    u5 = u4 + p[2]
    u6 = u3 - p[3]
    u7 = u3 + p[4]
    return np.block([[u1, u5], [u6, u7]])


def mm_flops(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def d6_flops(m: int, k: int, n: int) -> int:
    leaf = RADIX**4 * mm_flops(m // 16, k // 16, n // 16)
    transform = sum(
        RADIX**level * 11 * (m * k + k * n) // 4 ** (level + 1)
        for level in range(4)
    )
    reconstruct = sum(
        RADIX**level * 7 * m * n // 4 ** (level + 1)
        for level in range(4)
    )
    return leaf + transform + reconstruct


def inventory(meter) -> dict[str, object]:
    calls = Counter(record.op_name for record in meter.op_log)
    costs = Counter()
    for record in meter.op_log:
        costs[record.op_name] += int(record.flop_cost)
    return {
        "flops": int(meter.flops_used),
        "calls": {name: int(value) for name, value in sorted(calls.items())},
        "costs": {name: int(value) for name, value in sorted(costs.items())},
    }


def measured_roster(call, direct, order: str) -> tuple[float, float, dict, dict]:
    def candidate_arm():
        with flops.BudgetContext(flop_budget=2_000_000_000, quiet=True) as meter:
            for _ in range(ROSTER_CALLS):
                call()
        return float(meter.summary_dict()["residual_wall_time_s"]), inventory(meter)

    def direct_arm():
        with flops.BudgetContext(flop_budget=2_000_000_000, quiet=True) as meter:
            for _ in range(ROSTER_CALLS):
                direct()
        return float(meter.summary_dict()["residual_wall_time_s"]), inventory(meter)

    if order == "candidate_first":
        candidate_residual, candidate_inventory = candidate_arm()
        direct_residual, direct_inventory = direct_arm()
    else:
        direct_residual, direct_inventory = direct_arm()
        candidate_residual, candidate_inventory = candidate_arm()
    return candidate_residual, direct_residual, candidate_inventory, direct_inventory


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    assert sha256(SOURCE) == SOURCE_SHA256
    functions = load_kernel_functions()
    kernel = functions["_cp3_winograd_arena"]
    arena_elements, specs = tiny_specs(*MAX_SHAPE)
    arena = fnp.empty(arena_elements, dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True) as bind_meter:
        views = bind_views(arena, specs)

    rng = np.random.default_rng(608_260)
    max_left_np = rng.standard_normal(MAX_SHAPE[:2]).astype(np.float32)
    max_right_np = rng.standard_normal(MAX_SHAPE[1:]).astype(np.float32)
    max_left, max_right = fnp.asarray(max_left_np), fnp.asarray(max_right_np)
    max_out = views[24]
    max_call = lambda: kernel(max_left, max_right, max_out, views)
    direct_out = fnp.empty(MAX_SHAPE[::2], dtype=fnp.float32)
    direct_call = lambda: fnp.matmul(max_left, max_right, out=direct_out)

    gm, gk, gn = GAPPED_SHAPE
    gap_left_np = rng.standard_normal((gm, gk)).astype(np.float32)
    gap_right_np = rng.standard_normal((gk, gn)).astype(np.float32)
    gap_left, gap_right = fnp.asarray(gap_left_np), fnp.asarray(gap_right_np)
    gap_out = views[25][:gm, :gn]
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as gap_meter:
        returned = kernel(gap_left, gap_right, gap_out, views)
    gap_observed = np.asarray(returned).copy()
    gap_reference = recursive_d5(gap_left_np, gap_right_np, LEVELS)

    # Warm both spellings before the alternated, aggregate 58-call timing.
    with threadpool_limits(limits=THREADS):
        with flops.BudgetContext(flop_budget=100_000_000, quiet=True):
            max_call()
            direct_call()
        candidate_residuals = []
        direct_residuals = []
        deltas = []
        candidate_inventories = []
        direct_inventories = []
        for repetition in range(REPETITIONS):
            order = "candidate_first" if repetition % 2 == 0 else "direct_first"
            candidate_residual, direct_residual, candidate_inv, direct_inv = (
                measured_roster(max_call, direct_call, order)
            )
            candidate_residuals.append(candidate_residual)
            direct_residuals.append(direct_residual)
            deltas.append(candidate_residual - direct_residual)
            candidate_inventories.append(candidate_inv)
            direct_inventories.append(direct_inv)

    expected_candidate_calls = {
        "add": 28 * ROSTER_CALLS,
        "copyto": 56 * ROSTER_CALLS,
        "matmul": ROSTER_CALLS,
        "subtract": 32 * ROSTER_CALLS,
    }
    expected_direct_calls = {"matmul": ROSTER_CALLS}
    expected_candidate_flops = d6_flops(*MAX_SHAPE) * ROSTER_CALLS
    expected_direct_flops = mm_flops(*MAX_SHAPE) * ROSTER_CALLS
    median_candidate = statistics.median(candidate_residuals)
    median_direct = statistics.median(direct_residuals)
    median_delta = statistics.median(deltas)
    checks = {
        "source_hash_exact": sha256(SOURCE) == SOURCE_SHA256,
        "fifteen_physical_regions_twenty_six_views": len(specs) == 26,
        "binding_reshape_calls_exact": inventory(bind_meter)["calls"] == {"reshape": 26},
        "gapped_returned_exact_out_object": returned is gap_out,
        "gapped_output_byte_parity_to_recursive_d5": np.array_equal(
            gap_observed, gap_reference
        ),
        "gapped_output_is_strided": not bool(np.asarray(gap_out).flags.c_contiguous),
        "gapped_hot_inventory_exact": inventory(gap_meter)["calls"] == {
            "add": 28, "copyto": 56, "matmul": 1, "subtract": 32,
        },
        "all_candidate_roster_inventories_exact": all(
            item["calls"] == expected_candidate_calls
            and item["flops"] == expected_candidate_flops
            for item in candidate_inventories
        ),
        "all_direct_roster_inventories_exact": all(
            item["calls"] == expected_direct_calls
            and item["flops"] == expected_direct_flops
            for item in direct_inventories
        ),
        "all_values_finite": all(
            np.isfinite(value)
            for value in candidate_residuals + direct_residuals + deltas
        ),
    }
    assert all(checks.values()), checks
    payload = {
        "schema": "p2r0c1d6-independent-tiny-residual-diagnostic-v1",
        "status": (
            "PASS_RESIDUAL_GATE" if median_delta <= RESIDUAL_DELTA_GATE_S
            else "FAIL_RESIDUAL_GATE_ADVERSE_EVIDENCE"
        ),
        "candidate_source_sha256": SOURCE_SHA256,
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "flopscope_version": getattr(flops, "__version__", "unknown"),
        "threads": THREADS,
        "maximum_shape": list(MAX_SHAPE),
        "gapped_shape": list(GAPPED_SHAPE),
        "roster_calls_per_arm": ROSTER_CALLS,
        "alternated_repetitions": REPETITIONS,
        "binding_meter": inventory(bind_meter),
        "gapped_meter": inventory(gap_meter),
        "gapped_output_byte_parity_to_recursive_d5": True,
        "candidate_residual_wall_seconds": candidate_residuals,
        "direct_residual_wall_seconds": direct_residuals,
        "candidate_minus_direct_residual_seconds": deltas,
        "median_candidate_residual_wall_seconds": median_candidate,
        "median_direct_residual_wall_seconds": median_direct,
        "median_candidate_minus_direct_residual_seconds": median_delta,
        "residual_delta_gate_seconds": RESIDUAL_DELTA_GATE_S,
        "residual_gate_pass": median_delta <= RESIDUAL_DELTA_GATE_S,
        "expected_candidate_roster_flops": expected_candidate_flops,
        "expected_direct_roster_flops": expected_direct_flops,
        "expected_candidate_roster_calls": expected_candidate_calls,
        "expected_direct_roster_calls": expected_direct_calls,
        "checks": checks,
        "estimator_imported": False,
        "estimator_calls": 0,
        "fixture_calls": 0,
        "exact_shape_calls": 0,
        "targets_scores_names_board_opened": False,
        "common_driver_calls": 0,
        "submission": False,
        "network": False,
        "scope_note": (
            "Tiny same-source adverse evidence, not an exact-shape timing lower bound. "
            "The hot path has the same fixed 117 wrapper calls per contraction."
        ),
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
