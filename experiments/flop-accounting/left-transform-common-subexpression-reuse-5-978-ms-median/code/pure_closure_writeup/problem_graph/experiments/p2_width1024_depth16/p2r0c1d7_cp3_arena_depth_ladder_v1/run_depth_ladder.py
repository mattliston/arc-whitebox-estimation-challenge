#!/usr/bin/env python3
"""One-shot exact-runtime D1/D2/D3 fixed-arena utilization ladder."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
PRESEAL = HERE / "DEPTH_LADDER_PRESEAL.json"
RESULT = HERE / "DEPTH_LADDER_RECEIPT.json"
MAX_SHAPE = (64, 64, 64)
GAPPED_SHAPE = (64, 32, 48)
DEPTHS = (1, 2, 3)
RADIX = 7
ROSTER_CALLS = 58
REPETITIONS = 8
THREADS = 2
GATE_SECONDS = 0.010
WIDTH = 1024
MAX_PUSH_RANK = 15360
MAX_READ_RANK = 14336


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product(shape) -> int:
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def exact(value: Fraction) -> int:
    if value.denominator != 1:
        raise AssertionError(value)
    return value.numerator


def layout(m: int, k: int, n: int, levels: int):
    q = Fraction(7, 4)
    a, b, c = m * k, k * n, m * n
    regions = {}
    cursor = 0

    def add(name, size):
        nonlocal cursor
        regions[name] = (cursor, cursor + int(size))
        cursor += int(size)

    for level in range(1, levels + 1):
        add(f"a{level}", exact(q**level * a))
        add(f"b{level}", exact(q**level * b))
    cap = q ** (levels - 1)
    add("dependency", exact(cap * max(a, b)))
    add("leaf", exact(q**levels * c))
    add("product0", exact(cap * c))
    add("product1", exact(cap * c))
    add("u", exact(cap * c))
    add("output0", c)
    add("output1", c)
    specs = []

    def view(owner, shape):
        start, owner_stop = regions[owner]
        stop = start + product(shape)
        assert stop <= owner_stop
        specs.append((owner, start, stop, tuple(shape)))

    for level in range(1, levels + 1):
        prefix = (7,) * level
        view(f"a{level}", prefix + (m // 2**level, k // 2**level))
        view(f"b{level}", prefix + (k // 2**level, n // 2**level))
    for level in range(1, levels + 1):
        prefix = (7,) * (level - 1)
        view("dependency", (4,) + prefix + (m // 2**level, k // 2**level))
    for level in range(1, levels + 1):
        prefix = (7,) * (level - 1)
        view("dependency", (4,) + prefix + (k // 2**level, n // 2**level))
    view("leaf", (7,) * levels + (m // 2**levels, n // 2**levels))
    for unwind, remaining in enumerate(range(levels - 1, 0, -1)):
        view(
            "product0" if unwind % 2 == 0 else "product1",
            (7,) * remaining + (m // 2**remaining, n // 2**remaining),
        )
    for level in range(levels, 0, -1):
        view(
            "u",
            (3,) + (7,) * (level - 1) + (m // 2**level, n // 2**level),
        )
    view("output0", (m, n))
    view("output1", (m, n))
    assert len(specs) == 6 * levels + 2
    return cursor, regions, tuple(specs)


def bind(arena, specs):
    return tuple(fnp.reshape(arena[start:stop], shape) for _, start, stop, shape in specs)


def split_views(views, levels):
    cursor = 0
    a = views[cursor : cursor + 2 * levels : 2]
    b = views[cursor + 1 : cursor + 2 * levels : 2]
    cursor += 2 * levels
    dep_a = views[cursor : cursor + levels]
    cursor += levels
    dep_b = views[cursor : cursor + levels]
    cursor += levels
    leaf = views[cursor]
    cursor += 1
    intermediates = views[cursor : cursor + levels - 1]
    cursor += levels - 1
    u = views[cursor : cursor + levels]
    cursor += levels
    outputs = views[cursor : cursor + 2]
    assert cursor + 2 == len(views)
    return a, b, dep_a, dep_b, leaf, intermediates, u, outputs


def arena_product(left, right, out, split, levels):
    a_views, b_views, dep_as, dep_bs, leaf_view, intermediates, us, _outputs = split
    m, k, n = int(left.shape[0]), int(left.shape[1]), int(right.shape[1])
    current_a, current_b = left, right
    for index in range(levels):
        level = index + 1
        m2, k2, n2 = m // 2**level, k // 2**level, n // 2**level
        a11, a12 = current_a[..., :m2, :k2], current_a[..., :m2, k2 : 2 * k2]
        a21, a22 = current_a[..., m2 : 2 * m2, :k2], current_a[..., m2 : 2 * m2, k2 : 2 * k2]
        b11, b12 = current_b[..., :k2, :n2], current_b[..., :k2, n2 : 2 * n2]
        b21, b22 = current_b[..., k2 : 2 * k2, :n2], current_b[..., k2 : 2 * k2, n2 : 2 * n2]
        da = dep_as[index][..., :m2, :k2]
        fnp.add(a21, a22, out=da[0])
        fnp.subtract(da[0], a11, out=da[1])
        fnp.subtract(a11, a21, out=da[2])
        fnp.subtract(a12, da[1], out=da[3])
        na = a_views[index][..., :m2, :k2]
        for destination, source in zip(na, (a11, a12, da[3], a22, da[0], da[1], da[2])):
            fnp.copyto(destination, source)
        db = dep_bs[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=db[0])
        fnp.subtract(b22, db[0], out=db[1])
        fnp.subtract(b22, b12, out=db[2])
        fnp.subtract(db[1], b21, out=db[3])
        nb = b_views[index][..., :k2, :n2]
        for destination, source in zip(nb, (b11, b21, b22, db[3], db[0], db[1], db[2])):
            fnp.copyto(destination, source)
        current_a, current_b = na, nb
    leaf = leaf_view[..., : m // 2**levels, : n // 2**levels]
    fnp.matmul(current_a, current_b, out=leaf)
    current = leaf
    for unwind, level in enumerate(range(levels, 0, -1)):
        rows, cols = m // 2**level, n // 2**level
        destination = intermediates[unwind] if level > 1 else out
        destination = destination[..., : 2 * rows, : 2 * cols]
        p0, p1, p2, p3, p4, p5, p6 = (current[i] for i in range(7))
        scratch = us[unwind][..., :rows, :cols]
        tl = destination[..., :rows, :cols]
        tr = destination[..., :rows, cols : 2 * cols]
        bl = destination[..., rows : 2 * rows, :cols]
        br = destination[..., rows : 2 * rows, cols : 2 * cols]
        fnp.add(p0, p1, out=tl)
        fnp.add(p0, p5, out=scratch[0])
        fnp.add(scratch[0], p6, out=scratch[1])
        fnp.add(scratch[0], p4, out=scratch[2])
        fnp.add(scratch[2], p2, out=tr)
        fnp.subtract(scratch[1], p3, out=bl)
        fnp.add(scratch[1], p4, out=br)
        current = destination
    return out


def recursive_reference(a, b, levels):
    if levels == 0:
        return np.matmul(a, b)
    m2, k2, n2 = a.shape[-2] // 2, a.shape[-1] // 2, b.shape[-1] // 2
    a11, a12 = a[..., :m2, :k2], a[..., :m2, k2:]
    a21, a22 = a[..., m2:, :k2], a[..., m2:, k2:]
    b11, b12 = b[..., :k2, :n2], b[..., :k2, n2:]
    b21, b22 = b[..., k2:, :n2], b[..., k2:, n2:]
    s1, t1 = a21 + a22, b12 - b11
    s2, t2 = s1 - a11, b22 - t1
    s3, t3 = a11 - a21, b22 - b12
    s4, t4 = a12 - s2, t2 - b21
    p = recursive_reference(
        np.stack((a11, a12, s4, a22, s1, s2, s3), axis=0),
        np.stack((b11, b21, b22, t4, t1, t2, t3), axis=0),
        levels - 1,
    )
    u1, u2 = p[0] + p[1], p[0] + p[5]
    u3, u4 = u2 + p[6], u2 + p[4]
    return np.block([[u1, u4 + p[2]], [u3 - p[3], u3 + p[4]]])


def mm(m, k, n):
    return m * n * (2 * k - 1)


def arena_flops(m, k, n, levels):
    return 7**levels * mm(m // 2**levels, k // 2**levels, n // 2**levels) + sum(
        7**level * (11 * (m * k + k * n) + 7 * m * n) // 4 ** (level + 1)
        for level in range(levels)
    )


def operation_inventory(meter):
    calls, costs = Counter(), Counter()
    for record in meter.op_log:
        calls[record.op_name] += 1
        costs[record.op_name] += int(record.flop_cost)
    return {
        "flops": int(meter.flops_used),
        "calls": dict(sorted(calls.items())),
        "costs": dict(sorted(costs.items())),
    }


def production_roster():
    return (
        [(WIDTH, WIDTH, WIDTH * layer) for layer in range(1, 16) for _ in range(2)]
        + [(WIDTH, WIDTH * layer, WIDTH) for layer in range(1, 15) for _ in range(2)]
    )


def measure_arm(call):
    with flops.BudgetContext(flop_budget=2_000_000_000, quiet=True) as meter:
        for _ in range(ROSTER_CALLS):
            call()
    return float(meter.summary_dict()["residual_wall_time_s"]), operation_inventory(meter)


def main():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    preseal = json.loads(PRESEAL.read_text())
    assert preseal["status"] == "FROZEN_BEFORE_ONE_SHOT_DEPTH_LADDER"
    for relative, expected in preseal["sealed_files_sha256"].items():
        assert sha256(HERE / relative) == expected, relative
    rng = np.random.default_rng(608_260)
    left_np = rng.standard_normal(MAX_SHAPE[:2]).astype(np.float32)
    right_np = rng.standard_normal(MAX_SHAPE[1:]).astype(np.float32)
    gap_left_np = rng.standard_normal(GAPPED_SHAPE[:2]).astype(np.float32)
    gap_right_np = rng.standard_normal(GAPPED_SHAPE[1:]).astype(np.float32)
    left, right = fnp.asarray(left_np), fnp.asarray(right_np)
    direct_out = fnp.empty(MAX_SHAPE[::2], dtype=fnp.float32)
    direct_call = lambda: fnp.matmul(left, right, out=direct_out)
    records = []
    with threadpool_limits(limits=THREADS):
        for levels in DEPTHS:
            elements, regions, specs = layout(*MAX_SHAPE, levels)
            arena = fnp.empty(elements, dtype=fnp.float32)
            with flops.BudgetContext(flop_budget=10_000_000, quiet=True) as bind_meter:
                views = bind(arena, specs)
            split = split_views(views, levels)
            candidate_out = split[-1][0]
            candidate_call = lambda s=split, l=levels, o=candidate_out: arena_product(left, right, o, s, l)
            gm, gk, gn = GAPPED_SHAPE
            gap_left, gap_right = fnp.asarray(gap_left_np), fnp.asarray(gap_right_np)
            gap_out = split[-1][1][:gm, :gn]
            with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as gap_meter:
                returned = arena_product(gap_left, gap_right, gap_out, split, levels)
            gap_reference = recursive_reference(gap_left_np, gap_right_np, levels)
            with flops.BudgetContext(flop_budget=100_000_000, quiet=True):
                candidate_call()
                direct_call()
            candidate_residuals, direct_residuals, deltas = [], [], []
            candidate_inventories, direct_inventories = [], []
            for repetition in range(REPETITIONS):
                if repetition % 2 == 0:
                    cr, ci = measure_arm(candidate_call)
                    dr, di = measure_arm(direct_call)
                else:
                    dr, di = measure_arm(direct_call)
                    cr, ci = measure_arm(candidate_call)
                candidate_residuals.append(cr)
                direct_residuals.append(dr)
                deltas.append(cr - dr)
                candidate_inventories.append(ci)
                direct_inventories.append(di)
            expected_calls = {
                "add": 7 * levels * ROSTER_CALLS,
                "copyto": 14 * levels * ROSTER_CALLS,
                "matmul": ROSTER_CALLS,
                "subtract": 8 * levels * ROSTER_CALLS,
            }
            expected_flops = arena_flops(*MAX_SHAPE, levels) * ROSTER_CALLS
            direct_flops = mm(*MAX_SHAPE) * ROSTER_CALLS
            roster = production_roster()
            dense_prod = sum(mm(*shape) for shape in roster)
            arena_prod = sum(arena_flops(*shape, levels) for shape in roster)
            push_elements = layout(WIDTH, WIDTH, MAX_PUSH_RANK, levels)[0]
            read_elements = layout(WIDTH, MAX_READ_RANK, WIDTH, levels)[0]
            median_delta = statistics.median(deltas)
            checks = {
                "gapped_exact_out": returned is gap_out,
                "gapped_strided": not bool(np.asarray(gap_out).flags.c_contiguous),
                "gapped_byte_parity": np.array_equal(np.asarray(gap_out), gap_reference),
                "gapped_inventory": operation_inventory(gap_meter)["calls"] == {
                    "add": 7 * levels,
                    "copyto": 14 * levels,
                    "matmul": 1,
                    "subtract": 8 * levels,
                },
                "binding_calls": operation_inventory(bind_meter)["calls"] == {"reshape": 6 * levels + 2},
                "candidate_inventories": all(item["calls"] == expected_calls and item["flops"] == expected_flops for item in candidate_inventories),
                "direct_inventories": all(item["calls"] == {"matmul": ROSTER_CALLS} and item["flops"] == direct_flops for item in direct_inventories),
            }
            assert all(checks.values()), (levels, checks)
            records.append({
                "depth": levels,
                "status": "PASS_RESIDUAL_GATE" if median_delta <= GATE_SECONDS else "FAIL_RESIDUAL_GATE",
                "arena_elements_tiny": elements,
                "physical_region_count": len(regions),
                "persistent_view_count": len(specs),
                "binding_meter": operation_inventory(bind_meter),
                "gapped_meter": operation_inventory(gap_meter),
                "candidate_residual_wall_seconds": candidate_residuals,
                "direct_residual_wall_seconds": direct_residuals,
                "candidate_minus_direct_residual_seconds": deltas,
                "median_candidate_residual_wall_seconds": statistics.median(candidate_residuals),
                "median_direct_residual_wall_seconds": statistics.median(direct_residuals),
                "median_candidate_minus_direct_residual_seconds": median_delta,
                "residual_gate_pass": median_delta <= GATE_SECONDS,
                "tiny_roster_candidate_flops": expected_flops,
                "tiny_roster_direct_flops": direct_flops,
                "production_dense_roster_flops": dense_prod,
                "production_arena_roster_flops": arena_prod,
                "production_exact_hot_saving_flops": dense_prod - arena_prod,
                "production_saving_fraction": (dense_prod - arena_prod) / dense_prod,
                "production_worst_push_arena_elements": push_elements,
                "production_worst_push_arena_bytes": 4 * push_elements,
                "production_worst_read_arena_elements": read_elements,
                "production_worst_read_arena_bytes": 4 * read_elements,
                "expected_roster_calls": expected_calls,
                "checks": checks,
            })
    survivors = [record for record in records if record["residual_gate_pass"]]
    selected = max(survivors, key=lambda item: item["depth"]) if survivors else None
    receipt = {
        "schema": "p2r0c1d7-cp3-arena-depth-ladder-v1",
        "status": "PASS_DEPTH_SURVIVOR_IDENTIFIED" if selected else "NO_PYTHON_ARENA_DEPTH_SURVIVOR",
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "flopscope_version": getattr(flops, "__version__", "unknown"),
            "threads": THREADS,
        },
        "maximum_shape": list(MAX_SHAPE),
        "gapped_shape": list(GAPPED_SHAPE),
        "roster_calls_per_arm": ROSTER_CALLS,
        "alternated_repetitions": REPETITIONS,
        "residual_delta_gate_seconds": GATE_SECONDS,
        "depth_records": records,
        "selected_python_successor": None if selected is None else {
            "depth": selected["depth"],
            "production_exact_hot_saving_flops": selected["production_exact_hot_saving_flops"],
            "median_candidate_minus_direct_residual_seconds": selected["median_candidate_minus_direct_residual_seconds"],
        },
        "single_distinct_successor_if_none": None if selected else {
            "label": "legally registered literally billed fused Winograd primitive or equivalent native/custom backend",
            "mechanism": "collapse each complete channel contraction to one top-level counted out= dispatch while preserving the audited arena arithmetic, float32 association, deterministic output, and literal bill",
            "dispatches": {"python_d4_roster": 6786, "fused_roster_target": 58, "reduction_fraction": 6728 / 6786},
            "required_d4_excess_removal_seconds": 0.03961721133440733,
            "required_d4_excess_removal_fraction": 0.03961721133440733 / 0.04961721133440733,
            "external_constraint": "No lawful registered fused primitive/native provider is presently evidenced; unmetered NumPy/native work is forbidden.",
        },
        "dedupe": {
            "recursive_stack_block_d5": "closed by D5R1 resource failure",
            "iterative_depth4_d6": "closed by 58-roster residual failure",
            "depth_only_retunes": "decided by this exact-runtime D1/D2/D3 ladder",
            "streaming_or_paired_arenas": "previously deduped and resource-adverse",
            "direct_child_copy_reduction": "same Python wrapper family and insufficient to collapse dispatch",
        },
        "scope": {
            "estimator_imported_or_called": False,
            "fixture_access": False,
            "production_shape_calls": 0,
            "target_score_name_board_access": False,
            "common_driver_calls": 0,
            "graph_edits": 0,
            "network": False,
            "submission": False,
            "launch_authorized": False
        },
        "preseal_sha256": sha256(PRESEAL),
        "runner_sha256": sha256(Path(__file__)),
    }
    RESULT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
