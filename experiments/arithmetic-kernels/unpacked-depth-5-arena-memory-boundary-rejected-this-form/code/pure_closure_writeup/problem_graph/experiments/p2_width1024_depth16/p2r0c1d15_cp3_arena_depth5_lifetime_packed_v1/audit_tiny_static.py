#!/usr/bin/env python3
"""Changing-rank byte parity and physical bill for lifetime-packed D5."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from math import prod
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
KERNEL = HERE / "packed_d5_kernel.py"
LADDER = P2 / "p2r0c1d7_cp3_arena_depth_ladder_v1" / "run_depth_ladder.py"
D4_RESULT = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1" / "D4_CANDIDATE_RESULT.json"
RESULT = HERE / "PACKED_D5_STATIC_RECEIPT.json"
EXPECTED_LADDER_SHA256 = "50e2a26f110b4354630bf84f74e6e78433604a4d859b1d162b8f10e8e2144426"
EXPECTED_D4_RESULT_SHA256 = "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1"
PRODUCTION_WIDTH = 1024
PRODUCTION_PUSH_RANK = 15360
PRODUCTION_READ_RANK = 14336
PHASE2_BUDGET = 2**41


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(meter) -> dict[str, object]:
    summary = meter.summary_dict(by_namespace=False)
    return {
        "flops": int(summary["flops_used"]),
        "calls": {
            name: int(item["calls"])
            for name, item in sorted(summary["operations"].items())
        },
        "costs": {
            name: int(item["flop_cost"])
            for name, item in sorted(summary["operations"].items())
        },
    }


def standard_bind(ladder, arena, specs):
    return tuple(
        fnp.reshape(arena[start:stop], shape)
        for _owner, start, stop, shape in specs
    )


def arm(call):
    with flops.BudgetContext(flop_budget=10_000_000_000, quiet=True) as meter:
        value = call()
    return np.array(value, copy=True), inventory(meter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze and RESULT.exists():
        raise FileExistsError(RESULT)
    if sha256(LADDER) != EXPECTED_LADDER_SHA256:
        raise RuntimeError("depth-ladder authority changed")
    if sha256(D4_RESULT) != EXPECTED_D4_RESULT_SHA256:
        raise RuntimeError("passed D4 gate changed")
    packed = load("packed_d5_tiny_kernel", KERNEL)
    ladder = load("packed_d5_tiny_reference", LADDER)

    push_max = (64, 64, 128)
    read_max = (64, 128, 64)
    standard_push_elements, _regions, standard_push_specs = ladder.layout(
        *push_max, 5
    )
    standard_read_elements, _regions, standard_read_specs = ladder.layout(
        *read_max, 5
    )
    packed_push_elements, _regions, packed_push_specs = packed.packed_layout(
        *push_max
    )
    packed_read_elements, _regions, packed_read_specs = packed.packed_layout(
        *read_max
    )
    standard_arena = fnp.empty(
        max(standard_push_elements, standard_read_elements), dtype=fnp.float32
    )
    packed_arena = fnp.empty(
        max(packed_push_elements, packed_read_elements), dtype=fnp.float32
    )
    with flops.BudgetContext(flop_budget=1_000_000_000, quiet=True) as old_bind_meter:
        old_push = standard_bind(ladder, standard_arena, standard_push_specs)
        old_read = standard_bind(ladder, standard_arena, standard_read_specs)
    with flops.BudgetContext(flop_budget=1_000_000_000, quiet=True) as new_bind_meter:
        new_push = packed.bind(packed_arena, packed_push_specs)
        new_read = packed.bind(packed_arena, packed_read_specs)

    calls = (
        ("push64_first", 64, 64, 64, 815_101, old_push, new_push),
        ("read64_first", 64, 64, 64, 815_102, old_read, new_read),
        ("push128", 64, 64, 128, 815_103, old_push, new_push),
        ("read128", 64, 128, 64, 815_104, old_read, new_read),
        ("push64_repeat", 64, 64, 64, 815_101, old_push, new_push),
    )
    records = []
    first_candidate = None
    for label, m, k, n, seed, old_views, new_views in calls:
        rng = np.random.default_rng(seed)
        left_np = rng.standard_normal((m, k)).astype(np.float32)
        right_np = rng.standard_normal((k, n)).astype(np.float32)
        left = fnp.asarray(left_np)
        right = fnp.asarray(right_np)
        old_base = fnp.empty((m, n + 32), dtype=fnp.float32)
        new_base = fnp.empty((m, n + 32), dtype=fnp.float32)
        old_out = old_base[:, :n]
        new_out = new_base[:, :n]
        old_value, old_meter = arm(lambda: ladder.arena_product(
            left,
            right,
            old_out,
            ladder.split_views(old_views, 5),
            5,
        ))
        new_value, new_meter = arm(lambda: packed.packed_product(
            left, right, new_out, new_views
        ))
        byte_identical = bool(np.array_equal(old_value, new_value))
        meter_identical = old_meter == new_meter
        exact_out = old_value.shape == (m, n) and new_value.shape == (m, n)
        strided = not bool(np.asarray(old_out).flags.c_contiguous) and not bool(
            np.asarray(new_out).flags.c_contiguous
        )
        if not byte_identical or not meter_identical or not exact_out or not strided:
            raise AssertionError(
                (label, byte_identical, meter_identical, exact_out, strided)
            )
        if label == "push64_first":
            first_candidate = new_value
        repeat = label != "push64_repeat" or bool(
            np.array_equal(first_candidate, new_value)
        )
        if not repeat:
            raise AssertionError("changing-rank A/B/A repeat failed")
        records.append({
            "label": label,
            "shape": [m, k, n],
            "byte_identical": byte_identical,
            "meter_identical": meter_identical,
            "strided_external_output": strided,
            "repeat_byte_identical": repeat,
            "meter": new_meter,
        })

    width = PRODUCTION_WIDTH
    dense_roster = (
        [(width, width, width * layer) for layer in range(1, 16) for _ in range(2)]
        + [(width, width * layer, width) for layer in range(1, 15) for _ in range(2)]
    )
    d5_roster = sum(ladder.arena_flops(*shape, 5) for shape in dense_roster)
    d4_roster = 591_553_830_912
    dense_bill = sum(ladder.mm(*shape) for shape in dense_roster)
    push_elements, push_regions, push_specs = packed.packed_layout(
        width, width, PRODUCTION_PUSH_RANK
    )
    read_elements, read_regions, read_specs = packed.packed_layout(
        width, PRODUCTION_READ_RANK, width
    )
    scratch_binding = sum(
        prod(shape)
        for specs in (push_specs, read_specs)
        for _owner, _start, _stop, shape in specs
    )
    read_output_binding = 2 * width * width
    literal_binding = scratch_binding + read_output_binding
    d4_binding = 1_919_598_592
    hot_saving_vs_d4 = d4_roster - d5_roster
    cold_saving_vs_d4 = hot_saving_vs_d4 - (literal_binding - d4_binding) + 113
    d4 = json.loads(D4_RESULT.read_text())
    d4_peak = int(d4["aggregate"]["maximum_process_rss_bytes"])
    d4_arena = int(d4["depth4_accounting"]["push_arena_bytes"])
    packed_arena_bytes = 4 * max(push_elements, read_elements)
    projected_peak = d4_peak - d4_arena + packed_arena_bytes
    projected_margin = 8 * 2**30 - projected_peak
    if projected_margin <= 0:
        raise AssertionError(projected_margin)
    expected_calls = {
        "add": 7 * 5 * 58,
        "copyto": 14 * 5 * 58,
        "matmul": 58,
        "subtract": 8 * 5 * 58,
    }
    payload = {
        "schema": "p2r0c1-packed-d5-static-tiny-v1",
        "status": "PASS_TINY_CHANGING_RANK_BYTE_PARITY_AND_STATIC_BILL",
        "gate_eligibility": "GO_FREEZE_FULL_SOURCE_AND_RUNNER",
        "tiny_call_order": [item[0] for item in calls],
        "tiny_records": records,
        "tiny_binding_parent": inventory(old_bind_meter),
        "tiny_binding_candidate": inventory(new_bind_meter),
        "production": {
            "levels": 5,
            "dense_cp3_roster_flops": dense_bill,
            "packed_d5_cp3_roster_flops": d5_roster,
            "packed_d5_saving_vs_dense_flops": dense_bill - d5_roster,
            "d4_cp3_roster_flops": d4_roster,
            "packed_d5_hot_saving_vs_d4_flops": hot_saving_vs_d4,
            "packed_d5_cold_saving_vs_d4_flops": cold_saving_vs_d4,
            "literal_cold_binding_flops": literal_binding,
            "push_arena_elements": push_elements,
            "push_arena_bytes": 4 * push_elements,
            "read_arena_elements": read_elements,
            "read_arena_bytes": 4 * read_elements,
            "shared_arena_elements": max(push_elements, read_elements),
            "shared_arena_bytes": packed_arena_bytes,
            "full_d5_push_arena_bytes": 5_932_224_512,
            "arena_saving_vs_full_d5_push_bytes": 5_932_224_512 - packed_arena_bytes,
            "projected_full_process_peak_bytes_from_d4_gate": projected_peak,
            "projected_margin_below_8_gib_bytes": projected_margin,
            "expected_hot_kernel_wrapper_calls": expected_calls,
            "expected_incremental_kernel_wrappers_vs_d4": 29 * 58,
            "residual_delta_projection_seconds_from_depth_ladder": 0.012654314050450921,
            "projected_maximum_official_residual_seconds": (
                float(d4["aggregate"]["maximum_official_residual_wall_seconds"])
                + 0.012654314050450921
            ),
            "predicted_call_flops": [
                1_319_871_710_583 - cold_saving_vs_d4,
                1_356_595_288_584 - hot_saving_vs_d4,
                1_317_948_959_232 - hot_saving_vs_d4,
            ],
        },
        "lifetime_intervals": {
            "push": push_regions,
            "read": read_regions,
            "proof": [
                "odd/even transform buffers coexist only during level transition",
                "leaf overlaps only even/dependency buffers dead before matmul",
                "product0 and u overlap only odd transform buffers dead after matmul",
                "product1 overlaps leaf only after first unwind consumes leaf",
                "all live intervals are disjoint",
            ],
        },
        "board_baseline": {
            "submission_id": 328622,
            "raw_mse": 2.9409071835573285e-08,
            "adjusted_score": 1.783621478716186e-08,
        },
        "scope": {
            "estimator_calls": 0,
            "fixture_access": False,
            "target_or_score_access": False,
            "network": False,
            "submission": False,
        },
        "bindings": {
            "kernel_sha256": sha256(KERNEL),
            "audit_sha256": sha256(Path(__file__)),
            "depth_ladder_runner_sha256": sha256(LADDER),
            "d4_result_sha256": sha256(D4_RESULT),
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.freeze:
        RESULT.write_text(encoded)
    print(json.dumps({
        "status": payload["status"],
        "gate_eligibility": payload["gate_eligibility"],
        **payload["production"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
