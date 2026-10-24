#!/usr/bin/env python3
"""Tiny exact-CSE and alternating-dispatch discriminator for D25.

No fixture or full estimator is opened.  The experiment tests two mechanical
explanations for D25's residual miss:

* separate nearly-identical D4/D5 Python code objects versus one parameterized
  arena-product code object; and
* the duplicated D5 left/weight transform in each pair of CP push products.

The latter prepares the exact final Winograd left transform once, outside the
overwritten packed arena, then applies the unchanged right-transform, leaf,
and unwind order independently to p and q.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D25 = P2 / "p2r0c1d25_cp3_d5push_d4read_hybrid_v1"
D25_SOURCE = D25 / "estimator.py"
D25_RESULT = D25 / "D25_HYBRID_CANDIDATE_RESULT.json"
PACKED = D25 / "packed_d5_kernel.py"
LADDER = P2 / "p2r0c1d7_cp3_arena_depth_ladder_v1" / "run_depth_ladder.py"
RESULT = HERE / "D26_DISPATCH_CSE_TINY_RECEIPT.json"

PINS = {
    D25_SOURCE: "91f4f3501227666993897de154984e8d364bd335ae1ebc7c4137cef4c38b5978",
    D25_RESULT: "0513e30b06850b98d04af35df1812ca318fd13dfdfbb189a1a262333e6f521c8",
    PACKED: "f16a4953a365d789d170270758c5fdc08f8bb059e792a7f41b3d62ecd19e1891",
    LADDER: "50e2a26f110b4354630bf84f74e6e78433604a4d859b1d162b8f10e8e2144426",
}

LEVELS = 5
RADIX = 7
PRODUCTION_WIDTH = 1024
PRODUCTION_LAYERS = 15
PRODUCTION_HYBRID_ROSTER_FLOPS = 563_451_758_592
PRODUCTION_HYBRID_PROJECTED_PEAK_BYTES = 6_083_330_048


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
        "backend_seconds": float(summary["flopscope_backend_time_s"]),
        "overhead_seconds": float(summary["flopscope_overhead_time_s"]),
        "residual_seconds": float(summary["residual_wall_time_s"]),
        "wall_seconds": float(summary["wall_time_s"]),
    }


def arm(call, budget=10_000_000_000):
    with flops.BudgetContext(flop_budget=budget, quiet=True) as meter:
        value = call()
    return value, inventory(meter)


def prepare_left(left, frozen_left, arena_views, packed):
    """Produce the exact final D5 left transform once, without an extra copy."""
    (
        a_views,
        _b_views,
        dependency_a,
        _dependency_b,
        _leaf,
        _intermediates,
        _u_scratch,
    ) = packed.split_views(arena_views)
    m, k = int(left.shape[-2]), int(left.shape[-1])
    current = left
    for index in range(LEVELS):
        level = index + 1
        m2, k2 = m // 2**level, k // 2**level
        a11 = current[..., :m2, :k2]
        a12 = current[..., :m2, k2 : 2 * k2]
        a21 = current[..., m2 : 2 * m2, :k2]
        a22 = current[..., m2 : 2 * m2, k2 : 2 * k2]
        scratch = dependency_a[index][..., :m2, :k2]
        fnp.add(a21, a22, out=scratch[0])
        fnp.subtract(scratch[0], a11, out=scratch[1])
        fnp.subtract(a11, a21, out=scratch[2])
        fnp.subtract(a12, scratch[1], out=scratch[3])
        destination = (
            frozen_left if level == LEVELS
            else a_views[index][..., :m2, :k2]
        )
        for target, source in zip(
            destination,
            (a11, a12, scratch[3], a22, scratch[0], scratch[1], scratch[2]),
        ):
            fnp.copyto(target, source)
        current = destination
    return frozen_left


def apply_right(frozen_left, right, out, arena_views, packed):
    """Apply unchanged D5 right transform, leaf, and unwind to one RHS."""
    (
        _a_views,
        b_views,
        _dependency_a,
        dependency_b,
        leaf_view,
        intermediates,
        u_scratch,
    ) = packed.split_views(arena_views)
    m = int(out.shape[-2])
    k, n = int(right.shape[-2]), int(right.shape[-1])
    current_b = right
    for index in range(LEVELS):
        level = index + 1
        k2, n2 = k // 2**level, n // 2**level
        b11 = current_b[..., :k2, :n2]
        b12 = current_b[..., :k2, n2 : 2 * n2]
        b21 = current_b[..., k2 : 2 * k2, :n2]
        b22 = current_b[..., k2 : 2 * k2, n2 : 2 * n2]
        scratch = dependency_b[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=scratch[0])
        fnp.subtract(b22, scratch[0], out=scratch[1])
        fnp.subtract(b22, b12, out=scratch[2])
        fnp.subtract(scratch[1], b21, out=scratch[3])
        destination = b_views[index][..., :k2, :n2]
        for target, source in zip(
            destination,
            (b11, b21, b22, scratch[3], scratch[0], scratch[1], scratch[2]),
        ):
            fnp.copyto(target, source)
        current_b = destination

    current = leaf_view[..., : m // 32, : n // 32]
    fnp.matmul(frozen_left, current_b, out=current)
    for unwind, level in enumerate(range(LEVELS, 0, -1)):
        rows, columns = m // 2**level, n // 2**level
        destination = intermediates[unwind] if level > 1 else out
        destination = destination[..., : 2 * rows, : 2 * columns]
        p0, p1, p2, p3, p4, p5, p6 = (current[index] for index in range(7))
        scratch = u_scratch[unwind][..., :rows, :columns]
        top_left = destination[..., :rows, :columns]
        top_right = destination[..., :rows, columns : 2 * columns]
        bottom_left = destination[..., rows : 2 * rows, :columns]
        bottom_right = destination[..., rows : 2 * rows, columns : 2 * columns]
        fnp.add(p0, p1, out=top_left)
        fnp.add(p0, p5, out=scratch[0])
        fnp.add(scratch[0], p6, out=scratch[1])
        fnp.add(scratch[0], p4, out=scratch[2])
        fnp.add(scratch[2], p2, out=top_right)
        fnp.subtract(scratch[1], p3, out=bottom_left)
        fnp.add(scratch[1], p4, out=bottom_right)
        current = destination
    return out


def current_pair(left, right_p, right_q, out_p, out_q, views, candidate):
    candidate._cp3_winograd_arena_depth5(left, right_p, out_p, views)
    candidate._cp3_winograd_arena_depth5(left, right_q, out_q, views)
    return out_p, out_q


def cse_pair(left, right_p, right_q, out_p, out_q, views, frozen, packed):
    prepare_left(left, frozen, views, packed)
    apply_right(frozen, right_p, out_p, views, packed)
    apply_right(frozen, right_q, out_q, views, packed)
    return out_p, out_q


def expected_left_transform_flops(m: int, k: int) -> int:
    return sum(
        11 * RADIX ** (level - 1) * (m // 2**level) * (k // 2**level)
        for level in range(1, LEVELS + 1)
    )


def total_calls(record: dict[str, object]) -> int:
    return sum(record["calls"].values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze and RESULT.exists():
        raise FileExistsError(RESULT)
    for path, expected in PINS.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))

    candidate = load("d26_candidate", D25_SOURCE)
    packed = load("d26_packed", PACKED)
    ladder = load("d26_ladder", LADDER)

    # Changing-rank exact pair parity and literal inventory.
    max_shape = (64, 64, 128)
    arena_elements, _regions, specs = packed.packed_layout(*max_shape)
    reference_arena = fnp.empty(arena_elements, dtype=fnp.float32)
    candidate_arena = fnp.empty(arena_elements, dtype=fnp.float32)
    reference_views = packed.bind(reference_arena, specs)
    candidate_views = packed.bind(candidate_arena, specs)
    frozen = fnp.empty((7, 7, 7, 7, 7, 2, 2), dtype=fnp.float32)
    records = []
    first = None
    for label, n, seed in (
        ("rank64_first", 64, 826_101),
        ("rank128", 128, 826_102),
        ("rank64_repeat", 64, 826_101),
    ):
        rng = np.random.default_rng(seed)
        left = fnp.asarray(rng.standard_normal((64, 64)).astype(np.float32))
        right_p = fnp.asarray(rng.standard_normal((64, n)).astype(np.float32))
        right_q = fnp.asarray(rng.standard_normal((64, n)).astype(np.float32))
        ref_p_base = fnp.empty((64, n + 32), dtype=fnp.float32)
        ref_q_base = fnp.empty((64, n + 32), dtype=fnp.float32)
        cse_p_base = fnp.empty((64, n + 32), dtype=fnp.float32)
        cse_q_base = fnp.empty((64, n + 32), dtype=fnp.float32)
        ref_outputs, ref_meter = arm(lambda: current_pair(
            left,
            right_p,
            right_q,
            ref_p_base[:, :n],
            ref_q_base[:, :n],
            reference_views,
            candidate,
        ))
        cse_outputs, cse_meter = arm(lambda: cse_pair(
            left,
            right_p,
            right_q,
            cse_p_base[:, :n],
            cse_q_base[:, :n],
            candidate_views,
            frozen,
            packed,
        ))
        ref_native = tuple(np.array(value, copy=True) for value in ref_outputs)
        cse_native = tuple(np.array(value, copy=True) for value in cse_outputs)
        byte_identical = all(
            np.array_equal(old, new) for old, new in zip(ref_native, cse_native)
        )
        repeat = label != "rank64_repeat" or all(
            np.array_equal(old, new) for old, new in zip(first, cse_native)
        )
        if label == "rank64_first":
            first = cse_native
        expected_flops = expected_left_transform_flops(64, 64)
        call_delta = total_calls(ref_meter) - total_calls(cse_meter)
        operation_delta = {
            name: ref_meter["calls"].get(name, 0) - cse_meter["calls"].get(name, 0)
            for name in sorted(set(ref_meter["calls"]) | set(cse_meter["calls"]))
            if ref_meter["calls"].get(name, 0) != cse_meter["calls"].get(name, 0)
        }
        if (
            not byte_identical
            or not repeat
            or ref_meter["flops"] - cse_meter["flops"] != expected_flops
            or operation_delta != {"add": 5, "copyto": 35, "subtract": 15}
            or call_delta != 55
        ):
            raise AssertionError((label, byte_identical, repeat, operation_delta))
        records.append({
            "label": label,
            "shape": [64, 64, n],
            "both_outputs_byte_identical": byte_identical,
            "rank64_aba_repeat_byte_identical": repeat,
            "reference_meter": ref_meter,
            "cse_meter": cse_meter,
            "flop_saving": ref_meter["flops"] - cse_meter["flops"],
            "call_saving": call_delta,
            "operation_call_saving": operation_delta,
        })

    # Same production call order, tiny matrices: separate D5/D4 code objects
    # versus the ladder's one parameterized code object.  Both arms have exact
    # identical operation inventories; only Python dispatch structure differs.
    cube = (64, 64, 64)
    d5_elements, _regions, d5_specs = packed.packed_layout(*cube)
    d4_elements, _regions, d4_specs = ladder.layout(*cube, 4)
    separate_d5_arena = fnp.empty(d5_elements, dtype=fnp.float32)
    separate_d4_arena = fnp.empty(d4_elements, dtype=fnp.float32)
    unified_d5_arena = fnp.empty(d5_elements, dtype=fnp.float32)
    unified_d4_arena = fnp.empty(d4_elements, dtype=fnp.float32)
    separate_d5_views = packed.bind(separate_d5_arena, d5_specs)
    separate_d4_views = tuple(
        fnp.reshape(separate_d4_arena[start:stop], shape)
        for _owner, start, stop, shape in d4_specs
    )
    unified_d5_views = packed.bind(unified_d5_arena, d5_specs)
    unified_d4_views = tuple(
        fnp.reshape(unified_d4_arena[start:stop], shape)
        for _owner, start, stop, shape in d4_specs
    )
    dummy = fnp.empty((64, 64), dtype=fnp.float32)
    unified_d5_split = ladder.split_views(unified_d5_views + (dummy, dummy), 5)
    unified_d4_split = ladder.split_views(unified_d4_views, 4)
    rng = np.random.default_rng(826_201)
    left = fnp.asarray(rng.standard_normal((64, 64)).astype(np.float32))
    right = fnp.asarray(rng.standard_normal((64, 64)).astype(np.float32))
    separate_d5_out = fnp.empty((64, 64), dtype=fnp.float32)
    separate_d4_out = fnp.empty((64, 64), dtype=fnp.float32)
    unified_d5_out = fnp.empty((64, 64), dtype=fnp.float32)
    unified_d4_out = fnp.empty((64, 64), dtype=fnp.float32)
    schedule = tuple(
        role
        for _layer in range(14)
        for role in ("d5", "d5", "d4", "d4")
    ) + ("d5", "d5")

    def dispatch_arm(unified: bool):
        for role in schedule:
            if unified and role == "d5":
                ladder.arena_product(
                    left, right, unified_d5_out, unified_d5_split, 5
                )
            elif unified:
                ladder.arena_product(
                    left, right, unified_d4_out, unified_d4_split, 4
                )
            elif role == "d5":
                candidate._cp3_winograd_arena_depth5(
                    left, right, separate_d5_out, separate_d5_views
                )
            else:
                candidate._cp3_winograd_arena_depth4(
                    left, right, separate_d4_out, separate_d4_views
                )
        return unified_d5_out if unified else separate_d5_out

    # Warm specialization and allocator paths before paired measurement.
    arm(lambda: dispatch_arm(False), budget=100_000_000_000)
    arm(lambda: dispatch_arm(True), budget=100_000_000_000)
    dispatch_records = []
    for repetition in range(9):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        pair = {}
        for unified in order:
            _value, meter = arm(
                lambda unified=unified: dispatch_arm(unified),
                budget=100_000_000_000,
            )
            pair["unified" if unified else "separate"] = meter
        if (
            pair["unified"]["flops"] != pair["separate"]["flops"]
            or pair["unified"]["calls"] != pair["separate"]["calls"]
            or pair["unified"]["costs"] != pair["separate"]["costs"]
            or not np.array_equal(
                np.array(unified_d5_out), np.array(separate_d5_out)
            )
            or not np.array_equal(
                np.array(unified_d4_out), np.array(separate_d4_out)
            )
        ):
            raise AssertionError("unified dispatch changed arithmetic")
        dispatch_records.append({
            "repetition": repetition,
            "separate_residual_seconds": pair["separate"]["residual_seconds"],
            "unified_residual_seconds": pair["unified"]["residual_seconds"],
            "separate_minus_unified_residual_seconds": (
                pair["separate"]["residual_seconds"]
                - pair["unified"]["residual_seconds"]
            ),
        })

    # The CSE micro uses exactly the production count of 15 push pairs, so its
    # tiny residual delta directly measures the removed Python schedule work.
    cse_timing = []
    left = fnp.asarray(np.random.default_rng(826_301).standard_normal((64, 64)).astype(np.float32))
    right_p = fnp.asarray(np.random.default_rng(826_302).standard_normal((64, 64)).astype(np.float32))
    right_q = fnp.asarray(np.random.default_rng(826_303).standard_normal((64, 64)).astype(np.float32))
    ref_p = fnp.empty((64, 64), dtype=fnp.float32)
    ref_q = fnp.empty((64, 64), dtype=fnp.float32)
    new_p = fnp.empty((64, 64), dtype=fnp.float32)
    new_q = fnp.empty((64, 64), dtype=fnp.float32)

    def pair_arm(use_cse: bool):
        for _layer in range(15):
            if use_cse:
                cse_pair(
                    left, right_p, right_q, new_p, new_q,
                    candidate_views, frozen, packed,
                )
            else:
                current_pair(
                    left, right_p, right_q, ref_p, ref_q,
                    reference_views, candidate,
                )
        return new_p if use_cse else ref_p

    arm(lambda: pair_arm(False), budget=100_000_000_000)
    arm(lambda: pair_arm(True), budget=100_000_000_000)
    for repetition in range(9):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        pair = {}
        for use_cse in order:
            _value, meter = arm(
                lambda use_cse=use_cse: pair_arm(use_cse),
                budget=100_000_000_000,
            )
            pair["cse" if use_cse else "current"] = meter
        if not np.array_equal(np.array(ref_p), np.array(new_p)) or not np.array_equal(
            np.array(ref_q), np.array(new_q)
        ):
            raise AssertionError("timed CSE changed bytes")
        cse_timing.append({
            "repetition": repetition,
            "current_residual_seconds": pair["current"]["residual_seconds"],
            "cse_residual_seconds": pair["cse"]["residual_seconds"],
            "current_minus_cse_residual_seconds": (
                pair["current"]["residual_seconds"]
                - pair["cse"]["residual_seconds"]
            ),
        })

    dispatch_median = median(
        item["separate_minus_unified_residual_seconds"]
        for item in dispatch_records
    )
    cse_median = median(
        item["current_minus_cse_residual_seconds"] for item in cse_timing
    )
    production_pair_flop_saving = (
        expected_left_transform_flops(PRODUCTION_WIDTH, PRODUCTION_WIDTH)
        * PRODUCTION_LAYERS
    )
    production_call_saving = 55 * PRODUCTION_LAYERS
    frozen_elements = 7**5 * (PRODUCTION_WIDTH // 32) ** 2
    frozen_bytes = 4 * frozen_elements
    residual_credibility_seconds = max(dispatch_median, cse_median)
    credible = residual_credibility_seconds >= 0.015
    payload = {
        "schema": "p2r0c1-d5push-left-transform-cse-dispatch-tiny-v1",
        "status": (
            "PASS_EXACT_CSE_AND_MEASURED_RESIDUAL_CREDIBILITY"
            if credible else
            "PASS_EXACT_CSE_BUT_NO_15MS_RESIDUAL_CREDIBILITY"
        ),
        "full_estimator_successor_eligibility": (
            "GO_STATIC_BUILD_ONLY" if credible else "HOLD_NO_FULL_SUCCESSOR"
        ),
        "changing_rank_pair_parity": records,
        "unified_code_object_dispatch": {
            "production_call_order_length": len(schedule),
            "arithmetic_inventory_identical": True,
            "output_bytes_identical": True,
            "records": dispatch_records,
            "median_separate_minus_unified_residual_seconds": dispatch_median,
        },
        "left_transform_cse_timing": {
            "production_equivalent_pair_count": 15,
            "records": cse_timing,
            "median_current_minus_cse_residual_seconds": cse_median,
        },
        "production_projection": {
            "pair_count": PRODUCTION_LAYERS,
            "exact_flop_saving": production_pair_flop_saving,
            "exact_wrapper_call_saving": production_call_saving,
            "operation_call_saving": {
                "add": 5 * PRODUCTION_LAYERS,
                "copyto": 35 * PRODUCTION_LAYERS,
                "subtract": 15 * PRODUCTION_LAYERS,
            },
            "hybrid_roster_before_flops": PRODUCTION_HYBRID_ROSTER_FLOPS,
            "hybrid_roster_after_flops": (
                PRODUCTION_HYBRID_ROSTER_FLOPS - production_pair_flop_saving
            ),
            "frozen_left_elements": frozen_elements,
            "frozen_left_bytes": frozen_bytes,
            "projected_peak_bytes": (
                PRODUCTION_HYBRID_PROJECTED_PEAK_BYTES + frozen_bytes
            ),
            "projected_margin_below_8_gib_bytes": (
                8 * 2**30 - PRODUCTION_HYBRID_PROJECTED_PEAK_BYTES - frozen_bytes
            ),
            "measured_residual_credibility_threshold_seconds": 0.015,
            "measured_best_median_residual_saving_seconds": (
                residual_credibility_seconds
            ),
        },
        "reduction_order_proof": [
            "The left transform executes the same five levels, operand order, add/subtract order, and seven copy destinations as each original product.",
            "The final left transform is written directly to disjoint frozen storage; no arithmetic or extra rounding is introduced.",
            "Each right transform, leaf matmul, and five-level unwind remains an independent original-order execution for p and q.",
            "The frozen left transform is read-only between the two leaf matmuls.",
        ],
        "d25_residual_differential": {
            "d4": [0.3257444277405739, 0.33527638064697385, 0.35151344118639827],
            "homogeneous_d5": [0.3305413322523236, 0.3339466121979058, 0.35718477703630924],
            "heterogeneous_d25": [0.39205303182825446, 0.4121229983866215],
        },
        "scope": {
            "full_estimator_calls": 0,
            "fixtures_opened": [],
            "targets_opened": [],
            "scores_opened": [],
            "network": False,
            "package": False,
            "submission": False,
        },
        "bindings": {
            "audit_sha256": sha256(Path(__file__)),
            "d25_source_sha256": sha256(D25_SOURCE),
            "d25_result_sha256": sha256(D25_RESULT),
            "packed_kernel_sha256": sha256(PACKED),
            "depth_ladder_sha256": sha256(LADDER),
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.freeze:
        RESULT.write_text(encoded)
    print(json.dumps({
        "status": payload["status"],
        "full_estimator_successor_eligibility": payload[
            "full_estimator_successor_eligibility"
        ],
        "dispatch_median_seconds": dispatch_median,
        "cse_median_seconds": cse_median,
        "production_projection": payload["production_projection"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
