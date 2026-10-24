#!/usr/bin/env python3
"""Tiny shared-versus-split arena alias discriminator for D25.

Both arms execute the exact D25 D5,D5,D4,D4 production dispatch sequence,
with identical operations and output destinations.  The sole distinction is
whether the 30-view packed-D5 and 26-view D4 layouts alias one backing arena
or have disjoint backing arenas.
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
D26 = P2 / "p2r0c1d26_cp3_d5push_left_transform_cse_v1"
SOURCE = D25 / "estimator.py"
PACKED = D25 / "packed_d5_kernel.py"
LADDER = P2 / "p2r0c1d7_cp3_arena_depth_ladder_v1" / "run_depth_ladder.py"
D25_RESULT = D25 / "D25_HYBRID_CANDIDATE_RESULT.json"
D26_AUDIT = D26 / "audit_dispatch_cse_tiny.py"
D26_RECEIPT = D26 / "D26_DISPATCH_CSE_TINY_RECEIPT.json"
RESULT = HERE / "D27_SHARED_ALIAS_TINY_RECEIPT.json"

PINS = {
    SOURCE: "91f4f3501227666993897de154984e8d364bd335ae1ebc7c4137cef4c38b5978",
    PACKED: "f16a4953a365d789d170270758c5fdc08f8bb059e792a7f41b3d62ecd19e1891",
    LADDER: "50e2a26f110b4354630bf84f74e6e78433604a4d859b1d162b8f10e8e2144426",
    D25_RESULT: "0513e30b06850b98d04af35df1812ca318fd13dfdfbb189a1a262333e6f521c8",
    D26_AUDIT: "0b68b50d6703e451ad413668bb55df874528322cc373d8cdbc9cfdb2c0de29f6",
    D26_RECEIPT: "315da927287b35058917c0488196ef3640900cd077dc0e5764173f5bea5f5fd5",
}


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


def bind(arena, specs):
    return tuple(
        fnp.reshape(arena[start:stop], shape)
        for _owner, start, stop, shape in specs
    )


def inventory(meter):
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
        "residual_seconds": float(summary["residual_wall_time_s"]),
        "backend_seconds": float(summary["flopscope_backend_time_s"]),
        "overhead_seconds": float(summary["flopscope_overhead_time_s"]),
        "wall_seconds": float(summary["wall_time_s"]),
    }


def arm(call):
    with flops.BudgetContext(flop_budget=100_000_000_000, quiet=True) as meter:
        value = call()
    return value, inventory(meter)


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

    candidate = load("d27_candidate", SOURCE)
    packed = load("d27_packed", PACKED)
    ladder = load("d27_ladder", LADDER)
    shape = (64, 64, 64)
    d5_elements, _regions, d5_specs = packed.packed_layout(*shape)
    d4_elements, _regions, d4_specs = ladder.layout(*shape, 4)

    shared_arena = fnp.empty(max(d5_elements, d4_elements), dtype=fnp.float32)
    shared_d5_views = bind(shared_arena, d5_specs)
    shared_d4_views = bind(shared_arena, d4_specs)
    split_d5_arena = fnp.empty(d5_elements, dtype=fnp.float32)
    split_d4_arena = fnp.empty(d4_elements, dtype=fnp.float32)
    split_d5_views = bind(split_d5_arena, d5_specs)
    split_d4_views = bind(split_d4_arena, d4_specs)

    rng = np.random.default_rng(827_101)
    left = fnp.asarray(rng.standard_normal((64, 64)).astype(np.float32))
    right = fnp.asarray(rng.standard_normal((64, 64)).astype(np.float32))
    shared_d5_out = fnp.empty((64, 64), dtype=fnp.float32)
    shared_d4_out = fnp.empty((64, 64), dtype=fnp.float32)
    split_d5_out = fnp.empty((64, 64), dtype=fnp.float32)
    split_d4_out = fnp.empty((64, 64), dtype=fnp.float32)
    schedule = tuple(
        role
        for _layer in range(14)
        for role in ("d5", "d5", "d4", "d4")
    ) + ("d5", "d5")

    def run(shared: bool):
        d5_views = shared_d5_views if shared else split_d5_views
        d4_views = shared_d4_views if shared else split_d4_views
        d5_out = shared_d5_out if shared else split_d5_out
        d4_out = shared_d4_out if shared else split_d4_out
        for role in schedule:
            if role == "d5":
                candidate._cp3_winograd_arena_depth5(
                    left, right, d5_out, d5_views
                )
            else:
                candidate._cp3_winograd_arena_depth4(
                    left, right, d4_out, d4_views
                )
        return d5_out

    arm(lambda: run(True))
    arm(lambda: run(False))
    records = []
    for repetition in range(13):
        order = (True, False) if repetition % 2 == 0 else (False, True)
        pair = {}
        for shared in order:
            _value, meter = arm(lambda shared=shared: run(shared))
            pair["shared" if shared else "split"] = meter
        if (
            pair["shared"]["flops"] != pair["split"]["flops"]
            or pair["shared"]["calls"] != pair["split"]["calls"]
            or pair["shared"]["costs"] != pair["split"]["costs"]
            or not np.array_equal(np.array(shared_d5_out), np.array(split_d5_out))
            or not np.array_equal(np.array(shared_d4_out), np.array(split_d4_out))
        ):
            raise AssertionError("arena separation changed arithmetic")
        records.append({
            "repetition": repetition,
            "shared_residual_seconds": pair["shared"]["residual_seconds"],
            "split_residual_seconds": pair["split"]["residual_seconds"],
            "shared_minus_split_residual_seconds": (
                pair["shared"]["residual_seconds"]
                - pair["split"]["residual_seconds"]
            ),
        })

    residual_gain = median(
        item["shared_minus_split_residual_seconds"] for item in records
    )
    production_shared_arena_bytes = 2_725_920_768
    production_d5_push_arena_bytes = 2_320_941_056
    production_split_arena_bytes = (
        production_shared_arena_bytes + production_d5_push_arena_bytes
    )
    measured_d25_rank279_rss = 6_046_711_808
    projected_split_rank279_rss = (
        measured_d25_rank279_rss + production_d5_push_arena_bytes
    )
    margin = 8 * 2**30 - projected_split_rank279_rss
    credible = residual_gain >= 0.015 and margin > 0
    payload = {
        "schema": "p2r0c1-d25-hybrid-shared-alias-tiny-v1",
        "status": (
            "PASS_SPLIT_ARENA_15MS_CREDIBILITY"
            if credible else
            "PASS_ALIAS_ISOLATOR_NO_15MS_CREDIBILITY"
        ),
        "successor_eligibility": (
            "GO_STATIC_BUILD_ONLY" if credible else "HOLD_NO_SUCCESSOR"
        ),
        "production_call_order_length": len(schedule),
        "arithmetic_inventory_identical": True,
        "output_bytes_identical": True,
        "records": records,
        "median_shared_minus_split_residual_seconds": residual_gain,
        "threshold_seconds": 0.015,
        "production_projection": {
            "shared_arena_bytes": production_shared_arena_bytes,
            "split_d5_push_arena_bytes": production_d5_push_arena_bytes,
            "split_d4_read_arena_bytes": production_shared_arena_bytes,
            "split_total_arena_bytes": production_split_arena_bytes,
            "additional_bytes_vs_d25": production_d5_push_arena_bytes,
            "measured_d25_rank279_rss_bytes": measured_d25_rank279_rss,
            "projected_split_rank279_rss_bytes": projected_split_rank279_rss,
            "projected_margin_below_8_gib_bytes": margin,
            "flop_change": 0,
            "hot_operation_call_change": 0,
            "cold_empty_call_change": 1,
        },
        "reduction_order_proof": [
            "Both arms invoke the identical D25 D5 and D4 function objects in the identical 58-call order.",
            "Every metered operation count and cost is identical.",
            "Only the backing allocation identity of the already-bound arena views differs.",
        ],
        "scope": {
            "full_estimator_calls": 0,
            "fixtures_opened": [],
            "targets_opened": [],
            "scores_opened": [],
            "network": False,
            "package": False,
            "submission": False
        },
        "bindings": {
            "audit_sha256": sha256(Path(__file__)),
            "d25_source_sha256": sha256(SOURCE),
            "d25_result_sha256": sha256(D25_RESULT),
            "d26_receipt_sha256": sha256(D26_RECEIPT),
            "packed_kernel_sha256": sha256(PACKED),
            "depth_ladder_sha256": sha256(LADDER)
        }
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.freeze:
        RESULT.write_text(encoded)
    print(json.dumps({
        "status": payload["status"],
        "successor_eligibility": payload["successor_eligibility"],
        "median_shared_minus_split_residual_seconds": residual_gain,
        "production_projection": payload["production_projection"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
