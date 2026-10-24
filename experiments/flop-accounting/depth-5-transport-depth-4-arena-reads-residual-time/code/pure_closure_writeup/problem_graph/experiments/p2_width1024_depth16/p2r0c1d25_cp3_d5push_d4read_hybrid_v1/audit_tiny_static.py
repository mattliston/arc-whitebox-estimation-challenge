#!/usr/bin/env python3
"""Tiny parity plus literal physical bill for D5-push/D4-read hybrid.

This is target-free and makes no full-estimator call.  It proves that the
candidate source selects the established packed depth-5 schedule only for
push products and the passed depth-4 schedule only for read products, using
one shared arena sized by the larger depth-4 read layout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from math import prod
from pathlib import Path
import sys

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = HERE / "estimator.py"
PACKED = HERE / "packed_d5_kernel.py"
LADDER = P2 / "p2r0c1d7_cp3_arena_depth_ladder_v1" / "run_depth_ladder.py"
D4_DIR = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
D4_RESULT = D4_DIR / "D4_CANDIDATE_RESULT.json"
D24_DIR = P2 / "p2r0c1d24_d5_official_remote_aba_v1"
D24_RESULT = D24_DIR / "D24_D5_OFFICIAL_REMOTE_ABA_RESULT.json"
RESULT = HERE / "D25_HYBRID_TINY_STATIC_RECEIPT.json"

EXPECTED_SOURCE_SHA256 = "91f4f3501227666993897de154984e8d364bd335ae1ebc7c4137cef4c38b5978"
EXPECTED_PACKED_SHA256 = "f16a4953a365d789d170270758c5fdc08f8bb059e792a7f41b3d62ecd19e1891"
EXPECTED_LADDER_SHA256 = "50e2a26f110b4354630bf84f74e6e78433604a4d859b1d162b8f10e8e2144426"
EXPECTED_D4_RESULT_SHA256 = "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1"
EXPECTED_D24_RESULT_SHA256 = "a9a64cb519283caf23e6a6d390b81db5c0afe6afbb83a81fa58601339711c0d2"

WIDTH = 1024
PUSH_RANK = 15360
READ_RANK = 14336
D4_PUSH_FLOPS = 313_373_614_080
D4_READ_FLOPS = 278_180_216_832
D5_PUSH_FLOPS = 285_271_541_760
D5_READ_FLOPS = 256_770_519_040
D4_ROSTER_FLOPS = D4_PUSH_FLOPS + D4_READ_FLOPS
HYBRID_ROSTER_FLOPS = D5_PUSH_FLOPS + D4_READ_FLOPS
D4_BINDING_FLOPS = 1_919_598_592
D4_EXPECTED_FLOPS = (
    1_319_871_710_583,
    1_356_595_288_584,
    1_317_948_959_232,
)
D4_EXPECTED_CALLS = (15_416, 15_341, 15_331)


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
    }


def bind_plain(arena, specs):
    return tuple(
        fnp.reshape(arena[start:stop], shape)
        for _owner, start, stop, shape in specs
    )


def arm(call):
    with flops.BudgetContext(flop_budget=10_000_000_000, quiet=True) as meter:
        value = call()
    return np.array(value, copy=True), inventory(meter)


def called_names(function: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def source_dispatch_proof() -> dict[str, object]:
    tree = ast.parse(SOURCE.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    push_calls = called_names(functions["_cp_push"])
    read_calls = called_names(functions["_cp_read"])
    proof = {
        "cp_push_depth5_calls": push_calls.count("_cp3_winograd_arena_depth5"),
        "cp_push_depth4_calls": push_calls.count("_cp3_winograd_arena_depth4"),
        "cp_read_depth5_calls": read_calls.count("_cp3_winograd_arena_depth5"),
        "cp_read_depth4_calls": read_calls.count("_cp3_winograd_arena_depth4"),
    }
    if proof != {
        "cp_push_depth5_calls": 2,
        "cp_push_depth4_calls": 0,
        "cp_read_depth5_calls": 0,
        "cp_read_depth4_calls": 2,
    }:
        raise AssertionError(proof)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze and RESULT.exists():
        raise FileExistsError(RESULT)
    pins = {
        SOURCE: EXPECTED_SOURCE_SHA256,
        PACKED: EXPECTED_PACKED_SHA256,
        LADDER: EXPECTED_LADDER_SHA256,
        D4_RESULT: EXPECTED_D4_RESULT_SHA256,
        D24_RESULT: EXPECTED_D24_RESULT_SHA256,
    }
    for path, expected in pins.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError((str(path), observed, expected))

    candidate = load("d25_hybrid_candidate", SOURCE)
    packed = load("d25_packed_push_authority", PACKED)
    ladder = load("d25_d4_read_authority", LADDER)

    production_push_elements, push_regions, push_specs = packed.packed_layout(
        WIDTH, WIDTH, PUSH_RANK
    )
    production_read_elements, read_regions, read_specs = ladder.layout(
        WIDTH, READ_RANK, WIDTH, 4
    )
    normalized_push_specs = tuple(
        (start, stop, shape) for _owner, start, stop, shape in push_specs
    )
    normalized_read_specs = tuple(
        (start, stop, shape) for _owner, start, stop, shape in read_specs
    )
    layout_exact = bool(
        candidate._CP3_WINOGRAD_ARENA_ELEMENTS
        == max(production_push_elements, production_read_elements)
        and candidate._CP3_PUSH_ARENA_VIEW_SPECS == normalized_push_specs
        and candidate._CP3_READ_ARENA_VIEW_SPECS == normalized_read_specs
    )
    if not layout_exact:
        raise AssertionError("production arena/view binding differs from authorities")

    push_max = (64, 64, 128)
    read_max = (64, 128, 64)
    tiny_push_elements, _regions, tiny_push_specs = packed.packed_layout(*push_max)
    tiny_read_elements, _regions, tiny_read_specs = ladder.layout(*read_max, 4)
    candidate_arena = fnp.empty(
        max(tiny_push_elements, tiny_read_elements), dtype=fnp.float32
    )
    reference_push_arena = fnp.empty(tiny_push_elements, dtype=fnp.float32)
    reference_read_arena = fnp.empty(tiny_read_elements, dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=1_000_000_000, quiet=True) as bind_meter:
        candidate_push = bind_plain(candidate_arena, tiny_push_specs)
        candidate_read = bind_plain(candidate_arena, tiny_read_specs)
    reference_push = packed.bind(reference_push_arena, tiny_push_specs)
    reference_read = bind_plain(reference_read_arena, tiny_read_specs)

    calls = (
        ("push64_first", "push", 64, 64, 64, 815_101),
        ("read64_first", "read", 64, 64, 64, 815_102),
        ("push128", "push", 64, 64, 128, 815_103),
        ("read128", "read", 64, 128, 64, 815_104),
        ("push64_repeat", "push", 64, 64, 64, 815_101),
    )
    records = []
    first_candidate = None
    for label, role, m, k, n, seed in calls:
        rng = np.random.default_rng(seed)
        left = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32))
        right = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
        reference_base = fnp.empty((m, n + 32), dtype=fnp.float32)
        candidate_base = fnp.empty((m, n + 32), dtype=fnp.float32)
        reference_out = reference_base[:, :n]
        candidate_out = candidate_base[:, :n]
        if role == "push":
            reference_value, reference_meter = arm(
                lambda: packed.packed_product(
                    left, right, reference_out, reference_push
                )
            )
            candidate_value, candidate_meter = arm(
                lambda: candidate._cp3_winograd_arena_depth5(
                    left, right, candidate_out, candidate_push
                )
            )
        else:
            reference_value, reference_meter = arm(
                lambda: ladder.arena_product(
                    left,
                    right,
                    reference_out,
                    ladder.split_views(reference_read, 4),
                    4,
                )
            )
            candidate_value, candidate_meter = arm(
                lambda: candidate._cp3_winograd_arena_depth4(
                    left, right, candidate_out, candidate_read
                )
            )
        byte_identical = bool(np.array_equal(reference_value, candidate_value))
        meter_identical = reference_meter == candidate_meter
        strided = not bool(np.asarray(candidate_out).flags.c_contiguous)
        if not byte_identical or not meter_identical or not strided:
            raise AssertionError((label, byte_identical, meter_identical, strided))
        if label == "push64_first":
            first_candidate = candidate_value
        repeat = label != "push64_repeat" or bool(
            np.array_equal(first_candidate, candidate_value)
        )
        if not repeat:
            raise AssertionError("changing-rank A/B/A repeat failed")
        records.append({
            "label": label,
            "role": role,
            "shape": [m, k, n],
            "byte_identical_to_role_authority": byte_identical,
            "meter_identical_to_role_authority": meter_identical,
            "strided_external_output": strided,
            "repeat_byte_identical": repeat,
            "meter": candidate_meter,
        })

    push_binding = sum(prod(shape) for _owner, _start, _stop, shape in push_specs)
    read_binding = sum(prod(shape) for _owner, _start, _stop, shape in read_specs)
    hybrid_binding = push_binding + read_binding
    hot_saving = D4_ROSTER_FLOPS - HYBRID_ROSTER_FLOPS
    # Candidate inherits the D15 one-time removal of 226 dead payload FLOPs.
    cold_saving = hot_saving - (hybrid_binding - D4_BINDING_FLOPS) + 226
    predicted_flops = (
        D4_EXPECTED_FLOPS[0] - cold_saving,
        D4_EXPECTED_FLOPS[1] - hot_saving,
        D4_EXPECTED_FLOPS[2] - hot_saving,
    )
    kernel_wrapper_delta = {
        "add": 7 * 30,
        "copyto": 14 * 30,
        "subtract": 8 * 30,
        "total": 29 * 30,
    }
    predicted_call_deltas = (
        kernel_wrapper_delta["total"] + 4 - 10,
        kernel_wrapper_delta["total"],
        kernel_wrapper_delta["total"],
    )
    predicted_calls = tuple(
        parent + delta
        for parent, delta in zip(D4_EXPECTED_CALLS, predicted_call_deltas)
    )
    d4 = json.loads(D4_RESULT.read_text())
    d4_peak = int(d4["aggregate"]["maximum_process_rss_bytes"])
    d4_arena_bytes = int(d4["depth4_accounting"]["push_arena_bytes"])
    shared_arena_bytes = 4 * max(production_push_elements, production_read_elements)
    projected_peak = d4_peak - d4_arena_bytes + shared_arena_bytes
    projected_margin = 8 * 2**30 - projected_peak
    if (
        HYBRID_ROSTER_FLOPS != 563_451_758_592
        or hybrid_binding != 2_689_652_736
        or cold_saving != 27_332_018_402
        or predicted_flops
        != (1_292_539_692_181, 1_328_493_216_264, 1_289_846_886_912)
        or predicted_calls != (16_280, 16_211, 16_201)
        or projected_margin <= 0
    ):
        raise AssertionError("production static arithmetic changed")

    payload = {
        "schema": "p2r0c1-d5push-d4read-hybrid-static-tiny-v1",
        "status": "PASS_TINY_CHANGING_RANK_BYTE_PARITY_AND_STATIC_BILL",
        "gate_eligibility": "GO_FREEZE_FULL_RUNNER_HOLD_HEAVY",
        "official_environment": {
            "python": sys.version,
            "executable": sys.executable,
            "flopscope": importlib.metadata.version("flopscope"),
            "numpy": np.__version__,
        },
        "source_dispatch_proof": source_dispatch_proof(),
        "production_layout_exact": layout_exact,
        "tiny_call_order": [item[0] for item in calls],
        "tiny_records": records,
        "tiny_candidate_binding": inventory(bind_meter),
        "production": {
            "d4_push_flops": D4_PUSH_FLOPS,
            "d4_read_flops": D4_READ_FLOPS,
            "d4_roster_flops": D4_ROSTER_FLOPS,
            "d5_push_flops": D5_PUSH_FLOPS,
            "d5_read_flops_not_selected": D5_READ_FLOPS,
            "hybrid_roster_flops": HYBRID_ROSTER_FLOPS,
            "hot_saving_vs_d4_flops": hot_saving,
            "cold_saving_vs_d4_flops": cold_saving,
            "d4_binding_flops": D4_BINDING_FLOPS,
            "packed_d5_push_binding_flops": push_binding,
            "d4_read_binding_flops": read_binding,
            "hybrid_literal_cold_binding_flops": hybrid_binding,
            "dead_payload_cold_saving_flops": 226,
            "push_view_count": len(push_specs),
            "read_view_count": len(read_specs),
            "push_arena_elements": production_push_elements,
            "push_arena_bytes": 4 * production_push_elements,
            "read_arena_elements": production_read_elements,
            "read_arena_bytes": 4 * production_read_elements,
            "shared_arena_elements": max(
                production_push_elements, production_read_elements
            ),
            "shared_arena_bytes": shared_arena_bytes,
            "arena_saving_vs_d4_bytes": d4_arena_bytes - shared_arena_bytes,
            "arena_saving_vs_full_d5_bytes": 3_579_756_544 - shared_arena_bytes,
            "projected_process_peak_from_d4_gate_bytes": projected_peak,
            "projected_margin_below_8_gib_bytes": projected_margin,
            "incremental_hot_kernel_wrappers_vs_d4": kernel_wrapper_delta,
            "cold_non_kernel_call_delta_vs_d4": {
                "reshape_view_bindings": 4,
                "dead_asarray": -10,
            },
            "predicted_total_calls": list(predicted_calls),
            "predicted_call_flops": list(predicted_flops),
        },
        "lifetime_intervals": {
            "packed_d5_push": push_regions,
            "d4_read": read_regions,
            "proof": [
                "push and read execute sequentially and share one arena",
                "the arena extent is the maximum, not the sum, of role layouts",
                "each role's persistent views are bound once to its audited layout",
                "D5 push preserves packed-D5 reduction order",
                "D4 read preserves the board-passed D4 reduction order",
            ],
        },
        "scope": {
            "full_estimator_calls": 0,
            "fixture_access": False,
            "target_or_score_access": False,
            "network": False,
            "package": False,
            "submission": False,
        },
        "bindings": {
            "candidate_source_sha256": sha256(SOURCE),
            "packed_kernel_sha256": sha256(PACKED),
            "depth_ladder_runner_sha256": sha256(LADDER),
            "d4_result_sha256": sha256(D4_RESULT),
            "d24_result_sha256": sha256(D24_RESULT),
            "audit_sha256": sha256(Path(__file__)),
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.freeze:
        RESULT.write_text(encoded)
    print(json.dumps({
        "status": payload["status"],
        "gate_eligibility": payload["gate_eligibility"],
        "production": payload["production"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
