#!/usr/bin/env python3
"""Target-free changing-rank parity and exact static ledger for packed D4.

This audit never imports or calls an estimator.  It binds sealed-D4 evidence,
uses the frozen depth-ladder D4 arithmetic as the independent parent, and
executes only six width-16 matrix products under FlopScope.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from math import prod
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
KERNEL = HERE / "packed_d4_kernel.py"
LADDER = P2 / "p2r0c1d7_cp3_arena_depth_ladder_v1" / "run_depth_ladder.py"
D4_DIR = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
D4_SOURCE = D4_DIR / "estimator.py"
D4_RESULT = D4_DIR / "D4_CANDIDATE_RESULT.json"
D23_DIR = P2 / "p2r0c1d23_d4_remote_rank35_control_v1"
D23_RESULT = D23_DIR / "D23_D4_REMOTE_RANK35_CONTROL_RESULT.json"
D23_TELEMETRY = D23_DIR / "D23_SERVER_TELEMETRY.json"
D15_DIR = P2 / "p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1"
D15_KERNEL = D15_DIR / "packed_d5_kernel.py"
D15_STATIC = D15_DIR / "PACKED_D5_STATIC_RECEIPT.json"
D15_RESULT = D15_DIR / "D15_CANDIDATE_RESULT.json"

EXPECTED = {
    D4_SOURCE: "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e",
    D4_RESULT: "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1",
    D23_RESULT: "c432588f3ed5123c10d2aa36b393f4335fcbfc1a183339c6cd672df91aac488e",
    D23_TELEMETRY: "9685ac1a852bfbeb82201732fee8d7e98f6e2bf636feba38e1eb85190ca1f777",
    LADDER: "50e2a26f110b4354630bf84f74e6e78433604a4d859b1d162b8f10e8e2144426",
    D15_KERNEL: "f16a4953a365d789d170270758c5fdc08f8bb059e792a7f41b3d62ecd19e1891",
    D15_STATIC: "8db4e0e286a7f52b74a029981a46ddf2c29f4408411f74b868d0d572e5647065",
    D15_RESULT: "7571c8948ef992643642fb39731295b341fea9e7929aa8f9e53cbbe5c4623964",
}

WIDTH = 1024
PUSH_RANK = 15360
READ_RANK = 14336
PHASE2_BUDGET = 2**41
RSS_BUDGET = 8 * 2**30
RESIDUAL_BUDGET_SECONDS = 0.4
D4_BILLS = (1_319_871_710_583, 1_356_595_288_584, 1_317_948_959_232)


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


def arm(call):
    with flops.BudgetContext(flop_budget=1_000_000_000, quiet=True) as meter:
        value = call()
    return np.array(value, copy=True), inventory(meter)


def byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def main() -> None:
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"authority changed: {path}: {actual}")
    if importlib.metadata.version("flopscope") != "0.12.0":
        raise RuntimeError("this gate requires FlopScope distribution 0.12.0")
    if not str(flops.__version__).startswith("0.12.0"):
        raise RuntimeError("this gate requires FlopScope runtime 0.12.0")
    d23 = json.loads(D23_RESULT.read_text())
    official = d23["official_client_environment"]
    if official["flopscope_client_distribution"] != "0.12.0":
        raise RuntimeError("bound D23 authority is not official FlopScope 0.12")

    packed = load("packed_d4_tiny_kernel", KERNEL)
    ladder = load("packed_d4_tiny_reference", LADDER)

    # Width 16 is the smallest nonzero D4 shape.  One shared arena per arm is
    # deliberately retained across push/read rank changes and A/B/A repeats.
    push_max = (16, 16, 32)
    read_max = (16, 32, 16)
    old_push_elements, _old_push_regions, old_push_specs = ladder.layout(
        *push_max, 4
    )
    old_read_elements, _old_read_regions, old_read_specs = ladder.layout(
        *read_max, 4
    )
    new_push_elements, _new_push_regions, new_push_specs = packed.packed_layout(
        *push_max
    )
    new_read_elements, _new_read_regions, new_read_specs = packed.packed_layout(
        *read_max
    )
    old_arena = fnp.empty(max(old_push_elements, old_read_elements), dtype=fnp.float32)
    new_arena = fnp.empty(max(new_push_elements, new_read_elements), dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as old_bind_meter:
        old_push = ladder.bind(old_arena, old_push_specs)
        old_read = ladder.bind(old_arena, old_read_specs)
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as new_bind_meter:
        new_push = packed.bind(new_arena, new_push_specs)
        new_read = packed.bind(new_arena, new_read_specs)

    # Reuse two deliberately strided external-output banks across all calls.
    old_output_base = fnp.empty((16, 48), dtype=fnp.float32)
    new_output_base = fnp.empty((16, 48), dtype=fnp.float32)
    calls = (
        ("push16_first", 16, 16, 16, 826_101, old_push, new_push),
        ("read16_first", 16, 16, 16, 826_102, old_read, new_read),
        ("push32", 16, 16, 32, 826_103, old_push, new_push),
        ("read32", 16, 32, 16, 826_104, old_read, new_read),
        ("push16_repeat", 16, 16, 16, 826_101, old_push, new_push),
        ("read16_repeat", 16, 16, 16, 826_102, old_read, new_read),
    )
    records = []
    first_candidate = {}
    expected_kernel_calls = {"add": 28, "copyto": 56, "matmul": 1, "subtract": 32}
    for label, m, k, n, seed, old_views, new_views in calls:
        rng = np.random.default_rng(seed)
        left = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32))
        right = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
        old_out = old_output_base[:m, :n]
        new_out = new_output_base[:m, :n]
        old_value, old_meter = arm(
            lambda: ladder.arena_product(
                left,
                right,
                old_out,
                ladder.split_views(old_views, 4),
                4,
            )
        )
        new_value, new_meter = arm(
            lambda: packed.packed_product(left, right, new_out, new_views)
        )
        parity = byte_equal(old_value, new_value)
        meter_parity = old_meter == new_meter
        strided = not bool(np.asarray(old_out).flags.c_contiguous) and not bool(
            np.asarray(new_out).flags.c_contiguous
        )
        if new_meter["calls"] != expected_kernel_calls:
            raise AssertionError((label, new_meter["calls"]))
        if not parity or not meter_parity or not strided:
            raise AssertionError((label, parity, meter_parity, strided))
        family = "push" if label.startswith("push") else "read"
        if label.endswith("first"):
            first_candidate[family] = new_value
        repeat = not label.endswith("repeat") or byte_equal(
            first_candidate[family], new_value
        )
        if not repeat:
            raise AssertionError(f"{family} A/B/A repeat changed bytes")
        records.append({
            "label": label,
            "shape": [m, k, n],
            "byte_identical": parity,
            "meter_identical": meter_parity,
            "repeat_byte_identical": repeat,
            "strided_external_output": strided,
            "meter": new_meter,
        })

    # Exact production layout ledger.  No production array is allocated.
    old_push, _regions, old_push_specs = ladder.layout(WIDTH, WIDTH, PUSH_RANK, 4)
    old_read, _regions, old_read_specs = ladder.layout(WIDTH, READ_RANK, WIDTH, 4)
    new_push, push_intervals, new_push_specs = packed.packed_layout(
        WIDTH, WIDTH, PUSH_RANK
    )
    new_read, read_intervals, new_read_specs = packed.packed_layout(
        WIDTH, READ_RANK, WIDTH
    )
    old_binding_flops = sum(
        prod(shape)
        for specs in (old_push_specs, old_read_specs)
        for _owner, _start, _stop, shape in specs
    )
    new_binding_flops = sum(
        prod(shape)
        for specs in (new_push_specs, new_read_specs)
        for _owner, _start, _stop, shape in specs
    )
    separate_read_output_elements = 2 * WIDTH * WIDTH
    old_persistent_elements = max(old_push, old_read)
    new_persistent_elements = max(new_push, new_read) + separate_read_output_elements
    old_persistent_bytes = 4 * old_persistent_elements
    new_persistent_bytes = 4 * new_persistent_elements
    persistent_saving_bytes = old_persistent_bytes - new_persistent_bytes
    cold_binding_saving = old_binding_flops - new_binding_flops

    d4 = json.loads(D4_RESULT.read_text())
    accounting = d4["depth4_accounting"]
    aggregate = d4["aggregate"]
    if old_push != accounting["push_arena_elements"]:
        raise AssertionError((old_push, accounting["push_arena_elements"]))
    if old_read != accounting["read_arena_elements"]:
        raise AssertionError((old_read, accounting["read_arena_elements"]))
    if old_binding_flops != accounting["cold_view_binding_flops"]:
        raise AssertionError((old_binding_flops, accounting["cold_view_binding_flops"]))

    exact_expected = {
        "new_push_elements": 331_563_008,
        "new_read_elements": 511_393_792,
        "new_binding_flops": 1_886_044_160,
        "persistent_saving_bytes": 1_322_696_704,
        "cold_binding_saving_flops": 33_554_432,
    }
    actual = {
        "new_push_elements": new_push,
        "new_read_elements": new_read,
        "new_binding_flops": new_binding_flops,
        "persistent_saving_bytes": persistent_saving_bytes,
        "cold_binding_saving_flops": cold_binding_saving,
    }
    if actual != exact_expected:
        raise AssertionError((actual, exact_expected))

    candidate_bills = (
        D4_BILLS[0] - cold_binding_saving,
        D4_BILLS[1],
        D4_BILLS[2],
    )
    projected_peak = int(aggregate["maximum_process_rss_bytes"]) - persistent_saving_bytes
    maximum_residual = float(aggregate["maximum_official_residual_wall_seconds"])
    payload = {
        "schema": "p2r0c1-packed-d4-static-tiny-v1",
        "status": "PASS_TINY_CHANGING_RANK_BYTE_AND_METER_PARITY",
        "verdict": "GO_BUILD_PACKED_D4_FULL_FIVE_COMPONENT_GATE_ONLY",
        "tiny": {
            "width": 16,
            "call_order": [record["label"] for record in records],
            "records": records,
            "parent_binding": inventory(old_bind_meter),
            "candidate_binding": inventory(new_bind_meter),
        },
        "production_exact_static": {
            "old_shared_arena_elements": old_persistent_elements,
            "old_shared_arena_bytes": old_persistent_bytes,
            "new_push_arena_elements": new_push,
            "new_push_arena_bytes": 4 * new_push,
            "new_read_arena_elements": new_read,
            "new_read_arena_bytes": 4 * new_read,
            "new_shared_arena_elements": max(new_push, new_read),
            "new_shared_arena_bytes": 4 * max(new_push, new_read),
            "separate_read_output_elements": separate_read_output_elements,
            "separate_read_output_bytes": 4 * separate_read_output_elements,
            "new_total_persistent_scratch_bytes": new_persistent_bytes,
            "persistent_saving_bytes": persistent_saving_bytes,
            "old_binding_reshape_calls": len(old_push_specs) + len(old_read_specs),
            "new_binding_reshape_calls": len(new_push_specs) + len(new_read_specs),
            "old_binding_flops": old_binding_flops,
            "new_binding_flops": new_binding_flops,
            "cold_binding_saving_flops": cold_binding_saving,
            "hot_kernel_flops_delta": 0,
            "hot_kernel_inventory_delta": {},
            "candidate_call_flops": list(candidate_bills),
            "maximum_call_flops": max(candidate_bills),
            "maximum_flop_headroom": PHASE2_BUDGET - max(candidate_bills),
            "push_lifetime_intervals": push_intervals,
            "read_lifetime_intervals": read_intervals,
        },
        "projections_not_measurements": {
            "projected_process_peak_bytes": projected_peak,
            "projected_margin_below_8_gib_bytes": RSS_BUDGET - projected_peak,
            "measured_d4_maximum_residual_seconds_unchanged_as_envelope": maximum_residual,
            "remaining_residual_envelope_seconds": RESIDUAL_BUDGET_SECONDS - maximum_residual,
            "runtime_saving_claimed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "flopscope_distribution": importlib.metadata.version("flopscope"),
            "flopscope_runtime": str(flops.__version__),
            "bound_official_client_distribution": official["flopscope_client_distribution"],
            "bound_official_client_runtime": official["flopscope_client_runtime"],
        },
        "scope": {
            "estimator_imports": 0,
            "estimator_calls": 0,
            "production_arrays_allocated": False,
            "fixture_access": False,
            "target_or_score_access": False,
            "network": False,
            "submission": False,
        },
        "bindings": {
            "packed_kernel_sha256": sha256(KERNEL),
            "audit_sha256": sha256(Path(__file__)),
            **{path.name + "_sha256": digest for path, digest in EXPECTED.items()},
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
