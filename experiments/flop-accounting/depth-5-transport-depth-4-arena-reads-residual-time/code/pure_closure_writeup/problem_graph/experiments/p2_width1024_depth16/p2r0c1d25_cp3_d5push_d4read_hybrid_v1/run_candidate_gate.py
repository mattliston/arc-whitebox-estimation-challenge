#!/usr/bin/env python3
"""Held one-shot official-env [35,279,35] gate for D5-push/D4-read.

The scientific estimator is the Q24 closure.  Only its two CP3 push products
per nonfinal transition use packed depth five; the two CP3 read products keep
the board-passed depth-four association and arena.  This runner opens no
targets and performs no package, network, or submission action.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D18_DIR = P2 / "p2r0c1d18_d5_official_env_full_gate_v1"
D18_GATE = D18_DIR / "run_official_env_full_gate.py"
D18_RESULT = D18_DIR / "D18_OFFICIAL_ENV_FULL_GATE_RESULT.json"
D24_DIR = P2 / "p2r0c1d24_d5_official_remote_aba_v1"
D24_RESULT = D24_DIR / "D24_D5_OFFICIAL_REMOTE_ABA_RESULT.json"
D24_TELEMETRY = D24_DIR / "D24_SERVER_TELEMETRY.json"
D24_DIAGNOSTIC = D24_DIR / "D24_D5_REMOTE_NATIVE_DRIFT_AND_RESOURCE.json"
D4_DIR = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
D4_OUTPUTS = (
    D4_DIR / "candidate_rank35_first.npy",
    D4_DIR / "candidate_rank279.npy",
    D4_DIR / "candidate_rank35_repeat.npy",
)
AUDIT = HERE / "audit_tiny_static.py"
STATIC_RECEIPT = HERE / "D25_HYBRID_TINY_STATIC_RECEIPT.json"
PACKED_KERNEL = HERE / "packed_d5_kernel.py"


def load_d18_gate():
    spec = importlib.util.spec_from_file_location("d25_hybrid_parent", D18_GATE)
    if spec is None or spec.loader is None:
        raise ImportError(D18_GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


official = load_d18_gate()
d15 = official.d15
base = official.base
base.HERE = HERE
base.SOURCE = HERE / "estimator.py"
# The D15 builder is pinned as the scientific/packed-push lineage authority;
# the exact hybrid delta is proved by the D25 source AST and tiny receipt.
base.BUILDER = d15.HERE / "build_candidate.py"
base.RESULT = HERE / "D25_HYBRID_CANDIDATE_RESULT.json"
base.CANDIDATE_OUTPUTS = (
    HERE / "candidate_rank35_first.npy",
    HERE / "candidate_rank279.npy",
    HERE / "candidate_rank35_repeat.npy",
)
base.SOURCE_SHA256 = "91f4f3501227666993897de154984e8d364bd335ae1ebc7c4137cef4c38b5978"
base.BUILDER_SHA256 = "8788ad3eae331a97058559cec93c40740214593d8c93f13d9657eed7ba004988"
base.EXPECTED_CANDIDATE_FLOPS = (
    1_292_539_692_181,
    1_328_493_216_264,
    1_289_846_886_912,
)
base.EXPECTED_REFERENCE_MINUS_CANDIDATE = (
    399_945_212_130,
    402_634_864_640,
    402_634_864_640,
)
base.EXACT_HOT_SAVING = 402_634_864_640
base.COLD_VIEW_BINDING_BILL = 2_689_652_510

EXPECTED_TOTAL_CALLS = (16_280, 16_211, 16_201)
EXPECTED_WHERE_CALLS = 228
EXPECTED_WHERE_FLOPS = 1_768_423_424

inherited_verify_static_bindings = base.verify_static_bindings


def verify_static_bindings() -> None:
    # D18's static authority binds its own D17 call-0 bill through the shared
    # base object.  Restore that parent bill only while auditing the parent,
    # then reinstate the hybrid's independently proved bills.
    hybrid_flops = base.EXPECTED_CANDIDATE_FLOPS
    base.EXPECTED_CANDIDATE_FLOPS = (
        1_271_911_836_821,
        1_307_083_518_472,
        1_268_437_189_120,
    )
    try:
        inherited_verify_static_bindings()
    finally:
        base.EXPECTED_CANDIDATE_FLOPS = hybrid_flops
    pins = {
        AUDIT: "c54968ee80ecfcf62611cb894004e514e6aac50b89f97c51f598f088c9588c2b",
        STATIC_RECEIPT: "c455431dc6c167b690f3c8c8d764a34ac7fb3687c0cf36b172499f24776e19b3",
        PACKED_KERNEL: "f16a4953a365d789d170270758c5fdc08f8bb059e792a7f41b3d62ecd19e1891",
        D18_GATE: "5348ea0c6a034fdb1e1df4c1f765d4454085d1c9513c2bfe234a9bd7d3cf4e07",
        D18_RESULT: "c9573a5d60e0bc230e90d62ea3e779f87a6135f14b25508f3f5c0c2e44d0442b",
        D24_RESULT: "a9a64cb519283caf23e6a6d390b81db5c0afe6afbb83a81fa58601339711c0d2",
        D24_TELEMETRY: "df24b95488837af51665f90e7d2698436bc48bbc1be47bfa04d491b68b839aeb",
        D24_DIAGNOSTIC: "66aa81d777ba84dda573f24ddc399fbf8d11df4a8364b39149a47e7a95285a65",
    }
    for path, expected in pins.items():
        observed = base.sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))

    static = json.loads(STATIC_RECEIPT.read_text())
    production = static["production"]
    if (
        static.get("status")
        != "PASS_TINY_CHANGING_RANK_BYTE_PARITY_AND_STATIC_BILL"
        or not static.get("production_layout_exact")
        or tuple(production["predicted_call_flops"])
        != base.EXPECTED_CANDIDATE_FLOPS
        or tuple(production["predicted_total_calls"]) != EXPECTED_TOTAL_CALLS
        or int(production["shared_arena_bytes"]) != 2_725_920_768
        or int(production["projected_margin_below_8_gib_bytes"]) <= 0
    ):
        raise AssertionError("D25 static authority changed")
    d24 = json.loads(D24_RESULT.read_text())
    if (
        d24.get("status") != "PASS_FULL_REMOTE_ABA_RETURNED"
        or not d24["repeat_rank35_remote_byte_identical"]
        or not d24["all_completed_call_gates_pass"]
        or not all(
            trial["gates"]["literal_flops_exact"]
            and trial["gates"]["literal_calls_exact"]
            and trial["gates"]["output_all_finite"]
            for trial in d24["trials"]
        )
    ):
        raise AssertionError("D24 established remote authority changed")


base.verify_static_bindings = verify_static_bindings


def d4_lineage() -> list[dict[str, object]]:
    comparisons = []
    for candidate_path, d4_path in zip(base.CANDIDATE_OUTPUTS, D4_OUTPUTS):
        if not candidate_path.exists():
            break
        candidate = np.load(candidate_path, allow_pickle=False)
        d4 = np.load(d4_path, allow_pickle=False)
        delta = candidate.astype(np.float64) - d4.astype(np.float64)
        comparisons.append({
            "candidate_file": candidate_path.name,
            "d4_parent_file": d4_path.name,
            "byte_identical": bool(np.array_equal(candidate, d4)),
            "maximum_absolute_delta": float(np.max(np.abs(delta))),
            "rms_delta": float(np.sqrt(np.mean(delta * delta))),
            "d4_parent_file_sha256": base.sha256(d4_path),
        })
    return comparisons


def finish(status, trials, aggregate, reason=None):
    inventory_exact = True
    for ordinal, trial in enumerate(trials):
        operations = trial["literal_meter"]["operations"]
        total_calls = sum(int(item["calls"]) for item in operations.values())
        where = operations.get("where", {})
        call_ok = total_calls == EXPECTED_TOTAL_CALLS[ordinal]
        where_ok = (
            int(where.get("calls", -1)) == EXPECTED_WHERE_CALLS
            and int(where.get("flop_cost", -1)) == EXPECTED_WHERE_FLOPS
        )
        trial["gates"]["literal_total_calls_exact"] = call_ok
        trial["gates"]["official_where_inventory_exact"] = where_ok
        trial["gates"]["conjunctive_call_pass"] = bool(
            trial["gates"]["conjunctive_call_pass"] and call_ok and where_ok
        )
        inventory_exact = inventory_exact and call_ok and where_ok
    if aggregate:
        aggregate["literal_inventory_exact_all_calls"] = inventory_exact
        aggregate["conjunctive_pass"] = bool(
            aggregate.get("conjunctive_pass", False) and inventory_exact
        )
    if status == "PASS_TARGET_FREE_CANDIDATE_GATE" and not aggregate.get(
        "conjunctive_pass", False
    ):
        status = "FAIL_TERMINAL"
        reason = "D25 literal inventory or inherited gate failure"

    comparisons = d4_lineage()
    payload = {
        "schema": "p2r0c1-d5push-d4read-hybrid-full-gate-v1",
        "status": status,
        "call_order": list(base.CALL_ORDER),
        "reference_estimator_calls": 0,
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "training": False,
        "network": False,
        "trials": trials,
        "aggregate": aggregate,
        "d4_lineage_numerical_diagnostic": comparisons,
        "terminal_reason": reason,
        "hybrid_accounting": {
            "push_levels": 5,
            "read_levels": 4,
            "hot_cp3_roster_flops": 563_451_758_592,
            "hot_saving_vs_d4_flops": 28_102_072_320,
            "cold_saving_vs_d4_flops": 27_332_018_402,
            "literal_cold_binding_flops": 2_689_652_736,
            "dead_payload_cold_saving_flops": 226,
            "shared_arena_elements": 681_480_192,
            "shared_arena_bytes": 2_725_920_768,
            "arena_saving_vs_d4_bytes": 650_739_712,
            "arena_saving_vs_full_d5_bytes": 853_835_776,
            "push_view_count": 30,
            "read_view_count": 26,
            "incremental_hot_kernel_wrappers_vs_d4": 870,
            "expected_total_calls": list(EXPECTED_TOTAL_CALLS),
            "expected_call_flops": list(base.EXPECTED_CANDIDATE_FLOPS),
            "official_where_calls": EXPECTED_WHERE_CALLS,
            "official_where_flops": EXPECTED_WHERE_FLOPS,
        },
        "board_baseline": {
            "submission_id": 328622,
            "raw_mse": 2.9409071835573285e-08,
            "adjusted_score": 1.783621478716186e-08,
        },
        "bindings": {
            "runner_sha256": base.sha256(Path(__file__)),
            "candidate_source_sha256": base.sha256(base.SOURCE),
            "parent_d15_builder_sha256": base.sha256(base.BUILDER),
            "packed_kernel_sha256": base.sha256(PACKED_KERNEL),
            "tiny_audit_sha256": base.sha256(AUDIT),
            "static_receipt_sha256": base.sha256(STATIC_RECEIPT),
            "d18_runner_sha256": base.sha256(D18_GATE),
            "d18_result_sha256": base.sha256(D18_RESULT),
            "d24_result_sha256": base.sha256(D24_RESULT),
            "d24_telemetry_sha256": base.sha256(D24_TELEMETRY),
            "d24_diagnostic_sha256": base.sha256(D24_DIAGNOSTIC),
            "d4_parent_source_sha256": base.sha256(D4_DIR / "estimator.py"),
            "d4_parent_result_sha256": base.sha256(
                D4_DIR / "D4_CANDIDATE_RESULT.json"
            ),
            "incumbent_source_sha256": base.sha256(base.INCUMBENT),
        },
    }
    base.write_json_once(base.RESULT, payload)
    print(json.dumps({
        "status": status,
        "reason": reason,
        "aggregate": aggregate,
        "d4_lineage_numerical_diagnostic": comparisons,
    }, sort_keys=True), flush=True)


base.finish = finish


if __name__ == "__main__":
    base.main()
