#!/usr/bin/env python3
"""Reviewed one-shot official-environment [35,279,35] gate for packed D5.

The estimator is the unchanged #328627 payload.  This gate must run under the
data_gate environment (FlopScope 0.12.0, WhestBench 0.16.0, NumPy 2.4.6), whose
rank-35 discriminator restored the original D15 literal inventory.  It opens no
targets and does not package or submit.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D15_DIR = P2 / "p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1"
D15_GATE = D15_DIR / "run_candidate_gate.py"
D17_DIR = P2 / "p2r0c1d17_d5_official_env_failure_repro_v1"
D17_RESULT = D17_DIR / "OFFICIAL_ENV_RANK35_REPRO.json"
D17_ATTEMPT = D17_DIR / "ATTEMPT_MARKER.json"
D17_RUNNER = D17_DIR / "run_rank35_official_env_repro.py"
D16_DIR = P2 / "p2r0c1d16_cp3_arena_depth5_accounting_repair_v1"
SUBMIT_WATCH = D16_DIR / "SUBMIT_WATCH.log"


def load_d15_gate():
    spec = importlib.util.spec_from_file_location("d5_official_full_parent", D15_GATE)
    if spec is None or spec.loader is None:
        raise ImportError(D15_GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d15 = load_d15_gate()
base = d15.base
base.HERE = HERE
base.SOURCE = D15_DIR / "estimator.py"
base.BUILDER = D15_DIR / "build_candidate.py"
base.RESULT = HERE / "D18_OFFICIAL_ENV_FULL_GATE_RESULT.json"
base.CANDIDATE_OUTPUTS = (
    HERE / "candidate_rank35_first.npy",
    HERE / "candidate_rank279.npy",
    HERE / "candidate_rank35_repeat.npy",
)

# FlopScope 0.12 charges the unchanged source exactly as the original D15
# projection and D17 rank-35 observation, without root-env 0.10's extra where
# term.  Every scientific/resource threshold remains inherited unchanged.
base.EXPECTED_CANDIDATE_FLOPS = (
    1_271_911_836_821,
    1_307_083_518_472,
    1_268_437_189_120,
)
base.EXPECTED_REFERENCE_MINUS_CANDIDATE = (
    420_573_067_490,
    424_044_562_432,
    424_044_562_432,
)
base.EXACT_HOT_SAVING = 424_044_562_432
base.COLD_VIEW_BINDING_BILL = 3_471_494_942

D17_OUTPUT_ARRAY_SHA256 = "59b856811e01beabab2fc13d003b7a690d725eaea3914c83fe754deb8456a17a"
OFFICIAL_WHERE_CALLS = 228
OFFICIAL_WHERE_FLOPS = 1_768_423_424

inherited_verify_static_bindings = base.verify_static_bindings


def official_versions() -> dict[str, str | None]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "flopscope_runtime": getattr(__import__("flopscope"), "__version__", None),
        "flopscope_distribution": importlib.metadata.version("flopscope"),
        "whestbench_distribution": importlib.metadata.version("whestbench"),
        "numpy": np.__version__,
    }


def verify_static_bindings() -> None:
    inherited_verify_static_bindings()
    pins = {
        D17_RESULT: "462b4106b440568816ce3bbe3e967db87e9cbae99af313e29b48178706fdc5fc",
        D17_ATTEMPT: "8e700abe742b567a6d98e1c0549d5add8afdf7bc7a1352b2b26d766ed39aa1f2",
        D17_RUNNER: "d155484055fe4017fbcb2f0de15f6509b5c1a773c0c95ac08f0671535239798a",
        SUBMIT_WATCH: "38c824fc579fdf726d5dc5136f6510a3d92e6bff7ae87817943956e6532a4d19",
    }
    for path, expected in pins.items():
        observed = base.sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))

    versions = official_versions()
    if "data_gate/.venv/bin/python" not in str(versions["executable"]):
        raise AssertionError(("wrong full-gate interpreter", versions))
    if (
        versions["flopscope_distribution"] != "0.12.0"
        or versions["whestbench_distribution"] != "0.16.0"
        or versions["numpy"] != "2.4.6"
    ):
        raise AssertionError(("official environment version drift", versions))

    watch = json.loads(SUBMIT_WATCH.read_text().strip())
    submission = watch.get("submission", {})
    if (
        submission.get("id") != 328627
        or submission.get("grading_status_cd") != "failed"
        or submission.get("score") is not None
    ):
        raise AssertionError("#328627 failed status changed")
    d17_result = json.loads(D17_RESULT.read_text())
    if (
        d17_result.get("status") != "RETURNED_OUTPUT"
        or d17_result.get("exception") is not None
        or int(d17_result["literal_meter"]["flops_used"])
        != base.EXPECTED_CANDIDATE_FLOPS[0]
        or int(d17_result["literal_meter"]["operations"]["where"]["calls"])
        != OFFICIAL_WHERE_CALLS
        or int(d17_result["literal_meter"]["operations"]["where"]["flop_cost"])
        != OFFICIAL_WHERE_FLOPS
        or d17_result["output"]["array_sha256"] != D17_OUTPUT_ARRAY_SHA256
    ):
        raise AssertionError("D17 official-environment authority changed")


base.verify_static_bindings = verify_static_bindings


def finish(status, trials, aggregate, reason=None):
    official_where_inventory_exact = True
    for trial in trials:
        where = trial["literal_meter"]["operations"].get("where", {})
        where_ok = (
            int(where.get("calls", -1)) == OFFICIAL_WHERE_CALLS
            and int(where.get("flop_cost", -1)) == OFFICIAL_WHERE_FLOPS
        )
        trial["gates"]["official_flopscope_0_12_where_inventory_exact"] = where_ok
        trial["gates"]["conjunctive_call_pass"] = bool(
            trial["gates"]["conjunctive_call_pass"] and where_ok
        )
        official_where_inventory_exact = official_where_inventory_exact and where_ok

    d17_call0_output_parity = False
    if base.CANDIDATE_OUTPUTS[0].exists():
        first = np.load(base.CANDIDATE_OUTPUTS[0], allow_pickle=False)
        d17_call0_output_parity = base.array_sha256(first) == D17_OUTPUT_ARRAY_SHA256
    if aggregate:
        aggregate["official_flopscope_0_12_where_inventory_exact"] = (
            official_where_inventory_exact
        )
        aggregate["d17_rank35_output_byte_identical"] = d17_call0_output_parity
        aggregate["conjunctive_pass"] = bool(
            aggregate.get("conjunctive_pass", False)
            and official_where_inventory_exact
            and d17_call0_output_parity
        )
    if status == "PASS_TARGET_FREE_CANDIDATE_GATE" and (
        not aggregate.get("conjunctive_pass", False)
    ):
        status = "FAIL_TERMINAL"
        reason = "official-environment inventory or D17 output parity failure"

    comparisons = d15.parent_parity()
    payload = {
        "schema": "p2r0c1-packed-d5-official-env-full-gate-v1",
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
        "official_environment": official_versions(),
        "official_accounting": {
            "expected_candidate_flops": list(base.EXPECTED_CANDIDATE_FLOPS),
            "expected_reference_minus_candidate_flops": list(
                base.EXPECTED_REFERENCE_MINUS_CANDIDATE
            ),
            "hot_total_saving_flops": base.EXACT_HOT_SAVING,
            "cold_view_binding_bill_flops": base.COLD_VIEW_BINDING_BILL,
            "where_calls_per_prediction": OFFICIAL_WHERE_CALLS,
            "where_flops_per_prediction": OFFICIAL_WHERE_FLOPS,
            "root_flopscope_0_10_accounting_repair_applied": False,
        },
        "failed_submission_binding": {
            "id": 328627,
            "grading_status_cd": "failed",
            "score": None,
            "submit_watch_sha256": base.sha256(SUBMIT_WATCH),
        },
        "bindings": {
            "runner_sha256": base.sha256(Path(__file__)),
            "candidate_source_sha256": base.sha256(base.SOURCE),
            "builder_sha256": base.sha256(base.BUILDER),
            "packed_kernel_sha256": base.sha256(d15.PACKED_KERNEL),
            "d17_result_sha256": base.sha256(D17_RESULT),
            "d17_attempt_marker_sha256": base.sha256(D17_ATTEMPT),
            "d17_runner_sha256": base.sha256(D17_RUNNER),
            "d4_parent_result_sha256": base.sha256(d15.D4_RESULT),
            "incumbent_source_sha256": base.sha256(base.INCUMBENT),
            "depth_ladder_receipt_sha256": base.sha256(
                base.LADDER / "DEPTH_LADDER_RECEIPT.json"
            ),
            "recovered_reference_evidence_sha256": base.sha256(
                base.REFERENCE_EVIDENCE
            ),
        },
    }
    base.write_json_once(base.RESULT, payload)
    print(
        json.dumps(
            {
                "status": status,
                "reason": reason,
                "aggregate": aggregate,
                "d4_lineage_numerical_diagnostic": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )


base.finish = finish


if __name__ == "__main__":
    base.main()
