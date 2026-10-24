#!/usr/bin/env python3
"""One-shot target-free [35,279,35] gate for the D4 arena estimator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D3_GATE = P2 / "p2r0c1d10_cp3_arena_depth3_full_estimator_v1" / "run_candidate_gate.py"


def load_d3_gate():
    spec = importlib.util.spec_from_file_location("d4_gate_parent", D3_GATE)
    if spec is None or spec.loader is None:
        raise ImportError(D3_GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parent = load_d3_gate()
base = parent.base
base.HERE = HERE
base.SOURCE = HERE / "estimator.py"
base.BUILDER = HERE / "build_candidate.py"
base.RESULT = HERE / "D4_CANDIDATE_RESULT.json"
base.CANDIDATE_OUTPUTS = (
    HERE / "candidate_rank35_first.npy",
    HERE / "candidate_rank279.npy",
    HERE / "candidate_rank35_repeat.npy",
)
base.SOURCE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
base.BUILDER_SHA256 = "43e9d4e90d16f18e236962293f596f1c19175d4d04510e4677a33ad0495d59c5"
base.EXPECTED_CANDIDATE_FLOPS = (
    1_319_871_710_583,
    1_356_595_288_584,
    1_317_948_959_232,
)
base.EXPECTED_REFERENCE_MINUS_CANDIDATE = (
    372_613_193_728,
    374_532_792_320,
    374_532_792_320,
)
base.EXACT_HOT_SAVING = 374_532_792_320
base.COLD_VIEW_BINDING_BILL = 1_919_598_592


def finish(status, trials, aggregate, reason=None):
    payload = {
        "schema": "p2r0c1-d4-arena-candidate-only-gate-v1",
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
        "terminal_reason": reason,
        "depth4_accounting": {
            "levels": 4,
            "cp4_cap": 2048,
            "cold_view_binding_flops": base.COLD_VIEW_BINDING_BILL,
            "hot_cp3_roster_flops": 591_553_830_912,
            "exact_hot_saving_vs_dense_cp3": base.EXACT_HOT_SAVING,
            "push_arena_elements": 844_165_120,
            "push_arena_bytes": 3_376_660_480,
            "read_arena_elements": 681_480_192,
            "read_arena_bytes": 2_725_920_768,
            "persistent_view_count_per_layout": 26,
        },
        "bindings": {
            "runner_sha256": base.sha256(Path(__file__)),
            "candidate_source_sha256": base.sha256(base.SOURCE),
            "builder_sha256": base.sha256(base.BUILDER),
            "incumbent_source_sha256": base.sha256(base.INCUMBENT),
            "depth_ladder_receipt_sha256": base.sha256(
                base.LADDER / "DEPTH_LADDER_RECEIPT.json"
            ),
            "recovered_reference_evidence_sha256": base.sha256(
                base.REFERENCE_EVIDENCE
            ),
            "parent_d3_gate_runner_sha256": base.sha256(D3_GATE),
            "parent_d3_result_sha256": base.sha256(
                D3_GATE.parent / "D3_CANDIDATE_RESULT.json"
            ),
        },
    }
    base.write_json_once(base.RESULT, payload)
    print(
        json.dumps(
            {"status": status, "reason": reason, "aggregate": aggregate},
            sort_keys=True,
        ),
        flush=True,
    )


base.finish = finish


if __name__ == "__main__":
    base.main()
