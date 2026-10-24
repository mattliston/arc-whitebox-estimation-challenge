#!/usr/bin/env python3
"""One-shot target-free [35,279,35] gate for the standalone D3 estimator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE_PATH = (
    P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1" / "run_candidate_gate.py"
)


def load_base():
    spec = importlib.util.spec_from_file_location("d3_gate_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
base.HERE = HERE
base.SOURCE = HERE / "estimator.py"
base.BUILDER = HERE / "build_candidate.py"
base.RESULT = HERE / "D3_CANDIDATE_RESULT.json"
base.CANDIDATE_OUTPUTS = (
    HERE / "candidate_rank35_first.npy",
    HERE / "candidate_rank279.npy",
    HERE / "candidate_rank35_repeat.npy",
)
base.SOURCE_SHA256 = "1771a616c86507ce1aad75dced22be57caddb93f5121034c5022d4a11c62c8c4"
base.BUILDER_SHA256 = "8890c877e3c8ef44e6b2a104a85e934ee8f3b3715470bf5ac527058fff84f031"
base.EXPECTED_CANDIDATE_FLOPS = (
    1_387_721_303_415,
    1_425_349_655_048,
    1_386_703_325_696,
)
base.EXPECTED_REFERENCE_MINUS_CANDIDATE = (
    304_763_600_896,
    305_778_425_856,
    305_778_425_856,
)
base.EXACT_HOT_SAVING = 305_778_425_856
base.COLD_VIEW_BINDING_BILL = 1_014_824_960


def finish(status, trials, aggregate, reason=None):
    payload = {
        "schema": "p2r0c1-d3-arena-candidate-only-gate-v1",
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
        "depth3_accounting": {
            "levels": 3,
            "cp4_cap": 2048,
            "cold_view_binding_flops": base.COLD_VIEW_BINDING_BILL,
            "hot_cp3_roster_flops": 660_308_197_376,
            "exact_hot_saving_vs_dense_cp3": base.EXACT_HOT_SAVING,
            "push_arena_elements": 479_084_544,
            "push_arena_bytes": 1_916_338_176,
            "read_arena_elements": 360_955_904,
            "read_arena_bytes": 1_443_823_616,
            "persistent_view_count_per_layout": 20,
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
            "base_d1_gate_runner_sha256": base.sha256(BASE_PATH),
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
