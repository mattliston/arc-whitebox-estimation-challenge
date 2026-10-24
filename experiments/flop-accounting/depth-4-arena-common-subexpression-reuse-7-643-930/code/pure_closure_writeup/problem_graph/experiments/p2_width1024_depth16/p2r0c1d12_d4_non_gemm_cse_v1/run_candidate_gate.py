#!/usr/bin/env python3
"""One-shot [35,279,35] gate for D4 plus exact non-GEMM CSE."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D4_DIR = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
D4_GATE = D4_DIR / "run_candidate_gate.py"
D4_RESULT = D4_DIR / "D4_CANDIDATE_RESULT.json"
STATIC_RECEIPT = HERE / "CSE_STATIC_RECEIPT.json"
D4_OUTPUTS = (
    D4_DIR / "candidate_rank35_first.npy",
    D4_DIR / "candidate_rank279.npy",
    D4_DIR / "candidate_rank35_repeat.npy",
)


def load_d4_gate():
    spec = importlib.util.spec_from_file_location("d4_cse_gate_parent", D4_GATE)
    if spec is None or spec.loader is None:
        raise ImportError(D4_GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parent = load_d4_gate()
base = parent.base
base.HERE = HERE
base.SOURCE = HERE / "estimator.py"
base.BUILDER = HERE / "build_candidate.py"
base.RESULT = HERE / "D4_CSE_CANDIDATE_RESULT.json"
base.CANDIDATE_OUTPUTS = (
    HERE / "candidate_rank35_first.npy",
    HERE / "candidate_rank279.npy",
    HERE / "candidate_rank35_repeat.npy",
)
base.SOURCE_SHA256 = "d8e2c1261bd8b77ce9b54b434aaa41f21704146d48e215ccb690dd38224398bb"
base.BUILDER_SHA256 = "5a4270de78ad8dca9932e5a6a74f29576a89134b202700492124b82f5e77e945"
base.EXPECTED_CANDIDATE_FLOPS = (
    1_312_227_780_471,
    1_348_951_358_472,
    1_310_305_029_120,
)
base.EXPECTED_REFERENCE_MINUS_CANDIDATE = (
    380_257_123_840,
    382_176_722_432,
    382_176_722_432,
)
base.EXACT_HOT_SAVING = 382_176_722_432
base.COLD_VIEW_BINDING_BILL = 1_919_598_592


def parent_parity() -> list[dict[str, object]]:
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
    comparisons = parent_parity()
    if status == "PASS_TARGET_FREE_CANDIDATE_GATE":
        if len(comparisons) != 3 or not all(
            item["byte_identical"] for item in comparisons
        ):
            status = "FAIL_TERMINAL"
            reason = "D4 parent output byte parity failure"
    payload = {
        "schema": "p2r0c1-d4-nongemm-cse-candidate-gate-v1",
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
        "d4_parent_output_parity": comparisons,
        "terminal_reason": reason,
        "cse_accounting": {
            "pair_saving_per_full_reset": 542_623_488,
            "full_pair_reset_count": 14,
            "reset_plane_saving_per_reset": 3_146_752,
            "reset_count": 15,
            "exact_incremental_saving_flops": 7_643_930_112,
            "d4_hot_cp3_saving_flops": 374_532_792_320,
            "combined_hot_saving_flops": base.EXACT_HOT_SAVING,
        },
        "bindings": {
            "runner_sha256": base.sha256(Path(__file__)),
            "candidate_source_sha256": base.sha256(base.SOURCE),
            "builder_sha256": base.sha256(base.BUILDER),
            "static_receipt_sha256": base.sha256(STATIC_RECEIPT),
            "d4_parent_source_sha256": base.sha256(D4_DIR / "estimator.py"),
            "d4_parent_result_sha256": base.sha256(D4_RESULT),
            "d4_parent_runner_sha256": base.sha256(D4_GATE),
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
                "d4_parent_output_parity": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )


base.finish = finish


if __name__ == "__main__":
    base.main()
