#!/usr/bin/env python3
"""One-shot [35,279,35] gate for the lifetime-packed D5 estimator."""

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
STATIC_RECEIPT = HERE / "PACKED_D5_STATIC_RECEIPT.json"
ACCOUNTING_CORRECTION = HERE / "ACCOUNTING_CORRECTION.json"
PACKED_KERNEL = HERE / "packed_d5_kernel.py"
D4_OUTPUTS = (
    D4_DIR / "candidate_rank35_first.npy",
    D4_DIR / "candidate_rank279.npy",
    D4_DIR / "candidate_rank35_repeat.npy",
)


def load_d4_gate():
    spec = importlib.util.spec_from_file_location("packed_d5_gate_parent", D4_GATE)
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
base.RESULT = HERE / "D15_CANDIDATE_RESULT.json"
base.CANDIDATE_OUTPUTS = (
    HERE / "candidate_rank35_first.npy",
    HERE / "candidate_rank279.npy",
    HERE / "candidate_rank35_repeat.npy",
)
base.SOURCE_SHA256 = "74095215f842a0a993ad89c94cbaa27baf2cbc2f8cb2868e490972be8b9ab5e5"
base.BUILDER_SHA256 = "8788ad3eae331a97058559cec93c40740214593d8c93f13d9657eed7ba004988"
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
# Literal packed view binding is 3,471,495,168 FLOPs; the 226 dead-payload
# saving makes the cold-to-hot saving delta 3,471,494,942.
base.COLD_VIEW_BINDING_BILL = 3_471_494_942

inherited_verify_static_bindings = base.verify_static_bindings


def verify_static_bindings() -> None:
    inherited_verify_static_bindings()
    pins = {
        STATIC_RECEIPT: "8db4e0e286a7f52b74a029981a46ddf2c29f4408411f74b868d0d572e5647065",
        ACCOUNTING_CORRECTION: "2663625a7ddb373d872570920f33fe35f5c313d63c09025c8309ce72e25b349d",
        PACKED_KERNEL: "f16a4953a365d789d170270758c5fdc08f8bb059e792a7f41b3d62ecd19e1891",
        D4_DIR / "estimator.py": "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e",
        D4_GATE: "ac31a10c95a11d7a8e410a5e54469dd69d019a2ab46712826d15fdbb0b8e2207",
        D4_RESULT: "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1",
        D4_OUTPUTS[0]: "4a38b23d7fb995a005566336c0f7d3c5adac7cf46a5d02d6cba4b7c598b15290",
        D4_OUTPUTS[1]: "1586d3635eb865600b10418fffa61c89fbcad791e32c49e50fcf05fa783fd125",
        D4_OUTPUTS[2]: "4a38b23d7fb995a005566336c0f7d3c5adac7cf46a5d02d6cba4b7c598b15290",
    }
    for path, expected in pins.items():
        observed = base.sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))


base.verify_static_bindings = verify_static_bindings


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
    payload = {
        "schema": "p2r0c1-packed-d5-candidate-gate-v1",
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
        # The extra exact Winograd recursion level legitimately changes fp32
        # association.  These deltas document lineage only; scientific safety
        # remains the inherited incumbent drift gate plus A/B/A repeat bytes.
        "d4_lineage_numerical_diagnostic": comparisons,
        "terminal_reason": reason,
        "packed_d5_accounting": {
            "levels": 5,
            "hot_cp3_roster_flops": 542_042_060_800,
            "exact_hot_saving_vs_dense_cp3": base.EXACT_HOT_SAVING,
            "exact_hot_saving_vs_d4_flops": 49_511_770_112,
            "literal_cold_binding_flops": 3_471_495_168,
            "cold_to_hot_saving_delta_flops": base.COLD_VIEW_BINDING_BILL,
            "push_arena_elements": 580_235_264,
            "push_arena_bytes": 2_320_941_056,
            "read_arena_elements": 894_939_136,
            "read_arena_bytes": 3_579_756_544,
            "shared_arena_elements": 894_939_136,
            "shared_arena_bytes": 3_579_756_544,
            "persistent_view_count_per_layout": 30,
            "external_read_output_view_count": 2,
            "expected_incremental_kernel_wrappers_vs_d4": 1_682,
        },
        "board_baseline": {
            "submission_id": 328622,
            "raw_mse": 2.9409071835573285e-08,
            "adjusted_score": 1.783621478716186e-08,
        },
        "bindings": {
            "runner_sha256": base.sha256(Path(__file__)),
            "candidate_source_sha256": base.sha256(base.SOURCE),
            "builder_sha256": base.sha256(base.BUILDER),
            "packed_kernel_sha256": base.sha256(PACKED_KERNEL),
            "static_receipt_sha256": base.sha256(STATIC_RECEIPT),
            "accounting_correction_sha256": base.sha256(ACCOUNTING_CORRECTION),
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
                "d4_lineage_numerical_diagnostic": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )


base.finish = finish


if __name__ == "__main__":
    base.main()
