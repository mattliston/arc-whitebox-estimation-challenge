#!/usr/bin/env python3
"""Replay the sealed Phase 2 exact-layer foothold diagnostic.

This verifier reads only the already-sealed joint score summary.  It does not
load the dataset, predictions, weights, or target arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
RESULT = (
    P2
    / "data_gate"
    / "receipts"
    / "joint_target_open"
    / "0b3c04441938e2930f4d8fc92e9ca788d0dcfcc4f0107a8b152867df735d5a7e.joint-score.json"
)
EXPECTED_FILE_SHA256 = "992f8e27d8b44c4ce236051022b4baf43e59b6f610ca57aedb6eb4315b73045c"
EXPECTED_PAYLOAD_SHA256 = "16f93b625d8a779c663c540463deff268c75301bb4aa0e333a2fd7c778758035"
LABEL = "cp_cap_ladder/cp3_cap1792"
EXPECTED = (
    6.881554782955935e-10,
    2.192582806672553e-8,
    2.398551879011119e-7,
    3.722402415224689e-6,
)


def main() -> None:
    raw = RESULT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_FILE_SHA256
    result = json.loads(raw)
    assert result["schema"] == "phase2-single-joint-target-open-result-v1"
    assert result["status"] == "COMPLETE_SINGLE_JOINT_TARGET_OPEN"
    assert result["result_payload_sha256"] == EXPECTED_PAYLOAD_SHA256
    assert result["target_values_persisted"] == 0

    metric = result["candidate_metrics"][LABEL]
    curve = metric["per_layer_raw_mse"]
    assert len(curve) == 16
    observed = (curve[0], curve[1], curve[2], curve[15])
    assert observed == EXPECTED
    assert curve[0] < 1e-7
    assert curve[1] < 1e-7
    assert curve[2] > 1e-7
    assert all(value >= 0.0 for value in curve)
    assert metric["final_raw_mse"] == curve[-1]

    print(
        json.dumps(
            {
                "status": "PASS_EXACT_L1_NEAR_EXACT_L2_DRIFT_FROM_L3",
                "candidate": LABEL,
                "first_layer_above_1e-7": 3,
                "layer_1_raw_mse": curve[0],
                "layer_2_raw_mse": curve[1],
                "layer_3_raw_mse": curve[2],
                "layer_16_raw_mse": curve[15],
                "dataset_or_target_accesses": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
