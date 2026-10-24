#!/usr/bin/env python3
"""Read-only hash and invariant verifier for terminal P2R0C1D6 closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D5 = P2 / "p2r0c1d5_cp3_winograd_hybrid_v1"
TINY = P2 / "p2r0c1d6_tiny_prototype_v1"
RECEIPT = HERE / "D6_TERMINAL_RECEIPT.json"

EXPECTED_LOCAL = {
    "estimator.py": "e9ebe9abebe4d632e20d81396aea2bc64a511277d88bc6fd13fb6a80721ffd58",
    "ACCOUNTING_AND_LIVENESS_MANIFEST.json": "bced5170098ccef0a702414a212dea606ef3c16e1c823c2ed2973888d5f0558c",
    "BUILD_RECEIPT.json": "7a7c4f07975cca45a413656b44017571ccec7b65d4784ade0bfa54b365d28c2a",
    "SOURCE_DIFF.patch": "73dd4e286bf09160abc41102c080607aabbf981c8bd3822ce37250e9a663df58",
    "STATIC_VERIFICATION.json": "4756f8d22b032d3a96058caafab92ea18d3e76bb64a763642636b82944af2fe7",
    "SERIALIZATION_GUARD_RESULT.json": "9a3ffa19dab76b59443f86b2dbace7a1e2c172b66f3791f27d75d960fc01b45b",
    "run_independent_tiny_residual_diagnostic.py": "4e1f58c73d5d7d7fcc1622fdfb0c62f4f1d495f4f5d67a0a8ab8f00e49d45102",
    "INDEPENDENT_TINY_RESIDUAL_DIAGNOSTIC.json": "c08ce3f9a6ed054f2a3c5a8942ca0e4b93970ba5987d4699c6171765538ca7f8",
    "INDEPENDENT_REDTEAM_D6_IMPLEMENTATION_AUDIT.md": "5ae738215f76b008384346d58fdb6cef0db39c61a056a99f6b7f8f05d907807a",
    "verify_independent_d6_implementation_audit.py": "3cdc7a0d0a2c3006fa6f1d92db909716bc6c9fed57aefc33a4b941f29b288a4a",
    "INDEPENDENT_D6_IMPLEMENTATION_AUDIT_VERIFICATION.json": "ef4fca7d85fa89e2e9b4c46f02f0540bd437dce466e58932ba42db2b56352506",
    "run_exact_shape_kernel_resource_smoke.py": "50f073ee7aafd54cbe9d1625a803ea820459c372fbf74ab213592a847023751f",
    "EXACT_SHAPE_SMOKE_DRAFT_MANIFEST.json": "8b1e09d9255057886a353c98ca8364c9da39e6a6d4c48c1a25cc012e9907c111",
    "EXACT_SHAPE_SMOKE_TERMINAL_RECEIPT_SCHEMA.json": "b967980d31c3c94cdb0aa4d60e3339f4f4c6b9a60e63e923cdc2d5116800f16d",
    "verify_exact_shape_smoke_seal.py": "e84a667c30c8ce41984bf91822260359d9f82346ead28f3a9249732a34434d1d",
    "D6_TERMINAL_RECEIPT.json": "21cc6fe6f55144b87523f2dabac026724fae9f1583ad267f19bbd0edeb8285c6",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for relative, expected in EXPECTED_LOCAL.items():
        path = HERE / relative
        assert path.is_file(), path
        assert sha256(path) == expected, (relative, sha256(path), expected)
    assert sha256(D5 / "D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md") == (
        "808a41b8fa1429e1efaf978cd3f17f18ecf7ae7f21c20a9359546922b3696da7"
    )
    assert sha256(TINY / "TINY_PROTOTYPE_FINAL_SEAL_002.json") == (
        "2f2f8d26a4892b727ce7be43664bcf001e48fa61f35ff184b98f1d6f891b85ed"
    )

    receipt = json.loads(RECEIPT.read_text())
    diagnostic = json.loads((HERE / "INDEPENDENT_TINY_RESIDUAL_DIAGNOSTIC.json").read_text())
    audit_result = json.loads(
        (HERE / "INDEPENDENT_D6_IMPLEMENTATION_AUDIT_VERIFICATION.json").read_text()
    )
    smoke_manifest = json.loads((HERE / "EXACT_SHAPE_SMOKE_DRAFT_MANIFEST.json").read_text())
    assert receipt["status"] == "FAIL_RESIDUAL_GATE_TERMINAL_CLOSE_NO_RETRY"
    assert receipt["disposition"].startswith("CLOSED_D6_ARENA_IMPLEMENTATION_FORM_NO_RETRY")
    assert diagnostic["status"] == "FAIL_RESIDUAL_GATE_ADVERSE_EVIDENCE"
    assert all(diagnostic["checks"].values())
    deltas = diagnostic["candidate_minus_direct_residual_seconds"]
    assert len(deltas) == 8 and all(value > 0.010 for value in deltas)
    median = statistics.median(deltas)
    assert median == 0.04961721133440733
    assert receipt["causal_decomposition"][
        "candidate_minus_direct_median_residual_seconds"
    ] == median
    causal = receipt["causal_decomposition"]
    assert sum(causal["candidate_inventory"].values()) == 6786
    assert sum(causal["direct_inventory"].values()) == 58
    assert causal["excess_candidate_records"] == 6786 - 58 == 6728
    assert causal["gate_excess_seconds"] == median - 0.010
    assert causal["minimum_fraction_of_measured_delta_that_must_disappear"] == (
        median - 0.010
    ) / median
    assert audit_result["status"] == (
        "PASS_VERIFIED_TERMINAL_CLOSE_D6_RESIDUAL_GATE_FAILED"
    )
    assert audit_result["exact_shape_authorized"] is False
    assert audit_result["common_driver_authorized"] is False
    assert smoke_manifest["status"] == (
        "DRAFT_HOLD_TINY_EXACT_RUNTIME_RESIDUAL_GATE_FAILED"
    )
    prerequisites = smoke_manifest["prerequisites"]
    assert prerequisites["tiny_exact_runtime_smoke_sha256"] == EXPECTED_LOCAL[
        "INDEPENDENT_TINY_RESIDUAL_DIAGNOSTIC.json"
    ]
    assert prerequisites["tiny_residual_delta_seconds"] == median
    assert prerequisites["independent_implementation_audit_status"] == (
        "FAIL_RESIDUAL_GATE_TERMINAL_CLOSE"
    )
    assert not (HERE / "EXACT_SHAPE_SMOKE_SEAL.json").exists()
    assert not (HERE / "EXACT_SHAPE_SMOKE_RESULT.json").exists()
    assert tuple(HERE.glob("*.npy")) == ()
    access = receipt["access_declarations"]
    assert access["estimator_calls"] == access["exact_shape_calls"] == 0
    assert access["target_columns_opened"] == access["scores_opened"] == []
    assert access["submission"] is access["network"] is False
    print("PASS_P2R0C1D6_TERMINAL_CLOSURE")


if __name__ == "__main__":
    main()
