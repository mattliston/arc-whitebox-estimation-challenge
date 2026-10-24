#!/usr/bin/env python3
"""Static verifier for the target-free B2B2 IG-LC3 receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import enumerate_hypergraphs as enum


HERE = Path(__file__).resolve().parent
P2 = HERE.parent

BOUND = {
    P2 / "p2r0f3b2b_partition_complete_tt5_design_v1" / "DESIGN_RECEIPT.md":
        "eee173f564b655ff17aec9a8962ba490ec3f2b5013cebbf34f7ee1203b81e6ff",
    P2 / "p2r0f3b2b_partition_complete_tt5_independent_audit_v1" /
        "AUDIT_RECEIPT.json":
        "e7c34e7aec410f7b6c1ec9f94ce084afeb5030545ffda624aa39f56b6b04c99b",
    P2 / "public_trivariate_derivation_v1" / "L5_DERIVATION.md":
        "6c0dfdc022799953f8bd0f0372a7dbfdf8dbf9eeaf8649cb03f03600c49fee92",
    P2 / "public_trivariate_derivation_v1" / "DERIVATION.md":
        "5fec258709489fc5c36361bb6bb5fb3e0b8f780502d4ef9b48132bf040bb9444",
    HERE / "DERIVATION.md":
        "7c922b58d0baae6a90abd2edce1f19f03b5f5c0f219561630307ee07697332c1",
    HERE / "enumerate_hypergraphs.py":
        "25cb168fc2678dca4497fb1e326e06b02ca08c3b9fbb47006b2d4d592eeaf540",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path, expected in BOUND.items():
        got = sha256(path)
        assert got == expected, (path, got, expected)

    results = {}
    for width in range(1, 6):
        result = enum.enumerate_width(width)
        enum.verify_formula_counts(width, result)
        for key, expected in enum.EXPECTED[width].items():
            assert result[key] == expected
        results[width] = result
    costs = enum.verify_cost_witnesses(results)
    assert costs["optimistic_response_subtotal_excluding_carrier_and_log_kernel"] == 6808
    assert costs["direct_231_tt_reads"] == 362208
    assert costs["symmetry_prefix_trie_flops"] == 107744
    assert costs["optimistic_varying_fiber_flops"] == 61440

    receipt = json.loads((HERE / "DERIVATION_RECEIPT.json").read_text())
    assert receipt["status"] == (
        "PASS_IGLC3_ALGEBRA__KILL_GENERIC_TT_CARRIER__OPEN_FC0_ONLY"
    )
    assert receipt["operator"]["A_grade1"] == "Lambda2_offdiagonal + Lambda3"
    assert receipt["connected_support_counts_by_grade"]["s5"] == [0, 15, 2546]
    assert receipt["incidence_bin_counts_by_grade"]["s5"] == [0, 5, 126]
    assert receipt["recursive_leaf_split"]["generic_tt_carrier_leaf"]["status"] == (
        "CLOSED_UNDER_8192"
    )
    assert receipt["recursive_leaf_split"]["fused_carrier_identity_leaf_FC0"][
        "status"
    ] == "ONLY_OPEN_CARRIER_LEAF"
    assert receipt["access"]["targets_opened"] == 0
    assert receipt["access"]["graph_edits"] == 0
    assert receipt["access"]["experimental_runs"] == 0

    print("PASS_B2B2_IG_LC3_DERIVATION_RECEIPT")


if __name__ == "__main__":
    main()
