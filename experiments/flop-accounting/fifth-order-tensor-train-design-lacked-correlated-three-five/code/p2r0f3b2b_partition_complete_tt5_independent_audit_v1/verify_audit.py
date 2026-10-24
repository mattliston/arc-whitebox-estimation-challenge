#!/usr/bin/env python3
"""Static verifier for the additive PC-TT5 independent audit.

This script reads only bound source/audit files. It opens no dataset, target,
fit, score, board, estimator execution, or graph mutation surface.
"""

from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent

BOUND = {
    P2 / "p2r0f3b2b_partition_complete_tt5_design_v1" / "DESIGN_RECEIPT.md":
        "eee173f564b655ff17aec9a8962ba490ec3f2b5013cebbf34f7ee1203b81e6ff",
    P2 / "public_noeigh_card_v1" / "BUILD_RECEIPT.json":
        "3fd515406b6eec954523e7d553aba769999998318fb29fd987ce4637e2bbb0d3",
    P2 / "public_trim_card_v1" / "METER_RESULT.json":
        "bd2b33754585a4c5ffcd6877d7f7956f135a2f743c5dd9c96593b3b19569b006",
    P2 / "public_exact_ledger_v1" / "LEDGER_VERIFY_RESULTS.json":
        "f14b4964ab6d7da4917d4c4a77c08099477a840849819cc61ee46da0d842c938",
    P2 / "public_exact_ledger_v1" / "ledger_rig.py":
        "c74c92862a3afad7e44a077c80f87fdd09c66be4f02718db24a21ea7b5f4a581",
    P2 / "public_trivariate_derivation_v1" / "L5_DERIVATION.md":
        "6c0dfdc022799953f8bd0f0372a7dbfdf8dbf9eeaf8649cb03f03600c49fee92",
    P2 / "public_trivariate_derivation_v1" / "DERIVATION.md":
        "5fec258709489fc5c36361bb6bb5fb3e0b8f780502d4ef9b48132bf040bb9444",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recursive_values(obj, key):
    if isinstance(obj, dict):
        for name, value in obj.items():
            if name == key:
                yield value
            yield from recursive_values(value, key)
    elif isinstance(obj, list):
        for value in obj:
            yield from recursive_values(value, key)


def main() -> None:
    for path, expected in BOUND.items():
        got = sha256(path)
        assert got == expected, (path, got, expected)

    design = (P2 / "p2r0f3b2b_partition_complete_tt5_design_v1" /
              "DESIGN_RECEIPT.md").read_text()
    assert "n=4" in design
    assert "[11111]" in design
    assert "Lambda_5=0" in design
    assert "8,192" in design

    # A five-distinct-index class cannot be instantiated with width four.
    assert 4 < 5

    # Sentinel-sector counts and their width-1024 combinatorial imbalance.
    n = 1024
    assert 5 * n**4 / n**5 == 5 / n
    assert 10 * n**3 / n**5 == 10 / n**2
    assert 5 / n == 0.0048828125
    assert 10 / n**2 == 0.0000095367431640625

    # Number of unrestricted Hermite bond-count configurations through Q=24.
    expected_diagrams = {2: 25, 3: 2925, 4: 593775, 5: 131128140}
    for distinct, expected in expected_diagrams.items():
        edges = distinct * (distinct - 1) // 2
        assert comb(24 + edges, edges) == expected

    # Unique incoming cumulants on a generic set of five coordinates.
    assert comb(7, 3) + comb(8, 4) + comb(9, 5) == 231

    # Printed PC-TT5 arithmetic.
    nbar, layers, modes, rank = 1025, 15, 5, 16
    slots = nbar * (3 * rank**2 + 2 * rank)
    queries = 2 * layers * modes * nbar * rank**2
    query_work = queries * 8192
    qr_upper = 2 * layers * rank * modes * nbar * rank**3
    assert slots == 820000
    assert queries == 39360000
    assert query_work == 322437120000
    assert qr_upper == 10076160000

    affine = 25202700000
    allowance = 20000000000
    increment = affine + query_work + qr_upper + allowance
    assert increment == 377715980000
    trim = 2023256192375
    historical_eigh = 270582939648
    host = trim - historical_eigh
    total = host + increment
    margin = 2**41 - total
    assert host == 1752673252727
    assert total == 2130389232727
    assert margin == 68634022825

    no_eigh = json.loads((P2 / "public_noeigh_card_v1" /
                          "BUILD_RECEIPT.json").read_text())
    assert no_eigh["eigh_remaining"] == 1
    assert no_eigh["cholesky_free"] is True

    meter = json.loads((P2 / "public_trim_card_v1" /
                        "METER_RESULT.json").read_text())
    used = list(recursive_values(meter, "flops_used"))
    assert trim in used
    eigh_rows = list(recursive_values(meter, "linalg.eigh"))
    assert any(row.get("calls") == 14 and
               row.get("flop_cost") == historical_eigh
               for row in eigh_rows if isinstance(row, dict))

    receipt = json.loads((HERE / "AUDIT_RECEIPT.json").read_text())
    assert receipt["status"] == "HOLD_REPAIR_BEFORE_D0"
    assert receipt["access"]["targets_opened"] == 0
    assert receipt["access"]["graph_edits"] == 0
    assert receipt["access"]["runs_launched"] == 0
    for name, expected in receipt["audit_file_sha256"].items():
        assert sha256(HERE / name) == expected

    print("PASS_PC_TT5_INDEPENDENT_AUDIT_STATIC_VERIFICATION")


if __name__ == "__main__":
    main()
