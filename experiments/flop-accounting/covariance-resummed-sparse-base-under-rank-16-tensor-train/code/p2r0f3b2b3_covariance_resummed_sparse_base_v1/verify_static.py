#!/usr/bin/env python3
"""Static verifier for the B2B3 covariance-resummed constraint receipt."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    q = 25
    triples = sum(
        1
        for p in range(q)
        for r in range(q)
        for s in range(q)
        if (p + r + s) % 2 == 0
        and p + r >= s
        and p + s >= r
        and r + s >= p
    )
    if triples != 4147:
        raise AssertionError(f"triple-Hermite count changed: {triples}")

    path_cost = 3 * (2 * q * q) + 4 * q + 2 * q
    degree3_direct = 3 * triples
    if path_cost != 3900 or degree3_direct != 12441:
        raise AssertionError("tree contraction arithmetic changed")

    rbar = 0.7
    fourth_derivative_over_factorial = (
        (1.0 + 2.0 * rbar * rbar)
        / (2.0 * math.pi * (1.0 - rbar * rbar) ** 2.5)
        / math.factorial(4)
    )
    pair_eta = (1.0e-8 / fourth_derivative_over_factorial) ** 0.25
    five_tree_coefficient = 0.827735076388167
    five_eta = (1.0e-8 / five_tree_coefficient) ** 0.25
    if abs(fourth_derivative_over_factorial - 0.07068847946793874) > 1e-15:
        raise AssertionError("pair Taylor coefficient changed")
    if abs(pair_eta - 0.019393796758433206) > 1e-15:
        raise AssertionError("pair eta crossing changed")
    if abs(five_eta - 0.010484003569745515) > 1e-15:
        raise AssertionError("five-tree eta crossing changed")

    carrier = {
        "separate_scalar_tt": 362208,
        "shared_prefix_trie": 107744,
        "optimistic_varying_fiber": 61440,
    }
    ceiling = 9935
    if min(carrier.values()) <= ceiling:
        raise AssertionError("known carrier no longer exceeds ceiling")

    low_rank = {
        str(rank): {
            "nodes": q**rank,
            "raw_five_factor_subtotal": 5 * q**rank - 1,
        }
        for rank in (1, 2, 4)
    }
    expected_low_rank = {
        "1": {"nodes": 25, "raw_five_factor_subtotal": 124},
        "2": {"nodes": 625, "raw_five_factor_subtotal": 3124},
        "4": {"nodes": 390625, "raw_five_factor_subtotal": 1953124},
    }
    if low_rank != expected_low_rank:
        raise AssertionError("low-rank quadrature arithmetic changed")

    source_output = subprocess.check_output(
        [sys.executable, str(HERE / "summarize_existing_source_domain.py")],
        text=True,
    )
    source_actual = json.loads(source_output)
    source_expected = json.loads((HERE / "SOURCE_DOMAIN_SUMMARY.json").read_text())
    if source_actual != source_expected:
        raise AssertionError("source-domain summary does not reproduce")

    evidence = {
        "public_aug7_source_observable_tape_v1/GENERATION_RECEIPT.json":
            "d2e36181b09c0048791f130ba75897b2962c44eeab87b37086c8f4a6fcb5bfbd",
        "public_coincident_bond_census_v1/CENSUS_COUNTS.json":
            "a550aac38157cafd98c8693d165d5d9c2224e38075ade4cc87a4b0cde9c6a8d7",
        "p2r0f3b2b2_independent_gaussian_linked_cluster_grade3_v1/CONSTRAINT_RECEIPT.md":
            "d6275c2ace74294c4d9c6fcc6b5126dbbb9725e4dac51aeca1affba0e98c7cba",
        "p2r0f3b2b2_ig_lc3_reset_derivation_v1/DERIVATION.md":
            "7c922b58d0baae6a90abd2edce1f19f03b5f5c0f219561630307ee07697332c1",
        "public_q24_native_transport_energy_final_workspace_v4/estimator.py":
            "2fd2d40d4e30cc32269afb0113e244ee76c0a51005bce166c7625510f66a8a76",
    }
    for relative, expected in evidence.items():
        actual = sha256_file(P2 / relative)
        if actual != expected:
            raise AssertionError(f"evidence hash changed: {relative}")

    result = {
        "schema": "p2-b2b3-covariance-resummed-static-verification-v1",
        "status": "PASS_STATIC_CONSTRAINT_VERIFICATION",
        "targets_read": 0,
        "graph_edits": 0,
        "q24_orders": q,
        "tree": {
            "path_raw_moment_cost": path_cost,
            "degree3_nonzero_triples": triples,
            "degree3_direct_cost": degree3_direct,
        },
        "accuracy_controls": {
            "pair_fourth_derivative_over_factorial_at_rbar_0.7":
                fourth_derivative_over_factorial,
            "pair_eta_for_1e-8": pair_eta,
            "five_tree_coefficient": five_tree_coefficient,
            "five_eta_for_1e-8": five_eta,
        },
        "integrated_ceiling": ceiling,
        "carrier_costs": carrier,
        "minimum_known_carrier_over_ceiling": min(carrier.values()) / ceiling,
        "low_rank_q24": low_rank,
        "source_summary_reproduced": True,
        "evidence_hashes_verified": evidence,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
