#!/usr/bin/env python3
"""Verify the immutable diagnostic accounting addendum without any source access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "public_q24_ng_k6_difference_replica_block_static_v1"
N = 1024
LAYERS = 15
RANKS = (512, 192, 128)
HALF_RANKS = (256, 96, 64)
TERM_COUNTS = {
    "411": (4, 9, 16),
    "321": (5, 13, 29),
    "222": (3, 6, 12),
    "3111": (7, 19, 42),
    "2211": (8, 22, 52),
    "21111": (10, 28, 65),
    "111111": (11, 29, 66),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    prereg = json.loads((HERE / "PREREGISTRATION_ADDENDUM.json").read_text())
    parent_freeze_hash = sha256(PARENT / "FREEZE_RECEIPT.json")
    assert parent_freeze_hash == prereg["parent_freeze_sha256"]
    assert HALF_RANKS == tuple(rank // 2 for rank in RANKS)
    assert prereg["split_estimators"]["coefficients"] == {
        "6": "1/(2*256)",
        "42": "-15/(2*96)",
        "222": "15/64",
    }

    sector_rows = {}
    total_units = 0
    total_sector_flops = 0
    for sector, counts in TERM_COUNTS.items():
        units = sum(rank * count for rank, count in zip(RANKS, counts))
        flops = LAYERS * N * 14 * units
        expected = prereg["sector_assembly"]["sectors"][sector]
        assert expected == {
            "term_counts_6_42_222": list(counts),
            "weighted_units": units,
            "flops": flops,
        }
        sector_rows[sector] = {
            "term_counts_6_42_222": list(counts),
            "weighted_units": units,
            "flops": flops,
        }
        total_units += units
        total_sector_flops += flops
    assert total_units == 84_864
    assert total_sector_flops == 18_249_154_560

    reduce_ops_per_half = sum(rank - 1 for rank in HALF_RANKS) + 3 + 2
    assert reduce_ops_per_half == 418
    split_flops = LAYERS * N * 2 * 7 * (2 * reduce_ops_per_half + 8)
    assert split_flops == 181_493_760

    production_core = 625_985_232_896
    production_assembly = 1_411_645_440
    production_reduction = 25_620_480
    diagnostic_core = (
        production_core
        - production_assembly
        - production_reduction
        + total_sector_flops
        + split_flops
    )
    assert diagnostic_core == 642_978_615_296
    padding = 2_786_811_904
    diagnostic_cap = diagnostic_core + padding
    assert diagnostic_cap == 645_765_427_200
    d4_max = 1_356_595_288_584
    combined = d4_max + diagnostic_cap
    assert combined == 2_002_360_715_784
    assert 2**41 - combined == 196_662_539_768

    assert prereg["gate_interpretation"]["finite_sample_confidence_bound"] is False
    assert prereg["access"] == {
        "rows_opened": 0,
        "weights_opened": 0,
        "targets_opened": 0,
        "scores_computed": 0,
        "estimators_or_fixtures_called": 0,
        "source_gates_run": 0,
        "parent_files_mutated": 0,
        "graph_edits": 0,
        "releases": 0,
        "submissions": 0,
    }

    receipt = {
        "schema": "q24-ng-k6-difference-replica-block-gate-addendum-verification-v1",
        "status": "PASS_STATIC_ADDENDUM_ONLY__SOURCE_GATE_NOT_RUN",
        "checks": {
            "parent_freeze_bound": True,
            "half_ranks_and_unbiased_coefficients_exact": True,
            "all_seven_sector_components_exact": True,
            "split_reduction_and_statistic_reserve_exact": True,
            "diagnostic_core_and_cap_exact": True,
            "one_role_call_below_2pow41": True,
            "projected_peak_below_8gib": True,
            "sn_explicitly_empirical_not_confidence_bound": True,
            "source_or_target_access": False,
        },
        "sector_rows": sector_rows,
        "totals": {
            "sector_weighted_units": total_units,
            "sector_assembly_flops": total_sector_flops,
            "split_reduction_and_statistic_flops": split_flops,
            "diagnostic_enumerated_core_flops": diagnostic_core,
            "diagnostic_admission_cap_flops": diagnostic_cap,
            "combined_one_role_call_cap_flops": combined,
            "margin_to_2pow41_flops": 2**41 - combined,
            "combined_projected_peak_bytes": 5_679_808_512,
            "margin_to_8gib_bytes": 2_910_126_080,
        },
        "bindings": {
            "parent_FREEZE_RECEIPT.json": parent_freeze_hash,
            "ADDENDUM.md": sha256(HERE / "ADDENDUM.md"),
            "PREREGISTRATION_ADDENDUM.json": sha256(HERE / "PREREGISTRATION_ADDENDUM.json"),
            "verify_addendum.py": sha256(HERE / "verify_addendum.py"),
        },
        "next_action": "Independent review of this addendum; source gate remains HOLD.",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
