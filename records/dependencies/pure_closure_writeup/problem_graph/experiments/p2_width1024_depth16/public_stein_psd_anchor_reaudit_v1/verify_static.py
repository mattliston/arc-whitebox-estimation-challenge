#!/usr/bin/env python3
"""Verify the bounded Stein PSD-anchor re-audit without opening any source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RECEIPT = HERE / "STEIN_PSD_ANCHOR_REAUDIT.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["status"] == "PASS_BOUNDED_HOLD_EXTERNAL_SOURCE_PROVENANCE"
    assert receipt["decision"] == (
        "HOLD_NO_PRODUCTION_S0__FORMAL_SIGNED_FIRST_ORDER_JET_ONLY"
    )
    assert receipt["psd_obstruction"]["conclusion"].startswith("No PSD anchor")
    bounds = receipt["lower_bounds_not_complete_bills"]
    host = bounds["host_flops_used_for_projection"]
    assert host + 51_539_607_552 == bounds[
        "m512_r3_16384_r2_1024_score_total_flops_at_least"
    ]
    assert host + 103_079_215_104 == bounds[
        "m512_r3_16384_r2_1024_transport_total_flops_at_least"
    ]
    assert bounds["m512_transport_state_transport_flops"] == 2 * bounds[
        "m512_score_state_transport_flops"
    ]
    for relative, expected in receipt["evidence_sha256"].items():
        assert digest(ROOT / relative) == expected
    assert receipt["firewall"] == {
        "estimator_called": False,
        "graph_mutated_before_receipt": False,
        "rank_or_row_selected": False,
        "release_or_run_created": False,
        "submission_made": False,
        "target_score_weights_or_names_read": False,
    }
    print("PASS_BOUNDED_HOLD_EXTERNAL_SOURCE_PROVENANCE")


if __name__ == "__main__":
    main()
