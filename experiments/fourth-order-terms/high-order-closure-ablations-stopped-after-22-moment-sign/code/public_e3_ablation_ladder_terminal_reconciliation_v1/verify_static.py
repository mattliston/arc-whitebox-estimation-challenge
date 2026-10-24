#!/usr/bin/env python3
"""Verify the read-only E3 ablation-ladder terminal reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RECEIPT = HERE / "TERMINAL_RECONCILIATION.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["status"] == "PASS_READ_ONLY_GRAPH_RECONCILIATION_AUTHORITY"
    for relative, expected in receipt["evidence_sha256"].items():
        assert digest(ROOT / relative) == expected
    statuses = {
        node_id: row["new_status"]
        for node_id, row in receipt["stale_leaf_resolutions"].items()
    }
    assert statuses == {
        "P2R0E3A0": "solved",
        "P2R0E3A1": "closed_this_form",
        "P2R0E3A3": "closed_this_form",
        "P2R0E3A4": "closed_this_form",
        "P2R0E3A5": "closed_measured",
        "P2R0E3A7": "solved",
    }
    board = receipt["q24_board_result"]
    assert board["raw_final_mse"] / 1e-8 == board["ratio_to_1e_minus_8_gate"]
    assert receipt["firewall"] == {
        "estimator_or_row_called": False,
        "new_target_or_score_read": False,
        "release_or_run_created": False,
        "submission_made": False,
    }
    print("PASS_READ_ONLY_GRAPH_RECONCILIATION_AUTHORITY")


if __name__ == "__main__":
    main()
