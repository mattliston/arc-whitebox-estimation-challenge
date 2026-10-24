#!/usr/bin/env python3
"""Verify the frozen P2R0C1D5R1 package before independent review."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRESEAL = HERE / "RECOVERY_DESCENDANT_PRESEAL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    preseal = json.loads(PRESEAL.read_text())
    assert preseal["status"] == "FROZEN_FOR_INDEPENDENT_RECOVERY_DESCENDANT_REVIEW"
    assert preseal["launch_authorization"] == "HELD"
    assert preseal["node"] == "P2R0C1D5R1"
    assert preseal["parent_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
    assert preseal["reference_estimator_calls_in_descendant"] == 0
    assert preseal["original_candidate_calls"] == 0
    assert preseal["descendant_candidate_calls"] == 0
    for relative, expected in preseal["sha256"].items():
        assert sha256((HERE / relative).resolve()) == expected, relative

    manifest = json.loads((HERE / "RECOVERY_DESCENDANT_MANIFEST.json").read_text())
    assert manifest["candidate_node"] == "P2R0C1D5R1"
    assert manifest["call_order"] == [35, 279, 35]
    assert manifest["candidate_source_sha256"] == preseal["sha256"]["estimator.py"]
    assert manifest["candidate_runner_sha256"] == preseal["sha256"]["run_recovery_descendant_candidate_gate.py"]
    assert manifest["recovered_reference_evidence_sha256"] == preseal["sha256"]["RECOVERED_REFERENCE_EVIDENCE.json"]
    assert manifest["process_protocol"]["reference_estimator_calls_in_descendant"] == 0
    assert manifest["process_protocol"]["original_candidate_calls"] == 0
    assert manifest["process_protocol"]["descendant_candidate_calls_at_freeze"] == 0

    for name in (
        "run_recovery_descendant_candidate_gate.py",
        "build_recovery_descendant_runner.py",
    ):
        tree = ast.parse((HERE / name).read_text())
        forbidden = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in {"false", "true", "null"}
        }
        assert forbidden == set(), (name, forbidden)

    evidence = json.loads((HERE / "RECOVERED_REFERENCE_EVIDENCE.json").read_text())
    assert evidence["status"] == "RECOVERED_REFERENCE_SCIENTIFIC_EVIDENCE"
    assert evidence["original_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
    assert evidence["original_process_exit_code"] == 1
    assert evidence["recovery_reference_estimator_calls"] == 0
    assert evidence["original_candidate_calls"] == 0
    assert evidence["candidate_calls_at_recovery"] == 0
    assert evidence["call_order"] == [35, 279, 35]
    assert len(evidence["trials"]) == 3
    assert all(row["gates"]["conjunctive_pass"] for row in evidence["trials"])

    assert preseal["absence_gates"] == {
        "REFERENCE_RESULT.json": True,
        "COMMON_DRIVER_RESULT.json": True,
        "candidate_rank35_first.npy": True,
        "candidate_rank279.npy": True,
        "candidate_rank35_repeat.npy": True,
        "RECOVERY_DESCENDANT_RESULT.json": True,
        "recovery_candidate_rank35_first.npy": True,
        "recovery_candidate_rank279.npy": True,
        "recovery_candidate_rank35_repeat.npy": True,
        "RECOVERY_DESCENDANT_LAUNCH_SEAL.json": True,
    }
    for relative, expected_absent in preseal["absence_gates"].items():
        assert expected_absent is True
        assert not (HERE / relative).exists(), relative
    assert preseal["scope_guards"] == {
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "network": False,
        "training": False,
    }
    print("PASS_RECOVERY_DESCENDANT_PRESEAL")


if __name__ == "__main__":
    main()
