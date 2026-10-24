#!/usr/bin/env python3
"""Verify the final P2R0C1D5R1 launch seal before the only candidate run."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEAL = HERE / "RECOVERY_DESCENDANT_LAUNCH_SEAL.json"
PRESEAL_SHA256 = "a2d4a2263543d6d86c7e2db0c3f17baef6723ee93f49171a83ae83c94893f1e8"
INDEPENDENT_AUDIT_SHA256 = "d1229b645209a01ab0261119eb0386f4ccee012a2d23efe0c7dd52984e7f1beb"
INDEPENDENT_VERIFIER_SHA256 = "ebd0fa813bbca275058f69bfe04768dc9c4f2873000d7f116435af7317cb35df"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    seal = json.loads(SEAL.read_text())
    assert seal["status"] == "FROZEN_RECOVERY_DESCENDANT_AUTHORIZED"
    assert seal["authorization"] == "EXACT_ONE_SHOT_TARGET_FREE_CANDIDATE"
    assert seal["node"] == "P2R0C1D5R1"
    assert seal["parent_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
    assert seal["reference_estimator_calls_in_descendant"] == 0
    assert seal["candidate_calls_before_launch"] == 0
    assert seal["call_order"] == [35, 279, 35]
    assert seal["sha256"]["RECOVERY_DESCENDANT_PRESEAL.json"] == PRESEAL_SHA256
    assert seal["sha256"]["INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_PRERUN_AUDIT.md"] == INDEPENDENT_AUDIT_SHA256
    assert seal["sha256"]["verify_independent_recovery_descendant_prerun.py"] == INDEPENDENT_VERIFIER_SHA256
    for relative, expected in seal["sha256"].items():
        assert sha256((HERE / relative).resolve()) == expected, relative

    preseal = json.loads((HERE / "RECOVERY_DESCENDANT_PRESEAL.json").read_text())
    assert preseal["status"] == "FROZEN_FOR_INDEPENDENT_RECOVERY_DESCENDANT_REVIEW"
    assert preseal["launch_authorization"] == "HELD"
    assert preseal["reference_estimator_calls_in_descendant"] == 0
    assert preseal["descendant_candidate_calls"] == 0
    for relative, expected in preseal["sha256"].items():
        assert seal["sha256"][relative] == expected

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

    manifest = json.loads((HERE / "RECOVERY_DESCENDANT_MANIFEST.json").read_text())
    evidence = json.loads((HERE / "RECOVERED_REFERENCE_EVIDENCE.json").read_text())
    assert manifest["candidate_node"] == "P2R0C1D5R1"
    assert manifest["candidate_source_sha256"] == seal["sha256"]["estimator.py"]
    assert manifest["candidate_runner_sha256"] == seal["sha256"]["run_recovery_descendant_candidate_gate.py"]
    assert manifest["recovered_reference_evidence_sha256"] == seal["sha256"]["RECOVERED_REFERENCE_EVIDENCE.json"]
    assert evidence["original_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
    assert evidence["original_process_exit_code"] == 1
    assert evidence["recovery_reference_estimator_calls"] == 0
    assert evidence["original_candidate_calls"] == 0
    assert evidence["candidate_calls_at_recovery"] == 0
    assert evidence["call_order"] == [35, 279, 35]
    assert all(row["gates"]["conjunctive_pass"] for row in evidence["trials"])

    assert not (HERE / "REFERENCE_RESULT.json").exists()
    assert not (HERE / "COMMON_DRIVER_RESULT.json").exists()
    assert not (HERE / "RECOVERY_DESCENDANT_RESULT.json").exists()
    assert not any(HERE.glob("candidate_rank*.npy"))
    assert not any(HERE.glob("recovery_candidate_rank*.npy"))
    assert seal["scope_guards"] == {
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "network": False,
        "training": False,
    }
    print("PASS_RECOVERY_DESCENDANT_LAUNCH_SEAL_PRE_CANDIDATE")


if __name__ == "__main__":
    main()
