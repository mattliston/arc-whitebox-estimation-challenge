#!/usr/bin/env python3
"""Sequentially generate all 48 frozen D1/cap panel outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PREREGISTRATION = HERE / "PREREGISTRATION.json"
ROSTER_AUTHORITY = HERE.parent / "public_q24_cp4_cap_disjoint16_validation_v1/PREREGISTRATION.json"
RUNNER = HERE / "run_arm.py"
RESULT = HERE / "TARGET_FREE_RUN_RESULT.json"
PREREGISTRATION_SHA256 = "aa93dc769d3388512a6262d6ff77b10438b9796c183369ec6a946b19e40477fd"
ROSTER_AUTHORITY_SHA256 = "2cbaedbf506d345234e744ca1e5521faf6823a922659ea177f04781e847a6c51"
RUNNER_SHA256 = "ff30f1a901ce363425846c073a7e788d98df11fc58533e5c96c6ee0c9b9557fc"
ARMS = ("d1", "cap768", "cap512")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise AssertionError("preregistration changed")
    if sha256(ROSTER_AUTHORITY) != ROSTER_AUTHORITY_SHA256:
        raise AssertionError("roster authority changed")
    if sha256(RUNNER) != RUNNER_SHA256:
        raise AssertionError("isolated runner changed")
    roster = json.loads(ROSTER_AUTHORITY.read_text(encoding="utf-8"))["roster"]
    ranks = [int(row["role_rank"]) for row in roster]
    completed = []
    for arm in ARMS:
        for rank in ranks:
            subprocess.run(
                [sys.executable, str(RUNNER), "--arm", arm, "--rank", str(rank)],
                check=True,
            )
            receipt = HERE / "outputs" / f"{arm}_rank{rank}" / "RECEIPT.json"
            completed.append(
                {"arm": arm, "role_rank": rank, "receipt_sha256": sha256(receipt)}
            )
            if len(completed) % 8 == 0:
                print(json.dumps({"target_free_completed": len(completed), "total": 48}), flush=True)
    result = {
        "schema": "q24-d1-cp4-cap-disjoint16-target-free-run-v1",
        "status": "PASS_ALL_48_TARGET_FREE_OUTPUTS",
        "completed": completed,
        "bindings": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "roster_authority_sha256": sha256(ROSTER_AUTHORITY),
            "runner_sha256": sha256(RUNNER),
            "orchestrator_sha256": sha256(Path(__file__)),
        },
        "target_columns_opened": [],
        "board_submission": False,
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "completed": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
