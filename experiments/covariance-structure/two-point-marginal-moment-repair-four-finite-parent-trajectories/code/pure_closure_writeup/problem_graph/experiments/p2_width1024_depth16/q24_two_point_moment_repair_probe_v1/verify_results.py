#!/usr/bin/env python3
"""Independent artifact replay for the frozen two-point repair rejection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def digest(relative: str) -> str:
    return hashlib.sha256((HERE / relative).read_bytes()).hexdigest()


def main() -> int:
    receipt = json.loads((HERE / "REJECTION_RECEIPT_V1.json").read_text())
    results = json.loads((HERE / "runs/probe_v1/results.json").read_text())
    assert receipt["status"] == "REJECTED_THIS_FORM__BROADER_MOMENT_REPAIR_OPEN"
    for name in ("preregister", "runner", "results"):
        binding = receipt["bindings"][name]
        assert digest(binding["path"]) == binding["sha256"]
    assert results["status"] == "REJECT_MECHANISM_PROBE"
    gates = results["gates"]
    assert gates["mechanism_pass"] is False
    assert gates["no_parent_finite_to_repair_nonfinite_regression"] is False
    assert gates["all_repaired_post_A_and_B_represented_nonnegative"] is False
    assert gates["repair_activation_count"] == 4741
    assert gates["parent_failure_recovered_count"] == 0
    assert len(results["rows"]) == 10
    regressions = [
        row["fixture"] for row in results["rows"]
        if row["parent"]["outcome"] == "FINITE"
        and row["repair"]["outcome"] != "FINITE"
    ]
    assert regressions == receipt["decisive_failures"][
        "parent_finite_to_repair_nonfinite_regressions"
    ]
    post_a = sum(
        sum(row["repair"].get("post_negative_A_counts_by_reset", []))
        for row in results["rows"]
    )
    post_b = sum(
        sum(row["repair"].get("post_negative_B_counts_by_reset", []))
        for row in results["rows"]
    )
    assert post_a == receipt["decisive_failures"]["post_repair_negative_A_count_total"]
    assert post_b == receipt["decisive_failures"]["post_repair_negative_B_count_total"]
    assert all(value == 0 for value in receipt["access"].values())
    print("PASS_VERIFIED_TWO_POINT_REPAIR_REJECTION_V1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
