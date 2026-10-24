#!/usr/bin/env python3
"""Independent artifact replay for the frozen Gaussian-fallback rejection."""

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
    assert results["status"] == "REJECT_CONSTRUCTION_PROBE"
    gates = results["gates"]
    assert gates["construction_pass"] is False
    assert gates["no_parent_finite_regression"] is False
    assert gates["all_returned_r3_r4_represented_nonnegative"] is False
    assert gates["activation_count"] == 1609
    assert gates["parent_failure_recovered_count"] == 2
    assert len(results["rows"]) == 10
    regressions = [
        row["fixture"] for row in results["rows"]
        if row["parent"]["outcome"] == "FINITE"
        and row["fallback"]["outcome"] != "FINITE"
    ]
    assert regressions == receipt["decisive_failures"][
        "parent_finite_to_fallback_nonfinite_regressions"
    ]
    post_r3 = sum(
        sum(row["fallback"].get("negative_r3_counts_by_reset", []))
        for row in results["rows"]
    )
    post_r4 = sum(
        sum(row["fallback"].get("negative_r4_counts_by_reset", []))
        for row in results["rows"]
    )
    assert post_r3 == receipt["decisive_failures"]["post_fallback_negative_r3_count_total"]
    assert post_r4 == receipt["decisive_failures"]["post_fallback_negative_r4_count_total"]
    pure = next(row for row in results["rows"]
                if row["fixture"] == "pure_normal_seed17_scale0p12_float32")
    he32 = next(row for row in results["rows"]
                if row["fixture"] == "official_he_width32_seed20260825")
    assert pure["finite_pair_disagreement"]["final_layer_mse"] == receipt[
        "decisive_failures"
    ]["pure_normal_final_disagreement_mse"]
    assert he32["finite_pair_disagreement"]["final_layer_mse"] == receipt[
        "decisive_failures"
    ]["official_he_width32_seed20260825_final_disagreement_mse"]
    assert all(value == 0 for value in receipt["access"].values())
    print("PASS_VERIFIED_NEGATIVE_MOMENT_GAUSSIAN_FALLBACK_REJECTION_V1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
