#!/usr/bin/env python3
"""Independently verify the serialized mixed-pair rejection receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
EXPECTED_RESULTS_SHA256 = "91ddb8264df916ebd6ce4682b34c0f932df01cc5cfd92fe045c6428e4e021b6c"
EXPECTED_CANDIDATE_SHA256 = "66dcadb6d3229e83fcd57405f0618a51fd6d8f224d1881b2a5bc215de218646d"
PARENT_FLOPS = 2_179_742_954_469
BUDGET = 2_199_023_255_552


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-30)


def main(output: Path | None) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["results_hash"] = sha256(RESULTS) == EXPECTED_RESULTS_SHA256
    checks["candidate_hash"] = sha256(HERE / "estimator.py") == EXPECTED_CANDIDATE_SHA256
    checks["closed_no_hold"] = (
        data["status"] == "CLOSED_TARGET_FREE_GATE_FAILURE__NO_AUDIT_HOLD"
        and not data["verdict"]["source_only_audit_hold"]
    )
    checks["scope_clean"] = all(
        data["scope"][key] == 0
        for key in (
            "dataset_rows_opened", "targets_or_teachers_opened", "scores_computed",
            "fits", "samples", "production_shape_runs", "submissions",
            "canonical_graph_edits",
        )
    )
    checks["fixture_roster"] = [
        (row["name"], row["width"]) for row in data["rows"]
    ] == [
        ("official_he_w8_seed2026082408", 8),
        ("official_he_w32_seed2026082432", 32),
        ("official_he_w64_seed2026082464", 64),
        ("identity_float32", 8),
        ("near_identity_seed20260823_scale0p02_float32", 8),
        ("qr_seed20260823_scale0p35_float64", 8),
        ("canonical_pure_normal_seed17_scale0p12_float32", 8),
        ("alternate_identity_plus_qr_seed20260823_scale0p35_float64", 8),
        ("alternate_0p88_identity_plus_normal_seed17_scale0p12_over_sqrt8_float32", 8),
    ]
    by_name = {row["name"]: row for row in data["rows"]}
    checks["official_w8_eigh_failure"] = (
        not by_name["official_he_w8_seed2026082408"]["runs"]["candidate"]["success"]
        and by_name["official_he_w8_seed2026082408"]["runs"]["candidate"]["error"]["type"] == "LinAlgError"
        and by_name["official_he_w8_seed2026082408"]["runs"]["candidate"]["trace_events"] == 10
    )
    checks["official_w32_nonfinite"] = (
        not by_name["official_he_w32_seed2026082432"]["runs"]["candidate"]["success"]
        and by_name["official_he_w32_seed2026082432"]["runs"]["candidate"]["error"]["type"] == "FloatingPointError"
        and by_name["official_he_w32_seed2026082432"]["runs"]["candidate"]["trace_events"] == 15
    )
    checks["pure_normal_eigh_failure"] = (
        not by_name["canonical_pure_normal_seed17_scale0p12_float32"]["runs"]["candidate"]["success"]
        and by_name["canonical_pure_normal_seed17_scale0p12_float32"]["runs"]["candidate"]["trace_events"] == 6
    )
    w64 = by_name["official_he_w64_seed2026082464"]["output_metrics"]["candidate_minus_v4"]
    checks["official_w64_catastrophic_metric"] = (
        w64["available"]
        and close(w64["final_layer"]["mse"], 95_719_899_056.07812)
        and close(w64["final_layer"]["max_abs"], 1_140_393.8623046875)
        and close(w64["maximum_layer_mse"], 409_900_716_832.32983)
    )
    checks["failed_reference_matches_failure_class"] = all(
        row["runs"]["candidate"]["error"]["type"]
        == row["runs"]["independent_reference"]["error"]["type"]
        for row in data["rows"]
        if not row["runs"]["candidate"]["success"]
    )
    checks["successful_reference_bitwise"] = all(
        row["reference_candidate_output_bitwise_equal"]
        and row["reference_candidate_actions_bitwise_equal"]
        for row in data["rows"]
        if row["runs"]["candidate"]["success"]
    )
    checks["metric_rms_squared"] = all(
        close(layer["rms"] * layer["rms"], layer["mse"])
        for row in data["rows"]
        for comparison in row["output_metrics"].values()
        if comparison["available"]
        for layer in comparison["per_layer"]
    )
    projection = data["resource_projection"]
    a = int(projection["quadratic_delta"]["a_n2"])
    b = int(projection["quadratic_delta"]["b_n"])
    c = int(projection["quadratic_delta"]["c"])
    delta = a * 1024 * 1024 + b * 1024 + c
    checks["projection_arithmetic"] = (
        delta == projection["projected_candidate_minus_v4_flops"]
        and PARENT_FLOPS + delta == projection["projected_candidate_flops"]
        and BUDGET - (PARENT_FLOPS + delta) == projection["projected_headroom_flops"]
    )
    checks["projection_only_resource_pass"] = (
        projection["projected_candidate_flops"] <= BUDGET
        and projection["projected_peak_rss_bytes"] < 8 * 1024**3
        and data["gates"]["projected_flops_at_most_2pow41"]
        and data["gates"]["projected_static_peak_memory_below_8gib"]
    )
    checks["scientific_gates_fail"] = all(
        not data["gates"][name]
        for name in (
            "all_outputs_and_actions_finite",
            "official_law_scientific_materiality",
            "official_output_closer_than_q24",
            "stress_output_closer_than_q24",
            "all4_action_envelopes_closer_than_q24",
        )
    )
    checks["api_pass_only"] = data["gates"]["pinned_api_smoke"]

    failed = [name for name, passed in checks.items() if not passed]
    verification = {
        "schema": "fb-ec4-q32-fp32-pair-rejection-verification-v1",
        "status": "PASS_VERIFIED_REJECTION" if not failed else "FAIL",
        "results_sha256": sha256(RESULTS),
        "candidate_sha256": sha256(HERE / "estimator.py"),
        "checks": checks,
        "failed": failed,
        "claim": (
            "The exact mixed-pair spelling is source-legal and projected in-budget, "
            "but is scientifically rejected for official-law numerical instability "
            "and is not held for audit or promotion."
        ),
    }
    if output is not None:
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2, sort_keys=True))
    if failed:
        raise AssertionError(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.output)
