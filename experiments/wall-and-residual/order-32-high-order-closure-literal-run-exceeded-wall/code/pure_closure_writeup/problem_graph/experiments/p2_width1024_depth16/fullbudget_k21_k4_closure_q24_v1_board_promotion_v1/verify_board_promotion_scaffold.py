#!/usr/bin/env python3
"""Read-only verifier for the fail-closed Q24 board-promotion scaffold.

Default mode verifies that the immutable rationale is intact and all Q24
promotion fields remain null/HOLD. ``--require-ready`` is the later mandatory
gate: it validates populated hashes, resources, archive bytes, and exact argv,
and therefore intentionally fails on the present scaffold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
MANIFEST_PATH = HERE / "PROMOTION_MANIFEST.json"
TEMPLATE_PATH = HERE / "SUBMISSION_COMMAND_TEMPLATE.json"
AUDIT_PATH = HERE / "STATIC_SCAFFOLD_AUDIT.json"

HOLD_STATUS = "HOLD_Q24_SOURCE_METER_FREEZE_AUDIT_AND_ARCHIVE_NOT_BOUND__NOT_SUBMISSION_READY"
READY_STATUS = "READY_ONE_ATTEMPT_Q24_BOARD_VALIDATION"
EXPECTED_EVIDENCE = {
    "board_attempt_328199": "066574b553aecde40fd30213a362d3598efbf355b3b463b84af3282fd3b1aff8",
    "independent_board_result_audit": "455332232e76691d7a40349bbfbae890a3fc3fbbae8707fc4254debf6d609dc6",
    "a1_meter_result": "276c34a9ad3c8f5ccaec42f98b970543b21275d2cf31955333714483b00c8e27",
    "independent_a1_audit": "02d16c3f2bc1a6476dfb2ee18aa8a1add8ec8b8d5e7608f011d4923b54bc00ed",
}
ARGV_TEMPLATE = [
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/bin/whest",
    "submit",
    "${ARCHIVE_PATH_FROM_REPO_ROOT}",
    "--challenge",
    "arc-white-box-estimation-challenge-2026",
    "--description",
    "${FINAL_DESCRIPTION}",
    "--watch",
    "--watch-timeout",
    "3600",
    "--format",
    "json",
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def resolve_here(relative: str) -> Path:
    return (HERE / relative).resolve()


def nested_get(value: Any, dotted: str) -> Any:
    current = value
    for token in dotted.split("."):
        if "[" in token:
            name, index_text = token[:-1].split("[")
            current = current[name][int(index_text)]
        else:
            current = current[token]
    return current


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verify_common(manifest: dict[str, Any], template: dict[str, Any]) -> dict[str, bool]:
    evidence = manifest["rationale_evidence"]
    checks: dict[str, bool] = {}
    for key, expected_hash in EXPECTED_EVIDENCE.items():
        binding = evidence[key]
        path = resolve_here(binding["path"])
        checks[f"evidence_hash:{key}"] = (
            binding["sha256"] == expected_hash
            and path.is_file()
            and sha256_file(path) == expected_hash
        )

    board = load(resolve_here(evidence["board_attempt_328199"]["path"]))
    board_audit = load(resolve_here(evidence["independent_board_result_audit"]["path"]))
    a1 = load(resolve_here(evidence["a1_meter_result"]["path"]))
    a1_audit = load(resolve_here(evidence["independent_a1_audit"]["path"]))
    checks["board_partial_e8_not_aggregate"] = (
        board["submission"]["id"] == 328199
        and board["submission"]["status"] == "FAILED"
        and board["evaluation"]["progress"]["total_mlps"] == 100
        and board["evaluation"]["progress"]["mlps_completed"] == 2
        and board["evaluation"]["aggregate"] is None
        and board["submission"]["numeric_aggregate_board_score_obtained"] is False
        and [row["final_layer_mse"] for row in board["completed_rows"]] == [7.661e-8, 7.071e-8]
        and board["partial_completed_only_summary"]["mean_final_layer_mse"] == 7.366e-8
        and "NOT_AN_AGGREGATE" in board["partial_completed_only_summary"]["classification"]
        and board_audit["verdict_replay"]["scientific_or_aggregate_promotion_authorized"] is False
    )
    comparison = a1["comparison_to_frozen_v4"]
    checks["a1_wall_negative"] = (
        comparison["flop_delta"] < 0
        and comparison["backend_delta_seconds"] == 0.7140168910846114
        and comparison["wall_delta_seconds"] == 0.4547270843759179
        and a1["interpretation"]["board_wall_claim"].startswith("HOLD")
        and a1_audit["verdict"]["robust_wall_safe_claim"] == "HOLD"
        and "wall-targeted full-V4 Q-channel reduction" in a1_audit["verdict"]["recommended_priority"]
    )
    checks["criterion_frozen"] = manifest["primary_criterion"] == {
        "candidate_count": 1,
        "configured_mlps": 100,
        "required_completed_mlps": 100,
        "required_failed_mlps": 0,
        "metric": "observed_unrounded_board_aggregate_raw_final_layer_mse",
        "operator": "<",
        "threshold": 1e-8,
        "equality_passes": False,
        "partial_rows_can_rescue": False,
        "adjusted_or_secondary_metric_can_rescue": False,
    }
    checks["one_candidate_one_attempt_no_retry"] = (
        manifest["candidate"]["count"] == 1
        and manifest["authorization"]["max_attempts_this_preregistration"] == 1
        and manifest["outcome_law"]["tuning"] is False
        and manifest["outcome_law"]["fallback"] is False
        and manifest["outcome_law"]["retry"] is False
        and manifest["outcome_law"]["regrade_farming"] is False
    )
    checks["command_template_exact"] = (
        template["status"] == "HOLD_LITERAL_PLACEHOLDERS_NOT_EXECUTABLE"
        and template["argv_template"] == ARGV_TEMPLATE
        and template["attempt_limit"] == 1
        and template["retry"] is False
    )
    return checks


def verify_hold(manifest: dict[str, Any], audit: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["hold_status"] = manifest["status"] == HOLD_STATUS
    checks["all_required_final_fields_null"] = all(
        nested_get(manifest, dotted) is None for dotted in manifest["required_final_fields"]
    )
    checks["artifact_prerequisites_false"] = (
        manifest["authorization"]["artifact_prerequisites_satisfied"] is False
        and manifest["authorization"]["submission_authorized_while_hold"] is False
    )
    checks["submission_absent"] = manifest["submission"] == {
        "working_directory": "/Users/matt/arc-white-box-estimation",
        "command_argv": None,
        "attempt_state": "SCAFFOLD_HOLD_NOT_PREPARED",
        "submission_id": None,
        "board_result_path": None,
        "board_result_sha256": None,
    }
    checks["negative_controls"] = (
        audit["status"] == "PASS_SCAFFOLD_INTEGRITY__HOLD_NOT_SUBMISSION_READY"
        and all(value == 0 for value in audit["negative_controls"].values())
    )
    checks["forbidden_complete"] = {
        "TARGET_OR_DATASET_ACCESS",
        "SCORE_OR_PRIVATE_BOARD_ACCESS",
        "NETWORK_ACCESS",
        "CREDENTIAL_ACCESS",
        "PACKAGING",
        "SUBMISSION",
        "HASH_BINDING_FROM_UNFROZEN_WORKING_BYTES",
        "GRAPH_MUTATION",
    }.issubset(set(manifest["forbidden_now"]))
    return checks


def verify_ready(manifest: dict[str, Any], template: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["ready_status"] = manifest["status"] == READY_STATUS
    all_populated = all(
        nested_get(manifest, dotted) is not None for dotted in manifest["required_final_fields"]
    )
    checks["all_required_final_fields_populated"] = all_populated
    if not all_populated:
        return checks

    candidate = manifest["candidate"]
    source = resolve_here(candidate["source"]["path"])
    checks["source_bytes_hash"] = (
        source.is_file()
        and source.stat().st_size == candidate["source"]["bytes"]
        and sha256_file(source) == candidate["source"]["sha256"]
    )
    for key in ("source_preregister", "source_manifest", "freeze_receipt", "independent_source_and_meter_audit"):
        binding = candidate[key]
        path = resolve_here(binding["path"])
        checks[f"candidate_hash:{key}"] = path.is_file() and sha256_file(path) == binding["sha256"]
    action = candidate["canonical_v4_suite_action_envelope"]
    action_path = resolve_here(action["path"])
    checks["canonical_v4_suite_action_envelope_hash"] = (
        action_path.is_file() and sha256_file(action_path) == action["sha256"]
    )
    checks["canonical_v4_suite_registered_gate"] = (
        isinstance(action["status"], str)
        and bool(action["status"])
        and finite_number(action["final_prediction_disagreement_mse"])
        and action["registered_gate_pass"] is True
    )

    meter = manifest["meter_gate"]
    meter_path = resolve_here(meter["path"])
    checks["meter_receipt_hash"] = meter_path.is_file() and sha256_file(meter_path) == meter["sha256"]
    limits = manifest["phase2_interface"]
    checks["literal_resource_values"] = (
        meter["literal_phase2_resource_pass"] is True
        and meter["failed_mlps"] == 0
        and meter["output_shape"] == [16, 1024]
        and meter["output_dtype"] == "float32"
        and meter["output_all_finite"] is True
        and finite_number(meter["flops_used"])
        and meter["flops_used"] <= limits["budget_flops_max"]
        and finite_number(meter["wall_time_s"])
        and meter["wall_time_s"] < limits["wall_time_s_strict_max"]
        and finite_number(meter["residual_time_s"])
        and meter["residual_time_s"] <= limits["residual_time_s_max"]
        and finite_number(meter["descendant_peak_rss_bytes"])
        and meter["descendant_peak_rss_bytes"] < limits["descendant_rss_bytes_strict_max"]
    )
    robust_expected = (
        finite_number(meter["backend_time_s"])
        and meter["backend_time_s"] <= meter["robust_local_backend_s_max"]
        and meter["residual_time_s"] <= meter["robust_local_residual_s_max"]
    )
    checks["robust_classification_truthful"] = meter["robust_wall_engineering_classification"] == (
        "PASS_ROBUST_LOCAL_ENGINEERING_GATE" if robust_expected else "HOLD_ROBUST_LOCAL_ENGINEERING_GATE"
    )

    archive_binding = candidate["archive"]
    archive = REPO / archive_binding["path_from_repo_root"]
    checks["archive_bytes_hash"] = (
        archive.is_file()
        and archive.stat().st_size == archive_binding["bytes"]
        and sha256_file(archive) == archive_binding["sha256"]
    )
    if archive.is_file():
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            payloads = {
                member.name: handle.extractfile(member).read()
                for member in members
                if member.isfile()
            }
        expected_members = archive_binding["members_exact"]
        checks["archive_member_names_types"] = (
            [member.name for member in members] == ["estimator.py", "manifest.json"]
            and all(member.isfile() for member in members)
            and [item["name"] for item in expected_members] == ["estimator.py", "manifest.json"]
            and all(item["type"] == "regular" for item in expected_members)
        )
        checks["archive_member_bytes_hashes"] = all(
            len(payloads[item["name"]]) == item["bytes"]
            and sha256_bytes(payloads[item["name"]]) == item["sha256"]
            for item in expected_members
        )
        checks["archive_estimator_exact_source"] = payloads.get("estimator.py") == source.read_bytes()
    else:
        checks["archive_member_names_types"] = False
        checks["archive_member_bytes_hashes"] = False
        checks["archive_estimator_exact_source"] = False

    final_description = f"{candidate['tag']} one-attempt wall-safe rich-closure validation v1"
    expected_argv = [
        archive_binding["path_from_repo_root"] if value == "${ARCHIVE_PATH_FROM_REPO_ROOT}"
        else final_description if value == "${FINAL_DESCRIPTION}"
        else value
        for value in template["argv_template"]
    ]
    checks["exact_final_argv"] = manifest["submission"]["command_argv"] == expected_argv
    checks["ready_attempt_state"] = (
        manifest["submission"]["attempt_state"] == "PREPARED_NOT_ATTEMPTED"
        and manifest["submission"]["submission_id"] is None
        and manifest["authorization"]["artifact_prerequisites_satisfied"] is True
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    manifest = load(MANIFEST_PATH)
    template = load(TEMPLATE_PATH)
    audit = load(AUDIT_PATH)
    checks = verify_common(manifest, template)
    checks.update(verify_ready(manifest, template) if args.require_ready else verify_hold(manifest, audit))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Q24 board-promotion verification failed: {failed}")
    print(json.dumps({
        "status": (
            "PASS_Q24_BOARD_PROMOTION_READY" if args.require_ready
            else "PASS_Q24_BOARD_PROMOTION_SCAFFOLD__HOLD_NOT_READY"
        ),
        "checks": checks,
        "submission_attempt_state": manifest["submission"]["attempt_state"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
