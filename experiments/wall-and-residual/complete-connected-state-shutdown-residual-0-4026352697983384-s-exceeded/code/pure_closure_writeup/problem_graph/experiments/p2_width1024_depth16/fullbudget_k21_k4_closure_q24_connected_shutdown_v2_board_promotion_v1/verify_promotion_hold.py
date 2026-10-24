#!/usr/bin/env python3
"""Read-only fail-closed verifier for the CS2 board-promotion scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "pure_closure_writeup").is_dir())
MANIFEST_PATH = HERE / "PROMOTION_MANIFEST.json"
REJECTION_PATH = HERE / "BOARD_PROMOTION_GATE_RESULT_V1.json"

HOLD_STATUS = (
    "HOLD_FULLWIDTH_COMPARATOR_AND_LITERAL_METER_RECEIPTS_NULL__"
    "NO_PACKAGE_OR_SUBMISSION_AUTHORITY"
)
READY_STATUS = "READY_ONE_ATTEMPT_CS2_EXPLORATORY_BOARD_VALIDATION"
STRICT_AUDIT_STATUS = (
    "PASS_INDEPENDENT_EXACT_BYTE_SOURCE_AND_TARGET_FREE_CARRIER_AUDIT__"
    "IMMUTABLE_HOLD__FULLWIDTH_UNMEASURED"
)
COMPARATOR_STATUS = "PASS_TARGET_FREE_FULLWIDTH_STABILITY__LITERAL_METER_AND_ACCURACY_HOLD"
METER_STATUS = "PASS_PHASE2_SUBPROCESS_INTERFACE_AND_METER__ACCURACY_UNSCORED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve(path_text: str) -> Path:
    return (HERE / path_text).resolve()


def require(condition: bool, label: str, checks: dict[str, bool]) -> None:
    checks[label] = bool(condition)
    if not condition:
        raise AssertionError(label)


def verify_binding(binding: dict, label: str, checks: dict[str, bool]) -> Path:
    path = resolve(binding["path"])
    require(path.is_file(), f"{label}_exists", checks)
    require(sha256(path) == binding["sha256"], f"{label}_sha256", checks)
    return path


def nested_value(document: object, dotted: str) -> object:
    cursor = document
    for token in dotted.split("."):
        match = re.fullmatch(r"([^\[]+)(?:\[(\d+)\])?", token)
        if not match or not isinstance(cursor, dict):
            raise KeyError(dotted)
        cursor = cursor[match.group(1)]
        if match.group(2) is not None:
            if not isinstance(cursor, list):
                raise KeyError(dotted)
            cursor = cursor[int(match.group(2))]
    return cursor


def verify_core(manifest: dict, checks: dict[str, bool]) -> None:
    candidate = manifest["candidate"]
    require(candidate["count"] == 1, "exactly_one_candidate", checks)
    source_path = verify_binding(candidate["source"], "candidate_source", checks)
    require(source_path.stat().st_size == candidate["source"]["bytes"], "candidate_source_bytes", checks)

    source_prereg = verify_binding(candidate["source_preregister"], "source_preregister", checks)
    source_hold = verify_binding(candidate["source_hold"], "source_hold", checks)
    strict_audit = verify_binding(candidate["independent_strict_audit"], "strict_audit", checks)
    promotion_prereg = verify_binding(manifest["promotion_preregister"], "promotion_preregister", checks)
    command_template = verify_binding(manifest["command_template"], "command_template", checks)

    source_hold_json = load_json(source_hold)
    strict_json = load_json(strict_audit)
    require(source_hold_json["status"] == candidate["source_hold"]["status_exact"], "source_hold_status", checks)
    require(strict_json["status"] == STRICT_AUDIT_STATUS, "strict_audit_status", checks)
    require(
        strict_json["bindings"]["candidate_estimator"]["sha256"] == candidate["source"]["sha256"],
        "strict_audit_candidate_binding",
        checks,
    )
    require(
        strict_json["bindings"]["author_source_hold"]["sha256"] == candidate["source_hold"]["sha256"],
        "strict_audit_source_hold_binding",
        checks,
    )

    for key, label in (
        ("fullwidth_and_meter_preregister", "fullwidth_meter_preregister"),
        ("fullwidth_comparator_release", "fullwidth_comparator_release"),
        ("literal_meter_release_scaffold", "literal_meter_release_scaffold"),
        ("static_release_chain_verification", "static_release_chain_verification"),
    ):
        path = verify_binding(manifest["frozen_upstream_release_chain"][key], label, checks)
        expected = manifest["frozen_upstream_release_chain"][key].get("status_exact")
        if expected is not None:
            require(load_json(path)["status"] == expected, f"{label}_status", checks)

    q24_result = verify_binding(
        manifest["posthoc_operational_motivation"]["q24_board_result"],
        "q24_board_result",
        checks,
    )
    verify_binding(
        manifest["posthoc_operational_motivation"]["independent_q24_board_audit"],
        "q24_board_result_audit",
        checks,
    )
    q24_json = load_json(q24_result)
    require(q24_json["submission"]["id"] == 328210, "q24_submission_id", checks)
    require(q24_json["evaluation"]["progress"]["mlps_completed"] == 100, "q24_completed_100", checks)
    require(q24_json["evaluation"]["progress"]["mlps_failed"] == 0, "q24_failed_zero", checks)
    require(
        q24_json["submission"]["score_secondary_raw_final_exact"] == 5.765125013112993e-7,
        "q24_raw_exact",
        checks,
    )
    require(
        manifest["posthoc_operational_motivation"]["public_tail_used_to_design_or_fit_cs2"] is False,
        "q24_tail_posthoc_only",
        checks,
    )
    require(
        manifest["posthoc_operational_motivation"]["public_rows_or_scores_used_as_candidate_inputs"] == 0,
        "zero_public_candidate_inputs",
        checks,
    )

    prereg_text = promotion_prereg.read_text(encoding="utf-8")
    require("parent-versus-CS2 disagreement MSE must be finite and `<= 1e-8`" in prereg_text,
            "board_change_gate_literal_in_preregister", checks)
    require("zero parent-finite-to-CS2-nonfinite or raised regressions" in prereg_text,
            "zero_regression_gate_in_preregister", checks)
    require("combined-carrier invariant violations must be zero" in prereg_text,
            "zero_carrier_gate_in_preregister", checks)

    gate = manifest["fullwidth_comparator_gate"]["board_specific_gate_frozen_before_result"]
    require(manifest["fullwidth_comparator_gate"]["seeds_exact_in_order"] == [20260824, 20260825, 20260826],
            "fixed_three_seed_order", checks)
    require(gate["per_seed_operator"] == "<=", "change_gate_operator", checks)
    require(gate["per_seed_threshold"] == 1e-8, "change_gate_threshold", checks)
    require(gate["required_parent_finite_regression_count"] == 0, "required_zero_regressions", checks)
    require(gate["required_each_and_aggregate_carrier_violation_count"] == 0,
            "required_zero_carrier_violations", checks)
    require(gate["comparator_pass_alone_is_sufficient"] is False,
            "comparator_pass_not_sufficient", checks)

    criterion = manifest["primary_board_criterion"]
    require(criterion["configured_mlps"] == 100, "board_configured_100", checks)
    require(criterion["required_completed_mlps"] == 100, "board_requires_100", checks)
    require(criterion["required_failed_mlps"] == 0, "board_requires_zero_failures", checks)
    require(criterion["operator"] == "<" and criterion["threshold"] == 1e-8,
            "board_strict_raw_e8_gate", checks)
    require(manifest["authorization"]["max_attempts_this_preregistration"] == 1,
            "exactly_one_attempt", checks)
    require(manifest["outcome_law"]["retry"] is False, "no_retry", checks)

    template = load_json(command_template)
    require(template["status"] == "HOLD_LITERAL_PLACEHOLDERS_NOT_EXECUTABLE",
            "command_template_hold", checks)
    require(template["attempt_limit"] == 1 and template["retry"] is False,
            "command_template_one_attempt_no_retry", checks)
    require("${ARCHIVE_PATH_FROM_REPO_ROOT}" in template["argv_template"],
            "archive_placeholder_literal", checks)
    require("${FINAL_DESCRIPTION}" in template["argv_template"],
            "description_placeholder_literal", checks)

    forbidden = set(manifest["forbidden_now"])
    require({"PACKAGING", "SUBMISSION", "NETWORK_ACCESS", "CREDENTIAL_ACCESS", "GRAPH_MUTATION"} <= forbidden,
            "hold_forbidden_actions", checks)
    require(manifest["terminal_result_audit"]["independent_audit_required"] is True,
            "terminal_independent_audit_required", checks)
    require(manifest["terminal_result_audit"]["private_targets_or_unobserved_cause_forbidden"] is True,
            "terminal_private_and_causal_boundary", checks)


def verify_hold(manifest: dict, checks: dict[str, bool]) -> None:
    require(manifest["status"] == HOLD_STATUS, "hold_status_exact", checks)
    comp = manifest["fullwidth_comparator_gate"]
    meter = manifest["literal_meter_gate"]
    require(comp["receipt"]["sha256"] is None, "comparator_receipt_sha_null", checks)
    require(comp["receipt"]["status"] is None, "comparator_receipt_status_null", checks)
    require(comp["registered_gate_pass"] is None, "board_change_gate_result_null", checks)
    require(all(value is None for value in comp["board_specific_gate_frozen_before_result"][
        "per_seed_final_layer_parent_child_disagreement_mse"]), "per_seed_change_values_null", checks)
    require(meter["release"]["sha256"] is None, "meter_release_sha_null", checks)
    require(meter["receipt"]["sha256"] is None, "meter_receipt_sha_null", checks)
    require(meter["receipt"]["status"] is None, "meter_receipt_status_null", checks)
    require(meter["literal_phase2_resource_pass"] is None, "meter_result_null", checks)
    require(manifest["candidate"]["archive"]["sha256"] is None, "archive_sha_null", checks)
    require(manifest["package_and_smoke_gate"]["package_manifest"]["sha256"] is None,
            "package_manifest_sha_null", checks)
    require(manifest["package_and_smoke_gate"]["independent_package_smoke_audit"]["sha256"] is None,
            "package_smoke_audit_sha_null", checks)
    require(manifest["submission"]["command_argv"] is None, "submission_argv_null", checks)
    require(manifest["submission"]["attempt_state"] == "HOLD_NULL_PREREQUISITES",
            "attempt_state_hold", checks)
    require(manifest["authorization"]["artifact_prerequisites_satisfied"] is False,
            "prerequisites_unsatisfied", checks)
    require(manifest["authorization"]["submission_authorized_while_hold"] is False,
            "submission_unauthorized", checks)


def verify_rejection(manifest: dict, checks: dict[str, bool]) -> None:
    require(REJECTION_PATH.is_file(), "rejection_receipt_exists", checks)
    rejection = load_json(REJECTION_PATH)
    require(
        rejection["status"] ==
        "REJECT_BOARD_PROMOTION_CHANGE_SIZE_GATE__COMPARATOR_SOURCE_STABILITY_PASS__"
        "LITERAL_METER_TECHNICAL_ONLY",
        "rejection_status_exact",
        checks,
    )
    for key, label in (
        ("promotion_preregister", "rejection_preregister"),
        ("promotion_scaffold", "rejection_manifest"),
        ("candidate_estimator", "rejection_candidate"),
        ("source_hold", "rejection_source_hold"),
        ("independent_strict_audit", "rejection_strict_audit"),
        ("comparator_preregister", "rejection_comparator_preregister"),
        ("comparator_release", "rejection_comparator_release"),
        ("comparator_one_use_claim", "rejection_comparator_claim"),
        ("comparator_receipt", "rejection_comparator_receipt"),
    ):
        verify_binding(rejection["bindings"][key], label, checks)
    require(
        rejection["bindings"]["promotion_scaffold"]["sha256"] == sha256(MANIFEST_PATH),
        "rejection_binds_current_manifest",
        checks,
    )
    comparator = load_json(resolve(rejection["bindings"]["comparator_receipt"]["path"]))
    require(comparator["status"] == COMPARATOR_STATUS, "rejection_comparator_pass", checks)
    expected = [row["comparison"]["final_layer_disagreement_mse"] for row in comparator["rows"]]
    observed = [row["mse"] for row in rejection["board_specific_change_size_gate"]["rows"]]
    require(expected == observed, "rejection_exact_mse_replay", checks)
    require(all(value > 1e-8 for value in observed), "rejection_all_three_exceed_e8", checks)
    require(rejection["board_specific_change_size_gate"]["pass_count"] == 0,
            "rejection_zero_passes", checks)
    require(rejection["board_specific_change_size_gate"]["failure_count"] == 3,
            "rejection_three_failures", checks)
    totals = comparator["gates"]["carrier_violation_totals"]
    require(all(value == 0 for value in totals.values()), "rejection_zero_carrier_violations", checks)
    require(comparator["gates"]["parent_contracts_all_three"] is True and
            comparator["gates"]["child_contracts_all_three"] is True,
            "rejection_zero_parent_finite_regressions", checks)
    require(rejection["verdict"]["package_authorized"] is False,
            "rejection_package_forbidden", checks)
    require(rejection["verdict"]["submission_authorized"] is False,
            "rejection_submission_forbidden", checks)
    require(rejection["verdict"]["board_attempts_consumed"] == 0,
            "rejection_zero_board_attempts", checks)
    require(rejection["negative_controls"]["packages_created"] == 0,
            "rejection_no_package_created", checks)
    require(rejection["negative_controls"]["submissions_created"] == 0,
            "rejection_no_submission_created", checks)


def verify_archive(manifest: dict, checks: dict[str, bool]) -> None:
    archive = manifest["candidate"]["archive"]
    path = REPO / archive["path_from_repo_root"]
    require(path.is_file(), "archive_exists", checks)
    require(path.stat().st_size == archive["bytes"], "archive_bytes", checks)
    require(sha256(path) == archive["sha256"], "archive_sha256", checks)
    with tarfile.open(path, "r:gz") as tf:
        members = tf.getmembers()
        require([member.name for member in members] == ["estimator.py", "manifest.json"],
                "archive_member_names_order", checks)
        require(all(member.isfile() and "/" not in member.name for member in members),
                "archive_members_regular_top_level", checks)
        for expected, member in zip(archive["members_exact"], members):
            payload = tf.extractfile(member).read()
            require(len(payload) == expected["bytes"], f"archive_{member.name}_bytes", checks)
            require(hashlib.sha256(payload).hexdigest() == expected["sha256"],
                    f"archive_{member.name}_sha256", checks)
            if member.name == "estimator.py":
                require(expected["sha256"] == manifest["candidate"]["source"]["sha256"],
                        "archive_source_parity", checks)
            else:
                embedded = json.loads(payload)
                spec = manifest["package_and_smoke_gate"]["embedded_manifest"]
                require(embedded["schema_version"] == spec["schema_version"], "embedded_schema_version", checks)
                require(embedded["api_version"] == spec["api_version"], "embedded_api_version", checks)
                require(embedded["entrypoint"] == {"module": "estimator", "class": "Estimator"},
                        "embedded_entrypoint", checks)
                require(embedded["whestbench_version"] == spec["whestbench_version"],
                        "embedded_whestbench", checks)
                require(embedded["flopscope_version"] == spec["flopscope_version"],
                        "embedded_flopscope", checks)
                require(embedded["numpy_version"] == spec["numpy_version"], "embedded_numpy", checks)
                require(embedded["python_min_version"] == spec["python_min_version"],
                        "embedded_python", checks)


def verify_ready(manifest: dict, checks: dict[str, bool]) -> None:
    require(manifest["status"] == READY_STATUS, "ready_status_exact", checks)
    for field in manifest["required_final_fields"]:
        value = nested_value(manifest, field)
        require(value is not None, f"populated_{field}", checks)

    comp = manifest["fullwidth_comparator_gate"]
    comp_path = resolve(comp["receipt"]["path"])
    require(sha256(comp_path) == comp["receipt"]["sha256"], "comparator_receipt_sha256", checks)
    require(load_json(comp_path)["status"] == COMPARATOR_STATUS == comp["receipt"]["status"],
            "comparator_receipt_pass", checks)
    gate = comp["board_specific_gate_frozen_before_result"]
    values = gate["per_seed_final_layer_parent_child_disagreement_mse"]
    require(len(values) == 3 and all(math.isfinite(value) and value <= 1e-8 for value in values),
            "each_seed_change_mse_at_most_e8", checks)
    require(gate["parent_finite_regression_count"] == 0, "ready_zero_parent_finite_regressions", checks)
    for key in ("pairing_alpha_violation_count", "connected_star_violation_count",
                "cp_carrier_violation_count", "combined_carrier_violation_count",
                "aggregate_carrier_violation_count"):
        require(gate[key] == 0, f"ready_zero_{key}", checks)
    require(comp["registered_gate_pass"] is True, "board_change_gate_pass", checks)

    meter = manifest["literal_meter_gate"]
    meter_path = resolve(meter["receipt"]["path"])
    require(sha256(meter_path) == meter["receipt"]["sha256"], "meter_receipt_sha256", checks)
    require(load_json(meter_path)["status"] == METER_STATUS == meter["receipt"]["status"],
            "meter_receipt_pass", checks)
    limits = meter["limits"]
    require(meter["flops_used"] <= limits["flops_at_most"], "meter_flops", checks)
    require(meter["wall_time_s"] < limits["wall_time_s_strictly_below"], "meter_wall", checks)
    require(meter["residual_time_s"] <= limits["residual_time_s_at_most"], "meter_residual", checks)
    require(meter["descendant_peak_rss_bytes"] < limits["descendant_peak_rss_bytes_strictly_below"],
            "meter_rss", checks)
    require(meter["failed_mlps"] == 0, "meter_failed_zero", checks)
    require(meter["output_shape"] == [16, 1024], "meter_shape", checks)
    require(meter["output_dtype"] == "float32", "meter_dtype", checks)
    require(meter["output_all_finite"] is True and meter["literal_phase2_resource_pass"] is True,
            "meter_finite_resource_pass", checks)

    for key in ("package_manifest", "independent_package_smoke_audit"):
        binding = manifest["package_and_smoke_gate"][key]
        path = verify_binding(binding, key, checks)
        require(str(load_json(path)["status"]).startswith("PASS"), f"{key}_pass", checks)
    verify_archive(manifest, checks)

    template = load_json(resolve(manifest["command_template"]["path"]))
    argv = manifest["submission"]["command_argv"]
    expected = list(template["argv_template"])
    expected[2] = manifest["candidate"]["archive"]["path_from_repo_root"]
    require(argv[:6] == expected[:6] and argv[7:] == expected[7:], "submission_argv_shape", checks)
    require(manifest["candidate"]["tag"] in argv[6], "description_has_candidate_tag", checks)
    require(manifest["submission"]["attempt_state"] == "PREPARED_NOT_ATTEMPTED",
            "ready_attempt_state", checks)
    require(manifest["authorization"]["artifact_prerequisites_satisfied"] is True,
            "ready_prerequisites_satisfied", checks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    checks: dict[str, bool] = {}
    try:
        manifest = load_json(MANIFEST_PATH)
        verify_core(manifest, checks)
        if args.require_ready:
            verify_ready(manifest, checks)
            status = "PASS_CS2_BOARD_PROMOTION_READY"
        else:
            verify_hold(manifest, checks)
            if REJECTION_PATH.is_file():
                verify_rejection(manifest, checks)
                status = "PASS_CS2_BOARD_PROMOTION_REJECTED__NO_PACKAGE_OR_SUBMISSION"
            else:
                status = "PASS_CS2_BOARD_PROMOTION_SCAFFOLD__HOLD_NULL_PREREQUISITES"
    except Exception as exc:  # fail closed with a machine-readable diagnostic
        print(json.dumps({
            "status": "FAIL_CLOSED_CS2_BOARD_PROMOTION_VERIFICATION",
            "require_ready": args.require_ready,
            "error": f"{type(exc).__name__}: {exc}",
            "checks": checks,
        }, sort_keys=True, indent=2))
        return 1
    print(json.dumps({
        "status": status,
        "require_ready": args.require_ready,
        "checks": checks,
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
