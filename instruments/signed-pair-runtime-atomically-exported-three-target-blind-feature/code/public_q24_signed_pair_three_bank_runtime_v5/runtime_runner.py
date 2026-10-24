#!/usr/bin/env python3
"""Target-blind, exactly-once lean signed-pair three-bank generator runtime V5.

Only standard-library modules load before transaction ownership.  The runner
binds one immutable callable-bundle manifest.  An independent review binds
runner plus bundle; an external anchor binds review, runner, bundle, and the
ordered scope; a one-use capability binds the anchor but never the release;
and the final release binds anchor plus capability.  This graph is acyclic.

After authority validation the runner acquires a durable O_EXCL claim,
irreversibly consumes the capability, then and only then loads the token
module through its transient mint channel.  Parent, two source laws, adapter,
rows, and weights are opened only after a live process-local token exists.
Six binary artifacts and one canonical receipt publish by one atomic directory
rename.  The exact runner remains fail-closed until external GO/release/scope/
capability documents replace the deliberately null scaffolds.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys

# Patched once, after the five callable components and their immutable bundle
# manifest are finalized.  It is the runner's sole earlier code-root digest.
CALLABLE_BUNDLE_MANIFEST_SHA256 = (
    "5a5c7ea5fd83905ad4484cd74a0101eb076f1c6d2167a423f7000968ef1a2599"
)
EXECUTION_STACK_MANIFEST_SHA256 = (
    "e4204ec69ab30a5bb9e441558722edb7bcd80d45bba380ee9cc84217bfa6d433"
)
PRIOR_RUNTIME_V2_GO_SHA256 = (
    "7bf6664b2e27d863e5fc4a556373a217d05628ec681141757a4d0beec46c0b03"
)
MANIFEST_COMPATIBILITY_SCAN_RECEIPT_SHA256 = (
    "052842efefda1fd73e49d15f17256411436900393cbaf6ae61dac8098310ce0a"
)
PRIOR_RUNTIME_V3_TERMINAL_LINEAGE_SHA256 = (
    "b0756fd217c6bef5a0df473502109d18bb948fe8b12d0c69c4f15ab02f27ea70"
)
PRIOR_RUNTIME_V3_CLAIM_SHA256 = (
    "bfa9e8d800a5b1efc0ec4e47a62b53d70176e0bf8b9ec73f9981056974abd4e9"
)
PRIOR_RUNTIME_V3_TERMINAL_FAILURE_SHA256 = (
    "e64f998fc1513f906e90d2869340e44ce2ca548b91e31667aade69da9af1d50d"
)
PRIOR_RUNTIME_V3_CONSUMED_CAPABILITY_SHA256 = (
    "f7cd31ab124c7914d1bd2b413289fb84ced4a38d2e93e3f3d3d9a9e711d38a39"
)
PRIOR_RUNTIME_V4_TERMINAL_LINEAGE_SHA256 = (
    "0dfdfff19826c163958bfc0b6866b31a9f334a7a2f62262a447b5baa3f5d7ded"
)
PRIOR_RUNTIME_V4_CLAIM_SHA256 = (
    "d690aa5b42dd0bca12705795004640413368a4384bce467bad2a16c6560fd2b7"
)
PRIOR_RUNTIME_V4_TERMINAL_FAILURE_SHA256 = (
    "fbac37a97c64983fbc56627b67fdebb59dca07d0f43f3762798495022ed74dae"
)
PRIOR_RUNTIME_V4_CONSUMED_CAPABILITY_SHA256 = (
    "24db92e64b4129d3a8c002620017aa516e2287cb9183c5edf9c99ec35c6c4518"
)
INDEPENDENT_INCIDENT_ROOT_CAUSE_SHA256 = (
    "060e7bab3094a7929d246f4bdf5fbe6790ccb3aa9c3b05734395a6376b24cd6c"
)
FLOPSCOPE_RESET_VERIFICATION_SHA256 = (
    "9253d354914da94d193dadd43e118ef7b6afaa9a2b11769c8d599cbfd62b26af"
)
TOKEN_MODULE_IMPORT_NAME = "q24_three_bank_runtime_token_v2"
EXPECTED_RANKS = tuple(range(688))
SENTINEL_RANKS = (0, 343, 687)
EXPECTED_CAPTURE_LAYERS = tuple(range(2, 15))
EXPECTED_H = (16, 64, 256)
CALLABLE_COMPONENT_NAMES = (
    "runtime_token.py",
    "parent_capture.py",
    "signed_pair_literal.py",
    "signed_pair_reference.py",
    "capture_adapter.py",
)
INDEPENDENT_REVIEW_REQUIRED_FIELDS = (
    "schema",
    "status",
    "issued_by",
    "issued_utc",
    "reviewed_runner_sha256",
    "reviewed_callable_bundle_manifest_sha256",
    "reviewed_execution_stack_manifest_sha256",
    "prior_runtime_v1_hold_sha256",
    "prior_runtime_v2_go_sha256",
    "prior_runtime_v3_claim_sha256",
    "prior_runtime_v3_terminal_failure_sha256",
    "prior_runtime_v3_consumed_capability_sha256",
    "prior_runtime_v4_terminal_failure_lineage_sha256",
    "prior_runtime_v4_claim_sha256",
    "prior_runtime_v4_terminal_failure_sha256",
    "prior_runtime_v4_consumed_capability_sha256",
    "independent_incident_root_cause_sha256",
    "flopscope_reset_verification_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "r1_public_api_and_governance_chronology_closed",
    "hostile_interpreter_security_claim",
    "r2_authority_dag_acyclic",
    "r3_resource_veto_removed",
    "r4_dynamic_comparators_present",
    "r5_manifest_firewall_semantic_repair_verified",
    "r6_flopscope_reset_placement_and_invariance_verified",
    "v3_terminal_no_retry_closed",
    "v4_terminal_no_retry_closed",
    "source_data_target_execution_during_review",
)
CAPABILITY_REQUIRED_FIELDS = (
    "schema",
    "status",
    "capability_id",
    "issued_by",
    "issued_utc",
    "authority_anchor_sha256",
    "runner_sha256",
    "callable_bundle_manifest_sha256",
    "execution_stack_manifest_sha256",
    "ordered_688_row_scope_sha256",
    "ordered_688_record_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "one_use",
    "no_retry",
    "source_import_allowed",
    "row_access_allowed",
    "target_access_allowed",
    "training_allowed",
    "estimator_deployment_allowed",
    "package_allowed",
    "submission_allowed",
)
ANCHOR_REQUIRED_FIELDS = (
    "schema",
    "status",
    "issued_by",
    "issued_utc",
    "bindings",
    "target_access_allowed",
    "training_allowed",
    "estimator_deployment_allowed",
    "package_allowed",
    "submission_allowed",
)
RELEASE_REQUIRED_FIELDS = (
    "schema",
    "status",
    "issued_by",
    "issued_utc",
    "authority_anchor_sha256",
    "single_use_capability_sha256",
    "bindings",
    "one_use",
    "no_retry",
    "source_import_allowed",
    "row_access_allowed",
    "target_access_allowed",
    "training_allowed",
    "estimator_deployment_allowed",
    "package_allowed",
    "submission_allowed",
)
ROW_RECORD_REQUIRED_FIELDS = (
    "rank",
    "mlp_id",
    "row_manifest_path",
    "row_manifest_sha256",
    "weights_path",
    "weights_sha256",
    "weights_schema_sha256",
)
INTEGER_METER_NAMES = (
    "complete_flops",
    "complete_calls",
    "base_parent_flops",
    "base_parent_calls",
    "capture_parent_flops",
    "capture_parent_calls",
    "literal_source_hook_flops",
    "literal_source_hook_calls",
    "reference_source_hook_flops",
    "reference_source_hook_calls",
    "literal_workspace_flops",
    "literal_workspace_calls",
    "reference_workspace_flops",
    "reference_workspace_calls",
    "peak_rss_bytes",
    "source_hook_capture_count",
    "reference_source_hook_capture_count",
)
FLOAT_METER_NAMES = ("wall_seconds", "residual_seconds")
INTEGER_METER_VECTOR_NAMES = tuple(
    name + "_by_rank" for name in INTEGER_METER_NAMES
)
FLOAT_METER_VECTOR_NAMES = tuple(name + "_by_rank" for name in FLOAT_METER_NAMES)
FIVE_COMPONENT_PARENT_NAMES = (
    "host",
    "native_raw20",
    "native_sigma",
    "native_j_correction",
    "final_public_prediction",
)
EXACT_METER_VECTOR_SCALAR_PAIRS = tuple(
    (name + "_by_rank", name)
    for name in INTEGER_METER_NAMES + FLOAT_METER_NAMES
)
ROW_METER_REQUIRED_FIELDS = set(INTEGER_METER_NAMES) | set(FLOAT_METER_NAMES) | {"meter_scope"}
ROW_RESULT_REQUIRED_FIELDS = {
    "rank",
    "sentinel_rank",
    "comparator_mode",
    "mlp_id",
    "row_manifest_sha256",
    "weights_sha256",
    "weights_schema_sha256",
    "capture_layers_zero_based",
    "same_layer_preactivation_alpha",
    "capture_after_full_reset_before_reset_stars",
    "A15_exact_zero_float32",
    "cross_rung_H16_prefix_byte_identical",
    "h256_prefix_derivation",
    "source_action_comparator",
    "five_component_parent_comparator",
    "banks",
    "exact_meters",
    "resource_observation",
}


class RunnerReleaseHold(RuntimeError):
    """Raised before or during a fail-closed one-shot generation pass."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_nonsymlink_file(path, label):
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        information = os.lstat(current)
        if stat.S_ISLNK(information.st_mode):
            raise RunnerReleaseHold(label + " contains a symlink component")
    information = os.lstat(absolute)
    if not stat.S_ISREG(information.st_mode):
        raise RunnerReleaseHold(label + " is not a regular file")
    return absolute


def _require_nonsymlink_directory(path, label):
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        information = os.lstat(current)
        if stat.S_ISLNK(information.st_mode):
            raise RunnerReleaseHold(label + " contains a symlink component")
    if not stat.S_ISDIR(os.lstat(absolute).st_mode):
        raise RunnerReleaseHold(label + " is not a directory")
    return absolute


def _read_json(path):
    with Path(path).open("rb") as handle:
        raw = handle.read()
    def reject_constant(value):
        raise RunnerReleaseHold("nonfinite JSON constant: " + value)

    def exact_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise RunnerReleaseHold("duplicate JSON key: " + key)
            output[key] = value
        return output

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerReleaseHold("authority/scope JSON is not strict ASCII JSON") from error
    if raw != _canonical_json_bytes(value):
        raise RunnerReleaseHold("authority/scope JSON bytes are not canonical")
    return raw, value


def _canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _is_lower_hex_sha256(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_capability_text(name, value):
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) > 126 for character in value)
        or "/" in value
        or "\\" in value
    ):
        raise RunnerReleaseHold("invalid capability text field: " + name)


def _validate_issued_utc(value):
    if type(value) is not str:
        raise RunnerReleaseHold("issued_utc must be a canonical UTC string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise RunnerReleaseHold("issued_utc must be YYYY-MM-DDTHH:MM:SSZ") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise RunnerReleaseHold("issued_utc is not canonical UTC")


def _validate_exact_meter_scalars(meters):
    if not isinstance(meters, dict):
        raise RunnerReleaseHold("exact row meters must be a mapping")
    if set(meters) != ROW_METER_REQUIRED_FIELDS:
        raise RunnerReleaseHold("exact row meter fields are not exact")
    for name in INTEGER_METER_NAMES:
        value = meters.get(name)
        if type(value) is not int or value < 0:
            raise RunnerReleaseHold("invalid exact integer row meter: " + name)
    for name in FLOAT_METER_NAMES:
        value = meters.get(name)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise RunnerReleaseHold("invalid exact finite float row meter: " + name)


def _validate_exact_meter_vectors(meters):
    if not isinstance(meters, dict):
        raise RunnerReleaseHold("complete exact meters must be a mapping")
    if set(meters) != set(INTEGER_METER_VECTOR_NAMES) | set(FLOAT_METER_VECTOR_NAMES):
        raise RunnerReleaseHold("complete exact meter-vector fields are not exact")
    for name in INTEGER_METER_VECTOR_NAMES:
        vector = meters.get(name)
        if not isinstance(vector, list) or len(vector) != 688:
            raise RunnerReleaseHold("complete exact integer meter vector missing: " + name)
        if any(type(value) is not int or value < 0 for value in vector):
            raise RunnerReleaseHold("invalid complete exact integer meter vector: " + name)
    for name in FLOAT_METER_VECTOR_NAMES:
        vector = meters.get(name)
        if not isinstance(vector, list) or len(vector) != 688:
            raise RunnerReleaseHold("complete exact float meter vector missing: " + name)
        if any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in vector
        ):
            raise RunnerReleaseHold("invalid complete exact finite float meter vector: " + name)


def _validate_source_action_comparator(comparator, component_sha256):
    expected_fields = {
        "comparator",
        "literal_implementation_id",
        "reference_implementation_id",
        "literal_source_sha256",
        "reference_source_sha256",
        "literal",
        "reference",
        "all_action_and_A15_bytes_equal",
    }
    if not isinstance(comparator, dict) or set(comparator) != expected_fields:
        raise RunnerReleaseHold("source action comparator schema mismatch")
    if comparator.get("comparator") != (
        "SEPARATE_LITERAL_AND_REFERENCE_EXECUTIONS_BYTEWISE"
    ):
        raise RunnerReleaseHold("source action comparator identity mismatch")
    if comparator.get("literal_implementation_id") != (
        "ACCEPTED_LAW_RECEIVER_STRIDE_LITERAL_V2"
    ):
        raise RunnerReleaseHold("literal implementation identity mismatch")
    if comparator.get("reference_implementation_id") != (
        "INDEPENDENT_LOOP_AND_SECTOR_PACK_REFERENCE_V2"
    ):
        raise RunnerReleaseHold("reference implementation identity mismatch")
    if comparator["literal_implementation_id"] == comparator[
        "reference_implementation_id"
    ]:
        raise RunnerReleaseHold("source comparator implementations are aliased")
    if comparator.get("literal_source_sha256") != component_sha256[
        "signed_pair_literal.py"
    ]:
        raise RunnerReleaseHold("literal comparator source binding mismatch")
    if comparator.get("reference_source_sha256") != component_sha256[
        "signed_pair_reference.py"
    ]:
        raise RunnerReleaseHold("reference comparator source binding mismatch")
    if comparator.get("all_action_and_A15_bytes_equal") is not True:
        raise RunnerReleaseHold("literal/reference comparator failed")
    literal = comparator.get("literal")
    reference = comparator.get("reference")
    if (
        not isinstance(literal, dict)
        or not isinstance(reference, dict)
        or set(literal) != {"16", "64", "256"}
        or set(reference) != {"16", "64", "256"}
    ):
        raise RunnerReleaseHold("source comparator H keyset mismatch")
    for h in EXPECTED_H:
        key = str(h)
        for side in (literal, reference):
            if not isinstance(side[key], dict) or set(side[key]) != {
                "action_sha256",
                "A15_zero_sha256",
            }:
                raise RunnerReleaseHold("source comparator hash schema mismatch")
            if any(not _is_lower_hex_sha256(value) for value in side[key].values()):
                raise RunnerReleaseHold("source comparator has invalid digest")
        if literal[key] != reference[key]:
            raise RunnerReleaseHold("source comparator digests are not equal")


def _validate_five_component_parent_comparator(comparator):
    expected_fields = {
        "comparator",
        "component_names",
        "base_component_sha256",
        "capture_component_sha256",
        "all_five_component_bytes_equal",
    }
    if not isinstance(comparator, dict) or set(comparator) != expected_fields:
        raise RunnerReleaseHold("parent component comparator schema mismatch")
    if comparator.get("comparator") != (
        "UNHOOKED_BASE_VS_CAPTURED_PARENT_BYTEWISE"
    ):
        raise RunnerReleaseHold("parent component comparator identity mismatch")
    if comparator.get("component_names") != list(FIVE_COMPONENT_PARENT_NAMES):
        raise RunnerReleaseHold("parent component comparator name order mismatch")
    if comparator.get("all_five_component_bytes_equal") is not True:
        raise RunnerReleaseHold("parent component comparator failed")
    base = comparator.get("base_component_sha256")
    capture = comparator.get("capture_component_sha256")
    if (
        not isinstance(base, dict)
        or not isinstance(capture, dict)
        or set(base) != set(FIVE_COMPONENT_PARENT_NAMES)
        or set(capture) != set(FIVE_COMPONENT_PARENT_NAMES)
    ):
        raise RunnerReleaseHold("parent component comparator hash keyset mismatch")
    if any(not _is_lower_hex_sha256(value) for value in base.values()):
        raise RunnerReleaseHold("base parent component digest invalid")
    if any(not _is_lower_hex_sha256(value) for value in capture.values()):
        raise RunnerReleaseHold("capture parent component digest invalid")
    if base != capture:
        raise RunnerReleaseHold("base/capture parent component digests differ")


def _validate_exact_meter_vector_alignment(meters, row_receipts):
    _validate_exact_meter_vectors(meters)
    if not isinstance(row_receipts, list) or len(row_receipts) != len(EXPECTED_RANKS):
        raise RunnerReleaseHold("exact meter alignment requires 688 row receipts")
    for rank in EXPECTED_RANKS:
        row_receipt = row_receipts[rank]
        if (
            not isinstance(row_receipt, dict)
            or type(row_receipt.get("rank")) is not int
            or row_receipt.get("rank") != rank
        ):
            raise RunnerReleaseHold("exact meter row order mismatch")
        row_meters = row_receipt.get("exact_meters")
        _validate_exact_meter_scalars(row_meters)
        for vector_name, scalar_name in EXACT_METER_VECTOR_SCALAR_PAIRS:
            if meters[vector_name][rank] != row_meters[scalar_name]:
                raise RunnerReleaseHold(
                    "exact meter vector does not equal row scalar: " + vector_name
                )


def _validate_row_A15_hash_alignment(final, row_receipts):
    if not isinstance(row_receipts, list) or len(row_receipts) != len(EXPECTED_RANKS):
        raise RunnerReleaseHold("A15 alignment requires 688 row receipts")
    for h in EXPECTED_H:
        canonical_sha256 = final.get("banks", {}).get(str(h), {}).get(
            "A15_zero_sha256"
        )
        if not _is_lower_hex_sha256(canonical_sha256):
            raise RunnerReleaseHold("canonical A15 hash invalid for H" + str(h))
        for rank in EXPECTED_RANKS:
            row_receipt = row_receipts[rank]
            if (
                not isinstance(row_receipt, dict)
                or type(row_receipt.get("rank")) is not int
                or row_receipt.get("rank") != rank
            ):
                raise RunnerReleaseHold("A15 row order mismatch")
            row_sha256 = row_receipt.get("banks", {}).get(str(h), {}).get(
                "A15_zero_bytes_sha256"
            )
            if row_sha256 != canonical_sha256:
                raise RunnerReleaseHold(
                    "row A15 hash does not equal verified canonical file: H" + str(h)
                )


def _reject_forbidden_payload(value, location="root"):
    forbidden = ("target", "label", "ground_truth", "mse", "score", "board")
    allowed_false_policy_keys = (
        "target_access_allowed",
        "target_accessed",
        "source_data_target_execution_during_review",
        "score_access_allowed",
        "score_accessed",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            allowed_denial = lowered in allowed_false_policy_keys and child is False
            if any(token in lowered for token in forbidden) and not allowed_denial:
                raise RunnerReleaseHold(
                    "forbidden target/score field at " + location + "." + str(key)
                )
            _reject_forbidden_payload(child, location + "." + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_payload(child, location + "[" + str(index) + "]")


def _validate_exact_booleans(document, expected):
    for name, value in expected.items():
        if document.get(name) is not value:
            raise RunnerReleaseHold("invalid release boolean: " + name)


def _validate_callable_bundle(manifest_path, component_paths):
    raw, manifest = _read_json(manifest_path)
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != CALLABLE_BUNDLE_MANIFEST_SHA256:
        raise RunnerReleaseHold("callable-bundle manifest digest mismatch")
    if set(manifest) != {
        "schema",
        "status",
        "components",
        "source_lineage",
        "parent_lineage",
        "prior_runtime_v1_hold_sha256",
        "prior_runtime_v2_go_sha256",
        "prior_runtime_v3_terminal_failure_lineage_sha256",
        "manifest_compatibility_scan_receipt_sha256",
        "properties",
    }:
        raise RunnerReleaseHold("callable-bundle manifest fields are not exact")
    if manifest.get("schema") != "public-q24-three-bank-callable-bundle-v4":
        raise RunnerReleaseHold("callable-bundle schema mismatch")
    if manifest.get("status") != "FROZEN_IMMUTABLE_CALLABLE_CODE_ROOT":
        raise RunnerReleaseHold("callable-bundle status mismatch")
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != set(
        CALLABLE_COMPONENT_NAMES
    ):
        raise RunnerReleaseHold("callable-bundle component keyset mismatch")
    if not isinstance(component_paths, dict) or set(component_paths) != set(
        CALLABLE_COMPONENT_NAMES
    ):
        raise RunnerReleaseHold("runner component-path keyset mismatch")
    component_sha256 = {}
    for name in CALLABLE_COMPONENT_NAMES:
        item = components[name]
        if not isinstance(item, dict) or set(item) != {"role", "sha256"}:
            raise RunnerReleaseHold("callable component schema mismatch: " + name)
        if type(item["role"]) is not str or not item["role"]:
            raise RunnerReleaseHold("callable component role missing: " + name)
        if not _is_lower_hex_sha256(item["sha256"]):
            raise RunnerReleaseHold("callable component digest invalid: " + name)
        if _sha256(component_paths[name]) != item["sha256"]:
            raise RunnerReleaseHold("callable component bytes mismatch: " + name)
        component_sha256[name] = item["sha256"]
    for lineage_name in ("source_lineage", "parent_lineage"):
        lineage = manifest.get(lineage_name)
        if not isinstance(lineage, dict) or not lineage:
            raise RunnerReleaseHold("callable-bundle lineage missing")
        if any(not _is_lower_hex_sha256(value) for value in lineage.values()):
            raise RunnerReleaseHold("callable-bundle lineage digest invalid")
    if not _is_lower_hex_sha256(manifest.get("prior_runtime_v1_hold_sha256")):
        raise RunnerReleaseHold("prior V1 HOLD binding invalid")
    if manifest.get("prior_runtime_v2_go_sha256") != PRIOR_RUNTIME_V2_GO_SHA256:
        raise RunnerReleaseHold("prior runtime V2 GO binding invalid")
    if manifest.get(
        "manifest_compatibility_scan_receipt_sha256"
    ) != MANIFEST_COMPATIBILITY_SCAN_RECEIPT_SHA256:
        raise RunnerReleaseHold("manifest compatibility scan binding invalid")
    if manifest.get(
        "prior_runtime_v3_terminal_failure_lineage_sha256"
    ) != PRIOR_RUNTIME_V3_TERMINAL_LINEAGE_SHA256:
        raise RunnerReleaseHold("prior runtime V3 terminal lineage binding invalid")
    if manifest.get("properties") != {
        "bundle_contains_transaction_hashes": False,
        "public_numerical_entrypoints_require_process_local_token": True,
        "ordinary_import_exposes_mint": False,
        "hostile_interpreter_security_claim": False,
        "resource_thresholds_can_veto_accuracy_generation": False,
        "captured_parent_and_literal_source_execute_all_rows": True,
        "literal_reference_source_execution_is_separate_at_sentinels": True,
        "base_capture_parent_comparator_is_dynamic_at_sentinels": True,
        "h256_prefix_derivation_executes_all_rows": True,
        "sentinel_ranks_are_exactly_0_343_687": True,
        "manifest_firewall_allows_only_three_exact_zero_audit_paths": True,
        "manifest_root_schema_and_decoded_columns_are_exact": True,
        "v3_capability_is_terminally_consumed_and_not_retried": True,
    }:
        raise RunnerReleaseHold("callable-bundle properties mismatch")
    return manifest, manifest_sha256, component_sha256


def _validate_execution_stack(path):
    raw, manifest = _read_json(path)
    stack_sha256 = hashlib.sha256(raw).hexdigest()
    if stack_sha256 != EXECUTION_STACK_MANIFEST_SHA256:
        raise RunnerReleaseHold("execution-stack manifest digest mismatch")
    if set(manifest) != {
        "schema",
        "status",
        "lane",
        "interpreter",
        "distribution_versions",
        "module_files",
        "requirements_lock",
        "budget_context_contract",
    }:
        raise RunnerReleaseHold("execution-stack fields are not exact")
    if manifest.get("schema") != "public-q24-three-bank-execution-stack-v3":
        raise RunnerReleaseHold("execution-stack schema mismatch")
    if manifest.get("status") != "FROZEN_OFFICIAL_SHIPPING_LANE":
        raise RunnerReleaseHold("execution-stack status mismatch")
    if manifest.get("lane") != "OFFICIAL_FULL_NATIVE_SHIPPING_STACK":
        raise RunnerReleaseHold("execution-stack lane mismatch")
    interpreter = manifest.get("interpreter")
    if not isinstance(interpreter, dict) or set(interpreter) != {
        "implementation",
        "invocation_path",
        "invocation_symlink_target",
        "machine",
        "platform",
        "resolved_executable_sha256",
        "version",
        "version_full",
        "venv_prefix",
    }:
        raise RunnerReleaseHold("execution-stack interpreter schema mismatch")
    invocation = Path(interpreter["invocation_path"])
    if Path(sys.executable) != invocation:
        raise RunnerReleaseHold("runner is not using frozen venv interpreter")
    if not invocation.is_symlink() or os.readlink(invocation) != interpreter[
        "invocation_symlink_target"
    ]:
        raise RunnerReleaseHold("frozen interpreter symlink binding mismatch")
    if _sha256(invocation.resolve()) != interpreter["resolved_executable_sha256"]:
        raise RunnerReleaseHold("resolved interpreter digest mismatch")
    if sys.prefix != interpreter["venv_prefix"]:
        raise RunnerReleaseHold("venv prefix mismatch")
    if platform.python_implementation() != interpreter["implementation"]:
        raise RunnerReleaseHold("Python implementation mismatch")
    if platform.python_version() != interpreter["version"]:
        raise RunnerReleaseHold("Python version mismatch")
    if sys.version != interpreter["version_full"]:
        raise RunnerReleaseHold("full Python version mismatch")
    if sys.platform != interpreter["platform"] or platform.machine() != interpreter[
        "machine"
    ]:
        raise RunnerReleaseHold("execution platform mismatch")
    prefix = Path(interpreter["venv_prefix"])
    files = manifest.get("module_files")
    if not isinstance(files, dict) or not files:
        raise RunnerReleaseHold("execution-stack module file map missing")
    for relative, expected_sha256 in files.items():
        if (
            type(relative) is not str
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _is_lower_hex_sha256(expected_sha256)
        ):
            raise RunnerReleaseHold("execution-stack module path/hash invalid")
        if _sha256(prefix / relative) != expected_sha256:
            raise RunnerReleaseHold("execution-stack module digest mismatch: " + relative)
    lock = manifest.get("requirements_lock")
    if not isinstance(lock, dict) or set(lock) != {"path", "sha256"}:
        raise RunnerReleaseHold("execution-stack requirements lock schema mismatch")
    if _sha256(lock["path"]) != lock["sha256"]:
        raise RunnerReleaseHold("execution-stack requirements lock mismatch")
    if manifest.get("distribution_versions") != {
        "flopscope": "0.12.0",
        "flopscope-server": "0.12.0",
        "numpy": "2.4.6",
        "whestbench": "0.16.0",
    }:
        raise RunnerReleaseHold("execution-stack distribution version mismatch")
    budget = manifest.get("budget_context_contract")
    if budget != {
        "constructor_keywords": ["flop_budget", "quiet", "namespace"],
        "diagnostic_meter_ceiling": (1 << 63) - 1,
        "fixed_accuracy_row_flop_upper_bound": 4_000_000_000_000,
        "no_wall_deadline_by_omission": True,
        "remote_client_lane_allowed": False,
    }:
        raise RunnerReleaseHold("execution-stack budget contract mismatch")
    if budget["diagnostic_meter_ceiling"] <= (
        1_000_000 * budget["fixed_accuracy_row_flop_upper_bound"]
    ):
        raise RunnerReleaseHold("diagnostic meter ceiling lacks fixed-job margin")
    return manifest, stack_sha256


def _validate_independent_review(
    path, runner_sha256, bundle_sha256, stack_sha256, bundle_manifest
):
    raw, review = _read_json(path)
    review_sha256 = hashlib.sha256(raw).hexdigest()
    if set(review) != set(INDEPENDENT_REVIEW_REQUIRED_FIELDS):
        raise RunnerReleaseHold("independent review fields are not exact")
    if review.get("schema") != "public-q24-three-bank-runtime-static-review-v5":
        raise RunnerReleaseHold("independent review schema mismatch")
    if review.get("status") != "GO":
        raise RunnerReleaseHold("independent review is not GO")
    _validate_capability_text("review.issued_by", review.get("issued_by"))
    _validate_issued_utc(review.get("issued_utc"))
    if review.get("reviewed_runner_sha256") != runner_sha256:
        raise RunnerReleaseHold("independent review runner binding mismatch")
    if review.get("reviewed_callable_bundle_manifest_sha256") != bundle_sha256:
        raise RunnerReleaseHold("independent review bundle binding mismatch")
    if review.get("reviewed_execution_stack_manifest_sha256") != stack_sha256:
        raise RunnerReleaseHold("independent review stack binding mismatch")
    if review.get("prior_runtime_v1_hold_sha256") != bundle_manifest.get(
        "prior_runtime_v1_hold_sha256"
    ):
        raise RunnerReleaseHold("independent review V1 HOLD binding mismatch")
    if review.get("prior_runtime_v2_go_sha256") != PRIOR_RUNTIME_V2_GO_SHA256:
        raise RunnerReleaseHold("independent review V2 GO binding mismatch")
    if review.get("prior_runtime_v3_claim_sha256") != PRIOR_RUNTIME_V3_CLAIM_SHA256:
        raise RunnerReleaseHold("independent review V3 claim binding mismatch")
    if review.get(
        "prior_runtime_v3_terminal_failure_sha256"
    ) != PRIOR_RUNTIME_V3_TERMINAL_FAILURE_SHA256:
        raise RunnerReleaseHold("independent review V3 failure binding mismatch")
    if review.get(
        "prior_runtime_v3_consumed_capability_sha256"
    ) != PRIOR_RUNTIME_V3_CONSUMED_CAPABILITY_SHA256:
        raise RunnerReleaseHold("independent review V3 consumed capability mismatch")
    if review.get(
        "prior_runtime_v4_terminal_failure_lineage_sha256"
    ) != PRIOR_RUNTIME_V4_TERMINAL_LINEAGE_SHA256:
        raise RunnerReleaseHold("independent review V4 terminal lineage mismatch")
    if review.get("prior_runtime_v4_claim_sha256") != PRIOR_RUNTIME_V4_CLAIM_SHA256:
        raise RunnerReleaseHold("independent review V4 claim binding mismatch")
    if review.get(
        "prior_runtime_v4_terminal_failure_sha256"
    ) != PRIOR_RUNTIME_V4_TERMINAL_FAILURE_SHA256:
        raise RunnerReleaseHold("independent review V4 failure binding mismatch")
    if review.get(
        "prior_runtime_v4_consumed_capability_sha256"
    ) != PRIOR_RUNTIME_V4_CONSUMED_CAPABILITY_SHA256:
        raise RunnerReleaseHold("independent review V4 consumed capability mismatch")
    if review.get(
        "independent_incident_root_cause_sha256"
    ) != INDEPENDENT_INCIDENT_ROOT_CAUSE_SHA256:
        raise RunnerReleaseHold("independent review incident root-cause binding mismatch")
    if review.get(
        "flopscope_reset_verification_sha256"
    ) != FLOPSCOPE_RESET_VERIFICATION_SHA256:
        raise RunnerReleaseHold("independent review reset verification binding mismatch")
    if review.get(
        "manifest_compatibility_scan_receipt_sha256"
    ) != MANIFEST_COMPATIBILITY_SCAN_RECEIPT_SHA256:
        raise RunnerReleaseHold("independent review manifest scan binding mismatch")
    for name in (
        "r1_public_api_and_governance_chronology_closed",
        "r2_authority_dag_acyclic",
        "r3_resource_veto_removed",
        "r4_dynamic_comparators_present",
        "r5_manifest_firewall_semantic_repair_verified",
        "r6_flopscope_reset_placement_and_invariance_verified",
        "v3_terminal_no_retry_closed",
        "v4_terminal_no_retry_closed",
    ):
        if review.get(name) is not True:
            raise RunnerReleaseHold("independent review closure failed: " + name)
    if review.get("hostile_interpreter_security_claim") is not False:
        raise RunnerReleaseHold("review overclaims hostile-interpreter security")
    if review.get("source_data_target_execution_during_review") is not False:
        raise RunnerReleaseHold("independent static review executed protected data")
    _reject_forbidden_payload(review)
    return review, review_sha256


def _validate_anchor(
    anchor_path,
    *,
    runner_sha256,
    bundle_sha256,
    stack_sha256,
    review_sha256,
    scope_sha256,
    ordered_record_sha256,
):
    raw, anchor = _read_json(anchor_path)
    anchor_sha256 = hashlib.sha256(raw).hexdigest()
    if set(anchor) != set(ANCHOR_REQUIRED_FIELDS):
        raise RunnerReleaseHold("external authority-anchor fields are not exact")
    if anchor.get("schema") != "public-q24-three-bank-runtime-authority-anchor-v5":
        raise RunnerReleaseHold("external authority-anchor schema mismatch")
    if anchor.get("status") != "FROZEN_RUNTIME_AND_ORDERED_SCOPE_ALLOWLIST":
        raise RunnerReleaseHold("external authority-anchor status mismatch")
    _validate_capability_text("anchor.issued_by", anchor.get("issued_by"))
    _validate_issued_utc(anchor.get("issued_utc"))
    bindings = anchor.get("bindings")
    expected = {
        "runner_sha256": runner_sha256,
        "callable_bundle_manifest_sha256": bundle_sha256,
        "execution_stack_manifest_sha256": stack_sha256,
        "independent_review_sha256": review_sha256,
        "ordered_688_row_scope_sha256": scope_sha256,
        "ordered_688_record_sha256": ordered_record_sha256,
        "manifest_compatibility_scan_receipt_sha256": (
            MANIFEST_COMPATIBILITY_SCAN_RECEIPT_SHA256
        ),
    }
    if bindings != expected:
        raise RunnerReleaseHold("external authority-anchor bindings mismatch")
    _validate_exact_booleans(
        anchor,
        {
            "target_access_allowed": False,
            "training_allowed": False,
            "estimator_deployment_allowed": False,
            "package_allowed": False,
            "submission_allowed": False,
        },
    )
    _reject_forbidden_payload(anchor)
    return anchor, anchor_sha256


def _validate_release(
    release_path, anchor, anchor_sha256, capability_sha256
):
    raw, release = _read_json(release_path)
    release_sha256 = hashlib.sha256(raw).hexdigest()
    if set(release) != set(RELEASE_REQUIRED_FIELDS):
        raise RunnerReleaseHold("final release fields are not exact")
    if release.get("schema") != "public-q24-three-bank-runtime-final-release-v5":
        raise RunnerReleaseHold("release schema mismatch")
    if release.get("status") != "AUTHORIZE_ONE_TARGET_BLIND_688_BANK_PASS":
        raise RunnerReleaseHold("release status mismatch")
    _validate_capability_text("release.issued_by", release.get("issued_by"))
    _validate_issued_utc(release.get("issued_utc"))
    if release.get("authority_anchor_sha256") != anchor_sha256:
        raise RunnerReleaseHold("release authority-anchor binding mismatch")
    if release.get("single_use_capability_sha256") != capability_sha256:
        raise RunnerReleaseHold("release capability binding mismatch")
    bindings = release.get("bindings", {})
    if bindings != anchor.get("bindings"):
        raise RunnerReleaseHold("release does not exactly replay anchor bindings")
    _validate_exact_booleans(
        release,
        {
            "one_use": True,
            "no_retry": True,
            "source_import_allowed": True,
            "row_access_allowed": True,
            "target_access_allowed": False,
            "training_allowed": False,
            "estimator_deployment_allowed": False,
            "package_allowed": False,
            "submission_allowed": False,
        },
    )
    _reject_forbidden_payload(release)
    return release, release_sha256


def _validate_row_scope(path, expected_sha256, expected_record_sha256):
    raw, scope = _read_json(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RunnerReleaseHold("ordered row-scope digest mismatch")
    if scope.get("schema") != "public-q24-signed-pair-ordered-688-scope-v1":
        raise RunnerReleaseHold("ordered row-scope schema mismatch")
    if scope.get("status") != "TARGET_FREE_COMPLETE_688_SCOPE":
        raise RunnerReleaseHold("ordered row-scope status mismatch")
    if set(scope) != {
        "schema",
        "status",
        "ordered_688_record_sha256",
        "records",
    }:
        raise RunnerReleaseHold("ordered row-scope fields are not exact")
    records = scope.get("records")
    if not isinstance(records, list) or len(records) != 688:
        raise RunnerReleaseHold("row scope must contain exactly 688 records")
    aggregate = hashlib.sha256()
    mlp_ids = set()
    for expected_rank, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != set(ROW_RECORD_REQUIRED_FIELDS):
            raise RunnerReleaseHold("row record fields are not exact")
        rank = record["rank"]
        mlp_id = record["mlp_id"]
        if type(rank) is not int or rank != expected_rank:
            raise RunnerReleaseHold("row rank must be the exact non-bool integer order")
        if type(mlp_id) is not int or mlp_id < 0 or mlp_id in mlp_ids:
            raise RunnerReleaseHold("row mlp_id must be a unique nonnegative non-bool integer")
        mlp_ids.add(mlp_id)
        for name in ("row_manifest_path", "weights_path"):
            value = record[name]
            if type(value) is not str or not value or value.strip() != value:
                raise RunnerReleaseHold("row record path is not a nonempty canonical string")
        for name in (
            "row_manifest_sha256",
            "weights_sha256",
            "weights_schema_sha256",
        ):
            if not _is_lower_hex_sha256(record[name]):
                raise RunnerReleaseHold("row record has an invalid SHA256: " + name)
        aggregate.update(_canonical_json_bytes(record))
    record_sha256 = aggregate.hexdigest()
    if (
        not _is_lower_hex_sha256(expected_record_sha256)
        or scope.get("ordered_688_record_sha256") != record_sha256
        or record_sha256 != expected_record_sha256
    ):
        raise RunnerReleaseHold("ordered canonical 688-record digest mismatch")
    _reject_forbidden_payload(scope)
    return records, hashlib.sha256(raw).hexdigest(), record_sha256


def _validate_capability(
    path,
    *,
    anchor,
    anchor_sha256,
    runner_sha256,
    bundle_sha256,
    stack_sha256,
    row_scope_sha256,
    ordered_record_sha256,
):
    raw, capability = _read_json(path)
    capability_sha256 = hashlib.sha256(raw).hexdigest()
    if capability.get("schema") != "public-q24-signed-pair-bank-capability-v5":
        raise RunnerReleaseHold("capability schema mismatch")
    if capability.get("status") != "AUTHORIZE_ONE_TARGET_BLIND_688_BANK_PASS":
        raise RunnerReleaseHold("capability status mismatch")
    if set(capability) != set(CAPABILITY_REQUIRED_FIELDS):
        raise RunnerReleaseHold("capability fields are not exact")
    _validate_capability_text("capability_id", capability.get("capability_id"))
    _validate_capability_text("issued_by", capability.get("issued_by"))
    _validate_issued_utc(capability.get("issued_utc"))
    if capability.get("authority_anchor_sha256") != anchor_sha256:
        raise RunnerReleaseHold("capability authority-anchor binding mismatch")
    if "release_authority_sha256" in capability or "final_release_sha256" in capability:
        raise RunnerReleaseHold("capability must never bind the final release")
    if capability.get("ordered_688_row_scope_sha256") != row_scope_sha256:
        raise RunnerReleaseHold("capability row-scope binding mismatch")
    if capability.get("ordered_688_record_sha256") != ordered_record_sha256:
        raise RunnerReleaseHold("capability ordered-record binding mismatch")
    if capability.get(
        "manifest_compatibility_scan_receipt_sha256"
    ) != MANIFEST_COMPATIBILITY_SCAN_RECEIPT_SHA256:
        raise RunnerReleaseHold("capability manifest scan binding mismatch")
    if capability.get("runner_sha256") != runner_sha256:
        raise RunnerReleaseHold("capability runner binding mismatch")
    if capability.get("callable_bundle_manifest_sha256") != bundle_sha256:
        raise RunnerReleaseHold("capability bundle binding mismatch")
    if capability.get("execution_stack_manifest_sha256") != stack_sha256:
        raise RunnerReleaseHold("capability stack binding mismatch")
    if anchor.get("bindings") != {
        "runner_sha256": runner_sha256,
        "callable_bundle_manifest_sha256": bundle_sha256,
        "execution_stack_manifest_sha256": stack_sha256,
        "independent_review_sha256": anchor["bindings"][
            "independent_review_sha256"
        ],
        "ordered_688_row_scope_sha256": row_scope_sha256,
        "ordered_688_record_sha256": ordered_record_sha256,
        "manifest_compatibility_scan_receipt_sha256": (
            MANIFEST_COMPATIBILITY_SCAN_RECEIPT_SHA256
        ),
    }:
        raise RunnerReleaseHold("capability sees inconsistent anchor bindings")
    _validate_exact_booleans(
        capability,
        {
            "one_use": True,
            "no_retry": True,
            "source_import_allowed": True,
            "row_access_allowed": True,
            "target_access_allowed": False,
            "training_allowed": False,
            "estimator_deployment_allowed": False,
            "package_allowed": False,
            "submission_allowed": False,
        },
    )
    _reject_forbidden_payload(capability)
    return raw, capability, capability_sha256


def _consume_capability(path, raw):
    original = Path(path)
    before = os.lstat(original)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        raise RunnerReleaseHold(
            "capability must be one regular nonsymlink inode with one link"
        )
    consumed = original.with_name(original.name + ".consumed")
    if consumed.exists():
        raise RunnerReleaseHold("capability consumption path already exists")
    os.link(original, consumed)
    directory_fd = os.open(str(original.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
        os.unlink(original)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if consumed.read_bytes() != raw:
        raise RunnerReleaseHold("consumed capability bytes changed")
    after = os.lstat(consumed)
    if (
        not stat.S_ISREG(after.st_mode)
        or stat.S_ISLNK(after.st_mode)
        or after.st_nlink != 1
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
    ):
        raise RunnerReleaseHold("capability inode transition is not exactly one-use")
    return consumed, {
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "size": int(after.st_size),
        "initial_link_count": 1,
        "consumed_link_count": 1,
    }


def _fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_json(path, document):
    payload = _canonical_json_bytes(document)
    descriptor = os.open(
        str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise RunnerReleaseHold("short exclusive receipt write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(Path(path).parent)
    if Path(path).read_bytes() != payload:
        raise RunnerReleaseHold("durable exclusive receipt replay mismatch")
    return hashlib.sha256(payload).hexdigest()


def _claim_paths(transaction_root):
    root = Path(transaction_root)
    return {
        "root": root,
        "claim": root / "THREE_BANK_RUN.CLAIM.json",
        "failure": root / "THREE_BANK_RUN.FAILURE.json",
        "bundle": root / "THREE_BANK_BUNDLE",
        "staging": root / ".THREE_BANK_BUNDLE.staging",
    }


def _create_claim_exclusive(transaction_root, claim_document, ownership):
    if ownership != {"acquired": False}:
        raise RunnerReleaseHold("claim ownership state is not pristine")
    paths = _claim_paths(transaction_root)
    root = _require_nonsymlink_directory(
        paths["root"], "transaction root"
    )
    paths = _claim_paths(root)
    if any(root.iterdir()):
        raise RunnerReleaseHold("transaction root must be empty before exclusive claim")
    payload = _canonical_json_bytes(claim_document)
    claim_sha256 = hashlib.sha256(payload).hexdigest()
    descriptor = os.open(
        str(paths["claim"]), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    ownership.update(
        {
            "acquired": True,
            "expected_claim_sha256": claim_sha256,
            "claim_path": str(paths["claim"]),
        }
    )
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise RunnerReleaseHold("short exclusive claim write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(root)
    if paths["claim"].read_bytes() != payload:
        raise RunnerReleaseHold("durable exclusive claim replay mismatch")
    if {item.name for item in root.iterdir()} != {paths["claim"].name}:
        raise RunnerReleaseHold("unexpected transaction-root entry after claim")
    return paths, claim_sha256


def _write_owned_failure(paths, claim_sha256, error):
    if not paths["claim"].is_file() or paths["claim"].is_symlink():
        raise RunnerReleaseHold("claim owner has no regular claim inode")
    actual_claim_sha256 = _sha256(paths["claim"])
    document = {
        "schema": "public-q24-three-bank-runtime-terminal-failure-v1",
        "status": "TERMINAL_POST_CLAIM_FAILURE__NO_RETRY",
        "claim_sha256": claim_sha256,
        "actual_claim_sha256": actual_claim_sha256,
        "claim_complete": actual_claim_sha256 == claim_sha256,
        "error_type": type(error).__name__,
        "error_message": str(error)[:512],
        "published_bundle": paths["bundle"].exists(),
        "staging_preserved_for_forensics": paths["staging"].exists(),
    }
    _write_exclusive_json(paths["failure"], document)


def _publish_bundle(paths):
    if paths["bundle"].exists():
        raise RunnerReleaseHold("final bundle already exists")
    if {item.name for item in paths["root"].iterdir()} != {
        paths["claim"].name,
        paths["staging"].name,
    }:
        raise RunnerReleaseHold("transaction-root inventory drift before publish")
    expected = {
        "actions_H16.f32le",
        "actions_H64.f32le",
        "actions_H256.f32le",
        "A15_zero_H16.f32le",
        "A15_zero_H64.f32le",
        "A15_zero_H256.f32le",
        "THREE_BANK_GENERATION_RECEIPT.json",
    }
    entries = list(paths["staging"].iterdir())
    if (
        {item.name for item in entries} != expected
        or any(not item.is_file() or item.is_symlink() for item in entries)
    ):
        raise RunnerReleaseHold("atomic staging inventory is not exactly seven files")
    _fsync_directory(paths["staging"])
    os.rename(paths["staging"], paths["bundle"])
    _fsync_directory(paths["root"])
    if not paths["bundle"].is_dir() or paths["staging"].exists():
        raise RunnerReleaseHold("atomic bundle rename did not commit exactly once")


def _load_module(path, name, expected_sha256):
    if _sha256(path) != expected_sha256:
        raise RunnerReleaseHold(name + " digest mismatch")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RunnerReleaseHold("cannot create module specification: " + name)
    module = importlib.util.module_from_spec(specification)
    if name in sys.modules:
        raise RunnerReleaseHold("runtime module name is already occupied: " + name)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_token_and_mint(
    path,
    expected_sha256,
    *,
    claim_sha256,
    consumed_capability_sha256,
):
    """Load token code with one transient post-consume mint closure."""
    if _sha256(path) != expected_sha256:
        raise RunnerReleaseHold("runtime token digest mismatch")
    if TOKEN_MODULE_IMPORT_NAME in sys.modules:
        raise RunnerReleaseHold("runtime token module name is already occupied")
    specification = importlib.util.spec_from_file_location(
        TOKEN_MODULE_IMPORT_NAME, path
    )
    if specification is None or specification.loader is None:
        raise RunnerReleaseHold("cannot create runtime token specification")
    module = importlib.util.module_from_spec(specification)
    mint_sink = []
    module.__dict__["_Q24_POST_CONSUME_MINT_SINK"] = mint_sink
    sys.modules[TOKEN_MODULE_IMPORT_NAME] = module
    try:
        specification.loader.exec_module(module)
        if len(mint_sink) != 1 or not callable(mint_sink[0]):
            raise RunnerReleaseHold("token module did not yield one mint closure")
        if any(
            "mint" in name.lower()
            for name in module.__dict__
            if not name.startswith("__")
        ):
            raise RunnerReleaseHold("token module exposed a mint attribute")
        mint_once = mint_sink.pop()
        token = mint_once(
            claim_sha256=claim_sha256,
            consumed_capability_sha256=consumed_capability_sha256,
        )
        del mint_once
        if mint_sink:
            raise RunnerReleaseHold("transient mint sink did not clear")
        module.require_runtime_token(token)
        return module, token
    except BaseException:
        mint_sink.clear()
        sys.modules.pop(TOKEN_MODULE_IMPORT_NAME, None)
        raise


def _validate_row_result(result, record, component_sha256):
    rank = record["rank"]
    sentinel = rank in SENTINEL_RANKS
    if (
        not isinstance(result, dict)
        or set(result) != ROW_RESULT_REQUIRED_FIELDS
        or type(result.get("rank")) is not int
        or result.get("rank") != rank
    ):
        raise RunnerReleaseHold("adapter returned wrong row rank")
    for name in (
        "mlp_id",
        "row_manifest_sha256",
        "weights_sha256",
        "weights_schema_sha256",
    ):
        if result.get(name) != record.get(name):
            raise RunnerReleaseHold("adapter row identity mismatch: " + name)
    if result.get("capture_layers_zero_based") != list(EXPECTED_CAPTURE_LAYERS):
        raise RunnerReleaseHold("adapter returned wrong capture-layer ledger")
    if result.get("same_layer_preactivation_alpha") is not True:
        raise RunnerReleaseHold("same-layer preactivation alpha not attested")
    if result.get("capture_after_full_reset_before_reset_stars") is not True:
        raise RunnerReleaseHold("capture seam not attested")
    if result.get("A15_exact_zero_float32") is not True:
        raise RunnerReleaseHold("A15 zero action not attested")
    if result.get("cross_rung_H16_prefix_byte_identical") is not True:
        raise RunnerReleaseHold("cross-rung prefix parity failed")
    if result.get("sentinel_rank") is not sentinel:
        raise RunnerReleaseHold("row sentinel identity mismatch")
    expected_mode = (
        "SENTINEL_DUAL_PARITY" if sentinel else "LEAN_CAPTURE_LITERAL_ONLY"
    )
    if result.get("comparator_mode") != expected_mode:
        raise RunnerReleaseHold("row comparator mode mismatch")
    derivation = result.get("h256_prefix_derivation")
    if not isinstance(derivation, dict) or set(derivation) != {
        "method",
        "source_h",
        "derived_h",
        "all_derived_bytes_match_written_row",
        "sentinel_derived_matches_literal_and_reference",
    }:
        raise RunnerReleaseHold("H256 prefix derivation schema mismatch")
    if (
        derivation.get("method") != "EIGHT_SECTOR_RECEIVER_PREFIX_FROM_H256"
        or derivation.get("source_h") != 256
        or derivation.get("derived_h") != [16, 64]
        or derivation.get("all_derived_bytes_match_written_row") is not True
        or derivation.get("sentinel_derived_matches_literal_and_reference")
        is not (True if sentinel else None)
    ):
        raise RunnerReleaseHold("H256 prefix derivation evidence mismatch")
    source_comparator = result.get("source_action_comparator")
    parent_comparator = result.get("five_component_parent_comparator")
    if sentinel:
        _validate_source_action_comparator(source_comparator, component_sha256)
        _validate_five_component_parent_comparator(parent_comparator)
    elif source_comparator is not None or parent_comparator is not None:
        raise RunnerReleaseHold("non-sentinel row contains comparator evidence")
    bank = result.get("banks", {})
    if not isinstance(bank, dict) or set(bank) != {"16", "64", "256"}:
        raise RunnerReleaseHold("row bank keyset is not exact")
    expected_shapes = {
        "16": [13, 1024, 16],
        "64": [13, 1024, 64],
        "256": [13, 1024, 256],
    }
    for h in EXPECTED_H:
        item = bank.get(str(h), {})
        if set(item) != {
            "shape",
            "dtype",
            "finite",
            "action_bytes_sha256",
            "layer_action_bytes_sha256",
            "A15_zero_bytes_sha256",
        }:
            raise RunnerReleaseHold("row bank fields are not exact for H" + str(h))
        if item.get("shape") != expected_shapes[str(h)]:
            raise RunnerReleaseHold("wrong bank shape for H" + str(h))
        if item.get("dtype") != "float32" or item.get("finite") is not True:
            raise RunnerReleaseHold("invalid dtype/finiteness for H" + str(h))
        if not _is_lower_hex_sha256(item.get("action_bytes_sha256")):
            raise RunnerReleaseHold("missing row action hash for H" + str(h))
        layer_hashes = item.get("layer_action_bytes_sha256")
        if (
            not isinstance(layer_hashes, list)
            or len(layer_hashes) != len(EXPECTED_CAPTURE_LAYERS)
            or any(not _is_lower_hex_sha256(value) for value in layer_hashes)
        ):
            raise RunnerReleaseHold("missing canonical layer-action hashes for H" + str(h))
        if not _is_lower_hex_sha256(item.get("A15_zero_bytes_sha256")):
            raise RunnerReleaseHold("missing A15 zero hash for H" + str(h))
        if sentinel:
            comparator_item = source_comparator["literal"][str(h)]
            if comparator_item["action_sha256"] != item["action_bytes_sha256"]:
                raise RunnerReleaseHold("comparator/output action hash mismatch")
            if comparator_item["A15_zero_sha256"] != item["A15_zero_bytes_sha256"]:
                raise RunnerReleaseHold("comparator/output A15 hash mismatch")
    meters = result.get("exact_meters", {})
    _validate_exact_meter_scalars(meters)
    if type(meters.get("source_hook_capture_count")) is not int or meters.get(
        "source_hook_capture_count"
    ) != 13:
        raise RunnerReleaseHold("source hook meter does not span 13 captures")
    if meters.get("reference_source_hook_capture_count") != (13 if sentinel else 0):
        raise RunnerReleaseHold("reference hook sentinel meter mismatch")
    for name in (
        "base_parent_flops",
        "base_parent_calls",
        "reference_source_hook_flops",
        "reference_source_hook_calls",
        "reference_workspace_flops",
        "reference_workspace_calls",
    ):
        value = meters.get(name)
        if not sentinel and value != 0:
            raise RunnerReleaseHold("non-sentinel duplicate-work meter is nonzero: " + name)
    if meters.get("meter_scope") != (
        "CAPTURE_LITERAL_ALL_ROWS_BASE_REFERENCE_SENTINELS"
    ):
        raise RunnerReleaseHold("exact row meter scope mismatch")
    observation = result.get("resource_observation")
    if not isinstance(observation, dict) or set(observation) != {
        "classification",
        "deployment_flop_cap",
        "complete_flops_within_deployment_cap",
        "peak_rss_bytes",
        "wall_seconds",
        "residual_seconds",
        "accuracy_veto_allowed",
    }:
        raise RunnerReleaseHold("row resource observation schema mismatch")
    if observation.get("classification") != "OBSERVATIONAL_DEPLOYMENT_ONLY":
        raise RunnerReleaseHold("row resource observation classification drift")
    if observation.get("deployment_flop_cap") != 2_199_023_255_552:
        raise RunnerReleaseHold("row deployment cap observation drift")
    if type(observation.get("complete_flops_within_deployment_cap")) is not bool:
        raise RunnerReleaseHold("row FLOP observation must be an exact boolean")
    if observation.get("accuracy_veto_allowed") is not False:
        raise RunnerReleaseHold("resource observation attempted an accuracy veto")
    for name in ("peak_rss_bytes", "wall_seconds", "residual_seconds"):
        if observation.get(name) != meters[name]:
            raise RunnerReleaseHold("resource observation/meter mismatch: " + name)
    _reject_forbidden_payload(result)


def _validate_final_artifacts(output_directory, final, row_receipts):
    root = Path(output_directory).resolve()
    expected_tape_bytes = {
        "16": 586153984,
        "64": 2344615936,
        "256": 9378463744,
    }
    expected_A15_bytes = {
        "16": 65536,
        "64": 262144,
        "256": 1048576,
    }
    expected_files = set()
    for h in EXPECTED_H:
        bank = final["banks"][str(h)]
        if (
            type(bank.get("path")) is not str
            or Path(bank["path"]).name != bank["path"]
            or type(bank.get("A15_zero_path")) is not str
            or Path(bank["A15_zero_path"]).name != bank["A15_zero_path"]
        ):
            raise RunnerReleaseHold("bank artifact name is not one canonical leaf")
        tape_path = _require_regular_nonsymlink_file(
            root / bank["path"], "bank tape"
        )
        zero_path = _require_regular_nonsymlink_file(
            root / bank["A15_zero_path"], "A15 buffer"
        )
        if tape_path.parent != root or zero_path.parent != root:
            raise RunnerReleaseHold("bank artifact escapes output directory")
        if tape_path.stat().st_size != expected_tape_bytes[str(h)]:
            raise RunnerReleaseHold("complete bank byte count mismatch")
        if zero_path.stat().st_size != expected_A15_bytes[str(h)]:
            raise RunnerReleaseHold("A15 zero byte count mismatch")
        if _sha256(tape_path) != bank["sha256"]:
            raise RunnerReleaseHold("complete bank file digest mismatch")
        if _sha256(zero_path) != bank["A15_zero_sha256"]:
            raise RunnerReleaseHold("A15 zero file digest mismatch")
        row_bytes = len(EXPECTED_CAPTURE_LAYERS) * 1024 * h * 4
        layer_bytes = 1024 * h * 4
        with tape_path.open("rb") as handle:
            for rank in EXPECTED_RANKS:
                item = row_receipts[rank]["banks"][str(h)]
                handle.seek(rank * row_bytes)
                payload = handle.read(row_bytes)
                if len(payload) != row_bytes:
                    raise RunnerReleaseHold("short exact row read for H" + str(h))
                if hashlib.sha256(payload).hexdigest() != item["action_bytes_sha256"]:
                    raise RunnerReleaseHold("exact row action digest mismatch for H" + str(h))
                layer_hashes = item["layer_action_bytes_sha256"]
                for layer in EXPECTED_CAPTURE_LAYERS:
                    slot = layer - EXPECTED_CAPTURE_LAYERS[0]
                    start = slot * layer_bytes
                    layer_payload = payload[start : start + layer_bytes]
                    if hashlib.sha256(layer_payload).hexdigest() != layer_hashes[slot]:
                        raise RunnerReleaseHold(
                            "exact layer action digest mismatch for H" + str(h)
                        )
        with zero_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if any(chunk):
                    raise RunnerReleaseHold("A15 contains a nonzero byte")
        expected_files.add(tape_path)
        expected_files.add(zero_path)
    entries = list(root.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise RunnerReleaseHold("unexpected non-file bank artifact")
    actual_files = {path.resolve() for path in entries}
    if actual_files != expected_files:
        raise RunnerReleaseHold("unexpected or missing pre-receipt bank artifact")


def _validate_complete_cross_rung_prefixes(output_directory, final):
    root = Path(output_directory).resolve()
    paths = {
        h: (root / final["banks"][str(h)]["path"]).resolve()
        for h in EXPECTED_H
    }
    layer_bytes = {h: 1024 * h * 4 for h in EXPECTED_H}
    with (
        paths[16].open("rb") as handle16,
        paths[64].open("rb") as handle64,
        paths[256].open("rb") as handle256,
    ):
        for rank in EXPECTED_RANKS:
            for layer in EXPECTED_CAPTURE_LAYERS:
                narrow16 = handle16.read(layer_bytes[16])
                narrow64 = handle64.read(layer_bytes[64])
                wide = handle256.read(layer_bytes[256])
                if (
                    len(narrow16) != layer_bytes[16]
                    or len(narrow64) != layer_bytes[64]
                    or len(wide) != layer_bytes[256]
                ):
                    raise RunnerReleaseHold("short cross-rung layer read")
                expected16 = bytearray(layer_bytes[16])
                expected64 = bytearray(layer_bytes[64])
                for unit in range(1024):
                    for sector in range(8):
                        wide_start = (unit * 256 + sector * 32) * 4
                        start16 = (unit * 16 + sector * 2) * 4
                        start64 = (unit * 64 + sector * 8) * 4
                        expected16[start16:start16 + 8] = wide[
                            wide_start:wide_start + 8
                        ]
                        expected64[start64:start64 + 32] = wide[
                            wide_start:wide_start + 32
                        ]
                if narrow16 != expected16 or narrow64 != expected64:
                    raise RunnerReleaseHold(
                        "complete cross-rung prefix mismatch at rank/layer "
                        + str(rank) + "/" + str(layer)
                    )


def _write_receipt(
    output_directory,
    *,
    bundle_manifest_sha256,
    execution_stack_manifest_sha256,
    component_sha256,
    independent_review_sha256,
    anchor,
    anchor_sha256,
    capability_sha256,
    release,
    release_sha256,
    claim_sha256,
    token_receipt,
    scope_sha256,
    ordered_record_sha256,
    rows,
    final,
):
    receipt = {
        "schema": "public-q24-signed-pair-three-bank-generation-receipt-v4",
        "status": "SEALED_TARGET_BLIND_THREE_BANKS",
        "authority_chain": {
            "callable_bundle_manifest_sha256": bundle_manifest_sha256,
            "execution_stack_manifest_sha256": execution_stack_manifest_sha256,
            "independent_review_sha256": independent_review_sha256,
            "authority_anchor_sha256": anchor_sha256,
            "single_use_capability_sha256": capability_sha256,
            "final_release_sha256": release_sha256,
            "acyclic": True,
            "capability_binds_final_release": False,
            "bundle_binds_transaction_authority": False,
        },
        "callable_component_sha256": component_sha256,
        "claim_sha256": claim_sha256,
        "runtime_token_receipt": token_receipt,
        "release_bindings": release["bindings"],
        "anchor_bindings": anchor["bindings"],
        "ordered_688_row_scope_sha256": scope_sha256,
        "ordered_688_record_sha256": ordered_record_sha256,
        "ordered_ranks": list(EXPECTED_RANKS),
        "row_receipts": rows,
        "complete_banks": final,
        "lean_execution_policy": {
            "captured_parent_and_literal_source_all_ranks": True,
            "unhooked_base_and_reference_source_sentinel_ranks": list(
                SENTINEL_RANKS
            ),
            "literal_reference_parity_scope": "SENTINELS_ONLY",
            "base_capture_parity_scope": "SENTINELS_ONLY",
            "H16_H64_materialization": (
                "EXACT_EIGHT_SECTOR_RECEIVER_PREFIXES_FROM_H256_ALL_ROWS"
            ),
            "all_688_rows_banked": True,
        },
        "resource_policy": (
            "all FLOP/call/RSS/wall/residual meters are observational for "
            "accuracy science; any resource miss is deployment-only and cannot "
            "veto, erase, or alter a correct bank"
        ),
        "fourteen_parent_law_synchronizations_allowed_for_accuracy_science": True,
        "sentinel_duplicate_timing_is_deployment_only": True,
        "target_accessed": False,
        "training_run": False,
        "estimator_deployed": False,
        "package_built": False,
        "submission_made": False,
    }
    output = Path(output_directory) / "THREE_BANK_GENERATION_RECEIPT.json"
    payload = _canonical_json_bytes(receipt)
    descriptor = os.open(
        str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise RunnerReleaseHold("short generation-receipt write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(output.parent)
    if output.read_bytes() != payload:
        raise RunnerReleaseHold("generation receipt durable replay mismatch")
    return output


def _arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--callable-bundle", required=True)
    parser.add_argument("--execution-stack", required=True)
    parser.add_argument("--independent-review", required=True)
    parser.add_argument("--authority-anchor", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--capability", required=True)
    parser.add_argument("--row-scope", required=True)
    parser.add_argument("--runtime-token", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--literal-source", required=True)
    parser.add_argument("--reference-source", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--transaction-root", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _arguments(sys.argv[1:] if argv is None else argv)
    runner_path = Path(__file__).resolve()
    for path, label in (
        (args.callable_bundle, "callable-bundle manifest"),
        (args.execution_stack, "execution-stack manifest"),
        (args.independent_review, "independent review"),
        (args.authority_anchor, "authority anchor"),
        (args.release, "final release"),
        (args.capability, "single-use capability"),
        (args.row_scope, "ordered-688 scope"),
        (args.runtime_token, "runtime token"),
        (args.parent, "parent successor"),
        (args.literal_source, "literal source"),
        (args.reference_source, "reference source"),
        (args.adapter, "capture adapter"),
    ):
        _require_regular_nonsymlink_file(path, label)
    _require_nonsymlink_directory(args.transaction_root, "transaction root")

    # Only authority, scope, and immutable code bytes are opened before the
    # claim.  No numerical module is imported and no row/weight path is opened.
    runner_sha256 = _sha256(runner_path)
    component_paths = {
        "runtime_token.py": args.runtime_token,
        "parent_capture.py": args.parent,
        "signed_pair_literal.py": args.literal_source,
        "signed_pair_reference.py": args.reference_source,
        "capture_adapter.py": args.adapter,
    }
    bundle_manifest, bundle_manifest_sha256, component_sha256 = _validate_callable_bundle(
        args.callable_bundle, component_paths
    )
    _, execution_stack_manifest_sha256 = _validate_execution_stack(
        args.execution_stack
    )
    _, independent_review_sha256 = _validate_independent_review(
        args.independent_review,
        runner_sha256,
        bundle_manifest_sha256,
        execution_stack_manifest_sha256,
        bundle_manifest,
    )
    _, anchor_hint = _read_json(args.authority_anchor)
    hinted_scope_sha256 = anchor_hint.get("bindings", {}).get(
        "ordered_688_row_scope_sha256"
    )
    hinted_record_sha256 = anchor_hint.get("bindings", {}).get(
        "ordered_688_record_sha256"
    )
    if not _is_lower_hex_sha256(hinted_scope_sha256):
        raise RunnerReleaseHold("anchor has no valid row-scope hint")
    if not _is_lower_hex_sha256(hinted_record_sha256):
        raise RunnerReleaseHold("anchor has no valid ordered-record hint")
    records, scope_sha256, ordered_record_sha256 = _validate_row_scope(
        args.row_scope,
        hinted_scope_sha256,
        hinted_record_sha256,
    )
    anchor, anchor_sha256 = _validate_anchor(
        args.authority_anchor,
        runner_sha256=runner_sha256,
        bundle_sha256=bundle_manifest_sha256,
        stack_sha256=execution_stack_manifest_sha256,
        review_sha256=independent_review_sha256,
        scope_sha256=scope_sha256,
        ordered_record_sha256=ordered_record_sha256,
    )
    bindings = anchor["bindings"]
    capability_raw, capability, capability_sha256 = _validate_capability(
        args.capability,
        anchor=anchor,
        anchor_sha256=anchor_sha256,
        runner_sha256=runner_sha256,
        bundle_sha256=bundle_manifest_sha256,
        stack_sha256=execution_stack_manifest_sha256,
        row_scope_sha256=scope_sha256,
        ordered_record_sha256=ordered_record_sha256,
    )
    release, release_sha256 = _validate_release(
        args.release, anchor, anchor_sha256, capability_sha256
    )

    claim_document = {
        "schema": "public-q24-three-bank-runtime-exclusive-claim-v5",
        "status": "CLAIMED_ONE_TARGET_BLIND_688_BANK_PASS",
        "callable_bundle_manifest_sha256": bundle_manifest_sha256,
        "execution_stack_manifest_sha256": execution_stack_manifest_sha256,
        "independent_review_sha256": independent_review_sha256,
        "authority_anchor_sha256": anchor_sha256,
        "single_use_capability_sha256": capability_sha256,
        "final_release_sha256": release_sha256,
        "runner_sha256": runner_sha256,
        "callable_component_sha256": component_sha256,
        "ordered_688_row_scope_sha256": scope_sha256,
        "ordered_688_record_sha256": ordered_record_sha256,
        "capability_id": capability["capability_id"],
        "target_access_allowed": False,
        "training_allowed": False,
        "package_allowed": False,
        "submission_allowed": False,
    }
    claim_ownership = {"acquired": False}
    paths = _claim_paths(args.transaction_root)
    claim_sha256 = None
    try:
        paths, claim_sha256 = _create_claim_exclusive(
            args.transaction_root, claim_document, claim_ownership
        )

        # The irreversible transition precedes every numerical import and row
        # manifest/weight open.  A lock-race loser never reaches this point.
        consumed_capability, consumed_inode = _consume_capability(
            args.capability, capability_raw
        )
        consumed_capability_sha256 = _sha256(consumed_capability)
        if consumed_capability_sha256 != capability_sha256:
            raise RunnerReleaseHold("consumed capability digest drift")
        os.mkdir(paths["staging"], 0o700)
        _fsync_directory(paths["root"])

        # Token loading/minting is the first import after durable claim and
        # irreversible consume.  The mint closure never remains module-visible.
        token_module, runtime_token = _load_token_and_mint(
            args.runtime_token,
            component_sha256["runtime_token.py"],
            claim_sha256=claim_sha256,
            consumed_capability_sha256=consumed_capability_sha256,
        )
        token_receipt = token_module.runtime_token_receipt_fields(runtime_token)
        parent = _load_module(
            args.parent,
            "signed_pair_sync_minimized_parent_runtime_v3",
            component_sha256["parent_capture.py"],
        )
        literal_source = _load_module(
            args.literal_source,
            "signed_pair_three_bank_literal_runtime_v3",
            component_sha256["signed_pair_literal.py"],
        )
        reference_source = _load_module(
            args.reference_source,
            "signed_pair_three_bank_reference_runtime_v3",
            component_sha256["signed_pair_reference.py"],
        )
        adapter = _load_module(
            args.adapter,
            "signed_pair_three_bank_capture_adapter_runtime_v3",
            component_sha256["capture_adapter.py"],
        )

        session = adapter.begin_target_blind_three_bank_pass(
            staging_directory=paths["staging"],
            parent_module=parent,
            literal_source_module=literal_source,
            reference_source_module=reference_source,
            runtime_token=runtime_token,
            callable_component_sha256=component_sha256,
            expected_ranks=EXPECTED_RANKS,
        )
        # Reuse the exact module object loaded by the adapter; a second import
        # identity is neither needed nor allowed by the reset proof.
        flopscope = session["flops"]
        row_receipts = []
        for rank, record in enumerate(records):
            result = adapter.generate_one_target_blind_row(
                session=session,
                record=record,
                expected_rank=rank,
                runtime_token=runtime_token,
            )
            _validate_row_result(result, record, component_sha256)
            row_receipts.append(result)
            completed_rows = rank + 1
            if completed_rows % 8 == 0 or completed_rows == len(EXPECTED_RANKS):
                meters = result["exact_meters"]
                print(
                    "lean-bank-progress completed_rows=" + str(completed_rows)
                    + " last_rank=" + str(rank)
                    + " last_row_wall_seconds=" + str(meters["wall_seconds"])
                    + " peak_rss_bytes=" + str(meters["peak_rss_bytes"]),
                    flush=True,
                )
            # The adapter has returned only after its BudgetContext closed and
            # its detached summary, exact meters, bank bytes/hashes, row ledger,
            # and scheduled flush were materialized.  Validation, receipt append,
            # and optional progress are also complete.  Clearing only the
            # process-global diagnostic history here cannot mutate those values.
            flopscope.budget_reset()
        final = adapter.finish_target_blind_three_bank_pass(
            session=session, runtime_token=runtime_token
        )
        final["capability_consumption"] = {
            "capability_id": capability["capability_id"],
            "issued_by": capability["issued_by"],
            "issued_utc": capability["issued_utc"],
            "consumed_path": str(consumed_capability),
            "consumed_sha256": consumed_capability_sha256,
            "original_absent": not Path(args.capability).exists(),
            "one_use_inode": consumed_inode,
            "one_use": True,
            "no_retry": True,
        }
        if set(final) != {
            "row_count",
            "ranks",
            "sentinel_ranks",
            "banks",
            "exact_meters",
            "cross_rung_H16_prefix_byte_identical",
            "h256_prefix_derivation_complete_all_rows",
            "literal_reference_bytes_equal_sentinels",
            "base_capture_five_components_equal_sentinels",
            "resource_policy",
            "capability_consumption",
        }:
            raise RunnerReleaseHold("adapter final fields are not exact")
        if final.get("row_count") != 688:
            raise RunnerReleaseHold("adapter did not finish exactly 688 rows")
        if (
            not isinstance(final.get("ranks"), list)
            or len(final["ranks"]) != 688
            or any(type(value) is not int for value in final["ranks"])
            or final["ranks"] != list(EXPECTED_RANKS)
        ):
            raise RunnerReleaseHold("adapter final rank ledger mismatch")
        if final.get("sentinel_ranks") != list(SENTINEL_RANKS):
            raise RunnerReleaseHold("adapter sentinel-rank ledger mismatch")
        if final.get("cross_rung_H16_prefix_byte_identical") is not True:
            raise RunnerReleaseHold("complete-bank prefix parity failed")
        if final.get("h256_prefix_derivation_complete_all_rows") is not True:
            raise RunnerReleaseHold("complete H256 prefix derivation failed")
        if final.get("literal_reference_bytes_equal_sentinels") is not True:
            raise RunnerReleaseHold("sentinel literal/reference parity failed")
        if final.get("base_capture_five_components_equal_sentinels") is not True:
            raise RunnerReleaseHold("sentinel base/capture parity failed")
        if final.get("resource_policy") != (
            "RESOURCE_AND_RESIDUAL_OBSERVATIONAL_DEPLOYMENT_ONLY"
        ):
            raise RunnerReleaseHold("complete resource policy drift")
        expected_bank_shapes = {
            "16": [688, 13, 1024, 16],
            "64": [688, 13, 1024, 64],
            "256": [688, 13, 1024, 256],
        }
        if set(final.get("banks", {})) != {"16", "64", "256"}:
            raise RunnerReleaseHold("complete bank keyset is not exact")
        for h in EXPECTED_H:
            bank = final.get("banks", {}).get(str(h), {})
            if set(bank) != {
                "path",
                "shape",
                "dtype",
                "finite",
                "sha256",
                "A15_zero_path",
                "A15_zero_sha256",
            }:
                raise RunnerReleaseHold("complete bank fields are not exact")
            if (
                not _is_lower_hex_sha256(bank.get("sha256"))
                or not _is_lower_hex_sha256(bank.get("A15_zero_sha256"))
            ):
                raise RunnerReleaseHold("complete bank/A15 hash missing")
            if bank.get("shape") != expected_bank_shapes[str(h)]:
                raise RunnerReleaseHold("complete bank shape mismatch")
            if bank.get("dtype") != "float32" or bank.get("finite") is not True:
                raise RunnerReleaseHold("complete bank dtype/finiteness mismatch")
        _validate_exact_meter_vector_alignment(
            final.get("exact_meters", {}), row_receipts
        )
        _reject_forbidden_payload(final)
        _validate_final_artifacts(paths["staging"], final, row_receipts)
        _validate_complete_cross_rung_prefixes(paths["staging"], final)
        _validate_row_A15_hash_alignment(final, row_receipts)
        _write_receipt(
            paths["staging"],
            bundle_manifest_sha256=bundle_manifest_sha256,
            execution_stack_manifest_sha256=execution_stack_manifest_sha256,
            component_sha256=component_sha256,
            independent_review_sha256=independent_review_sha256,
            anchor=anchor,
            anchor_sha256=anchor_sha256,
            capability_sha256=capability_sha256,
            release=release,
            release_sha256=release_sha256,
            claim_sha256=claim_sha256,
            token_receipt=token_receipt,
            scope_sha256=scope_sha256,
            ordered_record_sha256=ordered_record_sha256,
            rows=row_receipts,
            final=final,
        )
        _publish_bundle(paths)
    except BaseException as error:
        # A process that lost O_EXCL never sets acquired and therefore may
        # not write a failure against the winner's transaction.
        if (
            claim_ownership.get("acquired") is True
            and not paths["failure"].exists()
        ):
            _write_owned_failure(
                paths,
                claim_ownership["expected_claim_sha256"],
                error,
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
