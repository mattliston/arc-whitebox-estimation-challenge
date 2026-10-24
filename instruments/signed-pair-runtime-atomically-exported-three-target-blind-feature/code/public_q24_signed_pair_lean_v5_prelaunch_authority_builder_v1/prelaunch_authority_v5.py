#!/usr/bin/env python3
"""Fail-closed control-plane builder for one fresh lean V5 bank attempt.

The static HOLD path opens no V4 staging entry, row manifest, weight, target,
or bank byte.  Live authority construction remains disabled until exact V4
terminal/reset evidence and a frozen independently reviewed V5 runtime are
bound.  V4 is evidence only and can never be retried or reused.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys
import tempfile


HERE = Path(__file__).absolute().parent
WORKSPACE_ROOT = Path("/Users/matt/arc-white-box-estimation")
P2 = WORKSPACE_ROOT / (
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16")
BINDINGS_SHA256 = "697dc4bc7c5afa22c57cec2eb7e9ca48750a1d922751e65f1cc0ac955c32d09a"

SCOPE_SHA256 = "2756e969a52dea280fdac2c47fe44713c0706896d5b4dc14d834e3c3c7d423f0"
RECORD_SHA256 = "902466f86013e2bf7e597518d124e4d98a5654fa63949ca8068522383d95bab3"
MANIFEST_BINDING_SHA256 = (
    "45334dcc9658529d9bf0b5a707a08e0cb456db3585bb9de597629ebc6f4ca52b")
MANIFEST_SCAN_SHA256 = (
    "052842efefda1fd73e49d15f17256411436900393cbaf6ae61dac8098310ce0a")

ACCEPTED_SOURCE = {
    "signed_pair_v2_independent_review_sha256": (
        "2a9270595bb33c0fd0f17a347131f02358a2aee00f0f38c6ee9e8484b0c45a0b"),
    "signed_pair_v2_manifest_sha256": (
        "85c1f3073ef54fa822748c8c579bd1d68d0823bb1169014f92d6e05152c5f895"),
    "three_bank_source_freeze_verification_sha256": (
        "708889d5db98ba27c642f15488bfd2551765dfc266472f9bd22ce4d7415fa6ea"),
    "three_bank_source_independent_review_sha256": (
        "e5bbea1e38a9c299ba7f708d7a2160c638473c86a3059970f516b535d26f1b4d"),
    "three_bank_source_manifest_sha256": (
        "ab3bc3b700f29dedb27c073c9cf8db995f1f0f275c31cc23c162810b2fbd3e90"),
}
ORDERED_SCOPE = {
    "manifest_compatibility_scan_receipt_sha256": MANIFEST_SCAN_SHA256,
    "ordered_688_record_sha256": RECORD_SHA256,
    "ordered_688_row_scope_sha256": SCOPE_SHA256,
    "ordered_manifest_binding_sha256": MANIFEST_BINDING_SHA256,
}
V4_FIXED = {
    "authority_anchor_sha256": (
        "9bcb90aa4279ae475d8ce6875eb9710dccd42351537ef9259145e99a25248b27"),
    "consumed_capability_sha256": (
        "24db92e64b4129d3a8c002620017aa516e2287cb9183c5edf9c99ec35c6c4518"),
    "exclusive_claim_sha256": (
        "d690aa5b42dd0bca12705795004640413368a4384bce467bad2a16c6560fd2b7"),
    "final_release_sha256": (
        "4d45b762c3515f58d6b2d8780407ceb5af5e98b45f5a40410cbbb57b33ae0f35"),
    "live_source_review_sha256": (
        "68dde823b5c333bcc4c5226d55e01e207889efe1fb82b7707b8c94ba55acfa00"),
    "source_authority_sha256": (
        "7e6ed0cb40e10497ee0d2ed93f0c0e1430ff056e7c481daef071dfb961493dee"),
    "terminal_failure_sha256": (
        "fbac37a97c64983fbc56627b67fdebb59dca07d0f43f3762798495022ed74dae"),
    "terminal_successor_record_sha256": (
        "836f6d5c71cfbf7ad488c086bf384eb3282561334783ffdf05e38a0a85b90978"),
}
V5_RUNTIME_KEYS = {
    "callable_bundle_manifest_sha256", "capture_adapter_sha256",
    "execution_stack_manifest_sha256", "freeze_receipt_sha256",
    "freeze_verification_sha256", "independent_review_request_sha256",
    "independent_review_sha256", "literal_source_sha256",
    "flopscope_reset_verification_sha256",
    "manifest_compatibility_scan_receipt_sha256", "package_manifest_sha256",
    "parent_capture_sha256", "reference_source_sha256", "runner_sha256",
    "runtime_token_sha256", "static_verification_sha256",
    "v3_terminal_failure_lineage_sha256",
    "v4_terminal_failure_lineage_sha256",
}
RESET_AUDIT_KEYS = {
    "acceptance_law", "access_accounting", "evidence_bindings", "incident",
    "issued_by", "issued_utc", "limitations", "memory_analysis",
    "reset_semantics", "root_cause", "schema", "status",
    "synthetic_pinned_stack_evidence",
}
RUNTIME_REVIEW_KEYS = {
    "schema", "status", "issued_by", "issued_utc", "reviewed_runner_sha256",
    "reviewed_callable_bundle_manifest_sha256",
    "reviewed_execution_stack_manifest_sha256", "prior_runtime_v1_hold_sha256",
    "prior_runtime_v2_go_sha256", "prior_runtime_v3_claim_sha256",
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
    "hostile_interpreter_security_claim", "r2_authority_dag_acyclic",
    "r3_resource_veto_removed", "r4_dynamic_comparators_present",
    "r5_manifest_firewall_semantic_repair_verified",
    "r6_flopscope_reset_placement_and_invariance_verified",
    "v3_terminal_no_retry_closed", "v4_terminal_no_retry_closed",
    "source_data_target_execution_during_review",
}
ANCHOR_KEYS = {
    "schema", "status", "issued_by", "issued_utc", "bindings",
    "target_access_allowed", "training_allowed",
    "estimator_deployment_allowed", "package_allowed", "submission_allowed",
}
CAPABILITY_KEYS = {
    "schema", "status", "capability_id", "issued_by", "issued_utc",
    "authority_anchor_sha256", "runner_sha256",
    "callable_bundle_manifest_sha256", "execution_stack_manifest_sha256",
    "ordered_688_row_scope_sha256", "ordered_688_record_sha256",
    "manifest_compatibility_scan_receipt_sha256", "one_use", "no_retry",
    "source_import_allowed", "row_access_allowed", "target_access_allowed",
    "training_allowed", "estimator_deployment_allowed", "package_allowed",
    "submission_allowed",
}
RELEASE_KEYS = {
    "schema", "status", "issued_by", "issued_utc",
    "authority_anchor_sha256", "single_use_capability_sha256", "bindings",
    "one_use", "no_retry", "source_import_allowed", "row_access_allowed",
    "target_access_allowed", "training_allowed", "estimator_deployment_allowed",
    "package_allowed", "submission_allowed",
}

V4_AUTHORITY = P2 / "public_q24_signed_pair_lean_v4_three_bank_authority_instance_v1"
V4_TRANSACTION = P2 / (
    "public_q24_signed_pair_lean_v4_three_bank_generation_transaction_v1")
V4_SOURCE = P2 / "public_q24_signed_pair_lean_v4_bank_source_authority_instance_v1"
V4_SOURCE_REVIEW = P2 / (
    "public_q24_signed_pair_lean_v4_bank_source_authority_review_v1")
V5_RUNTIME = P2 / "public_q24_signed_pair_three_bank_runtime_v5"
V5_RUNTIME_REVIEW = P2 / "public_q24_signed_pair_three_bank_runtime_review_v5"
V5_AUTHORITY = P2 / "public_q24_signed_pair_lean_v5_three_bank_authority_instance_v1"
V5_TRANSACTION = P2 / (
    "public_q24_signed_pair_lean_v5_three_bank_generation_transaction_v1")
V5_SOURCE = P2 / "public_q24_signed_pair_lean_v5_bank_source_authority_instance_v1"
V5_SOURCE_REVIEW = P2 / (
    "public_q24_signed_pair_lean_v5_bank_source_authority_review_v1")
V5_BUILDER_REVIEW_ROOT = P2 / (
    "public_q24_signed_pair_lean_v5_prelaunch_authority_builder_review_v1")
V5_BUILDER_REVIEW = V5_BUILDER_REVIEW_ROOT / "INDEPENDENT_STATIC_REVIEW_V1.json"
ROW_SCOPE = P2 / "public_q24_signed_pair_ordered_688_scope_v1/ORDERED_688_SCOPE.json"
PYTHON = P2 / "data_gate/.venv/bin/python"

V5_RUNTIME_PATHS = {
    "package_manifest_sha256": V5_RUNTIME / "PACKAGE_MANIFEST.json",
    "runner_sha256": V5_RUNTIME / "runtime_runner.py",
    "callable_bundle_manifest_sha256": V5_RUNTIME / "CALLABLE_BUNDLE_MANIFEST.json",
    "execution_stack_manifest_sha256": V5_RUNTIME / "EXECUTION_STACK_MANIFEST.json",
    "freeze_receipt_sha256": V5_RUNTIME / "FREEZE_RECEIPT.json",
    "freeze_verification_sha256": V5_RUNTIME / "FREEZE_VERIFICATION.json",
    "static_verification_sha256": V5_RUNTIME / "STATIC_VERIFICATION.json",
    "flopscope_reset_verification_sha256": (
        V5_RUNTIME / "FLOPSCOPE_RESET_VERIFICATION_V5.json"),
    "independent_review_request_sha256": V5_RUNTIME / "INDEPENDENT_REVIEW_REQUEST.md",
    "independent_review_sha256": V5_RUNTIME_REVIEW / "INDEPENDENT_STATIC_REVIEW_V5.json",
    "runtime_token_sha256": V5_RUNTIME / "runtime_token.py",
    "parent_capture_sha256": V5_RUNTIME / "parent_capture.py",
    "literal_source_sha256": V5_RUNTIME / "signed_pair_literal.py",
    "reference_source_sha256": V5_RUNTIME / "signed_pair_reference.py",
    "capture_adapter_sha256": V5_RUNTIME / "capture_adapter.py",
    "v3_terminal_failure_lineage_sha256": (
        V5_RUNTIME / "V3_TERMINAL_FAILURE_LINEAGE_V4.json"),
    "v4_terminal_failure_lineage_sha256": (
        V5_RUNTIME / "V4_TERMINAL_FAILURE_LINEAGE_V5.json"),
    "manifest_compatibility_scan_receipt_sha256": (
        V5_RUNTIME / "MANIFEST_COMPATIBILITY_SCAN_RECEIPT_V4.json"),
}
RUNTIME_FILENAMES = {
    "anchor": "RUNTIME_AUTHORITY_ANCHOR_V5.json",
    "capability": "SINGLE_USE_CAPABILITY_V5.json",
    "release": "FINAL_EXECUTION_RELEASE_V5.json",
    "command": "EXACTLY_ONE_LEAN_V5_LAUNCH_COMMAND_V1.json",
    "receipt": "LEAN_V5_AUTHORITY_INSTANCE_RECEIPT_V1.json",
}
RUNTIME_CONFIRMATION = (
    "ISSUE_EXACTLY_ONE_FRESH_LEAN_V5_TARGET_BLIND_688_BANK_PASS")

V4_PATHS = {
    "authority_anchor": V4_AUTHORITY / "RUNTIME_AUTHORITY_ANCHOR_V4.json",
    "consumed_capability": V4_AUTHORITY / "SINGLE_USE_CAPABILITY_V4.json.consumed",
    "final_release": V4_AUTHORITY / "FINAL_EXECUTION_RELEASE_V4.json",
    "source_authority": V4_SOURCE / "BANK_SOURCE_AUTHORITY_V4.json",
    "live_source_review": V4_SOURCE_REVIEW / "INDEPENDENT_LIVE_REVIEW_V4.json",
    "terminal_successor_record": V4_SOURCE / "V3_TERMINAL_SUCCESSOR_RECORD_V1.json",
    "exclusive_claim": V4_TRANSACTION / "THREE_BANK_RUN.CLAIM.json",
    "terminal_failure": V4_TRANSACTION / "THREE_BANK_RUN.FAILURE.json",
}
V4_STAGING = V4_TRANSACTION / ".THREE_BANK_BUNDLE.staging"
V4_BUNDLE = V4_TRANSACTION / "THREE_BANK_BUNDLE"
V4_RESET_AUDIT = P2 / (
    "public_q24_signed_pair_lean_v4_flopscope_accumulator_incident_audit_v1/"
    "INDEPENDENT_INCIDENT_ROOT_CAUSE_V1.json")

PRIOR_V2_ROOTS = {
    P2 / "public_q24_signed_pair_three_bank_authority_instance_v1",
    P2 / "public_q24_signed_pair_bank_source_authority_instance_v1",
    P2 / "public_q24_signed_pair_bank_source_authority_review_v2",
    P2 / "public_q24_signed_pair_three_bank_generation_transaction_v1",
}
PRIOR_V3_ROOTS = {
    P2 / "public_q24_signed_pair_lean_three_bank_authority_instance_v1",
    P2 / "public_q24_signed_pair_lean_bank_source_authority_instance_v1",
    P2 / "public_q24_signed_pair_lean_bank_source_authority_review_v3",
    P2 / "public_q24_signed_pair_lean_three_bank_generation_transaction_v1",
}
V5_NEW_ROOTS = {V5_AUTHORITY, V5_TRANSACTION, V5_SOURCE, V5_SOURCE_REVIEW}
BUILDER_REVIEW_KEYS = {
    "schema", "status", "review_id", "issued_by", "issued_utc",
    "package_manifest_sha256", "freeze_receipt_sha256",
    "freeze_verification_sha256", "static_verification_sha256",
    "helper_sha256", "bindings_sha256",
    "runtime_v5_package_manifest_sha256",
    "runtime_v5_independent_review_sha256",
    "v4_terminal_and_incident_bound",
    "path_overlap_and_exact_argv_verified",
    "source_and_live_review_gates_verified",
    "frozen_static_replay_verified",
    "reviewed_without_execution_or_protected_data",
    "reviewer_different_from_package_author",
    "hostile_interpreter_security_claim",
}

ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class AuthorityError(RuntimeError):
    pass


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def exact_keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise AuthorityError(label + " fields are not exact")
    return value


def decode_strict(raw, label):
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise AuthorityError(label + " duplicate key")
            output[key] = value
        return output
    try:
        return json.loads(
            raw.decode("ascii"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AuthorityError(label + " nonfinite " + token)))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorityError(label + " invalid strict JSON") from error


def lexical_absolute(value, label):
    if type(value) is not str or not value or "\x00" in value:
        raise AuthorityError(label + " malformed")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise AuthorityError(label + " must be absolute normalized")
    return path


def path_overlaps(left, right):
    left, right = Path(left), Path(right)
    return left == right or left in right.parents or right in left.parents


def protected_evidence_roots():
    roots = set(PRIOR_V2_ROOTS) | set(PRIOR_V3_ROOTS)
    roots.update({
        V4_AUTHORITY, V4_TRANSACTION, V4_SOURCE, V4_SOURCE_REVIEW,
        V4_RESET_AUDIT.parent, V5_RUNTIME, V5_RUNTIME_REVIEW,
        V5_BUILDER_REVIEW_ROOT, ROW_SCOPE.parent, PYTHON.parent, HERE,
    })
    roots.update(path.parent for path in V4_PATHS.values())
    roots.update(path.parent for path in V5_RUNTIME_PATHS.values())
    return roots


def require_new_roots_disjoint(new_roots, protected, label):
    new_roots = [Path(value) for value in new_roots]
    for index, left in enumerate(new_roots):
        for right in new_roots[index + 1:]:
            if path_overlaps(left, right):
                raise AuthorityError(label + " new roots overlap")
        for frozen in protected:
            if path_overlaps(left, frozen):
                raise AuthorityError(
                    label + " overlaps protected frozen evidence/input root")


def validate_fixed_root_disjointness():
    require_new_roots_disjoint(
        sorted(V5_NEW_ROOTS, key=os.fspath), protected_evidence_roots(),
        "fixed V5 authority/source/transaction roots")
    return True


def lstat_chain(path_value, label):
    path = lexical_absolute(os.fspath(path_value), label)
    current = Path(path.anchor)
    metadata = None
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise AuthorityError(label + " absent: " + str(current)) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise AuthorityError(label + " contains symlink")
    return path, metadata


def regular_one_link(path_value, label):
    path, metadata = lstat_chain(path_value, label)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise AuthorityError(label + " must be regular one-link")
    return path


def nonsymlink_directory(path_value, label):
    path, metadata = lstat_chain(path_value, label)
    if not stat.S_ISDIR(metadata.st_mode):
        raise AuthorityError(label + " must be directory")
    return path


def read_canonical(path_value, label):
    path = regular_one_link(path_value, label)
    raw = path.read_bytes()
    document = decode_strict(raw, label)
    if raw != canonical_bytes(document):
        raise AuthorityError(label + " bytes are not canonical")
    return raw, document


def parse_utc(value, label):
    if type(value) is not str:
        raise AuthorityError(label + " must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except ValueError as error:
        raise AuthorityError(label + " must be YYYY-MM-DDTHH:MM:SSZ") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise AuthorityError(label + " changed")
    return parsed


def require_id(value, label, prefix=None):
    if type(value) is not str or ID_RE.fullmatch(value) is None:
        raise AuthorityError(label + " is not canonical")
    if prefix is not None and not value.startswith(prefix):
        raise AuthorityError(label + " prefix changed")
    return value


def builder_frozen_roots():
    paths = {
        "package_manifest_sha256": HERE / "PACKAGE_MANIFEST.json",
        "freeze_receipt_sha256": HERE / "FREEZE_RECEIPT.json",
        "freeze_verification_sha256": HERE / "FREEZE_VERIFICATION.json",
        "static_verification_sha256": HERE / "STATIC_VERIFICATION.json",
        "helper_sha256": HERE / "prelaunch_authority_v5.py",
        "bindings_sha256": HERE / "BINDINGS_HOLD.json",
    }
    documents = {}
    roots = {}
    for key, path in paths.items():
        if path.suffix == ".json":
            raw, document = read_canonical(path, "frozen builder " + key)
            roots[key] = sha256_bytes(raw)
            documents[key] = document
        else:
            roots[key] = sha256_file(regular_one_link(
                path, "frozen builder " + key))
    package = documents["package_manifest_sha256"]
    freeze = documents["freeze_receipt_sha256"]
    verification = documents["freeze_verification_sha256"]
    static_receipt = documents["static_verification_sha256"]
    components = package.get("components")
    if (package.get("schema")
            != "public-q24-lean-v5-prelaunch-authority-package-manifest-v1"
            or package.get("status")
            != "FROZEN_STATIC_READY_FOR_INDEPENDENT_CONTROL_REVIEW"
            or type(components) is not dict
            or components.get("prelaunch_authority_v5.py") != roots["helper_sha256"]
            or components.get("BINDINGS_HOLD.json") != roots["bindings_sha256"]
            or components.get("STATIC_VERIFICATION.json")
            != roots["static_verification_sha256"]):
        raise AuthorityError("frozen builder package roots changed")
    if (freeze.get("schema")
            != "public-q24-lean-v5-prelaunch-authority-freeze-receipt-v1"
            or freeze.get("status") != "FROZEN_NO_LIVE_AUTHORITY_NO_EXECUTION"
            or freeze.get("package_manifest_sha256")
            != roots["package_manifest_sha256"]
            or freeze.get("static_verification_sha256")
            != roots["static_verification_sha256"]
            or freeze.get("helper_sha256") != roots["helper_sha256"]
            or freeze.get("bindings_sha256") != roots["bindings_sha256"]):
        raise AuthorityError("frozen builder receipt roots changed")
    if (verification.get("schema")
            != "public-q24-lean-v5-prelaunch-authority-freeze-verification-v1"
            or verification.get("status")
            != "PASS_BYTE_IDENTICAL_STATIC_REPLAY_NO_EXECUTION"
            or verification.get("package_manifest_sha256")
            != roots["package_manifest_sha256"]
            or verification.get("freeze_receipt_sha256")
            != roots["freeze_receipt_sha256"]
            or verification.get("static_verification_sha256")
            != roots["static_verification_sha256"]
            or verification.get("live_authority_created") is not False
            or verification.get("numerical_execution") is not False):
        raise AuthorityError("frozen builder verification roots changed")
    if (static_receipt.get("helper_sha256") != roots["helper_sha256"]
            or static_receipt.get("bindings_sha256") != roots["bindings_sha256"]
            or static_receipt.get("runtime_independent_review_sha256")
            != "26af3b9cf9cb4320b9dfff1917723655ee7ba4046dc99a302fcd2f1608cb2f8e"):
        raise AuthorityError("frozen builder static receipt changed")
    return roots, freeze


def validate_builder_review_inventory(names):
    if set(names) != {V5_BUILDER_REVIEW.name}:
        raise AuthorityError("V5 builder independent review inventory changed")
    return True


def validate_builder_go_document(review, roots, freeze):
    exact_keys(review, BUILDER_REVIEW_KEYS, "V5 builder independent GO")
    expected = {
        **roots,
        "runtime_v5_package_manifest_sha256": (
            "6307642b0ebe6397e9ffb7c5d163fe29970b5644fb39f76a240fe59afbd4935d"),
        "runtime_v5_independent_review_sha256": (
            "26af3b9cf9cb4320b9dfff1917723655ee7ba4046dc99a302fcd2f1608cb2f8e"),
    }
    if (review["schema"]
            != "public-q24-lean-v5-prelaunch-authority-independent-static-review-v1"
            or review["status"]
            != "GO_FROZEN_FAIL_CLOSED_V5_PRELAUNCH_AUTHORITY_BUILDER"
            or review["issued_by"] != "independent_reviewer"
            or any(review[key] != value for key, value in expected.items())):
        raise AuthorityError("V5 builder independent review identity changed")
    require_id(review["review_id"], "V5 builder review id",
               "q24-sp3b-lean-v5-authority-builder-review-")
    if parse_utc(review["issued_utc"], "V5 builder review issued_utc") < parse_utc(
            freeze["issued_utc"], "V5 builder freeze issued_utc"):
        raise AuthorityError("V5 builder review predates freeze")
    for key in (
            "v4_terminal_and_incident_bound",
            "path_overlap_and_exact_argv_verified",
            "source_and_live_review_gates_verified",
            "frozen_static_replay_verified",
            "reviewed_without_execution_or_protected_data",
            "reviewer_different_from_package_author"):
        if review[key] is not True:
            raise AuthorityError("V5 builder review closure failed: " + key)
    if review["hostile_interpreter_security_claim"] is not False:
        raise AuthorityError("V5 builder review overclaims hostile interpreter")
    return True


def validate_builder_go():
    roots, freeze = builder_frozen_roots()
    review_root = nonsymlink_directory(
        V5_BUILDER_REVIEW_ROOT, "V5 builder independent review root")
    with os.scandir(review_root) as iterator:
        validate_builder_review_inventory({entry.name for entry in iterator})
    raw, review = read_canonical(V5_BUILDER_REVIEW, "V5 builder independent GO")
    validate_builder_go_document(review, roots, freeze)
    return {**roots, "independent_review_sha256": sha256_bytes(raw)}


def validate_bindings(document, live):
    exact_keys(document, {
        "schema", "status", "accepted_source", "ordered_scope",
        "v4_terminal", "v5_runtime",
    }, "V5 prelaunch bindings")
    if document["schema"] != "public-q24-lean-v5-prelaunch-authority-bindings-v1":
        raise AuthorityError("V5 binding schema changed")
    if document["accepted_source"] != ACCEPTED_SOURCE:
        raise AuthorityError("accepted source roots changed")
    if document["ordered_scope"] != ORDERED_SCOPE:
        raise AuthorityError("ordered scope roots changed")
    terminal = exact_keys(document["v4_terminal"], set(V4_FIXED) | {
        "reset_receipt_sha256"}, "V4 terminal bindings")
    for key, expected in V4_FIXED.items():
        if terminal[key] != expected:
            raise AuthorityError("V4 terminal root changed: " + key)
    runtime = exact_keys(document["v5_runtime"], V5_RUNTIME_KEYS,
                         "V5 runtime bindings")
    dynamic = [terminal["reset_receipt_sha256"], *runtime.values()]
    if live:
        if document["status"] != "FROZEN_READY_AFTER_V4_RESET_AND_V5_RUNTIME_GO":
            raise AuthorityError("live V5 bindings remain HOLD")
        if any(not is_sha(value) for value in dynamic):
            raise AuthorityError("live V5 binding is null or malformed")
        if len(set(dynamic)) != len(dynamic):
            raise AuthorityError("distinct V5 evidence roots unexpectedly collide")
    else:
        if any(value is not None and not is_sha(value) for value in dynamic):
            raise AuthorityError("HOLD V5 binding malformed")
        if any(value is None for value in dynamic) and document["status"] != (
                "HOLD_PENDING_EXACT_V4_RESET_RECEIPT_AND_FROZEN_V5_RUNTIME_GO"):
            raise AuthorityError("null V5 bindings require HOLD")
    return document


def load_bindings(path_value, live=False):
    validate_fixed_root_disjointness()
    path = lexical_absolute(os.fspath(path_value), "V5 bindings path")
    if path != HERE / "BINDINGS_HOLD.json":
        raise AuthorityError("V5 bindings path changed")
    raw, document = read_canonical(path, "V5 prelaunch bindings")
    if sha256_bytes(raw) != BINDINGS_SHA256:
        raise AuthorityError("V5 prelaunch bindings hash changed")
    return validate_bindings(document, live)


def validate_reset_audit(path_value, expected_sha256):
    path = lexical_absolute(os.fspath(path_value), "V4 incident/reset audit path")
    if path != V4_RESET_AUDIT:
        raise AuthorityError("V4 incident/reset audit path changed")
    path = regular_one_link(path, "V4 incident/reset audit")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise AuthorityError("V4 incident/reset audit hash changed")
    audit = decode_strict(raw, "V4 incident/reset audit")
    exact_keys(audit, RESET_AUDIT_KEYS, "V4 incident/reset audit")
    if (audit["schema"]
            != "public-q24-lean-v4-flopscope-accumulator-incident-independent-audit-v1"
            or audit["status"]
            != "PASS_ROOT_CAUSE_CONFIRMED__V5_PER_ROW_POST_COMMIT_RESET_REQUIRED"
            or audit["issued_by"] != "independent Codex incident auditor"):
        raise AuthorityError("V4 incident/reset audit identity changed")
    parse_utc(audit["issued_utc"], "V4 incident/reset audit issued_utc")
    incident = audit["incident"]
    if incident != {
            "claim_digest_crosswalk_valid": True,
            "error_message": "",
            "error_type": "KeyboardInterrupt",
            "published_bundle": False,
            "staging_preserved_for_forensics": True,
            "terminal_status": "TERMINAL_POST_CLAIM_FAILURE__NO_RETRY",
    }:
        raise AuthorityError("V4 incident facts changed")
    evidence = audit["evidence_bindings"]
    if (evidence.get("claim", {}).get("sha256")
            != V4_FIXED["exclusive_claim_sha256"]
            or evidence.get("failure", {}).get("sha256")
            != V4_FIXED["terminal_failure_sha256"]):
        raise AuthorityError("V4 incident evidence roots changed")
    access = audit["access_accounting"]
    if (type(access) is not dict or not access
            or any(type(value) is not int or value != 0
                   for value in access.values())):
        raise AuthorityError("V4 incident audit opened protected artifacts")
    acceptance = audit["acceptance_law"]
    if (acceptance.get("reset_count") != {
            "failed_or_unvalidated_rows": 0,
            "successful_validated_rows": 688,
            }
            or acceptance.get("single_admitted_runtime_order", [])[-2:] != [
                "runner invokes exactly one budget_reset through session flops",
                "the next row begins, or after rank 687 the finish pass begins",
            ]):
        raise AuthorityError("V5 reset acceptance law changed")
    root_cause = audit["root_cause"]
    if (root_cause.get("classification")
            != "FLOPSCOPE_0_12_PROCESS_GLOBAL_DIAGNOSTIC_HISTORY_RETENTION"
            or root_cause.get("v4_reset_call_count") != 0):
        raise AuthorityError("V4 root cause changed")
    for label, evidence_item in audit[
            "synthetic_pinned_stack_evidence"].items():
        if label == "protected_execution_performed":
            if evidence_item is not False:
                raise AuthorityError("reset audit performed protected execution")
            continue
        if (evidence_item.get("repeat_count") != 1000
                or evidence_item.get(
                    "maximum_accumulator_records_before_each_reset") != 1
                and evidence_item.get(
                    "maximum_global_records_before_each_reset") != 1
                or evidence_item.get("final_accumulator_record_count") != 0
                and evidence_item.get("global_records_after_reset") != 0
                or evidence_item.get("context_summary_unchanged") is not True
                or evidence_item.get("detached_summary_unchanged") is not True
                or evidence_item.get("output_value_before_and_after_reset") != 3.0):
            raise AuthorityError("reset audit synthetic invariance changed: " + label)
    if audit["limitations"].get("protected_execution_performed") is not False:
        raise AuthorityError("reset audit limitations changed")
    return audit


def validate_v4_terminal_metadata(bindings, reset_path=None):
    terminal = bindings["v4_terminal"]
    for key, path in V4_PATHS.items():
        expected_key = key + "_sha256"
        if key == "consumed_capability":
            expected_key = "consumed_capability_sha256"
        observed = sha256_file(regular_one_link(path, "V4 terminal " + key))
        if observed != terminal[expected_key]:
            raise AuthorityError("V4 terminal artifact changed: " + key)
    if os.path.lexists(V4_AUTHORITY / "SINGLE_USE_CAPABILITY_V4.json"):
        raise AuthorityError("V4 consumed capability name reappeared")
    if os.path.lexists(V4_BUNDLE):
        raise AuthorityError("V4 unexpectedly published a bundle")
    staging = nonsymlink_directory(V4_STAGING, "V4 forensic staging root")
    del staging  # Never descend: the preserved V4 staging tree is not an input.
    _, claim = read_canonical(V4_PATHS["exclusive_claim"], "V4 claim")
    _, failure = read_canonical(V4_PATHS["terminal_failure"], "V4 failure")
    if (claim.get("schema") != "public-q24-three-bank-runtime-exclusive-claim-v4"
            or claim.get("status") != "CLAIMED_ONE_TARGET_BLIND_688_BANK_PASS"
            or failure != {
                "actual_claim_sha256": V4_FIXED["exclusive_claim_sha256"],
                "claim_complete": True,
                "claim_sha256": V4_FIXED["exclusive_claim_sha256"],
                "error_message": "",
                "error_type": "KeyboardInterrupt",
                "published_bundle": False,
                "schema": "public-q24-three-bank-runtime-terminal-failure-v1",
                "staging_preserved_for_forensics": True,
                "status": "TERMINAL_POST_CLAIM_FAILURE__NO_RETRY",
            }):
        raise AuthorityError("V4 terminal claim/failure semantics changed")
    if terminal["reset_receipt_sha256"] is not None:
        validate_reset_audit(
            V4_RESET_AUDIT if reset_path is None else reset_path,
            terminal["reset_receipt_sha256"])
    return True


def validate_runtime_review(review, runtime):
    exact_keys(review, RUNTIME_REVIEW_KEYS, "V5 runtime independent review")
    if (review["schema"] != "public-q24-three-bank-runtime-static-review-v5"
            or review["status"] != "GO"):
        raise AuthorityError("V5 runtime independent review is not GO")
    require_id(review["issued_by"], "V5 runtime reviewer")
    parse_utc(review["issued_utc"], "V5 runtime review issued_utc")
    expected = {
        "reviewed_runner_sha256": runtime["runner_sha256"],
        "reviewed_callable_bundle_manifest_sha256": runtime[
            "callable_bundle_manifest_sha256"],
        "reviewed_execution_stack_manifest_sha256": runtime[
            "execution_stack_manifest_sha256"],
        "prior_runtime_v1_hold_sha256": (
            "49e51e300c158229f8401a5df032feebc6ae5cc8ae48ab1f2e55b2a407646d32"),
        "prior_runtime_v2_go_sha256": (
            "7bf6664b2e27d863e5fc4a556373a217d05628ec681141757a4d0beec46c0b03"),
        "prior_runtime_v3_claim_sha256": (
            "bfa9e8d800a5b1efc0ec4e47a62b53d70176e0bf8b9ec73f9981056974abd4e9"),
        "prior_runtime_v3_terminal_failure_sha256": (
            "e64f998fc1513f906e90d2869340e44ce2ca548b91e31667aade69da9af1d50d"),
        "prior_runtime_v3_consumed_capability_sha256": (
            "f7cd31ab124c7914d1bd2b413289fb84ced4a38d2e93e3f3d3d9a9e711d38a39"),
        "prior_runtime_v4_terminal_failure_lineage_sha256": runtime[
            "v4_terminal_failure_lineage_sha256"],
        "prior_runtime_v4_claim_sha256": V4_FIXED["exclusive_claim_sha256"],
        "prior_runtime_v4_terminal_failure_sha256": V4_FIXED[
            "terminal_failure_sha256"],
        "prior_runtime_v4_consumed_capability_sha256": V4_FIXED[
            "consumed_capability_sha256"],
        "independent_incident_root_cause_sha256": (
            "060e7bab3094a7929d246f4bdf5fbe6790ccb3aa9c3b05734395a6376b24cd6c"),
        "flopscope_reset_verification_sha256": runtime[
            "flopscope_reset_verification_sha256"],
        "manifest_compatibility_scan_receipt_sha256": MANIFEST_SCAN_SHA256,
    }
    if any(review[key] != value for key, value in expected.items()):
        raise AuthorityError("V5 runtime review binding changed")
    for key in (
            "r1_public_api_and_governance_chronology_closed",
            "r2_authority_dag_acyclic", "r3_resource_veto_removed",
            "r4_dynamic_comparators_present",
            "r5_manifest_firewall_semantic_repair_verified",
            "r6_flopscope_reset_placement_and_invariance_verified",
            "v3_terminal_no_retry_closed", "v4_terminal_no_retry_closed"):
        if review[key] is not True:
            raise AuthorityError("V5 runtime review closure failed: " + key)
    if (review["hostile_interpreter_security_claim"] is not False
            or review["source_data_target_execution_during_review"] is not False):
        raise AuthorityError("V5 runtime review overclaims or opened protected data")
    return review


def validate_runtime_artifacts(bindings):
    runtime = bindings["v5_runtime"]
    if set(V5_RUNTIME_PATHS) != V5_RUNTIME_KEYS:
        raise AuthorityError("internal V5 runtime path inventory changed")
    documents = {}
    for key, path in V5_RUNTIME_PATHS.items():
        observed = sha256_file(regular_one_link(path, "V5 runtime " + key))
        if observed != runtime[key]:
            raise AuthorityError("V5 runtime artifact changed: " + key)
        if path.suffix == ".json":
            documents[key] = decode_strict(path.read_bytes(), "V5 runtime " + key)
    lineage = documents["v4_terminal_failure_lineage_sha256"]
    if (lineage.get("schema")
            != "public-q24-three-bank-runtime-v4-terminal-failure-lineage-v5"
            or lineage.get("status") != "V4_TERMINAL_POST_CLAIM_FAILURE__NO_RETRY"
            or lineage.get("claim_sha256") != V4_FIXED["exclusive_claim_sha256"]
            or lineage.get("terminal_failure_sha256")
            != V4_FIXED["terminal_failure_sha256"]
            or lineage.get("consumed_capability_sha256")
            != V4_FIXED["consumed_capability_sha256"]
            or lineage.get("independent_incident_root_cause_sha256")
            != bindings["v4_terminal"]["reset_receipt_sha256"]
            or lineage.get("no_retry") is not True
            or lineage.get("published_bundle") is not False
            or lineage.get("staging_preserved_for_forensics") is not True
            or lineage.get("v4_attempt_reuse_allowed") is not False
            or lineage.get("v5_fresh_authority_capability_and_transaction_required")
            is not True):
        raise AuthorityError("V5 V4-terminal lineage changed")
    review = documents["independent_review_sha256"]
    validate_runtime_review(review, runtime)
    if runtime["callable_bundle_manifest_sha256"] != (
            "5a5c7ea5fd83905ad4484cd74a0101eb076f1c6d2167a423f7000968ef1a2599"):
        raise AuthorityError("V5 callable scientific bundle changed")
    if runtime["execution_stack_manifest_sha256"] != (
            "e4204ec69ab30a5bb9e441558722edb7bcd80d45bba380ee9cc84217bfa6d433"):
        raise AuthorityError("V5 execution stack changed")
    if runtime["manifest_compatibility_scan_receipt_sha256"] != (
            MANIFEST_SCAN_SHA256):
        raise AuthorityError("V5 manifest scan changed")
    return review


def runtime_binding_map(bindings):
    runtime = bindings["v5_runtime"]
    return {
        "runner_sha256": runtime["runner_sha256"],
        "callable_bundle_manifest_sha256": runtime[
            "callable_bundle_manifest_sha256"],
        "execution_stack_manifest_sha256": runtime[
            "execution_stack_manifest_sha256"],
        "independent_review_sha256": runtime["independent_review_sha256"],
        "ordered_688_row_scope_sha256": SCOPE_SHA256,
        "ordered_688_record_sha256": RECORD_SHA256,
        "manifest_compatibility_scan_receipt_sha256": MANIFEST_SCAN_SHA256,
    }


def runtime_request(document, require_confirmation):
    exact_keys(document, {
        "schema", "instance_dir", "transaction_root", "working_directory",
        "capability_id", "issued_by", "anchor_issued_utc",
        "capability_issued_utc", "release_issued_utc", "confirmation",
    }, "V5 runtime issuance request")
    if document["schema"] != "public-q24-lean-v5-runtime-authority-issuance-request-v1":
        raise AuthorityError("V5 runtime request schema changed")
    if require_confirmation and document["confirmation"] != RUNTIME_CONFIRMATION:
        raise AuthorityError("V5 runtime confirmation missing")
    if not require_confirmation and document["confirmation"] not in (
            None, RUNTIME_CONFIRMATION):
        raise AuthorityError("V5 runtime confirmation malformed")
    if document["instance_dir"] != os.fspath(V5_AUTHORITY):
        raise AuthorityError("V5 authority root changed")
    if document["transaction_root"] != os.fspath(V5_TRANSACTION):
        raise AuthorityError("V5 transaction root changed")
    if document["working_directory"] != os.fspath(WORKSPACE_ROOT):
        raise AuthorityError("V5 working directory changed")
    capability_id = require_id(
        document["capability_id"], "V5 capability id", "q24-sp3b-lean-v5-")
    suffix = capability_id.removeprefix("q24-sp3b-lean-v5-")
    if len(suffix) != 32 or any(c not in "0123456789abcdef" for c in suffix):
        raise AuthorityError("V5 capability id lacks fresh 128-bit suffix")
    require_id(document["issued_by"], "V5 runtime issuer")
    times = [parse_utc(document[key], key) for key in (
        "anchor_issued_utc", "capability_issued_utc", "release_issued_utc")]
    if times != sorted(times):
        raise AuthorityError("V5 runtime chronology changed")
    if require_confirmation:
        now = datetime.now(timezone.utc)
        for key, parsed in zip(("anchor", "capability", "release"), times):
            delta = (now - parsed).total_seconds()
            if delta < -60 or delta > 300:
                raise AuthorityError("V5 " + key + " timestamp is not recent")
    return document


def expected_runtime_argv():
    instance = V5_AUTHORITY
    return [
        os.fspath(PYTHON), os.fspath(V5_RUNTIME / "runtime_runner.py"),
        "--callable-bundle", os.fspath(V5_RUNTIME / "CALLABLE_BUNDLE_MANIFEST.json"),
        "--execution-stack", os.fspath(V5_RUNTIME / "EXECUTION_STACK_MANIFEST.json"),
        "--independent-review", os.fspath(
            V5_RUNTIME_REVIEW / "INDEPENDENT_STATIC_REVIEW_V5.json"),
        "--authority-anchor", os.fspath(instance / RUNTIME_FILENAMES["anchor"]),
        "--release", os.fspath(instance / RUNTIME_FILENAMES["release"]),
        "--capability", os.fspath(instance / RUNTIME_FILENAMES["capability"]),
        "--row-scope", os.fspath(ROW_SCOPE),
        "--runtime-token", os.fspath(V5_RUNTIME / "runtime_token.py"),
        "--parent", os.fspath(V5_RUNTIME / "parent_capture.py"),
        "--literal-source", os.fspath(V5_RUNTIME / "signed_pair_literal.py"),
        "--reference-source", os.fspath(V5_RUNTIME / "signed_pair_reference.py"),
        "--adapter", os.fspath(V5_RUNTIME / "capture_adapter.py"),
        "--transaction-root", os.fspath(V5_TRANSACTION),
    ]


def make_runtime_documents(bindings, request, review):
    validate_bindings(bindings, True)
    runtime_request(request, False)
    if parse_utc(review["issued_utc"], "review issued_utc") > parse_utc(
            request["anchor_issued_utc"], "anchor issued_utc"):
        raise AuthorityError("V5 anchor predates independent runtime review")
    mapping = runtime_binding_map(bindings)
    anchor = {
        "schema": "public-q24-three-bank-runtime-authority-anchor-v5",
        "status": "FROZEN_RUNTIME_AND_ORDERED_SCOPE_ALLOWLIST",
        "issued_by": request["issued_by"],
        "issued_utc": request["anchor_issued_utc"],
        "bindings": mapping,
        "target_access_allowed": False,
        "training_allowed": False,
        "estimator_deployment_allowed": False,
        "package_allowed": False,
        "submission_allowed": False,
    }
    anchor_sha = sha256_bytes(canonical_bytes(anchor))
    runtime = bindings["v5_runtime"]
    capability = {
        "schema": "public-q24-signed-pair-bank-capability-v5",
        "status": "AUTHORIZE_ONE_TARGET_BLIND_688_BANK_PASS",
        "capability_id": request["capability_id"],
        "issued_by": request["issued_by"],
        "issued_utc": request["capability_issued_utc"],
        "authority_anchor_sha256": anchor_sha,
        "runner_sha256": runtime["runner_sha256"],
        "callable_bundle_manifest_sha256": runtime[
            "callable_bundle_manifest_sha256"],
        "execution_stack_manifest_sha256": runtime[
            "execution_stack_manifest_sha256"],
        "ordered_688_row_scope_sha256": SCOPE_SHA256,
        "ordered_688_record_sha256": RECORD_SHA256,
        "manifest_compatibility_scan_receipt_sha256": MANIFEST_SCAN_SHA256,
        "one_use": True,
        "no_retry": True,
        "source_import_allowed": True,
        "row_access_allowed": True,
        "target_access_allowed": False,
        "training_allowed": False,
        "estimator_deployment_allowed": False,
        "package_allowed": False,
        "submission_allowed": False,
    }
    capability_sha = sha256_bytes(canonical_bytes(capability))
    release = {
        "schema": "public-q24-three-bank-runtime-final-release-v5",
        "status": "AUTHORIZE_ONE_TARGET_BLIND_688_BANK_PASS",
        "issued_by": request["issued_by"],
        "issued_utc": request["release_issued_utc"],
        "authority_anchor_sha256": anchor_sha,
        "single_use_capability_sha256": capability_sha,
        "bindings": mapping,
        "one_use": True,
        "no_retry": True,
        "source_import_allowed": True,
        "row_access_allowed": True,
        "target_access_allowed": False,
        "training_allowed": False,
        "estimator_deployment_allowed": False,
        "package_allowed": False,
        "submission_allowed": False,
    }
    argv = expected_runtime_argv()
    command = {
        "schema": "public-q24-lean-v5-three-bank-exactly-one-launch-command-v1",
        "status": "PREPARED_NOT_AUTHORIZED_UNTIL_V5_SOURCE_REVIEW_GO",
        "command_count": 1,
        "argv": argv,
        "shell_rendering_for_display_only": shlex.join(argv),
        "working_directory": os.fspath(WORKSPACE_ROOT),
        "authority_anchor_sha256": anchor_sha,
        "single_use_capability_sha256": capability_sha,
        "final_release_sha256": sha256_bytes(canonical_bytes(release)),
        "manifest_compatibility_scan_receipt_sha256": MANIFEST_SCAN_SHA256,
        "v4_terminal_attempt_reused": False,
        "v4_staging_used_as_input": False,
        "transaction_root_must_be_existing_empty_nonsymlink_directory": True,
        "execute_via_argv_without_shell": True,
        "no_retry": True,
        "source_authority_and_independent_live_review_required_before_launch": True,
    }
    documents = {"anchor": anchor, "capability": capability,
                 "release": release, "command": command}
    validate_runtime_documents(documents, bindings)
    return documents


def validate_runtime_documents(documents, bindings):
    exact_keys(documents, {"anchor", "capability", "release", "command"},
               "V5 runtime documents")
    anchor = exact_keys(documents["anchor"], ANCHOR_KEYS, "V5 anchor")
    capability = exact_keys(
        documents["capability"], CAPABILITY_KEYS, "V5 capability")
    release = exact_keys(documents["release"], RELEASE_KEYS, "V5 release")
    command = exact_keys(documents["command"], {
        "schema", "status", "command_count", "argv",
        "shell_rendering_for_display_only", "working_directory",
        "authority_anchor_sha256", "single_use_capability_sha256",
        "final_release_sha256", "manifest_compatibility_scan_receipt_sha256",
        "v4_terminal_attempt_reused", "v4_staging_used_as_input",
        "transaction_root_must_be_existing_empty_nonsymlink_directory",
        "execute_via_argv_without_shell", "no_retry",
        "source_authority_and_independent_live_review_required_before_launch",
    }, "V5 launch command")
    if (anchor["schema"] != "public-q24-three-bank-runtime-authority-anchor-v5"
            or anchor["status"] != "FROZEN_RUNTIME_AND_ORDERED_SCOPE_ALLOWLIST"
            or capability["schema"] != "public-q24-signed-pair-bank-capability-v5"
            or capability["status"] != "AUTHORIZE_ONE_TARGET_BLIND_688_BANK_PASS"
            or release["schema"] != "public-q24-three-bank-runtime-final-release-v5"
            or release["status"] != "AUTHORIZE_ONE_TARGET_BLIND_688_BANK_PASS"):
        raise AuthorityError("V5 runtime document identity changed")
    mapping = runtime_binding_map(bindings)
    if anchor["bindings"] != mapping or release["bindings"] != mapping:
        raise AuthorityError("V5 anchor/release binding map changed")
    anchor_sha = sha256_bytes(canonical_bytes(anchor))
    capability_sha = sha256_bytes(canonical_bytes(capability))
    release_sha = sha256_bytes(canonical_bytes(release))
    if (capability["authority_anchor_sha256"] != anchor_sha
            or release["authority_anchor_sha256"] != anchor_sha
            or release["single_use_capability_sha256"] != capability_sha):
        raise AuthorityError("V5 authority DAG changed")
    runtime = bindings["v5_runtime"]
    direct = {
        "runner_sha256": runtime["runner_sha256"],
        "callable_bundle_manifest_sha256": runtime[
            "callable_bundle_manifest_sha256"],
        "execution_stack_manifest_sha256": runtime[
            "execution_stack_manifest_sha256"],
        "ordered_688_row_scope_sha256": SCOPE_SHA256,
        "ordered_688_record_sha256": RECORD_SHA256,
        "manifest_compatibility_scan_receipt_sha256": MANIFEST_SCAN_SHA256,
    }
    if any(capability[key] != value for key, value in direct.items()):
        raise AuthorityError("V5 capability direct binding changed")
    for document in (anchor, capability, release):
        if any(document[key] is not False for key in (
                "target_access_allowed", "training_allowed",
                "estimator_deployment_allowed", "package_allowed",
                "submission_allowed")):
            raise AuthorityError("V5 denial permission changed")
    for document in (capability, release):
        if (document["one_use"] is not True or document["no_retry"] is not True
                or document["source_import_allowed"] is not True
                or document["row_access_allowed"] is not True):
            raise AuthorityError("V5 one-use/source permission changed")
    if (command["schema"]
            != "public-q24-lean-v5-three-bank-exactly-one-launch-command-v1"
            or command["status"]
            != "PREPARED_NOT_AUTHORIZED_UNTIL_V5_SOURCE_REVIEW_GO"
            or command["command_count"] != 1
            or command["authority_anchor_sha256"] != anchor_sha
            or command["single_use_capability_sha256"] != capability_sha
            or command["final_release_sha256"] != release_sha
            or command["manifest_compatibility_scan_receipt_sha256"]
            != MANIFEST_SCAN_SHA256
            or command["v4_terminal_attempt_reused"] is not False
            or command["v4_staging_used_as_input"] is not False
            or command["execute_via_argv_without_shell"] is not True
            or command["no_retry"] is not True
            or command[
                "transaction_root_must_be_existing_empty_nonsymlink_directory"]
            is not True
            or command[
                "source_authority_and_independent_live_review_required_before_launch"]
            is not True):
        raise AuthorityError("V5 command gate changed")
    argv = command["argv"]
    if (argv != expected_runtime_argv()
            or command["shell_rendering_for_display_only"] != shlex.join(argv)
            or command["working_directory"] != os.fspath(WORKSPACE_ROOT)
            or any(type(item) is not str for item in argv)):
        raise AuthorityError("V5 command argv changed or reuses V4")
    return {
        "anchor_sha256": anchor_sha,
        "capability_sha256": capability_sha,
        "release_sha256": release_sha,
        "capability_id": capability["capability_id"],
        "release_issued_utc": release["issued_utc"],
    }


def write_exclusive(path_value, document, mode=0o400):
    path = lexical_absolute(os.fspath(path_value), "exclusive output")
    payload = canonical_bytes(document)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise AuthorityError("short exclusive write")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_bytes(payload)


def atomic_publish(destination, documents, ordered_names):
    destination = lexical_absolute(os.fspath(destination), "publish destination")
    if destination.exists() or destination.is_symlink():
        raise AuthorityError("publish destination must be absent")
    parent = nonsymlink_directory(destination.parent, "publish parent")
    staging = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".staging-", dir=parent))
    os.chmod(staging, 0o700)
    hashes = {}
    for name in ordered_names:
        hashes[name] = write_exclusive(staging / name, documents[name])
    os.rename(staging, destination)
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashes


def require_fresh_transaction(authority_must_be_absent=True):
    root = nonsymlink_directory(V5_TRANSACTION, "fresh V5 transaction root")
    with os.scandir(root) as iterator:
        if any(True for _ in iterator):
            raise AuthorityError("fresh V5 transaction root is not empty")
    if (authority_must_be_absent
            and (V5_AUTHORITY.exists() or V5_AUTHORITY.is_symlink())):
        raise AuthorityError("V5 authority instance already exists")
    if authority_must_be_absent and (
            V5_SOURCE.exists() or V5_SOURCE.is_symlink()
            or V5_SOURCE_REVIEW.exists() or V5_SOURCE_REVIEW.is_symlink()):
        raise AuthorityError("future V5 source/review destination already exists")
    return root


def runtime_context(bindings_path, request_path, require_confirmation):
    bindings = load_bindings(bindings_path, True)
    validate_v4_terminal_metadata(bindings)
    review = validate_runtime_artifacts(bindings)
    builder = validate_builder_go()
    _, request = read_canonical(request_path, "V5 runtime issuance request")
    runtime_request(request, require_confirmation)
    require_fresh_transaction()
    documents = make_runtime_documents(bindings, request, review)
    return bindings, request, documents, builder


def runtime_preflight(bindings_path, request_path):
    _, _, documents, builder = runtime_context(
        bindings_path, request_path, False)
    roots = validate_runtime_documents(
        documents, load_bindings(bindings_path, True))
    return {
        "schema": "public-q24-lean-v5-runtime-authority-preflight-v1",
        "status": "PASS_NO_AUTHORITY_NO_CAPABILITY_NO_EXECUTION",
        "prospective_roots": roots,
        "authority_builder_independent_review_sha256": builder[
            "independent_review_sha256"],
        "v4_staging_entries_opened": 0,
        "target_access": False,
        "numerical_execution": False,
    }


def runtime_receipt(bindings, roots, builder):
    return {
        "schema": "public-q24-lean-v5-authority-instance-receipt-v1",
        "status": "ISSUED_FRESH_V5_CONTROL_CHAIN__AWAITING_SOURCE_REVIEW",
        "authority_anchor_sha256": roots["anchor_sha256"],
        "single_use_capability_sha256": roots["capability_sha256"],
        "final_release_sha256": roots["release_sha256"],
        "v4_claim_sha256": V4_FIXED["exclusive_claim_sha256"],
        "v4_terminal_failure_sha256": V4_FIXED["terminal_failure_sha256"],
        "v4_consumed_capability_sha256": V4_FIXED[
            "consumed_capability_sha256"],
        "independent_incident_root_cause_sha256": bindings[
            "v4_terminal"]["reset_receipt_sha256"],
        "flopscope_reset_verification_sha256": bindings["v5_runtime"][
            "flopscope_reset_verification_sha256"],
        "v5_runtime_package_manifest_sha256": bindings["v5_runtime"][
            "package_manifest_sha256"],
        "v5_runtime_independent_review_sha256": bindings["v5_runtime"][
            "independent_review_sha256"],
        "authority_builder_package_manifest_sha256": builder[
            "package_manifest_sha256"],
        "authority_builder_freeze_receipt_sha256": builder[
            "freeze_receipt_sha256"],
        "authority_builder_freeze_verification_sha256": builder[
            "freeze_verification_sha256"],
        "authority_builder_static_verification_sha256": builder[
            "static_verification_sha256"],
        "authority_builder_helper_sha256": builder["helper_sha256"],
        "authority_builder_bindings_sha256": builder["bindings_sha256"],
        "authority_builder_independent_review_sha256": builder[
            "independent_review_sha256"],
        "v4_retry": False,
        "v4_staging_used_as_input": False,
        "fresh_v5_transaction_empty": True,
        "source_review_exists": False,
        "execution_authorized_by_receipt": False,
    }


def runtime_issue(bindings_path, request_path):
    bindings, _, documents, builder = runtime_context(
        bindings_path, request_path, True)
    roots = validate_runtime_documents(documents, bindings)
    receipt = runtime_receipt(bindings, roots, builder)
    output = {
        RUNTIME_FILENAMES["anchor"]: documents["anchor"],
        RUNTIME_FILENAMES["capability"]: documents["capability"],
        RUNTIME_FILENAMES["release"]: documents["release"],
        RUNTIME_FILENAMES["command"]: documents["command"],
        RUNTIME_FILENAMES["receipt"]: receipt,
    }
    hashes = atomic_publish(V5_AUTHORITY, output, list(output))
    return {
        "schema": "public-q24-lean-v5-runtime-authority-issuance-result-v1",
        "status": "ISSUED__HOLD_FOR_FRESH_V5_SOURCE_AUTHORITY_AND_REVIEW",
        "artifact_sha256": hashes,
        "execution_authorized": False,
        "v4_staging_entries_opened": 0,
        "numerical_execution": False,
    }


SOURCE_FILENAMES = {
    "successor": "V4_TERMINAL_SUCCESSOR_RECORD_V1.json",
    "authority": "BANK_SOURCE_AUTHORITY_V5.json",
    "receipt": "BANK_SOURCE_AUTHORITY_ISSUANCE_RECEIPT_V5.json",
    "request": "BANK_SOURCE_AUTHORITY_LIVE_REVIEW_REQUEST_V5.json",
}
SOURCE_CONFIRMATION = (
    "ISSUE_EXACT_LEAN_V5_PREGEN_SOURCE_AUTHORITY_AFTER_V4_TERMINAL_RESET_GO")


def row_scope_bindings():
    return {
        "ordered_688_record_sha256": RECORD_SHA256,
        "ordered_688_row_scope_sha256": SCOPE_SHA256,
        "ordered_ranks_int32le_sha256": (
            "a4f610ffb6ff6479212f61f4a8d28c0cb308712aff840a8fc42dec6cc98914a4"),
        "row_count": 688,
        "sentinel_ranks": [0, 343, 687],
        "sentinel_ranks_int32le_sha256": (
            "9605c6e3790102dbb4418fa765c420f39e4300ff6983159dfe0345cbcc84c86a"),
    }


def manifest_contract():
    return {
        "decoded_columns": ["mlp_id", "mlp_name", "mlp_seed", "weights"],
        "exact_python_int_zero_fields": [
            "scores", "target_columns_projected", "target_values_decoded",
            "fits", "optimizers_constructed"],
        "exact_single_manifest_keyset_required": True,
        "exact_single_manifest_schema_required": True,
        "independent_runtime_review_must_authenticate_receipt": True,
        "numerical_module_imported": False,
        "receipt_schema": "public-q24-row-manifest-compatibility-scan-receipt-v4",
        "receipt_status": "PASS_EXACT_REAL_SCHEMA_TARGET_BLIND_METADATA_ONLY",
        "row_manifest_hashes_must_match_ordered_scope": True,
        "targets_opened": False,
        "weights_opened": False,
    }


def generation_law():
    return {
        "all_688_rows_captured_d4_parent_once": True,
        "all_688_rows_literal_h256_once": True,
        "all_row_base_capture_parity_claimed": False,
        "all_row_literal_reference_parity_claimed": False,
        "bank_inventory": [
            "actions_H16.f32le", "actions_H64.f32le", "actions_H256.f32le",
            "A15_zero_H16.f32le", "A15_zero_H64.f32le",
            "A15_zero_H256.f32le", "THREE_BANK_GENERATION_RECEIPT.json"],
        "flopscope_reset_law": {
            "call": "session['flops'].budget_reset()",
            "exact_count_after_successful_validated_rows": 688,
            "exact_count_after_failed_or_unvalidated_rows": 0,
            "placement": "AFTER_VALIDATE_APPEND_OPTIONAL_PROGRESS_BEFORE_NEXT_ROW",
            "inside_active_BudgetContext": False,
            "detached_row_result_and_output_bytes_invariant": True,
            "reset_failure_fails_closed": True,
        },
        "host_prefix_derivation": {
            "H16_receivers_per_sector": 2,
            "H256_receivers_per_sector": 32,
            "H64_receivers_per_sector": 8,
            "exact_final_float32_byte_copy": True,
            "sector_count": 8,
            "sector_major": True,
        },
        "manifest_compatibility_scan_required_before_capability": True,
        "resource_observations_can_veto_generation": False,
        "sentinel_checks": {
            "all_five_parent_components_base_vs_capture_byte_equal": True,
            "derived_h16_h64_vs_literal_byte_equal": True,
            "literal_vs_independent_reference_h16_h64_h256_and_a15_byte_equal": True,
            "ranks": [0, 343, 687],
            "selection_target_independent": True,
        },
        "v3_terminal_attempt_may_be_retried": False,
        "v4_terminal_attempt_may_be_retried": False,
        "v4_staging_is_forensic_only_and_not_an_input": True,
        "v5_is_a_new_distinct_attempt": True,
    }


def permissions():
    return {
        "estimator_deployment_allowed": False,
        "network_allowed": False,
        "one_shot": True,
        "package_allowed": False,
        "retry": False,
        "row_access_allowed": True,
        "source_import_allowed": True,
        "submission_allowed": False,
        "target_access_allowed": False,
        "training_allowed": False,
    }


def load_runtime_instance(bindings, builder=None):
    if builder is None:
        builder = validate_builder_go()
    documents = {}
    for key in ("anchor", "capability", "release", "command"):
        _, documents[key] = read_canonical(
            V5_AUTHORITY / RUNTIME_FILENAMES[key], "live V5 " + key)
    roots = validate_runtime_documents(documents, bindings)
    expected_inventory = set(RUNTIME_FILENAMES.values())
    with os.scandir(nonsymlink_directory(V5_AUTHORITY, "V5 authority root")) as it:
        if {entry.name for entry in it} != expected_inventory:
            raise AuthorityError("V5 authority instance inventory changed")
    cap = V5_AUTHORITY / RUNTIME_FILENAMES["capability"]
    if sha256_file(regular_one_link(cap, "unconsumed V5 capability")) != roots[
            "capability_sha256"]:
        raise AuthorityError("V5 capability changed")
    if any(item.name.startswith(RUNTIME_FILENAMES["capability"] + ".consumed")
           for item in V5_AUTHORITY.iterdir()):
        raise AuthorityError("V5 capability already consumed")
    _, receipt = read_canonical(
        V5_AUTHORITY / RUNTIME_FILENAMES["receipt"], "live V5 receipt")
    if receipt != runtime_receipt(bindings, roots, builder):
        raise AuthorityError("V5 runtime issuance receipt changed")
    return documents, roots


def successor_record(bindings, roots, issued_utc):
    document = {
        "schema": "public-q24-signed-pair-lean-v4-terminal-successor-v1",
        "status": "V4_TERMINAL_FAILED_NO_RETRY__V5_IS_NEW_DISTINCT_ATTEMPT",
        "issued_utc": issued_utc,
        "reason": "V4_FLOPSCOPE_GLOBAL_DIAGNOSTIC_HISTORY_REPAIRED_IN_FRESH_V5",
        "v4_terminal": deepcopy(bindings["v4_terminal"]),
        "v4_capability_consumed": True,
        "v4_claim_complete": True,
        "v4_terminal_failure_no_retry": True,
        "v4_bundle_published": False,
        "v4_staging_forensic_only": True,
        "v4_staging_used_as_input": False,
        "v4_authority_or_transaction_reused": False,
        "v5_authority_anchor_sha256": roots["anchor_sha256"],
        "v5_single_use_capability_sha256": roots["capability_sha256"],
        "v5_final_release_sha256": roots["release_sha256"],
        "v5_runtime_package_manifest_sha256": bindings["v5_runtime"][
            "package_manifest_sha256"],
        "v5_runtime_independent_review_sha256": bindings["v5_runtime"][
            "independent_review_sha256"],
    }
    validate_successor(document, bindings, roots)
    return document


def validate_successor(document, bindings, roots):
    exact_keys(document, {
        "schema", "status", "issued_utc", "reason", "v4_terminal",
        "v4_capability_consumed", "v4_claim_complete",
        "v4_terminal_failure_no_retry", "v4_bundle_published",
        "v4_staging_forensic_only", "v4_staging_used_as_input",
        "v4_authority_or_transaction_reused", "v5_authority_anchor_sha256",
        "v5_single_use_capability_sha256", "v5_final_release_sha256",
        "v5_runtime_package_manifest_sha256",
        "v5_runtime_independent_review_sha256",
    }, "V4 terminal successor")
    if (document["schema"] != "public-q24-signed-pair-lean-v4-terminal-successor-v1"
            or document["status"]
            != "V4_TERMINAL_FAILED_NO_RETRY__V5_IS_NEW_DISTINCT_ATTEMPT"
            or document["reason"]
            != "V4_FLOPSCOPE_GLOBAL_DIAGNOSTIC_HISTORY_REPAIRED_IN_FRESH_V5"
            or document["v4_terminal"] != bindings["v4_terminal"]):
        raise AuthorityError("V4 successor identity changed")
    parse_utc(document["issued_utc"], "V4 successor issued_utc")
    booleans = {
        "v4_capability_consumed": True,
        "v4_claim_complete": True,
        "v4_terminal_failure_no_retry": True,
        "v4_bundle_published": False,
        "v4_staging_forensic_only": True,
        "v4_staging_used_as_input": False,
        "v4_authority_or_transaction_reused": False,
    }
    if any(document[key] is not value for key, value in booleans.items()):
        raise AuthorityError("V4 successor terminal facts changed")
    expected = {
        "v5_authority_anchor_sha256": roots["anchor_sha256"],
        "v5_single_use_capability_sha256": roots["capability_sha256"],
        "v5_final_release_sha256": roots["release_sha256"],
        "v5_runtime_package_manifest_sha256": bindings["v5_runtime"][
            "package_manifest_sha256"],
        "v5_runtime_independent_review_sha256": bindings["v5_runtime"][
            "independent_review_sha256"],
    }
    if any(document[key] != value for key, value in expected.items()):
        raise AuthorityError("V4 successor V5 roots changed")
    return document


def source_bindings(bindings, roots, successor_sha, builder):
    runtime = bindings["v5_runtime"]
    return {
        **ACCEPTED_SOURCE,
        "v5_runtime_package_manifest_sha256": runtime["package_manifest_sha256"],
        "v5_runtime_freeze_receipt_sha256": runtime["freeze_receipt_sha256"],
        "v5_runtime_freeze_verification_sha256": runtime[
            "freeze_verification_sha256"],
        "v5_runtime_static_verification_sha256": runtime[
            "static_verification_sha256"],
        "v5_runtime_independent_review_sha256": runtime[
            "independent_review_sha256"],
        "flopscope_reset_verification_sha256": runtime[
            "flopscope_reset_verification_sha256"],
        "authority_builder_package_manifest_sha256": builder[
            "package_manifest_sha256"],
        "authority_builder_freeze_receipt_sha256": builder[
            "freeze_receipt_sha256"],
        "authority_builder_freeze_verification_sha256": builder[
            "freeze_verification_sha256"],
        "authority_builder_static_verification_sha256": builder[
            "static_verification_sha256"],
        "authority_builder_helper_sha256": builder["helper_sha256"],
        "authority_builder_bindings_sha256": builder["bindings_sha256"],
        "authority_builder_independent_review_sha256": builder[
            "independent_review_sha256"],
        "generation_execution_anchor_sha256": roots["anchor_sha256"],
        "generation_single_use_capability_sha256": roots["capability_sha256"],
        "generation_final_release_sha256": roots["release_sha256"],
        "generation_runner_sha256": runtime["runner_sha256"],
        "callable_bundle_manifest_sha256": runtime[
            "callable_bundle_manifest_sha256"],
        "execution_stack_manifest_sha256": runtime[
            "execution_stack_manifest_sha256"],
        "runtime_token_sha256": runtime["runtime_token_sha256"],
        "parent_capture_sha256": runtime["parent_capture_sha256"],
        "literal_source_sha256": runtime["literal_source_sha256"],
        "independent_reference_source_sha256": runtime[
            "reference_source_sha256"],
        "capture_adapter_sha256": runtime["capture_adapter_sha256"],
        "v3_terminal_failure_lineage_sha256": runtime[
            "v3_terminal_failure_lineage_sha256"],
        "v4_terminal_failure_lineage_sha256": runtime[
            "v4_terminal_failure_lineage_sha256"],
        "independent_incident_root_cause_sha256": bindings[
            "v4_terminal"]["reset_receipt_sha256"],
        "manifest_compatibility_scan_receipt_sha256": MANIFEST_SCAN_SHA256,
        "v4_exclusive_claim_sha256": V4_FIXED["exclusive_claim_sha256"],
        "v4_terminal_failure_sha256": V4_FIXED["terminal_failure_sha256"],
        "v4_consumed_capability_sha256": V4_FIXED[
            "consumed_capability_sha256"],
        "v4_terminal_successor_record_sha256": successor_sha,
    }


def make_source_outputs(bindings, runtime_documents, builder, authority_id,
                        issued_utc):
    roots = validate_runtime_documents(runtime_documents, bindings)
    successor = successor_record(bindings, roots, issued_utc)
    successor_sha = sha256_bytes(canonical_bytes(successor))
    authority = {
        "schema": "public-q24-signed-pair-lean-bank-source-authority-v5",
        "status": "FROZEN_REVIEWED_V5_ALLOWLIST_BEFORE_BANK_GENERATION",
        "authority_id": authority_id,
        "issued_by": "root",
        "issued_utc": issued_utc,
        "frozen_before_generation": True,
        "source_bindings": source_bindings(
            bindings, roots, successor_sha, builder),
        "row_scope_bindings": row_scope_bindings(),
        "manifest_compatibility_contract": manifest_contract(),
        "generation_law": generation_law(),
        "permissions": permissions(),
    }
    authority_sha = sha256_bytes(canonical_bytes(authority))
    receipt = {
        "schema": "public-q24-signed-pair-lean-bank-source-authority-issuance-receipt-v5",
        "status": "ISSUED_BEFORE_GENERATION__AWAITING_INDEPENDENT_V5_LIVE_REVIEW",
        "source_authority_sha256": authority_sha,
        "v4_terminal_successor_record_sha256": successor_sha,
        "generation_execution_anchor_sha256": roots["anchor_sha256"],
        "generation_single_use_capability_sha256": roots["capability_sha256"],
        "generation_final_release_sha256": roots["release_sha256"],
        "independent_incident_root_cause_sha256": bindings[
            "v4_terminal"]["reset_receipt_sha256"],
        "flopscope_reset_verification_sha256": bindings["v5_runtime"][
            "flopscope_reset_verification_sha256"],
        "authority_builder_independent_review_sha256": builder[
            "independent_review_sha256"],
        "fresh_capability_id": roots["capability_id"],
        "v4_terminal_state_observed": True,
        "v4_staging_used_as_input": False,
        "fresh_v5_capability_observed_unconsumed": True,
        "fresh_v5_transaction_root_observed_empty": True,
        "generation_started": False,
        "independent_review_exists": False,
        "generation_authorized_by_this_receipt": False,
    }
    request = {
        "schema": "public-q24-signed-pair-lean-bank-source-authority-live-review-request-v5",
        "status": "REVIEW_REQUESTED__NO_GENERATION_UNTIL_SEPARATE_V5_GO",
        "source_authority_sha256": authority_sha,
        "v4_terminal_successor_record_sha256": successor_sha,
        "issuance_receipt_sha256": sha256_bytes(canonical_bytes(receipt)),
        "generation_final_release_sha256": roots["release_sha256"],
        "independent_incident_root_cause_sha256": bindings[
            "v4_terminal"]["reset_receipt_sha256"],
        "flopscope_reset_verification_sha256": bindings["v5_runtime"][
            "flopscope_reset_verification_sha256"],
        "authority_builder_independent_review_sha256": builder[
            "independent_review_sha256"],
        "required_review_schema": (
            "public-q24-signed-pair-lean-bank-source-authority-independent-review-v5"),
        "required_review_status": (
            "GO_LEAN_V5_THREE_BANK_GENERATION_AUTHORITY_ALLOWLIST"),
        "reviewer_must_rehash_static_and_live_roots": True,
        "reviewer_must_reverify_v4_terminal_reset_and_v5_fresh_transaction": True,
        "helper_emits_independent_go": False,
    }
    outputs = {
        SOURCE_FILENAMES["successor"]: successor,
        SOURCE_FILENAMES["authority"]: authority,
        SOURCE_FILENAMES["receipt"]: receipt,
        SOURCE_FILENAMES["request"]: request,
    }
    validate_source_outputs(outputs, bindings, runtime_documents, builder)
    return outputs


def validate_source_outputs(outputs, bindings, runtime_documents, builder):
    if set(outputs) != set(SOURCE_FILENAMES.values()):
        raise AuthorityError("V5 source output inventory changed")
    roots = validate_runtime_documents(runtime_documents, bindings)
    successor = outputs[SOURCE_FILENAMES["successor"]]
    validate_successor(successor, bindings, roots)
    successor_sha = sha256_bytes(canonical_bytes(successor))
    authority = exact_keys(outputs[SOURCE_FILENAMES["authority"]], {
        "schema", "status", "authority_id", "issued_by", "issued_utc",
        "frozen_before_generation", "source_bindings", "row_scope_bindings",
        "manifest_compatibility_contract", "generation_law", "permissions",
    }, "V5 source authority")
    if (authority["schema"]
            != "public-q24-signed-pair-lean-bank-source-authority-v5"
            or authority["status"]
            != "FROZEN_REVIEWED_V5_ALLOWLIST_BEFORE_BANK_GENERATION"
            or authority["issued_by"] != "root"
            or authority["frozen_before_generation"] is not True
            or authority["source_bindings"] != source_bindings(
                bindings, roots, successor_sha, builder)
            or authority["row_scope_bindings"] != row_scope_bindings()
            or authority["manifest_compatibility_contract"] != manifest_contract()
            or authority["generation_law"] != generation_law()
            or authority["permissions"] != permissions()):
        raise AuthorityError("V5 source authority changed")
    require_id(authority["authority_id"], "V5 source authority id")
    parse_utc(authority["issued_utc"], "V5 source issued_utc")
    authority_sha = sha256_bytes(canonical_bytes(authority))
    receipt = exact_keys(outputs[SOURCE_FILENAMES["receipt"]], {
        "schema", "status", "source_authority_sha256",
        "v4_terminal_successor_record_sha256",
        "generation_execution_anchor_sha256",
        "generation_single_use_capability_sha256",
        "generation_final_release_sha256",
        "independent_incident_root_cause_sha256",
        "flopscope_reset_verification_sha256",
        "authority_builder_independent_review_sha256", "fresh_capability_id",
        "v4_terminal_state_observed", "v4_staging_used_as_input",
        "fresh_v5_capability_observed_unconsumed",
        "fresh_v5_transaction_root_observed_empty", "generation_started",
        "independent_review_exists", "generation_authorized_by_this_receipt",
    }, "V5 source receipt")
    expected_receipt = {
        "schema": "public-q24-signed-pair-lean-bank-source-authority-issuance-receipt-v5",
        "status": "ISSUED_BEFORE_GENERATION__AWAITING_INDEPENDENT_V5_LIVE_REVIEW",
        "source_authority_sha256": authority_sha,
        "v4_terminal_successor_record_sha256": successor_sha,
        "generation_execution_anchor_sha256": roots["anchor_sha256"],
        "generation_single_use_capability_sha256": roots["capability_sha256"],
        "generation_final_release_sha256": roots["release_sha256"],
        "independent_incident_root_cause_sha256": bindings[
            "v4_terminal"]["reset_receipt_sha256"],
        "flopscope_reset_verification_sha256": bindings["v5_runtime"][
            "flopscope_reset_verification_sha256"],
        "authority_builder_independent_review_sha256": builder[
            "independent_review_sha256"],
        "fresh_capability_id": roots["capability_id"],
        "v4_terminal_state_observed": True,
        "v4_staging_used_as_input": False,
        "fresh_v5_capability_observed_unconsumed": True,
        "fresh_v5_transaction_root_observed_empty": True,
        "generation_started": False,
        "independent_review_exists": False,
        "generation_authorized_by_this_receipt": False,
    }
    if receipt != expected_receipt:
        raise AuthorityError("V5 source receipt changed or overclaims")
    request = exact_keys(outputs[SOURCE_FILENAMES["request"]], {
        "schema", "status", "source_authority_sha256",
        "v4_terminal_successor_record_sha256", "issuance_receipt_sha256",
        "generation_final_release_sha256",
        "independent_incident_root_cause_sha256",
        "flopscope_reset_verification_sha256",
        "authority_builder_independent_review_sha256",
        "required_review_schema",
        "required_review_status", "reviewer_must_rehash_static_and_live_roots",
        "reviewer_must_reverify_v4_terminal_reset_and_v5_fresh_transaction",
        "helper_emits_independent_go",
    }, "V5 source review request")
    if (request["schema"]
            != "public-q24-signed-pair-lean-bank-source-authority-live-review-request-v5"
            or request["status"]
            != "REVIEW_REQUESTED__NO_GENERATION_UNTIL_SEPARATE_V5_GO"
            or request["source_authority_sha256"] != authority_sha
            or request["v4_terminal_successor_record_sha256"] != successor_sha
            or request["issuance_receipt_sha256"]
            != sha256_bytes(canonical_bytes(receipt))
            or request["generation_final_release_sha256"] != roots["release_sha256"]
            or request["independent_incident_root_cause_sha256"]
            != bindings["v4_terminal"]["reset_receipt_sha256"]
            or request["flopscope_reset_verification_sha256"]
            != bindings["v5_runtime"]["flopscope_reset_verification_sha256"]
            or request["authority_builder_independent_review_sha256"]
            != builder["independent_review_sha256"]
            or request["required_review_schema"]
            != "public-q24-signed-pair-lean-bank-source-authority-independent-review-v5"
            or request["required_review_status"]
            != "GO_LEAN_V5_THREE_BANK_GENERATION_AUTHORITY_ALLOWLIST"
            or request["reviewer_must_rehash_static_and_live_roots"] is not True
            or request[
                "reviewer_must_reverify_v4_terminal_reset_and_v5_fresh_transaction"]
            is not True
            or request["helper_emits_independent_go"] is not False):
        raise AuthorityError("V5 source review request changed")
    return outputs


def load_source_instance(bindings, runtime_documents, builder):
    root = nonsymlink_directory(V5_SOURCE, "V5 source authority root")
    with os.scandir(root) as iterator:
        if {entry.name for entry in iterator} != set(SOURCE_FILENAMES.values()):
            raise AuthorityError("V5 source authority inventory changed")
    outputs = {}
    for name in SOURCE_FILENAMES.values():
        _, outputs[name] = read_canonical(root / name, "live V5 source " + name)
    validate_source_outputs(outputs, bindings, runtime_documents, builder)
    return outputs


def source_request(document, require_confirmation):
    exact_keys(document, {
        "schema", "source_instance_dir", "authority_id", "issued_utc",
        "confirmation",
    }, "V5 source issuance request")
    if document["schema"] != "public-q24-lean-v5-source-authority-issuance-request-v1":
        raise AuthorityError("V5 source request schema changed")
    if document["source_instance_dir"] != os.fspath(V5_SOURCE):
        raise AuthorityError("V5 source authority root changed")
    if require_confirmation and document["confirmation"] != SOURCE_CONFIRMATION:
        raise AuthorityError("V5 source confirmation missing")
    if not require_confirmation and document["confirmation"] not in (
            None, SOURCE_CONFIRMATION):
        raise AuthorityError("V5 source confirmation malformed")
    require_id(document["authority_id"], "V5 source authority id",
               "q24-sp3b-lean-v5-source-")
    parsed = parse_utc(document["issued_utc"], "V5 source issued_utc")
    if require_confirmation:
        delta = (datetime.now(timezone.utc) - parsed).total_seconds()
        if delta < -60 or delta > 300:
            raise AuthorityError("V5 source timestamp is not recent")
    return document


def source_context(bindings_path, request_path, require_confirmation):
    bindings = load_bindings(bindings_path, True)
    validate_v4_terminal_metadata(bindings)
    validate_runtime_artifacts(bindings)
    builder = validate_builder_go()
    _, request = read_canonical(request_path, "V5 source issuance request")
    source_request(request, require_confirmation)
    require_fresh_transaction(False)
    runtime_documents, roots = load_runtime_instance(bindings, builder)
    if parse_utc(request["issued_utc"], "source issued_utc") < parse_utc(
            roots["release_issued_utc"], "release issued_utc"):
        raise AuthorityError("V5 source authority predates runtime release")
    if V5_SOURCE.exists() or V5_SOURCE.is_symlink():
        raise AuthorityError("V5 source authority root already exists")
    if V5_SOURCE_REVIEW.exists() or V5_SOURCE_REVIEW.is_symlink():
        raise AuthorityError("V5 source review root already exists")
    outputs = make_source_outputs(
        bindings, runtime_documents, builder, request["authority_id"],
        request["issued_utc"])
    return bindings, request, outputs


def source_preflight(bindings_path, request_path):
    bindings, _, outputs = source_context(bindings_path, request_path, False)
    authority = outputs[SOURCE_FILENAMES["authority"]]
    successor = outputs[SOURCE_FILENAMES["successor"]]
    return {
        "schema": "public-q24-lean-v5-source-authority-preflight-v1",
        "status": "PASS_NO_SOURCE_AUTHORITY_NO_REVIEW_NO_EXECUTION",
        "prospective_source_authority_sha256": sha256_bytes(
            canonical_bytes(authority)),
        "prospective_successor_sha256": sha256_bytes(canonical_bytes(successor)),
        "incident_root_cause_sha256": bindings["v4_terminal"][
            "reset_receipt_sha256"],
        "v4_staging_entries_opened": 0,
        "execution_authorized": False,
    }


def source_issue(bindings_path, request_path):
    _, _, outputs = source_context(bindings_path, request_path, True)
    hashes = atomic_publish(V5_SOURCE, outputs, list(outputs))
    return {
        "schema": "public-q24-lean-v5-source-authority-issuance-result-v1",
        "status": "ISSUED__HOLD_FOR_INDEPENDENT_V5_LIVE_REVIEW",
        "artifact_sha256": hashes,
        "generation_authorized": False,
        "v4_staging_entries_opened": 0,
        "numerical_execution": False,
    }


def validate_live_source_review(review, authority, successor, source_issued_utc,
                                incident_sha256, reset_verification_sha256,
                                builder_review_sha256):
    exact_keys(review, {
        "schema", "status", "review_id", "issued_by", "issued_utc",
        "authority_sha256", "v4_terminal_successor_record_sha256",
        "independent_incident_root_cause_sha256",
        "flopscope_reset_verification_sha256",
        "authority_builder_independent_review_sha256",
        "reviewed_before_generation",
        "target_free_scope_verified", "exact_source_allowlist_verified",
        "exact_generation_law_verified", "sentinel_contract_verified",
        "manifest_compatibility_scan_verified",
        "flopscope_post_commit_reset_law_verified",
        "v4_terminal_claim_failure_consumption_and_reset_verified",
        "v4_staging_forensic_only_and_not_input", "fresh_v5_transaction_root_empty",
        "generation_authorized",
    }, "V5 live source review")
    if (review["schema"]
            != "public-q24-signed-pair-lean-bank-source-authority-independent-review-v5"
            or review["status"]
            != "GO_LEAN_V5_THREE_BANK_GENERATION_AUTHORITY_ALLOWLIST"
            or review["issued_by"] != "independent_reviewer"
            or review["authority_sha256"]
            != sha256_bytes(canonical_bytes(authority))
            or review["v4_terminal_successor_record_sha256"]
            != sha256_bytes(canonical_bytes(successor))
            or review["independent_incident_root_cause_sha256"] != incident_sha256
            or review["flopscope_reset_verification_sha256"]
            != reset_verification_sha256
            or review["authority_builder_independent_review_sha256"]
            != builder_review_sha256):
        raise AuthorityError("V5 live source review identity changed")
    require_id(review["review_id"], "V5 live review id")
    if parse_utc(review["issued_utc"], "V5 live review issued_utc") < parse_utc(
            source_issued_utc, "V5 source issued_utc"):
        raise AuthorityError("V5 live review predates source authority")
    for key in (
            "reviewed_before_generation", "target_free_scope_verified",
            "exact_source_allowlist_verified", "exact_generation_law_verified",
            "sentinel_contract_verified", "manifest_compatibility_scan_verified",
            "flopscope_post_commit_reset_law_verified",
            "v4_terminal_claim_failure_consumption_and_reset_verified",
            "v4_staging_forensic_only_and_not_input",
            "fresh_v5_transaction_root_empty", "generation_authorized"):
        if review[key] is not True:
            raise AuthorityError("V5 live source review closure failed: " + key)
    return review


def final_gate(bindings_path):
    bindings = load_bindings(bindings_path, True)
    validate_v4_terminal_metadata(bindings)
    validate_runtime_artifacts(bindings)
    builder = validate_builder_go()
    require_fresh_transaction(False)
    runtime_documents, roots = load_runtime_instance(bindings, builder)
    source_outputs = load_source_instance(bindings, runtime_documents, builder)
    authority = source_outputs[SOURCE_FILENAMES["authority"]]
    successor = source_outputs[SOURCE_FILENAMES["successor"]]
    review_root = nonsymlink_directory(V5_SOURCE_REVIEW, "V5 source review root")
    with os.scandir(review_root) as iterator:
        if {entry.name for entry in iterator} != {"INDEPENDENT_LIVE_REVIEW_V5.json"}:
            raise AuthorityError("V5 source review inventory changed")
    _, review = read_canonical(
        review_root / "INDEPENDENT_LIVE_REVIEW_V5.json",
        "V5 live source review")
    validate_live_source_review(
        review, authority, successor, authority["issued_utc"],
        bindings["v4_terminal"]["reset_receipt_sha256"],
        bindings["v5_runtime"]["flopscope_reset_verification_sha256"],
        builder["independent_review_sha256"])
    command = runtime_documents["command"]
    return {
        "schema": "public-q24-lean-v5-final-process-gate-v1",
        "status": "GO_EXACTLY_ONE_FRESH_LEAN_V5_TARGET_BLIND_688_BANK_PASS",
        "authority_anchor_sha256": roots["anchor_sha256"],
        "single_use_capability_sha256": roots["capability_sha256"],
        "final_release_sha256": roots["release_sha256"],
        "source_authority_sha256": sha256_bytes(canonical_bytes(authority)),
        "source_review_sha256": sha256_bytes(canonical_bytes(review)),
        "authority_builder_independent_review_sha256": builder[
            "independent_review_sha256"],
        "stored_command_sha256": sha256_bytes(canonical_bytes(command)),
        "v4_retry": False,
        "v4_staging_used_as_input": False,
        "fresh_v5_transaction_empty": True,
        "fresh_v5_capability_unconsumed": True,
        "target_access_by_gate": False,
        "numerical_execution": False,
        "one_shot": True,
        "no_retry": True,
    }


def synthetic_live_bindings(hold):
    document = deepcopy(hold)
    document["status"] = "FROZEN_READY_AFTER_V4_RESET_AND_V5_RUNTIME_GO"
    document["v4_terminal"]["reset_receipt_sha256"] = hashlib.sha256(
        b"fixture:reset").hexdigest()
    for key in sorted(V5_RUNTIME_KEYS):
        document["v5_runtime"][key] = hashlib.sha256(
            ("fixture:" + key).encode("ascii")).hexdigest()
    return validate_bindings(document, True)


def synthetic_runtime_review(bindings):
    runtime = bindings["v5_runtime"]
    review = {
        "schema": "public-q24-three-bank-runtime-static-review-v5",
        "status": "GO",
        "issued_by": "independent_reviewer",
        "issued_utc": "2026-08-28T00:40:00Z",
        "reviewed_runner_sha256": runtime["runner_sha256"],
        "reviewed_callable_bundle_manifest_sha256": runtime[
            "callable_bundle_manifest_sha256"],
        "reviewed_execution_stack_manifest_sha256": runtime[
            "execution_stack_manifest_sha256"],
        "prior_runtime_v1_hold_sha256": (
            "49e51e300c158229f8401a5df032feebc6ae5cc8ae48ab1f2e55b2a407646d32"),
        "prior_runtime_v2_go_sha256": (
            "7bf6664b2e27d863e5fc4a556373a217d05628ec681141757a4d0beec46c0b03"),
        "prior_runtime_v3_claim_sha256": (
            "bfa9e8d800a5b1efc0ec4e47a62b53d70176e0bf8b9ec73f9981056974abd4e9"),
        "prior_runtime_v3_terminal_failure_sha256": (
            "e64f998fc1513f906e90d2869340e44ce2ca548b91e31667aade69da9af1d50d"),
        "prior_runtime_v3_consumed_capability_sha256": (
            "f7cd31ab124c7914d1bd2b413289fb84ced4a38d2e93e3f3d3d9a9e711d38a39"),
        "prior_runtime_v4_terminal_failure_lineage_sha256": runtime[
            "v4_terminal_failure_lineage_sha256"],
        "prior_runtime_v4_claim_sha256": V4_FIXED["exclusive_claim_sha256"],
        "prior_runtime_v4_terminal_failure_sha256": V4_FIXED[
            "terminal_failure_sha256"],
        "prior_runtime_v4_consumed_capability_sha256": V4_FIXED[
            "consumed_capability_sha256"],
        "independent_incident_root_cause_sha256": (
            "060e7bab3094a7929d246f4bdf5fbe6790ccb3aa9c3b05734395a6376b24cd6c"),
        "flopscope_reset_verification_sha256": runtime[
            "flopscope_reset_verification_sha256"],
        "manifest_compatibility_scan_receipt_sha256": MANIFEST_SCAN_SHA256,
        "r1_public_api_and_governance_chronology_closed": True,
        "hostile_interpreter_security_claim": False,
        "r2_authority_dag_acyclic": True,
        "r3_resource_veto_removed": True,
        "r4_dynamic_comparators_present": True,
        "r5_manifest_firewall_semantic_repair_verified": True,
        "r6_flopscope_reset_placement_and_invariance_verified": True,
        "v3_terminal_no_retry_closed": True,
        "v4_terminal_no_retry_closed": True,
        "source_data_target_execution_during_review": False,
    }
    return validate_runtime_review(review, runtime)


def synthetic_builder_provenance():
    return {
        key: hashlib.sha256(("fixture:builder:" + key).encode("ascii")).hexdigest()
        for key in (
            "package_manifest_sha256", "freeze_receipt_sha256",
            "freeze_verification_sha256", "static_verification_sha256",
            "helper_sha256", "bindings_sha256", "independent_review_sha256")
    }


def expected_failure(label, operation):
    try:
        operation()
    except AuthorityError:
        return label
    raise AuthorityError("negative fixture passed: " + label)


def self_test():
    bindings = load_bindings(HERE / "BINDINGS_HOLD.json", False)
    validate_v4_terminal_metadata(bindings)
    null_live = deepcopy(bindings)
    null_live["v5_runtime"]["independent_review_sha256"] = None
    failures = [expected_failure(
        "null-live-bindings", lambda: validate_bindings(null_live, True))]
    live = synthetic_live_bindings(bindings)
    review = synthetic_runtime_review(live)
    request = {
        "schema": "public-q24-lean-v5-runtime-authority-issuance-request-v1",
        "instance_dir": os.fspath(V5_AUTHORITY),
        "transaction_root": os.fspath(V5_TRANSACTION),
        "working_directory": os.fspath(WORKSPACE_ROOT),
        "capability_id": "q24-sp3b-lean-v5-0123456789abcdef0123456789abcdef",
        "issued_by": "root",
        "anchor_issued_utc": "2026-08-28T00:41:00Z",
        "capability_issued_utc": "2026-08-28T00:42:00Z",
        "release_issued_utc": "2026-08-28T00:43:00Z",
        "confirmation": None,
    }
    runtime_request(request, False)
    runtime_documents = make_runtime_documents(live, request, review)
    builder = synthetic_builder_provenance()
    builder_roots = {key: value for key, value in builder.items()
                     if key != "independent_review_sha256"}
    builder_freeze = {"issued_utc": "2026-08-28T00:49:00Z"}
    builder_review = {
        "schema": (
            "public-q24-lean-v5-prelaunch-authority-independent-static-review-v1"),
        "status": "GO_FROZEN_FAIL_CLOSED_V5_PRELAUNCH_AUTHORITY_BUILDER",
        "review_id": "q24-sp3b-lean-v5-authority-builder-review-fixture",
        "issued_by": "independent_reviewer",
        "issued_utc": "2026-08-28T00:50:00Z",
        **builder_roots,
        "runtime_v5_package_manifest_sha256": (
            "6307642b0ebe6397e9ffb7c5d163fe29970b5644fb39f76a240fe59afbd4935d"),
        "runtime_v5_independent_review_sha256": (
            "26af3b9cf9cb4320b9dfff1917723655ee7ba4046dc99a302fcd2f1608cb2f8e"),
        "v4_terminal_and_incident_bound": True,
        "path_overlap_and_exact_argv_verified": True,
        "source_and_live_review_gates_verified": True,
        "frozen_static_replay_verified": True,
        "reviewed_without_execution_or_protected_data": True,
        "reviewer_different_from_package_author": True,
        "hostile_interpreter_security_claim": False,
    }
    validate_builder_go_document(builder_review, builder_roots, builder_freeze)
    for label, key, value in (
            ("builder-GO-package-root", "package_manifest_sha256", "0" * 64),
            ("builder-GO-helper-root", "helper_sha256", "0" * 64),
            ("builder-GO-static-root", "static_verification_sha256", "0" * 64),
            ("builder-GO-bindings-root", "bindings_sha256", "0" * 64),
            ("builder-GO-self-authored", "issued_by", "root"),
            ("builder-GO-false-closure", "frozen_static_replay_verified", False),
            ("builder-GO-hostile-overclaim", "hostile_interpreter_security_claim",
             True),
            ("builder-GO-pre-freeze", "issued_utc", "2026-08-28T00:48:59Z")):
        candidate = deepcopy(builder_review)
        candidate[key] = value
        failures.append(expected_failure(
            label, lambda candidate=candidate: validate_builder_go_document(
                candidate, builder_roots, builder_freeze)))
    failures.append(expected_failure(
        "builder-GO-missing-inventory", lambda: validate_builder_review_inventory(
            set())))
    failures.append(expected_failure(
        "builder-GO-extra-inventory", lambda: validate_builder_review_inventory(
            {V5_BUILDER_REVIEW.name, "EXTRA.json"})))
    roots = validate_runtime_documents(runtime_documents, live)
    if not all(is_sha(roots[key]) for key in (
            "anchor_sha256", "capability_sha256", "release_sha256")):
        raise AuthorityError("synthetic V5 authority roots malformed")
    source_outputs = make_source_outputs(
        live, runtime_documents, builder, "q24-sp3b-lean-v5-source-fixture",
        "2026-08-28T00:44:00Z")
    validate_source_outputs(source_outputs, live, runtime_documents, builder)
    source_authority = source_outputs[SOURCE_FILENAMES["authority"]]
    source_successor = source_outputs[SOURCE_FILENAMES["successor"]]
    source_review = {
        "schema": (
            "public-q24-signed-pair-lean-bank-source-authority-independent-review-v5"),
        "status": "GO_LEAN_V5_THREE_BANK_GENERATION_AUTHORITY_ALLOWLIST",
        "review_id": "q24-sp3b-lean-v5-source-review-fixture",
        "issued_by": "independent_reviewer",
        "issued_utc": "2026-08-28T00:45:00Z",
        "authority_sha256": sha256_bytes(canonical_bytes(source_authority)),
        "v4_terminal_successor_record_sha256": sha256_bytes(
            canonical_bytes(source_successor)),
        "independent_incident_root_cause_sha256": live["v4_terminal"][
            "reset_receipt_sha256"],
        "flopscope_reset_verification_sha256": live["v5_runtime"][
            "flopscope_reset_verification_sha256"],
        "authority_builder_independent_review_sha256": builder[
            "independent_review_sha256"],
        "reviewed_before_generation": True,
        "target_free_scope_verified": True,
        "exact_source_allowlist_verified": True,
        "exact_generation_law_verified": True,
        "sentinel_contract_verified": True,
        "manifest_compatibility_scan_verified": True,
        "flopscope_post_commit_reset_law_verified": True,
        "v4_terminal_claim_failure_consumption_and_reset_verified": True,
        "v4_staging_forensic_only_and_not_input": True,
        "fresh_v5_transaction_root_empty": True,
        "generation_authorized": True,
    }
    validate_live_source_review(
        source_review, source_authority, source_successor,
        source_authority["issued_utc"], live["v4_terminal"][
            "reset_receipt_sha256"], live["v5_runtime"][
                "flopscope_reset_verification_sha256"], builder[
                    "independent_review_sha256"])
    for label, section, key, value in (
            ("V4-failure", "v4_terminal", "terminal_failure_sha256", "0" * 64),
            ("V4-reset", "v4_terminal", "reset_receipt_sha256", None),
            ("V5-runner", "v5_runtime", "runner_sha256", "0" * 63),
            ("V5-review", "v5_runtime", "independent_review_sha256", None),
            ("V5-lineage", "v5_runtime", "v4_terminal_failure_lineage_sha256", None),
    ):
        candidate = deepcopy(live)
        candidate[section][key] = value
        failures.append(expected_failure(
            label, lambda candidate=candidate: validate_bindings(candidate, True)))
    duplicate = deepcopy(live)
    duplicate["v5_runtime"]["runner_sha256"] = duplicate[
        "v5_runtime"]["static_verification_sha256"]
    failures.append(expected_failure(
        "runtime-root-collision", lambda: validate_bindings(duplicate, True)))
    protected_cases = (
        ("root-child-V2-transaction",
         next(root for root in PRIOR_V2_ROOTS if "transaction" in root.name)),
        ("root-child-V3-authority",
         next(root for root in PRIOR_V3_ROOTS if "authority_instance" in root.name)),
        ("root-child-V4-transaction", V4_TRANSACTION),
        ("root-child-V4-staging", V4_STAGING),
        ("root-child-V5-runtime", V5_RUNTIME),
        ("root-child-ordered-scope", ROW_SCOPE.parent),
        ("root-child-authority-builder", HERE),
    )
    protected = protected_evidence_roots()
    for label, parent in protected_cases:
        candidate = parent / "forbidden-v5-child"
        failures.append(expected_failure(
            label, lambda candidate=candidate: require_new_roots_disjoint(
                [candidate, V5_TRANSACTION], protected, label)))
    failures.append(expected_failure(
        "new-root-ancestor-overlap", lambda: require_new_roots_disjoint(
            [V5_AUTHORITY, V5_AUTHORITY / "forbidden-v5-child"], set(),
            "new-root-ancestor-overlap")))
    bad_review = deepcopy(review)
    bad_review["r6_flopscope_reset_placement_and_invariance_verified"] = False
    failures.append(expected_failure(
        "review-reset-hold",
        lambda: validate_runtime_review(bad_review, live["v5_runtime"])))
    for label, forbidden in (
            ("argv-V2-reuse", next(root for root in PRIOR_V2_ROOTS
                                   if "transaction" in root.name)),
            ("argv-V3-reuse", next(root for root in PRIOR_V3_ROOTS
                                   if "transaction" in root.name)),
            ("argv-V4-reuse", V4_TRANSACTION)):
        bad_argv = deepcopy(runtime_documents)
        bad_argv["command"]["argv"][-1] = os.fspath(forbidden)
        bad_argv["command"]["shell_rendering_for_display_only"] = shlex.join(
            bad_argv["command"]["argv"])
        failures.append(expected_failure(
            label, lambda bad_argv=bad_argv: validate_runtime_documents(
                bad_argv, live)))
    for label, section, key, value in (
            ("anchor-version", "anchor", "schema",
             "public-q24-three-bank-runtime-authority-anchor-v4"),
            ("capability-retry", "capability", "no_retry", False),
            ("capability-release-cycle", "capability", "final_release_sha256",
             "0" * 64),
            ("release-capability", "release", "single_use_capability_sha256",
             "0" * 64),
            ("command-staging-input", "command", "v4_staging_used_as_input", True),
    ):
        candidate = deepcopy(runtime_documents)
        candidate[section][key] = value
        failures.append(expected_failure(
            label, lambda candidate=candidate: validate_runtime_documents(
                candidate, live)))
    bad_source = deepcopy(source_outputs)
    bad_source[SOURCE_FILENAMES["authority"]]["generation_law"][
        "resource_observations_can_veto_generation"] = True
    failures.append(expected_failure(
        "source-resource-veto", lambda: validate_source_outputs(
            bad_source, live, runtime_documents, builder)))
    bad_source = deepcopy(source_outputs)
    bad_source[SOURCE_FILENAMES["authority"]]["generation_law"][
        "v4_staging_is_forensic_only_and_not_an_input"] = False
    failures.append(expected_failure(
        "source-v4-staging-input", lambda: validate_source_outputs(
            bad_source, live, runtime_documents, builder)))
    bad_live_review = deepcopy(source_review)
    bad_live_review["flopscope_post_commit_reset_law_verified"] = False
    failures.append(expected_failure(
        "source-review-reset-hold", lambda: validate_live_source_review(
            bad_live_review, source_authority, source_successor,
            source_authority["issued_utc"], live["v4_terminal"][
                "reset_receipt_sha256"], live["v5_runtime"][
                    "flopscope_reset_verification_sha256"], builder[
                        "independent_review_sha256"])))
    return {
        "schema": "public-q24-lean-v5-prelaunch-authority-self-test-v1",
        "status": "PASS_STATIC_HOLD_NO_V5_AUTHORITY_NO_EXECUTION",
        "adversarial_fixture_count": len(failures),
        "adversarial_evidence_sha256": sha256_bytes(canonical_bytes(failures)),
        "v4_claim_sha256": V4_FIXED["exclusive_claim_sha256"],
        "v4_terminal_failure_sha256": V4_FIXED["terminal_failure_sha256"],
        "v4_consumed_capability_sha256": V4_FIXED[
            "consumed_capability_sha256"],
        "v4_retry_allowed": False,
        "v4_staging_forensic_only": True,
        "v4_staging_entries_opened": 0,
        "fresh_v5_transaction_required": True,
        "synthetic_v5_authority_roots": roots,
        "mps_authorized_for_v5_run": False,
        "resource_or_novelty_veto_allowed": False,
        "target_bytes_opened": False,
        "weights_or_row_manifests_opened": False,
        "numerical_execution": False,
        "live_authority_created": False,
    }


def hold_status(bindings_path):
    bindings = load_bindings(bindings_path, False)
    return {
        "schema": "public-q24-lean-v5-prelaunch-authority-hold-v1",
        "status": bindings["status"],
        "reset_receipt_sha256": bindings["v4_terminal"][
            "reset_receipt_sha256"],
        "v5_runtime": deepcopy(bindings["v5_runtime"]),
        "execution_authorized": False,
        "v4_retry_allowed": False,
        "v4_staging_is_input": False,
        "target_access_allowed": False,
        "numerical_execution": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    hold = sub.add_parser("hold-status")
    hold.add_argument("--bindings", required=True)
    for name in ("preflight-runtime", "issue-runtime", "preflight-source",
                 "issue-source"):
        child = sub.add_parser(name)
        child.add_argument("--bindings", required=True)
        child.add_argument("--request", required=True)
    gate = sub.add_parser("final-gate")
    gate.add_argument("--bindings", required=True)
    args = parser.parse_args(argv)
    if args.command == "self-test":
        result = self_test()
    elif args.command == "hold-status":
        result = hold_status(args.bindings)
    elif args.command == "preflight-runtime":
        result = runtime_preflight(args.bindings, args.request)
    elif args.command == "issue-runtime":
        result = runtime_issue(args.bindings, args.request)
    elif args.command == "preflight-source":
        result = source_preflight(args.bindings, args.request)
    elif args.command == "issue-source":
        result = source_issue(args.bindings, args.request)
    else:
        result = final_gate(args.bindings)
    sys.stdout.buffer.write(canonical_bytes(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuthorityError as error:
        print("HOLD: " + str(error), file=sys.stderr)
        raise SystemExit(2)
