#!/usr/bin/env python3
"""Metadata-only V4 gate for the production fit/selection target bundle.

Import and ``preflight`` never open or hash either role Parquet.  Activation
requires a future independent row-level PASS plus separately issued reader and
one-use training authorizations.  The training process must call
``validate_activation`` before its irreversible claim and may import the
sealed reader only after that claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BINDING_PATH = HERE / "BUNDLE_BINDING_V4.json"
DATA_GATE = HERE.parents[1] / "data_gate"
EXPECTED_BINDING_SCHEMA = "phase2-observer-v4-sealed-role-bundle-binding-v1"
EXPECTED_PASS_STATUS = (
    "PASS_PRODUCTION_CUSTODIAN_V3_POST_SEAL_PAYLOAD_INTEGRITY_AND_CUSTODY_BOUNDARY"
)
EXPECTED_POST_SEAL_SHA256 = (
    "09a8c90833fedd44c541f9b61c74c1d4b4414f52b5b009cfe1ac630791a505ab"
)
EXPECTED_LAUNCH_AUDIT_STATUS = "PASS_PHASE2_OBSERVER_V4_TARGET_FREE_LAUNCH_SCAFFOLD"
TRAINING_AUTHORIZATION_SCHEMA = "phase2-observer-v4-one-use-training-authorization-v1"
EXPECTED_ROLES = ("observer_fit", "observer_selection")
WITHHELD_ROLES = (
    "observer_audit",
    "observer_lockbox",
    "retired_teacher_diagnostics",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _binding() -> dict[str, Any]:
    payload = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != EXPECTED_BINDING_SCHEMA:
        raise AssertionError("V4 bundle binding schema changed")
    if payload.get("status") != (
        "POST_SEAL_PASS_BOUND__INACTIVE_PENDING_INDEPENDENT_LAUNCH_AUDIT_AND_ONE_USE_AUTHORIZATION"
    ):
        raise AssertionError("target-free binding must remain inactive")
    return payload


def preflight() -> dict[str, Any]:
    """Check only JSON/source hashes and directory membership, never Parquets."""

    binding = _binding()
    custody = binding["production_custody"]
    receipt = DATA_GATE / "PRODUCTION_CUSTODIAN_V3_SEAL_RECEIPT.json"
    manifest = DATA_GATE / "sealed_role_target_bundle" / "manifest.json"
    reader = DATA_GATE / "sealed_role_targets_v2.py"
    audit = DATA_GATE / "INDEPENDENT_PRODUCTION_CUSTODIAN_V3_POST_SEAL_AUDIT_PASS.json"
    expected = (
        (receipt, custody["seal_receipt_sha256"]),
        (manifest, custody["bundle_manifest_sha256"]),
        (reader, custody["sealed_reader_sha256"]),
        (audit, EXPECTED_POST_SEAL_SHA256),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise AssertionError(f"metadata/source binding changed: {path.name}")
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    if receipt_payload.get("status") != "PASS_ONE_USE_PRODUCTION_V3_ROLE_TARGET_SEAL_AND_METADATA_VERIFICATION":
        raise AssertionError("production seal receipt status changed")
    if receipt_payload.get("ordinary_training_reader_authorized") is not False:
        raise AssertionError("production receipt unexpectedly authorizes training")
    if set(manifest_payload.get("artifacts", {})) != set(EXPECTED_ROLES):
        raise AssertionError("bundle metadata does not contain exactly fit and selection")
    if manifest_payload.get("withheld_roles_included") is not False:
        raise AssertionError("bundle metadata includes a withheld role")
    if audit_payload.get("status") != EXPECTED_PASS_STATUS:
        raise AssertionError("independent post-seal audit status changed")
    published = audit_payload.get("published_bundle_integrity", {})
    if published.get("manifest", {}).get("sha256") != custody["bundle_manifest_sha256"]:
        raise AssertionError("independent post-seal audit binds another manifest")
    access = audit_payload.get("independent_audit_access_and_mutation", {})
    if access.get("target_values_used_for_fit_prediction_score_or_scientific_metric") != 0:
        raise AssertionError("independent custody audit used targets scientifically")
    if audit_payload.get("verdict", {}).get("this_receipt_is_not_training_or_execution_authority") is not True:
        raise AssertionError("custody audit unexpectedly grants execution authority")
    bundle_dir = manifest.parent
    if {path.name for path in bundle_dir.iterdir()} != {
        "manifest.json", "observer_fit.parquet", "observer_selection.parquet"
    }:
        raise AssertionError("bundle directory membership changed")
    # Deliberately do not stat, hash, footer-open, or decode either Parquet.
    return {
        "status": "PASS_POST_SEAL_METADATA_BINDING__INACTIVE_PENDING_LAUNCH_AUDIT_AND_AUTHORIZATION",
        "binding_sha256": sha256_file(BINDING_PATH),
        "seal_receipt_sha256": custody["seal_receipt_sha256"],
        "bundle_manifest_sha256": custody["bundle_manifest_sha256"],
        "sealed_reader_sha256": custody["sealed_reader_sha256"],
        "independent_post_seal_audit_sha256": EXPECTED_POST_SEAL_SHA256,
        "role_parquet_files_opened_or_hashed": 0,
        "record_batches_read": 0,
        "target_values_decoded": 0,
        "training_authorized": False,
    }


def validate_activation(
    *,
    independent_post_seal_receipt: Path,
    independent_launch_audit_receipt: Path,
    reader_authorization: Path,
    training_authorization: Path,
    trainer_source: Path,
    training_core_source: Path,
    converter_source: Path,
    feature_contract_source: Path,
    feature_provider_source: Path,
    implementation_manifest: Path,
    feature_manifest: Path,
) -> dict[str, Any]:
    """Validate future activation metadata without importing/using the reader."""

    binding = _binding()
    audit = json.loads(independent_post_seal_receipt.read_text(encoding="utf-8"))
    launch_audit = json.loads(independent_launch_audit_receipt.read_text(encoding="utf-8"))
    reader_auth = json.loads(reader_authorization.read_text(encoding="utf-8"))
    training_auth = json.loads(training_authorization.read_text(encoding="utf-8"))
    audit_sha = sha256_file(independent_post_seal_receipt)
    launch_audit_sha = sha256_file(independent_launch_audit_receipt)
    reader_auth_sha = sha256_file(reader_authorization)
    feature_sha = sha256_file(feature_manifest)
    if audit.get("status") != EXPECTED_PASS_STATUS:
        raise PermissionError("independent row-level post-seal PASS is absent")
    if audit_sha != EXPECTED_POST_SEAL_SHA256:
        raise PermissionError("independent post-seal receipt bytes changed")
    if audit.get("published_bundle_integrity", {}).get("manifest", {}).get("sha256") != binding["production_custody"]["bundle_manifest_sha256"]:
        raise PermissionError("independent audit is bound to another bundle")
    if audit.get("independent_audit_access_and_mutation", {}).get(
        "target_values_used_for_fit_prediction_score_or_scientific_metric"
    ) != 0:
        raise PermissionError("post-seal audit mixed integrity checking with fitting/scoring")
    if launch_audit.get("status") != EXPECTED_LAUNCH_AUDIT_STATUS:
        raise PermissionError("independent target-free launch audit is absent")
    if launch_audit.get("target_values_decoded") != 0 or launch_audit.get("fits") != 0:
        raise PermissionError("launch audit is not target-free")
    if reader_auth.get("schema") != "whest-phase2-sealed-role-target-reader-v1":
        raise PermissionError("wrong sealed-reader authorization schema")
    if reader_auth.get("authorized") is not True:
        raise PermissionError("sealed reader remains inactive")
    if reader_auth.get("purpose") != "observer_fit_and_selection_training":
        raise PermissionError("reader authorization purpose changed")
    if reader_auth.get("allowed_roles") != list(EXPECTED_ROLES):
        raise PermissionError("reader authorization roles changed")
    if reader_auth.get("allowed_columns") != ["mlp_id", "all_layer_means"]:
        raise PermissionError("reader authorization columns changed")
    if reader_auth.get("bundle_manifest_sha256") != binding["production_custody"]["bundle_manifest_sha256"]:
        raise PermissionError("reader authorization is bound to another bundle")
    if training_auth.get("schema") != TRAINING_AUTHORIZATION_SCHEMA or training_auth.get("authorized") is not True:
        raise PermissionError("one-use observer training is not authorized")
    exact = {
        "bundle_binding_sha256": sha256_file(BINDING_PATH),
        "production_seal_receipt_sha256": binding["production_custody"]["seal_receipt_sha256"],
        "bundle_manifest_sha256": binding["production_custody"]["bundle_manifest_sha256"],
        "fit_artifact_sha256": binding["role_artifacts"]["observer_fit"]["sha256"],
        "selection_artifact_sha256": binding["role_artifacts"]["observer_selection"]["sha256"],
        "sealed_reader_sha256": binding["production_custody"]["sealed_reader_sha256"],
        "independent_post_seal_receipt_sha256": audit_sha,
        "independent_launch_audit_receipt_sha256": launch_audit_sha,
        "reader_authorization_sha256": reader_auth_sha,
        "trainer_source_sha256": sha256_file(trainer_source),
        "training_core_source_sha256": sha256_file(training_core_source),
        "converter_source_sha256": sha256_file(converter_source),
        "feature_contract_source_sha256": sha256_file(feature_contract_source),
        "feature_provider_source_sha256": sha256_file(feature_provider_source),
        "implementation_manifest_sha256": sha256_file(implementation_manifest),
        "feature_manifest_sha256": feature_sha,
        "roles": list(EXPECTED_ROLES),
        "host_caps": [1536, 1792],
        "training_device": "cpu",
        "execution_attempts_authorized": 1,
        "retry_authorized": False,
        "mps_authorized": False,
        "audit_or_lockbox_access_authorized": False,
    }
    for key, value in exact.items():
        if training_auth.get(key) != value:
            raise PermissionError(f"training authorization binding mismatch: {key}")
    return {
        "status": "PASS_FUTURE_ACTIVATION_METADATA__NO_TARGET_OPEN",
        **exact,
        "target_values_decoded": 0,
    }


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2, sort_keys=True))
