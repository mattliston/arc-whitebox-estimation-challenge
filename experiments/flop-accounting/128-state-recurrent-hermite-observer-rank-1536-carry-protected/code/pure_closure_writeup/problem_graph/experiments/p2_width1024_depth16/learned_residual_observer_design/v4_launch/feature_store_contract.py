#!/usr/bin/env python3
"""Metadata-only contract for a future target-blind V4 feature store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "FEATURE_STORE_CONTRACT_V4.json"
EXPECTED_CONTRACT_SCHEMA = "phase2-observer-v4-target-blind-feature-store-contract-v1"
EXPECTED_MANIFEST_SCHEMA = "phase2-observer-v4-sealed-target-blind-feature-store-v1"
EXPECTED_ROLES = {
    "observer_fit": (688, "ba3d59f54ffacf4fe11d8ae6cd0f7ee1726486176217acc68002acb93be89de0"),
    "observer_selection": (88, "d1c719c608cadf5d4f7d852d23a9d935a23de0eae09d5dc1710257963aa1e755"),
}
EXPECTED_CAPS = (1536, 1792)
EXPECTED_FIELDS = (
    "mlp_id", "raw_locals", "edge_matrices", "host_rows", "sigma",
    "entering_cp_state_hashes", "host_latch",
)
EXPECTED_TARGET_BLIND_SOURCE_SHA256 = (
    "6828136668cece104521f1b34460bf9d06de427649852653352bb09d9e801bef"
)
EXPECTED_FULL_CACHE_AUDIT_SHA256 = (
    "43857f77f86f2ce95f058bda7d314d7c836b60aedab5a40d37b7208f291494fd"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("schema") != EXPECTED_CONTRACT_SCHEMA:
        raise AssertionError("feature-store contract schema changed")
    if contract.get("status") != "CONTRACT_ONLY__NO_FEATURE_STORE_GENERATED":
        raise AssertionError("target-free preflight cannot claim a generated store")
    if contract.get("current_artifacts") != 0:
        raise AssertionError("feature artifacts unexpectedly appeared in the contract")
    return {
        "status": "PASS_FEATURE_STORE_CONTRACT__HOLD_GENERATION_AND_TRAINING",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "feature_store_manifest_present": False,
        "feature_artifacts_opened_or_hashed": 0,
        "role_parquet_files_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "fits": 0,
    }


def validate_future_manifest(
    *, manifest_path: Path, provider_source: Path
) -> dict[str, Any]:
    """Validate only future manifest/source bytes; artifact reads stay post-claim."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        raise PermissionError("wrong target-blind feature manifest schema")
    if manifest.get("status") != "SEALED_TARGET_BLIND_FEATURE_STORE":
        raise PermissionError("target-blind feature store is not sealed")
    if manifest.get("provider_source_sha256") != sha256_file(provider_source):
        raise PermissionError("feature manifest binds another provider source")
    if manifest.get("host_caps") != list(EXPECTED_CAPS):
        raise PermissionError("feature manifest host caps changed")
    if manifest.get("fields") != list(EXPECTED_FIELDS):
        raise PermissionError("feature manifest fields changed")
    roles = manifest.get("roles", {})
    if set(roles) != set(EXPECTED_ROLES):
        raise PermissionError("feature manifest roles changed")
    for role, (rows, ids_sha) in EXPECTED_ROLES.items():
        if roles[role].get("rows") != rows or roles[role].get("ordered_mlp_ids_sha256") != ids_sha:
            raise PermissionError(f"feature manifest role binding changed: {role}")
    if manifest.get("target_columns_decoded") != 0:
        raise PermissionError("feature generation decoded a target column")
    if manifest.get("raw_full_dataset_target_projection") != "forbidden":
        raise PermissionError("feature store permits a raw target projection")
    if manifest.get("target_blind_full_source_sha256") != EXPECTED_TARGET_BLIND_SOURCE_SHA256:
        raise PermissionError("feature store used another full-weight reader")
    if manifest.get("full_cache_audit_sha256") != EXPECTED_FULL_CACHE_AUDIT_SHA256:
        raise PermissionError("feature store is bound to another full cache")
    if manifest.get("independent_feature_store_audit_status") != "PASS":
        raise PermissionError("feature store lacks independent source/artifact PASS")
    return {
        "status": "PASS_FUTURE_TARGET_BLIND_FEATURE_MANIFEST_METADATA",
        "feature_manifest_sha256": sha256_file(manifest_path),
        "provider_source_sha256": sha256_file(provider_source),
        "feature_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
    }


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2, sort_keys=True))
