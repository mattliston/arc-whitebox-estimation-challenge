#!/usr/bin/env python3
"""PyArrow 11/25-compatible reader for sealed Phase 2 role targets v2.

This ordinary-training interface never imports or resolves the raw 63-shard
full dataset.  It accepts only a custodian-produced bundle that physically
contains the observer-fit and observer-selection artifacts, verifies all
source and file hashes before decoding, and projects exactly ``mlp_id`` and
``all_layer_means``.  Audit, lockbox, diagnostic, weight, and auxiliary data
are denied.

No role bundle exists yet.  Import and ``preflight`` are target-free; the
iterator remains inert until both a sealed bundle and an explicit active
authorization are supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
AUTHORITY_PATH = HERE / "authority.json"
EXTENSION_PATH = HERE / "FULL_SPLIT_AUTHORITY_EXTENSION.json"
SPLIT_MANIFEST = HERE / "full_network_splits.json"
ADDRESSABILITY_AUDIT = HERE / "FULL_SPLIT_ROLE_ADDRESSABILITY_AUDIT.json"
DEFAULT_BUNDLE_MANIFEST = HERE / "sealed_role_target_bundle" / "manifest.json"
AUTHORIZATION_SCHEMA = "whest-phase2-sealed-role-target-reader-v1"
BUNDLE_SCHEMA = "whest-phase2-training-role-target-bundle-v1"
TARGET_COLUMNS = ("mlp_id", "all_layer_means")
DENIED_COLUMNS = frozenset(
    {
        "mlp_name",
        "mlp_seed",
        "weights",
        "final_means",
        "avg_variance",
        "sampling_budget_breakdown",
    }
)
TRAINING_ROLES = frozenset(("observer_fit", "observer_selection"))
WITHHELD_ROLES = frozenset(
    ("observer_audit", "observer_lockbox", "retired_teacher_diagnostics")
)


@dataclass(frozen=True)
class SealedRoleTargetRow:
    role: str
    role_rank: int
    mlp_id: int
    all_layer_means: np.ndarray
    decoded_columns: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_static_context() -> tuple[dict[str, object], dict[str, object]]:
    authority_raw = AUTHORITY_PATH.read_bytes()
    authority = json.loads(authority_raw)
    extension = json.loads(EXTENSION_PATH.read_text(encoding="utf-8"))
    if hashlib.sha256(authority_raw).hexdigest() != extension["parent_authority"][
        "sha256"
    ]:
        raise AssertionError("full extension is not bound to the frozen authority")
    split_hash = _sha256_file(SPLIT_MANIFEST)
    addressability = json.loads(ADDRESSABILITY_AUDIT.read_text(encoding="utf-8"))
    if addressability.get("status") != (
        "EXTERNAL_CONSTRAINT__RAW_PARQUET_CANNOT_PRESERVE_ROLE_TARGET_SEALS"
    ):
        raise AssertionError("raw target-seal constraint is not frozen")
    if addressability.get("raw_role_projection_physically_safe") is not False:
        raise AssertionError("raw role target projection unexpectedly marked safe")
    if addressability.get("split_manifest_sha256") != split_hash:
        raise AssertionError("addressability audit is bound to another roster")
    return authority, extension


def preflight() -> dict[str, object]:
    """Validate the inert design and frozen bindings without opening targets."""

    authority, extension = _load_static_context()
    if frozenset(TARGET_COLUMNS) & DENIED_COLUMNS:
        raise AssertionError("sealed target projection intersects deny-list")
    if frozenset(TARGET_COLUMNS) | DENIED_COLUMNS != frozenset(
        authority["features"]
    ):
        raise AssertionError("sealed target projection and deny-list do not partition schema")
    return {
        "status": "PASS_INERT_SEALED_ROLE_TARGET_READER_PREFLIGHT",
        "repository": authority["repository"],
        "commit": authority["pinned_commit_sha"],
        "full_shard_manifest_sha256": extension["full"]["manifest_sha256"],
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
        "role_addressability_audit_sha256": _sha256_file(ADDRESSABILITY_AUDIT),
        "allowed_roles_after_authorization": sorted(TRAINING_ROLES),
        "withheld_roles": sorted(WITHHELD_ROLES),
        "projected_columns_after_authorization": list(TARGET_COLUMNS),
        "denied_columns": sorted(DENIED_COLUMNS),
        "raw_full_dataset_access": "ABSENT_AND_FORBIDDEN",
        "authorization_required": True,
        "target_values_decoded": 0,
    }


def _expected_role_ids() -> dict[str, list[int]]:
    roster = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if roster.get("schema") != "whest-phase2-network-split-v1":
        raise AssertionError("wrong frozen roster schema")
    result: dict[str, list[int]] = {}
    for role in TRAINING_ROLES:
        rows = sorted(
            roster["observer_roles"][role],
            key=lambda row: int(row["global_hash_rank"]),
        )
        result[role] = [int(row["mlp_id"]) for row in rows]
    if set(result["observer_fit"]) & set(result["observer_selection"]):
        raise AssertionError("training roles overlap in frozen roster")
    return result


def _ordered_ids_sha256(ids: list[int]) -> str:
    encoded = "".join(f"{mlp_id}\n" for mlp_id in ids).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_authorization(path: Path, role: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != AUTHORIZATION_SCHEMA:
        raise PermissionError("wrong sealed-role target authorization schema")
    if payload.get("authorized") is not True:
        raise PermissionError("sealed-role target access is not explicitly authorized")
    if role not in TRAINING_ROLES:
        raise PermissionError(f"role is withheld from ordinary training: {role}")
    allowed_roles = payload.get("allowed_roles")
    if (
        not isinstance(allowed_roles, list)
        or len(allowed_roles) != len(set(allowed_roles))
        or not set(allowed_roles) <= TRAINING_ROLES
        or role not in allowed_roles
    ):
        raise PermissionError("authorization role scope is invalid")
    authority, extension = _load_static_context()
    expected = {
        "purpose": "observer_fit_and_selection_training",
        "repository": authority["repository"],
        "commit": authority["pinned_commit_sha"],
        "full_shard_manifest_sha256": extension["full"]["manifest_sha256"],
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
        "role_addressability_audit_sha256": _sha256_file(ADDRESSABILITY_AUDIT),
        "allowed_columns": list(TARGET_COLUMNS),
        "raw_full_dataset_access": "forbidden",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"authorization binding mismatch: {key}")
    for required_text in (
        "authorized_by",
        "authorized_at_utc",
        "nonce",
        "bundle_manifest_sha256",
    ):
        value = payload.get(required_text)
        if not isinstance(value, str) or not value:
            raise PermissionError(f"authorization lacks non-empty {required_text}")
    return payload


def _load_bundle(
    manifest_path: Path, authorization: dict[str, object]
) -> tuple[Path, dict[str, object]]:
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise AssertionError("bundle manifest must be a regular non-symlink file")
    if _sha256_file(manifest_path) != authorization["bundle_manifest_sha256"]:
        raise PermissionError("sealed-role bundle hash differs from authorization")
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise AssertionError("wrong sealed-role bundle schema")
    expected = {
        "status": "SEALED_BY_TRUSTED_DATA_CUSTODIAN",
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
        "role_addressability_audit_sha256": _sha256_file(ADDRESSABILITY_AUDIT),
        "target_columns": list(TARGET_COLUMNS),
        "raw_full_dataset_included": False,
        "withheld_roles_included": False,
    }
    for key, value in expected.items():
        if bundle.get(key) != value:
            raise AssertionError(f"sealed-role bundle binding mismatch: {key}")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != TRAINING_ROLES:
        raise AssertionError("bundle must contain exactly the two training roles")

    bundle_dir = manifest_path.resolve().parent
    expected_entries = {manifest_path.resolve()}
    for role, artifact in artifacts.items():
        relative = Path(str(artifact["path"]))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise AssertionError(f"unsafe artifact path for {role}")
        artifact_path = bundle_dir / relative
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise AssertionError(f"artifact must be a regular non-symlink file: {role}")
        expected_entries.add(artifact_path.resolve())
    actual_entries = {path.resolve() for path in bundle_dir.iterdir()}
    if actual_entries != expected_entries:
        raise AssertionError("bundle directory contains unmanifested or missing files")
    return bundle_dir, bundle


def _all_layer_view(batch: pa.RecordBatch) -> np.ndarray:
    storage = batch.column(batch.schema.get_field_index("all_layer_means"))
    storage = getattr(storage, "storage", storage)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    values = storage.to_numpy(zero_copy_only=True).reshape(16, 1024)
    if values.dtype != np.float32 or not values.flags.c_contiguous:
        raise AssertionError("sealed all_layer_means has unexpected dtype/layout")
    if not np.isfinite(values).all():
        raise AssertionError("non-finite sealed target")
    values.setflags(write=False)
    return values


def _validate_artifact_schema(parquet: pq.ParquetFile) -> None:
    schema = parquet.schema_arrow
    if schema.names != list(TARGET_COLUMNS):
        raise AssertionError("sealed role artifact contains unexpected columns")
    if schema.field("mlp_id").type != pa.int32():
        raise AssertionError("sealed role mlp_id is not int32")
    target = schema.field("all_layer_means")
    metadata = target.metadata or {}
    extension_name = metadata.get(b"ARROW:extension:name", b"").decode("utf-8")
    extension_shape = metadata.get(b"ARROW:extension:metadata")
    if not extension_name.endswith("Array2DExtensionType"):
        raise AssertionError("sealed all_layer_means lacks Array2D extension metadata")
    if extension_shape is None or json.loads(extension_shape) != [
        [16, 1024],
        "float32",
    ]:
        raise AssertionError("sealed all_layer_means geometry/dtype metadata changed")


def iter_authorized_sealed_role_targets(
    *,
    role: str,
    authorization_path: Path,
    bundle_manifest_path: Path = DEFAULT_BUNDLE_MANIFEST,
) -> Iterator[SealedRoleTargetRow]:
    """Yield one explicitly authorized training role from its sealed artifact."""

    preflight()
    authorization = _load_authorization(authorization_path, role)
    bundle_dir, bundle = _load_bundle(bundle_manifest_path, authorization)
    expected_ids = _expected_role_ids()[role]
    artifact = bundle["artifacts"][role]
    if int(artifact.get("rows", -1)) != len(expected_ids):
        raise AssertionError("sealed role row count differs from roster")
    if artifact.get("ordered_mlp_ids_sha256") != _ordered_ids_sha256(expected_ids):
        raise AssertionError("sealed role ID commitment differs from roster")
    artifact_path = bundle_dir / str(artifact["path"])
    if _sha256_file(artifact_path) != artifact.get("sha256"):
        raise AssertionError("sealed role artifact SHA-256 mismatch")

    parquet = pq.ParquetFile(artifact_path)
    _validate_artifact_schema(parquet)
    if int(parquet.metadata.num_rows) != len(expected_ids):
        raise AssertionError("sealed role Parquet row count differs from roster")
    batches = parquet.iter_batches(
        batch_size=1,
        columns=list(TARGET_COLUMNS),
        use_threads=False,
    )
    for role_rank, (batch, expected_id) in enumerate(zip(batches, expected_ids)):
        if batch.num_rows != 1 or tuple(batch.schema.names) != TARGET_COLUMNS:
            raise AssertionError("sealed role physical projection violated contract")
        if set(batch.schema.names) & DENIED_COLUMNS:
            raise AssertionError("deny-listed column reached sealed target batch")
        mlp_id = int(batch.column(batch.schema.get_field_index("mlp_id"))[0].as_py())
        if mlp_id != expected_id:
            raise AssertionError("sealed role row order/identity differs from roster")
        yield SealedRoleTargetRow(
            role=role,
            role_rank=role_rank,
            mlp_id=mlp_id,
            all_layer_means=_all_layer_view(batch),
            decoded_columns=TARGET_COLUMNS,
        )



SYNTHETIC_COMPAT_ENV = "WHEST_PHASE2_SYNTHETIC_COMPAT"
SYNTHETIC_COMPAT_TOKEN = "isolated-synthetic-parquet-v1"
SYNTHETIC_MAX_ARTIFACT_BYTES = 64 << 20
SYNTHETIC_MAX_ROLE_ROWS = 16


def _assert_synthetic_bundle_root(root: Path, manifest_path: Path) -> Path:
    if os.environ.get(SYNTHETIC_COMPAT_ENV) != SYNTHETIC_COMPAT_TOKEN:
        raise PermissionError("synthetic compatibility boundary is not asserted")
    resolved_root = root.resolve(strict=True)
    resolved_manifest = manifest_path.resolve(strict=True)
    if resolved_root == HERE or HERE in resolved_root.parents:
        raise PermissionError("synthetic compatibility root overlaps data gate")
    if not resolved_root.name.startswith("whest-p2-synthetic-compat-"):
        raise PermissionError("synthetic compatibility root lacks fixed prefix")
    if resolved_root not in resolved_manifest.parents:
        raise PermissionError("synthetic bundle manifest escapes compatibility root")
    manifest_stat = resolved_manifest.stat()
    if (
        not resolved_manifest.is_file()
        or manifest_path.is_symlink()
        or manifest_stat.st_nlink != 1
        or manifest_stat.st_size <= 0
        or manifest_stat.st_size > (1 << 20)
    ):
        raise PermissionError("synthetic bundle manifest is not a regular file")
    return resolved_root


def iter_synthetic_test_role_targets(
    *,
    role: str,
    expected_ids: list[int],
    bundle_manifest_path: Path,
    synthetic_root: Path,
) -> Iterator[SealedRoleTargetRow]:
    """Read one compatibility-only role artifact outside production paths."""

    resolved_root = _assert_synthetic_bundle_root(
        synthetic_root, bundle_manifest_path
    )
    if role not in TRAINING_ROLES:
        raise PermissionError(f"synthetic reader denies non-training role: {role}")
    if (
        not expected_ids
        or len(expected_ids) > SYNTHETIC_MAX_ROLE_ROWS
        or len(expected_ids) != len(set(expected_ids))
    ):
        raise AssertionError("synthetic expected IDs are empty or duplicated")
    bundle = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    if (
        bundle.get("schema") != BUNDLE_SCHEMA
        or bundle.get("status") != "SYNTHETIC_COMPATIBILITY_ONLY_NOT_A_RELEASE"
        or bundle.get("synthetic_only") is not True
        or bundle.get("target_columns") != list(TARGET_COLUMNS)
        or bundle.get("raw_full_dataset_included") is not False
        or bundle.get("withheld_roles_included") is not False
    ):
        raise AssertionError("synthetic bundle contract changed")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != TRAINING_ROLES:
        raise AssertionError("synthetic bundle must contain exactly fit and selection")
    bundle_dir = bundle_manifest_path.resolve().parent
    expected_entries = {bundle_manifest_path.resolve()}
    for artifact_role, artifact in artifacts.items():
        relative = Path(str(artifact["path"]))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise AssertionError("unsafe synthetic artifact path")
        artifact_path = bundle_dir / relative
        artifact_stat = artifact_path.stat()
        if (
            resolved_root not in artifact_path.resolve(strict=True).parents
            or not artifact_path.is_file()
            or artifact_path.is_symlink()
            or artifact_stat.st_nlink != 1
            or artifact_stat.st_size <= 0
            or artifact_stat.st_size > SYNTHETIC_MAX_ARTIFACT_BYTES
        ):
            raise AssertionError("synthetic artifact escapes root or is not regular")
        expected_entries.add(artifact_path.resolve())
        if artifact_role in WITHHELD_ROLES:
            raise AssertionError("withheld role entered synthetic bundle")
    if {path.resolve() for path in bundle_dir.iterdir()} != expected_entries:
        raise AssertionError("synthetic bundle contains an unexpected file")

    artifact = artifacts[role]
    artifact_path = bundle_dir / str(artifact["path"])
    if (
        int(artifact.get("rows", -1)) != len(expected_ids)
        or artifact.get("ordered_mlp_ids_sha256")
        != _ordered_ids_sha256(expected_ids)
        or _sha256_file(artifact_path) != artifact.get("sha256")
    ):
        raise AssertionError("synthetic artifact hash/ID/count commitment changed")
    parquet = pq.ParquetFile(artifact_path)
    _validate_artifact_schema(parquet)
    if int(parquet.metadata.num_rows) != len(expected_ids):
        raise AssertionError("synthetic artifact Parquet row count changed")
    batches = parquet.iter_batches(
        batch_size=1,
        columns=list(TARGET_COLUMNS),
        use_threads=False,
    )
    yielded = 0
    for role_rank, (batch, expected_id) in enumerate(zip(batches, expected_ids)):
        names = tuple(batch.schema.names)
        if batch.num_rows != 1 or names != TARGET_COLUMNS:
            raise AssertionError("synthetic sealed projection violated contract")
        if set(batch.schema.names) & DENIED_COLUMNS:
            raise AssertionError("deny-listed column reached synthetic sealed batch")
        mlp_id = int(batch.column(batch.schema.get_field_index("mlp_id"))[0].as_py())
        if mlp_id != expected_id:
            raise AssertionError("synthetic sealed row identity/order changed")
        yielded += 1
        yield SealedRoleTargetRow(
            role=role,
            role_rank=role_rank,
            mlp_id=mlp_id,
            all_layer_means=_all_layer_view(batch),
            decoded_columns=TARGET_COLUMNS,
        )
    if yielded != len(expected_ids):
        raise AssertionError("synthetic sealed reader did not exhaust expected IDs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not args.preflight:
        raise SystemExit("refusing target access: use --preflight (decodes zero targets)")
    print(json.dumps(preflight(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
