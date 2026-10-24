#!/usr/bin/env python3
"""PyArrow 11/25-compatible trusted-custodian raw target projection v2.

Importing this module and running its preflight decode nothing.  The iterator
requires an explicit authorization JSON bound to the immutable dataset and a
frozen split-manifest and physical-addressability audit.  When separately
authorized later, a trusted custodian may physically project only ``mlp_id``
and ``all_layer_means`` in order to create role-specific sealed artifacts.
Weights, redundant final means, and all auxiliary columns are deny-listed.

This is not a training iterator.  The audited upstream files have 63 mixed
row groups, no target page indexes, zero role-pure groups, and zero
single-network groups.  Therefore authorization covers the complete full-split
``all_layer_means`` column, and a completed traversal yields all 1,000 rows.
Ordinary training must consume separately sealed role artifacts produced by
this custodian boundary; it must never import this raw iterator.
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

from full_data_gate import AUTHORITY, FULL, HERE, _attest_shard_pointer, load_full_manifest, pinned_snapshot_path


AUTHORIZATION_SCHEMA = "whest-phase2-custodian-target-access-v1"
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
ALL_COLUMNS = frozenset(AUTHORITY["features"])
DEFAULT_SPLIT_MANIFEST = HERE / "full_network_splits.json"
ADDRESSABILITY_AUDIT = HERE / "FULL_SPLIT_ROLE_ADDRESSABILITY_AUDIT.json"


@dataclass(frozen=True)
class CustodianTargetRow:
    source_ordinal: int
    mlp_id: int
    primary_split: str
    observer_role: str
    all_layer_means: np.ndarray
    decoded_columns: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict[str, object]:
    if frozenset(TARGET_COLUMNS) & DENIED_COLUMNS:
        raise AssertionError("target projection intersects deny-list")
    if frozenset(TARGET_COLUMNS) | DENIED_COLUMNS != ALL_COLUMNS:
        raise AssertionError("target projection and deny-list do not partition schema")
    load_full_manifest()
    addressability = json.loads(ADDRESSABILITY_AUDIT.read_text(encoding="utf-8"))
    if addressability.get("status") != (
        "EXTERNAL_CONSTRAINT__RAW_PARQUET_CANNOT_PRESERVE_ROLE_TARGET_SEALS"
    ):
        raise AssertionError("raw role-addressability constraint is not frozen")
    if addressability.get("raw_role_projection_physically_safe") is not False:
        raise AssertionError("raw role projection unexpectedly marked safe")
    if addressability.get("split_manifest_sha256") != _sha256_file(
        DEFAULT_SPLIT_MANIFEST
    ):
        raise AssertionError("role-addressability receipt is bound to another roster")
    if addressability.get("full_shard_manifest_sha256") != FULL["manifest_sha256"]:
        raise AssertionError("role-addressability receipt is bound to another full split")
    return {
        "status": "PASS_INERT_TRUSTED_CUSTODIAN_TARGET_PREFLIGHT",
        "projected_columns_after_authorization": list(TARGET_COLUMNS),
        "denied_columns": sorted(DENIED_COLUMNS),
        "authorization_scope": "complete full-split all_layer_means physical column",
        "ordinary_training_raw_access": "FORBIDDEN",
        "required_output": "separately sealed role-specific target artifacts",
        "role_addressability_audit_sha256": _sha256_file(ADDRESSABILITY_AUDIT),
        "authorization_required": True,
        "target_values_decoded": 0,
    }


def _load_authorization(path: Path, split_manifest: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != AUTHORIZATION_SCHEMA:
        raise PermissionError("wrong target authorization schema")
    if payload.get("authorized") is not True:
        raise PermissionError("target access is not explicitly authorized")
    split_hash = _sha256_file(split_manifest)
    addressability = json.loads(ADDRESSABILITY_AUDIT.read_text(encoding="utf-8"))
    if split_hash != addressability.get("split_manifest_sha256"):
        raise PermissionError("custodian roster differs from addressability audit")
    expected = {
        "repository": AUTHORITY["repository"],
        "revision": AUTHORITY["revision"],
        "commit": AUTHORITY["pinned_commit_sha"],
        "full_shard_manifest_sha256": FULL["manifest_sha256"],
        "physical_target_scope": "complete_full_all_layer_means_column",
        "execution_boundary": "trusted_data_custodian",
        "purpose": "materialize_separately_sealed_role_target_artifacts",
        "output_contract": "role_specific_artifacts_with_audit_and_lockbox_withheld",
        "allowed_columns": list(TARGET_COLUMNS),
        "split_manifest_sha256": split_hash,
        "role_addressability_audit_sha256": _sha256_file(ADDRESSABILITY_AUDIT),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"authorization binding mismatch: {key}")
    for required_text in ("authorized_by", "authorized_at_utc", "nonce"):
        if not isinstance(payload.get(required_text), str) or not payload[required_text]:
            raise PermissionError(f"authorization lacks non-empty {required_text}")
    return payload


def _load_complete_membership(split_manifest: Path) -> dict[int, tuple[str, str]]:
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != "whest-phase2-network-split-v1":
        raise AssertionError("wrong network split schema")
    membership: dict[int, tuple[str, str]] = {}
    for split_name, collection in payload["splits"].items():
        for row in collection:
            mlp_id = int(row["mlp_id"])
            if mlp_id in membership:
                raise AssertionError(f"duplicate split mlp_id: {mlp_id}")
            membership[mlp_id] = (split_name, str(row["observer_role"]))
    if len(membership) != int(FULL["n_mlps"]):
        raise AssertionError("custodian membership does not cover full split")
    return membership


def _all_layer_view(batch: pa.RecordBatch) -> np.ndarray:
    storage = batch.column(batch.schema.get_field_index("all_layer_means"))
    storage = getattr(storage, "storage", storage)
    for _ in range(2):
        storage = storage.flatten()
    values = storage.to_numpy(zero_copy_only=True).reshape(16, 1024)
    if values.dtype != np.float32:
        raise AssertionError("all_layer_means is not float32")
    if not np.isfinite(values).all():
        raise AssertionError("non-finite authorized target")
    values.setflags(write=False)
    return values


def iter_trusted_custodian_all_layer_means(
    *,
    authorization_path: Path,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
) -> Iterator[CustodianTargetRow]:
    """Yield all 1,000 rows inside the authorized trusted boundary only."""

    preflight()
    _load_authorization(authorization_path, split_manifest_path)
    membership = _load_complete_membership(split_manifest_path)
    seen: set[int] = set()
    source_ordinal = 0
    snapshot = pinned_snapshot_path()
    for expected in load_full_manifest():
        resolved = _attest_shard_pointer(snapshot / str(expected["path"]), expected)
        batches = pq.ParquetFile(resolved).iter_batches(
            batch_size=1,
            columns=list(TARGET_COLUMNS),
            use_threads=False,
        )
        for batch in batches:
            if batch.num_rows != 1 or tuple(batch.schema.names) != TARGET_COLUMNS:
                raise AssertionError("authorized target projection violated contract")
            if set(batch.schema.names) & DENIED_COLUMNS:
                raise AssertionError("deny-listed column reached target batch")
            mlp_id = int(batch.column(batch.schema.get_field_index("mlp_id"))[0].as_py())
            if mlp_id in seen:
                raise AssertionError(f"duplicate custodian target mlp_id: {mlp_id}")
            if mlp_id not in membership:
                raise AssertionError(f"custodian target ID absent from roster: {mlp_id}")
            seen.add(mlp_id)
            primary_split, observer_role = membership[mlp_id]
            yield CustodianTargetRow(
                source_ordinal=source_ordinal,
                mlp_id=mlp_id,
                primary_split=primary_split,
                observer_role=observer_role,
                all_layer_means=_all_layer_view(batch),
                decoded_columns=TARGET_COLUMNS,
            )
            source_ordinal += 1
    if source_ordinal != int(FULL["n_mlps"]):
        raise AssertionError("authorized target iterator source count mismatch")
    if seen != set(membership):
        raise AssertionError("custodian target iterator missed roster IDs")



SYNTHETIC_COMPAT_ENV = "WHEST_PHASE2_SYNTHETIC_COMPAT"
SYNTHETIC_COMPAT_TOKEN = "isolated-synthetic-parquet-v1"
SYNTHETIC_MAX_BYTES = 64 << 20
SYNTHETIC_MAX_ROWS = 32


def _assert_synthetic_root(root: Path, paths: tuple[Path, ...]) -> Path:
    if os.environ.get(SYNTHETIC_COMPAT_ENV) != SYNTHETIC_COMPAT_TOKEN:
        raise PermissionError("synthetic compatibility boundary is not asserted")
    resolved_root = root.resolve(strict=True)
    if resolved_root == HERE or HERE in resolved_root.parents:
        raise PermissionError("synthetic compatibility root overlaps data gate")
    if not resolved_root.name.startswith("whest-p2-synthetic-compat-"):
        raise PermissionError("synthetic compatibility root lacks fixed prefix")
    if len(paths) != 1:
        raise PermissionError("synthetic compatibility accepts exactly one source file")
    for path in paths:
        resolved = path.resolve(strict=True)
        if resolved_root not in resolved.parents:
            raise PermissionError("synthetic Parquet escapes compatibility root")
        stat = resolved.stat()
        if not resolved.is_file() or path.is_symlink() or stat.st_nlink != 1:
            raise PermissionError("synthetic Parquet must be a regular non-symlink file")
        if stat.st_size <= 0 or stat.st_size > SYNTHETIC_MAX_BYTES:
            raise PermissionError("synthetic Parquet exceeds compatibility size cap")
    return resolved_root


def iter_synthetic_test_all_layer_means(
    *,
    parquet_paths: tuple[Path, ...],
    membership: dict[int, tuple[str, str]],
    synthetic_root: Path,
) -> Iterator[CustodianTargetRow]:
    """Compatibility-only projection over caller-created synthetic Parquet.

    This cannot address the production cache: it requires a fixed environment
    token and a temporary root outside the data-gate tree.
    """

    _assert_synthetic_root(synthetic_root, parquet_paths)
    if not membership or len(membership) > SYNTHETIC_MAX_ROWS:
        raise AssertionError("synthetic membership is empty or exceeds test cap")
    seen: set[int] = set()
    ordinal = 0
    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        if (
            int(parquet.metadata.num_rows) != len(membership)
            or int(parquet.metadata.num_row_groups) != 1
            or parquet.schema_arrow.names != list(AUTHORITY["features"])
        ):
            raise AssertionError("synthetic source shape/schema envelope changed")
        batches = parquet.iter_batches(
            batch_size=1,
            columns=list(TARGET_COLUMNS),
            use_threads=False,
        )
        for batch in batches:
            names = tuple(batch.schema.names)
            if batch.num_rows != 1 or names != TARGET_COLUMNS:
                raise AssertionError("synthetic raw projection violated contract")
            if set(batch.schema.names) & DENIED_COLUMNS:
                raise AssertionError("deny-listed column reached synthetic raw batch")
            mlp_id = int(batch.column(batch.schema.get_field_index("mlp_id"))[0].as_py())
            if mlp_id in seen or mlp_id not in membership:
                raise AssertionError("synthetic target ID is duplicate or unregistered")
            seen.add(mlp_id)
            primary_split, observer_role = membership[mlp_id]
            yield CustodianTargetRow(
                source_ordinal=ordinal,
                mlp_id=mlp_id,
                primary_split=primary_split,
                observer_role=observer_role,
                all_layer_means=_all_layer_view(batch),
                decoded_columns=TARGET_COLUMNS,
            )
            ordinal += 1
    if ordinal != len(membership) or seen != set(membership):
        raise AssertionError("synthetic raw iterator did not exhaust membership")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Static and manifest checks only; this is the sole CLI mode.",
    )
    args = parser.parse_args()
    if not args.preflight:
        raise SystemExit("refusing target access: use --preflight (decodes zero targets)")
    print(json.dumps(preflight(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
