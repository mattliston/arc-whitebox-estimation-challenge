#!/usr/bin/env python3
"""One-use trusted-custodian sealer for Phase 2 training-role targets.

THIS FILE IS FROZEN DESIGN, NOT AN AUTHORIZATION TO RUN IT.

An authorized future execution consumes a fixed O_EXCL claim before creating
or advancing the raw target iterator.  It then traverses all 1,000 raw target
rows inside an isolated custodian boundary because the upstream row groups are
physically role-mixed.  Only observer-fit and observer-selection values are
copied and written.  Audit, lockbox, and retired-diagnostic values are counted
as transient physical decodes and are never persisted.

The claim is never deleted.  Any crash after claim creation consumes the one
attempt permanently; there is deliberately no cleanup or retry path.  A
successful run atomically publishes exactly two role Parquets and one manifest
compatible with ``sealed_role_targets.py``.  This program performs no fitting,
prediction, scoring, or feature generation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from authorized_full_targets import (
    TARGET_COLUMNS as RAW_TARGET_COLUMNS,
    iter_trusted_custodian_all_layer_means,
)


HERE = Path(__file__).resolve().parent
AUTHORITY_PATH = HERE / "authority.json"
EXTENSION_PATH = HERE / "FULL_SPLIT_AUTHORITY_EXTENSION.json"
ROSTER_PATH = HERE / "full_network_splits.json"
ADDRESSABILITY_PATH = HERE / "FULL_SPLIT_ROLE_ADDRESSABILITY_AUDIT.json"
CACHE_AUDIT_PATH = HERE / "FULL_SPLIT_CACHE_AUDIT.json"
RAW_READER_PATH = HERE / "authorized_full_targets.py"
TRAINING_READER_PATH = HERE / "sealed_role_targets.py"

# These execution paths are intentionally fixed and have no CLI override.
FIXED_CLAIM_PATH = HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE.CLAIM"
FIXED_STAGING_PATH = HERE / "sealed_role_target_bundle.staging"
FIXED_OUTPUT_PATH = HERE / "sealed_role_target_bundle"

EXPECTED_AUTHORITY_SHA256 = (
    "b33aea75ee83be5cc7c2778aefd4cc35d731dcd702d8b258269fc2c895ef4ec5"
)
EXPECTED_EXTENSION_SHA256 = (
    "d61677658462f22949d8242f7bb92b24341502678f499d793184002fd3b6027a"
)
EXPECTED_ROSTER_SHA256 = (
    "ba325d5c9ee4382ebf856b2e8d15330852fba5ac5627665fcd81dd9a5ef461d1"
)
EXPECTED_ADDRESSABILITY_SHA256 = (
    "b73e6ccf24434d27cc1bc34cc5ebcd87b9b38b504c3412bf9afad571ecb4b49c"
)
EXPECTED_CACHE_AUDIT_SHA256 = (
    "43857f77f86f2ce95f058bda7d314d7c836b60aedab5a40d37b7208f291494fd"
)
EXPECTED_RAW_READER_SHA256 = (
    "dfea9dc4f6e1096e9bd078f5594bc94a4dc5f85e550077f6e954a4f56512dd96"
)
EXPECTED_TRAINING_READER_SHA256 = (
    "194d41ea7afbeb4a2171c0ff55fe1a4321c0c817a6c23466cc21fcb23b850e91"
)
EXPECTED_COMMIT = "ce926b7d1ced65faa668c01b94a43f8c4f43783d"
EXPECTED_FULL_MANIFEST_SHA256 = (
    "51357aceca60f0a0365b26d7b25e136d1e4e713d44451a5f7c079e3ba9ea2794"
)
ASSIGNMENT_DOMAIN = b"p2-teacher-v1|v2-phase2|"
ASSIGNMENT_TEST_VECTOR_SEED_ZERO = (
    "22ba5941825921f557dc0abb7efc979b168e63300302421f7a2d9eaabffb92ac"
)
ROLE_SEGMENTS = (
    ("retired_teacher_diagnostics", 0, 12),
    ("observer_fit", 12, 700),
    ("retired_teacher_diagnostics", 700, 712),
    ("observer_selection", 712, 800),
    ("observer_audit", 800, 900),
    ("observer_lockbox", 900, 1000),
)
EXPECTED_ROLE_COUNTS = {
    "observer_fit": 688,
    "observer_selection": 88,
    "observer_audit": 100,
    "observer_lockbox": 100,
    "retired_teacher_diagnostics": 24,
}
PERSISTED_ROLES = frozenset(("observer_fit", "observer_selection"))
WITHHELD_ROLES = frozenset(
    ("observer_audit", "observer_lockbox", "retired_teacher_diagnostics")
)
TARGET_SHAPE = (16, 1024)
TARGET_DTYPE = np.dtype("float32")
BOUNDARY_ENVIRONMENT_VARIABLE = "WHEST_PHASE2_CUSTODIAN_BOUNDARY"
EXPECTED_BOUNDARY_VALUE = "isolated-target-sealer-v1"
AUTHORIZATION_SCHEMA = "whest-phase2-custodian-target-access-v1"
BUNDLE_SCHEMA = "whest-phase2-training-role-target-bundle-v1"
ARRAY2D_FIELD_METADATA = {
    b"ARROW:extension:name": b"datasets.features.features.Array2DExtensionType",
    b"ARROW:extension:metadata": b'[[16, 1024], "float32"]',
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_fsynced(path: Path, payload: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating durable exclusive file")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise AssertionError(f"expected regular non-symlink file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assignment_hash(mlp_seed: int) -> str:
    return hashlib.sha256(
        ASSIGNMENT_DOMAIN + str(mlp_seed).encode("ascii")
    ).hexdigest()


def _role_for_rank(rank: int) -> str:
    matches = [role for role, begin, end in ROLE_SEGMENTS if begin <= rank < end]
    if len(matches) != 1:
        raise AssertionError(f"role segments do not uniquely cover rank {rank}")
    return matches[0]


def _ordered_ids_sha256(ids: list[int]) -> str:
    encoded = "".join(f"{mlp_id}\n" for mlp_id in ids).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_roster() -> tuple[dict[int, str], dict[str, list[int]]]:
    if _sha256_file(ROSTER_PATH) != EXPECTED_ROSTER_SHA256:
        raise AssertionError("frozen roster SHA-256 changed")
    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    if roster.get("schema") != "whest-phase2-network-split-v1":
        raise AssertionError("frozen roster schema changed")
    if roster.get("commit") != EXPECTED_COMMIT:
        raise AssertionError("frozen roster commit changed")
    if roster.get("full_shard_manifest_sha256") != EXPECTED_FULL_MANIFEST_SHA256:
        raise AssertionError("frozen roster full manifest binding changed")
    assignment = roster.get("assignment", {})
    if bytes.fromhex(assignment.get("domain_hex", "")) != ASSIGNMENT_DOMAIN:
        raise AssertionError("frozen roster assignment domain changed")
    if assignment.get("assignment_inputs") != ["mlp_seed"]:
        raise AssertionError("non-seed input entered frozen assignment law")
    if assignment.get("attestations_not_assignment_inputs") != [
        "mlp_id",
        "weights_sha256",
    ]:
        raise AssertionError("roster assignment/attestation boundary changed")
    if _assignment_hash(0) != ASSIGNMENT_TEST_VECTOR_SEED_ZERO:
        raise AssertionError("assignment-law test vector failed")

    ranked: list[dict[str, Any]] = []
    split_sizes = (("train", 700), ("calibration", 100), ("validation", 100), ("test", 100))
    for split_name, expected_count in split_sizes:
        rows = roster["splits"][split_name]
        if len(rows) != expected_count:
            raise AssertionError(f"wrong frozen primary split size: {split_name}")
        for split_rank, row in enumerate(rows):
            if row["primary_split"] != split_name or int(row["split_rank"]) != split_rank:
                raise AssertionError("frozen primary split annotations changed")
        ranked.extend(rows)
    if len(ranked) != 1000:
        raise AssertionError("frozen roster does not contain 1,000 networks")
    if [int(row["global_hash_rank"]) for row in ranked] != list(range(1000)):
        raise AssertionError("frozen global ranks changed")

    ids: list[int] = []
    seeds: list[int] = []
    digests: list[str] = []
    role_by_id: dict[int, str] = {}
    role_ids: dict[str, list[int]] = {role: [] for role in EXPECTED_ROLE_COUNTS}
    for rank, row in enumerate(ranked):
        mlp_id = int(row["mlp_id"])
        mlp_seed = int(row["mlp_seed"])
        digest = _assignment_hash(mlp_seed)
        role = _role_for_rank(rank)
        if row["assignment_sha256"] != digest:
            raise AssertionError("roster digest is not reproducible from mlp_seed")
        if row["observer_role"] != role:
            raise AssertionError("roster observer-role segment changed")
        ids.append(mlp_id)
        seeds.append(mlp_seed)
        digests.append(digest)
        if mlp_id in role_by_id:
            raise AssertionError("duplicate mlp_id in frozen roster")
        role_by_id[mlp_id] = role
        role_ids[role].append(mlp_id)
    if len(set(ids)) != 1000 or len(set(seeds)) != 1000 or len(set(digests)) != 1000:
        raise AssertionError("roster IDs, seeds, or digests are not unique")
    if digests != sorted(digests):
        raise AssertionError("frozen roster is not digest-ascending")
    if {role: len(values) for role, values in role_ids.items()} != EXPECTED_ROLE_COUNTS:
        raise AssertionError("frozen observer role counts changed")
    if roster.get("observer_role_counts") != EXPECTED_ROLE_COUNTS:
        raise AssertionError("roster role-count attestation changed")
    for role, expected_ids in role_ids.items():
        duplicated_ids = [
            int(row["mlp_id"]) for row in roster["observer_roles"][role]
        ]
        if duplicated_ids != expected_ids:
            raise AssertionError(f"duplicated roster role index changed: {role}")
    return role_by_id, role_ids


def _validate_frozen_inputs() -> tuple[dict[int, str], dict[str, list[int]]]:
    expected_hashes = {
        AUTHORITY_PATH: EXPECTED_AUTHORITY_SHA256,
        EXTENSION_PATH: EXPECTED_EXTENSION_SHA256,
        ROSTER_PATH: EXPECTED_ROSTER_SHA256,
        ADDRESSABILITY_PATH: EXPECTED_ADDRESSABILITY_SHA256,
        CACHE_AUDIT_PATH: EXPECTED_CACHE_AUDIT_SHA256,
        RAW_READER_PATH: EXPECTED_RAW_READER_SHA256,
        TRAINING_READER_PATH: EXPECTED_TRAINING_READER_SHA256,
    }
    for path, expected in expected_hashes.items():
        if _sha256_file(path) != expected:
            raise AssertionError(f"frozen custodian input changed: {path.name}")
    authority = json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))
    extension = json.loads(EXTENSION_PATH.read_text(encoding="utf-8"))
    if authority.get("pinned_commit_sha") != EXPECTED_COMMIT:
        raise AssertionError("authority commit changed")
    if extension["parent_authority"]["sha256"] != EXPECTED_AUTHORITY_SHA256:
        raise AssertionError("extension parent authority binding changed")
    if extension["full"]["manifest_sha256"] != EXPECTED_FULL_MANIFEST_SHA256:
        raise AssertionError("extension full manifest binding changed")
    addressability = json.loads(ADDRESSABILITY_PATH.read_text(encoding="utf-8"))
    if addressability.get("status") != (
        "EXTERNAL_CONSTRAINT__RAW_PARQUET_CANNOT_PRESERVE_ROLE_TARGET_SEALS"
    ):
        raise AssertionError("raw physical-seal constraint changed")
    if addressability.get("raw_role_projection_physically_safe") is not False:
        raise AssertionError("raw roles unexpectedly marked physically safe")
    if addressability.get("training_role_row_groups_colliding_with_audit_or_lockbox") is None:
        raise AssertionError("mixed-row-group evidence is absent")
    cache_audit = json.loads(CACHE_AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        cache_audit.get("status") != "PASS_FULL_CACHE_PAYLOAD_AND_SCHEMA"
        or cache_audit.get("payload_hashes_verified") is not True
        or cache_audit.get("target_values_decoded") != 0
    ):
        raise AssertionError("full cache admission receipt changed")
    return _validate_roster()


def _validate_authorization(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PermissionError("authorization must be a regular non-symlink file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != AUTHORIZATION_SCHEMA:
        raise PermissionError("wrong custodian authorization schema")
    if payload.get("authorized") is not True:
        raise PermissionError("custodian execution is not explicitly authorized")
    expected = {
        "repository": "aicrowd/arc-whestbench-public-2026",
        "revision": "v2-phase2",
        "commit": EXPECTED_COMMIT,
        "full_shard_manifest_sha256": EXPECTED_FULL_MANIFEST_SHA256,
        "physical_target_scope": "complete_full_all_layer_means_column",
        "execution_boundary": "trusted_data_custodian",
        "purpose": "materialize_separately_sealed_role_target_artifacts",
        "output_contract": "role_specific_artifacts_with_audit_and_lockbox_withheld",
        "allowed_columns": list(RAW_TARGET_COLUMNS),
        "split_manifest_sha256": EXPECTED_ROSTER_SHA256,
        "role_addressability_audit_sha256": EXPECTED_ADDRESSABILITY_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"custodian authorization binding mismatch: {key}")
    for key in ("authorized_by", "authorized_at_utc", "nonce"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise PermissionError(f"custodian authorization lacks non-empty {key}")
    return payload


def _assert_execution_environment_is_fresh() -> None:
    if os.environ.get(BOUNDARY_ENVIRONMENT_VARIABLE) != EXPECTED_BOUNDARY_VALUE:
        raise PermissionError("isolated custodian boundary environment is not asserted")
    if FIXED_CLAIM_PATH.exists() or FIXED_CLAIM_PATH.is_symlink():
        raise FileExistsError("one-use custodian claim is already consumed")
    if FIXED_STAGING_PATH.exists() or FIXED_STAGING_PATH.is_symlink():
        raise FileExistsError("custodian staging path already exists")
    if FIXED_OUTPUT_PATH.exists() or FIXED_OUTPUT_PATH.is_symlink():
        raise FileExistsError("sealed training bundle already exists")


def _consume_fixed_claim(authorization_path: Path, authorization: dict[str, Any]) -> str:
    claim = {
        "schema": "whest-phase2-training-target-sealer-claim-v1",
        "status": "CONSUMED_BEFORE_RAW_TARGET_ITERATOR",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_nonce": authorization["nonce"],
        "authorization_sha256": _sha256_file(authorization_path),
        "sealer_source_sha256": _sha256_file(Path(__file__).resolve()),
        "fixed_output_path": FIXED_OUTPUT_PATH.name,
        "crash_policy": "claim_is_never_removed; any crash consumes the attempt; no retry",
        "target_values_decoded_before_claim": 0,
    }
    _write_exclusive_fsynced(FIXED_CLAIM_PATH, _canonical_json(claim), 0o400)
    _fsync_directory(HERE)
    return _sha256_file(FIXED_CLAIM_PATH)


def _validate_target_array(values: np.ndarray) -> None:
    if values.shape != TARGET_SHAPE:
        raise AssertionError(f"raw target shape changed: {values.shape}")
    if values.dtype != TARGET_DTYPE:
        raise AssertionError(f"raw target dtype changed: {values.dtype}")
    if not values.flags.c_contiguous:
        raise AssertionError("raw target is not C-contiguous")
    if not np.isfinite(values).all():
        raise AssertionError("raw target contains non-finite values")


def _target_array(ids: list[int], retained: dict[int, np.ndarray]) -> np.ndarray:
    if set(retained) != set(ids) or len(retained) != len(ids):
        raise AssertionError("retained role target IDs differ from frozen roster")
    values = np.stack([retained[mlp_id] for mlp_id in ids], axis=0)
    if values.shape != (len(ids), *TARGET_SHAPE):
        raise AssertionError("stacked role target geometry changed")
    if values.dtype != TARGET_DTYPE or not values.flags.c_contiguous:
        raise AssertionError("stacked role target dtype/layout changed")
    return values


def _array2d_storage(values: np.ndarray) -> pa.FixedSizeListArray:
    primitive = pa.array(values.reshape(-1), type=pa.float32(), from_pandas=False)
    inner = pa.FixedSizeListArray.from_arrays(primitive, TARGET_SHAPE[1])
    return pa.FixedSizeListArray.from_arrays(inner, TARGET_SHAPE[0])


def _validate_written_parquet(path: Path, expected_rows: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if schema.names != list(RAW_TARGET_COLUMNS):
        raise AssertionError("role artifact columns changed")
    if schema.field("mlp_id").type != pa.int32():
        raise AssertionError("role artifact mlp_id is not int32")
    metadata = schema.field("all_layer_means").metadata or {}
    if metadata != ARRAY2D_FIELD_METADATA:
        raise AssertionError("role artifact Array2D metadata changed")
    if int(parquet.metadata.num_rows) != expected_rows:
        raise AssertionError("role artifact row count changed")
    if int(parquet.metadata.num_row_groups) != 1:
        raise AssertionError("role artifact must contain exactly one row group")
    return {
        "physical_arrow_type": str(schema.field("all_layer_means").type),
        "parquet_row_groups": int(parquet.metadata.num_row_groups),
    }


def _write_role_artifact(
    role: str,
    ids: list[int],
    retained: dict[int, np.ndarray],
) -> dict[str, Any]:
    if role not in PERSISTED_ROLES:
        raise AssertionError(f"refusing to persist withheld role: {role}")
    values = _target_array(ids, retained)
    id_array = pa.array(ids, type=pa.int32())
    target_storage = _array2d_storage(values)
    schema = pa.schema(
        (
            pa.field("mlp_id", pa.int32(), nullable=False),
            pa.field(
                "all_layer_means",
                target_storage.type,
                nullable=False,
                metadata=ARRAY2D_FIELD_METADATA,
            ),
        )
    )
    table = pa.Table.from_arrays((id_array, target_storage), schema=schema)
    final_name = f"{role}.parquet"
    final_path = FIXED_STAGING_PATH / final_name
    temporary_path = FIXED_STAGING_PATH / f".{final_name}.tmp"
    if final_path.exists() or temporary_path.exists():
        raise FileExistsError(f"role artifact path already exists: {role}")
    pq.write_table(
        table,
        temporary_path,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=len(ids),
        data_page_size=1 << 20,
        version="2.6",
    )
    os.chmod(temporary_path, 0o400)
    _fsync_regular_file(temporary_path)
    physical = _validate_written_parquet(temporary_path, len(ids))
    os.replace(temporary_path, final_path)
    _fsync_directory(FIXED_STAGING_PATH)
    artifact = {
        "path": final_name,
        "sha256": _sha256_file(final_path),
        "bytes": final_path.stat().st_size,
        "rows": len(ids),
        "columns": list(RAW_TARGET_COLUMNS),
        "mlp_id_dtype": "int32",
        "ordered_mlp_ids_sha256": _ordered_ids_sha256(ids),
        "all_layer_means_shape_per_row": list(TARGET_SHAPE),
        "all_layer_means_dtype": "float32",
        "all_layer_means_layout": "C",
        "all_layer_means_c_order_sha256": hashlib.sha256(
            memoryview(values).cast("B")
        ).hexdigest(),
        **physical,
    }
    del table, target_storage, id_array, values
    return artifact


def _write_manifest_atomic(payload: dict[str, Any]) -> None:
    final_path = FIXED_STAGING_PATH / "manifest.json"
    temporary_path = FIXED_STAGING_PATH / ".manifest.json.tmp"
    _write_exclusive_fsynced(temporary_path, _canonical_json(payload), 0o400)
    os.replace(temporary_path, final_path)
    _fsync_directory(FIXED_STAGING_PATH)


def execute(authorization_path: Path) -> dict[str, Any]:
    role_by_id, role_ids = _validate_frozen_inputs()
    authorization = _validate_authorization(authorization_path)
    authorization_sha256 = _sha256_file(authorization_path)
    _assert_execution_environment_is_fresh()

    # Irreversible one-use boundary.  Nothing below may remove this claim.
    claim_sha256 = _consume_fixed_claim(authorization_path, authorization)
    FIXED_STAGING_PATH.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(HERE)

    retained: dict[str, dict[int, np.ndarray]] = {
        "observer_fit": {},
        "observer_selection": {},
    }
    decoded_counts: Counter[str] = Counter()
    seen_ids: set[int] = set()

    # The claim above must exist before this generator is even constructed.
    if (
        _sha256_file(authorization_path) != authorization_sha256
        or _sha256_file(ROSTER_PATH) != EXPECTED_ROSTER_SHA256
    ):
        raise AssertionError("authorization or roster changed across claim boundary")
    raw_iterator = iter_trusted_custodian_all_layer_means(
        authorization_path=authorization_path,
        split_manifest_path=ROSTER_PATH,
    )
    for raw_row in raw_iterator:
        if raw_row.decoded_columns != RAW_TARGET_COLUMNS:
            raise AssertionError("raw custodian projection changed")
        if raw_row.mlp_id in seen_ids or raw_row.mlp_id not in role_by_id:
            raise AssertionError("raw target ID is duplicate or absent from roster")
        seen_ids.add(raw_row.mlp_id)
        role = role_by_id[raw_row.mlp_id]
        if raw_row.observer_role != role:
            raise AssertionError("raw iterator role differs from frozen roster")
        _validate_target_array(raw_row.all_layer_means)
        decoded_counts[role] += 1
        if role == "observer_fit":
            retained["observer_fit"][raw_row.mlp_id] = np.array(
                raw_row.all_layer_means, dtype=TARGET_DTYPE, order="C", copy=True
            )
        elif role == "observer_selection":
            retained["observer_selection"][raw_row.mlp_id] = np.array(
                raw_row.all_layer_means, dtype=TARGET_DTYPE, order="C", copy=True
            )
        elif role not in WITHHELD_ROLES:
            raise AssertionError(f"unknown observer role: {role}")
        # Withheld arrays are intentionally neither copied nor appended.
    del raw_iterator, raw_row

    if (
        _sha256_file(authorization_path) != authorization_sha256
        or _sha256_file(ROSTER_PATH) != EXPECTED_ROSTER_SHA256
    ):
        raise AssertionError("authorization or roster changed during raw traversal")

    if seen_ids != set(role_by_id) or len(seen_ids) != 1000:
        raise AssertionError("raw custodian traversal did not exhaust all 1,000 rows")
    if dict(decoded_counts) != EXPECTED_ROLE_COUNTS:
        raise AssertionError("honest raw physical-decode accounting changed")
    if set(retained) != PERSISTED_ROLES:
        raise AssertionError("retained role set changed")

    # The only two target-bearing write calls in this program.
    artifacts = {
        "observer_fit": _write_role_artifact(
            "observer_fit", role_ids["observer_fit"], retained["observer_fit"]
        ),
        "observer_selection": _write_role_artifact(
            "observer_selection",
            role_ids["observer_selection"],
            retained["observer_selection"],
        ),
    }
    del retained
    persisted_rows = sum(int(artifact["rows"]) for artifact in artifacts.values())
    withheld_decoded_rows = sum(decoded_counts[role] for role in WITHHELD_ROLES)
    if persisted_rows != 776 or withheld_decoded_rows != 224:
        raise AssertionError("persisted/transient target accounting changed")

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "status": "SEALED_BY_TRUSTED_DATA_CUSTODIAN",
        "repository": "aicrowd/arc-whestbench-public-2026",
        "revision": "v2-phase2",
        "commit": EXPECTED_COMMIT,
        "parent_authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "full_shard_manifest_sha256": EXPECTED_FULL_MANIFEST_SHA256,
        "split_manifest_sha256": EXPECTED_ROSTER_SHA256,
        "role_addressability_audit_sha256": EXPECTED_ADDRESSABILITY_SHA256,
        "custodian_claim_sha256": claim_sha256,
        "custodian_authorization_sha256": authorization_sha256,
        "custodian_sealer_source_sha256": _sha256_file(Path(__file__).resolve()),
        "target_columns": list(RAW_TARGET_COLUMNS),
        "raw_full_dataset_included": False,
        "withheld_roles_included": False,
        "artifacts": artifacts,
        "custodian_decode_accounting": {
            "raw_rows_physically_decoded": 1000,
            "raw_role_counts": dict(sorted(decoded_counts.items())),
            "persisted_training_rows": persisted_rows,
            "persisted_roles": sorted(PERSISTED_ROLES),
            "transient_withheld_rows_decoded_due_to_raw_row_group_mixing": (
                withheld_decoded_rows
            ),
            "transient_withheld_role_counts": {
                role: decoded_counts[role] for role in sorted(WITHHELD_ROLES)
            },
            "withheld_rows_written": 0,
        },
        "fit_or_score_performed": False,
        "host_features_generated": False,
        "crash_retry_allowed": False,
    }
    _write_manifest_atomic(manifest)
    expected_entries = {
        "manifest.json",
        "observer_fit.parquet",
        "observer_selection.parquet",
    }
    if {path.name for path in FIXED_STAGING_PATH.iterdir()} != expected_entries:
        raise AssertionError("staging directory contains an unexpected file")
    _fsync_directory(FIXED_STAGING_PATH)
    os.chmod(FIXED_STAGING_PATH, 0o500)
    os.rename(FIXED_STAGING_PATH, FIXED_OUTPUT_PATH)
    _fsync_directory(HERE)
    return {
        "status": "PUBLISHED_SEALED_FIT_AND_SELECTION_TARGET_BUNDLE",
        "bundle_path": str(FIXED_OUTPUT_PATH),
        "bundle_manifest_sha256": _sha256_file(FIXED_OUTPUT_PATH / "manifest.json"),
        "claim_path": str(FIXED_CLAIM_PATH),
        "claim_sha256": claim_sha256,
        "persisted_rows": persisted_rows,
        "transient_withheld_rows_decoded": withheld_decoded_rows,
        "withheld_rows_written": 0,
        "fit_or_score_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--acknowledge-crash-consumes", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing: this sealer has no executable preflight mode")
    if not args.acknowledge_crash_consumes:
        raise SystemExit("refusing: --acknowledge-crash-consumes is required")
    print(json.dumps(execute(args.authorization), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
