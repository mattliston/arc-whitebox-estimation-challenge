#!/usr/bin/env python3
"""Pinned-runtime, claim-before-raw-import Phase 2 custodian sealer v3.

THIS FILE IS A FROZEN EXECUTION DESIGN, NOT AUTHORIZATION TO RUN IT.

The only production execution path first validates every immutable input,
the exact Python/PyArrow runtime, and the required PyArrow API surface.  It
then durably consumes a fixed O_EXCL claim before importing the raw target
reader, resolving its iterator, constructing it, or decoding a target row.
Because the source row groups mix roles, all 1,000 target rows are physically
decoded inside the custodian boundary.  Only observer-fit (688) and
observer-selection (88) arrays are retained and written.  Audit, lockbox, and
retired diagnostic arrays are transiently decoded but never persisted.

The claim is never removed.  A crash consumes the attempt with no retry.
The program performs no fitting, prediction, scoring, or host-feature work.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


RAW_TARGET_MODULE_NAME = "authorized_full_targets_v2"
RAW_TARGET_FACTORY_NAME = "iter_trusted_custodian_all_layer_means"
ARTIFACT_CODEC_MODULE_NAME = "role_target_artifact_codec_v1"
RAW_TARGET_COLUMNS = ("mlp_id", "all_layer_means")

HERE = Path(__file__).resolve().parent
AUTHORITY_PATH = HERE / "authority.json"
EXTENSION_PATH = HERE / "FULL_SPLIT_AUTHORITY_EXTENSION.json"
ROSTER_PATH = HERE / "full_network_splits.json"
ADDRESSABILITY_PATH = HERE / "FULL_SPLIT_ROLE_ADDRESSABILITY_AUDIT.json"
CACHE_AUDIT_PATH = HERE / "FULL_SPLIT_CACHE_AUDIT.json"
RAW_READER_PATH = HERE / "authorized_full_targets_v2.py"
TRAINING_READER_PATH = HERE / "sealed_role_targets_v2.py"
MINI_READER_PATH = HERE / "target_blind_mini_v2.py"
ARTIFACT_CODEC_PATH = HERE / "role_target_artifact_codec_v1.py"
EXPECTED_EXECUTABLE = HERE / ".venv/bin/python"

# Fixed production paths; there are deliberately no CLI overrides.
FIXED_CLAIM_PATH = HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE_V3.CLAIM"
FIXED_STAGING_PATH = HERE / "sealed_role_target_bundle.v3.staging"
FIXED_OUTPUT_PATH = HERE / "sealed_role_target_bundle"
EARLIER_RUNTIME_PATHS = (
    HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE.CLAIM",
    HERE / "sealed_role_target_bundle.staging",
    HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE_V2.CLAIM",
    HERE / "sealed_role_target_bundle.v2.staging",
)

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
    "95ffd5ba0e55823a3176a35e42af052feebc4cb462dcfa9738b135d2110a7c83"
)
EXPECTED_TRAINING_READER_SHA256 = (
    "43cdc72d6bcd7b534d6c661d670b5e718af75bf482aff954ca6507d8d640cdbe"
)
EXPECTED_MINI_READER_SHA256 = (
    "6778a3216377a4a7e588df71a08fb0e182dc7a14d42ab393c2ac80200cccf0e1"
)
EXPECTED_ARTIFACT_CODEC_SHA256 = (
    "f09855d8843492ead6b3117535d19487a45988d6cd9fbacd2e642798e4c79b17"
)
EXPECTED_COMMIT = "ce926b7d1ced65faa668c01b94a43f8c4f43783d"
EXPECTED_FULL_MANIFEST_SHA256 = (
    "51357aceca60f0a0365b26d7b25e136d1e4e713d44451a5f7c079e3ba9ea2794"
)
EXPECTED_PYTHON = (3, 11, 5)
EXPECTED_PYARROW = "25.0.1"
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
EXPECTED_BOUNDARY_VALUE = "isolated-target-sealer-v3"
AUTHORIZATION_SCHEMA = "whest-phase2-custodian-target-access-v1"
BUNDLE_SCHEMA = "whest-phase2-training-role-target-bundle-v1"


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


def _assignment_hash(mlp_seed: int) -> str:
    return hashlib.sha256(
        ASSIGNMENT_DOMAIN + str(mlp_seed).encode("ascii")
    ).hexdigest()


def _role_for_rank(rank: int) -> str:
    matches = [role for role, begin, end in ROLE_SEGMENTS if begin <= rank < end]
    if len(matches) != 1:
        raise AssertionError(f"role segments do not uniquely cover rank {rank}")
    return matches[0]


def _validate_roster() -> tuple[dict[int, str], dict[str, list[int]]]:
    if _sha256_file(ROSTER_PATH) != EXPECTED_ROSTER_SHA256:
        raise AssertionError("frozen roster SHA-256 changed")
    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    if roster.get("schema") != "whest-phase2-network-split-v1":
        raise AssertionError("frozen roster schema changed")
    if roster.get("commit") != EXPECTED_COMMIT:
        raise AssertionError("frozen roster commit changed")
    if roster.get("full_shard_manifest_sha256") != EXPECTED_FULL_MANIFEST_SHA256:
        raise AssertionError("frozen roster full-manifest binding changed")
    assignment = roster.get("assignment", {})
    if bytes.fromhex(assignment.get("domain_hex", "")) != ASSIGNMENT_DOMAIN:
        raise AssertionError("frozen roster assignment domain changed")
    if assignment.get("assignment_inputs") != ["mlp_seed"]:
        raise AssertionError("non-seed input entered frozen assignment law")
    if assignment.get("attestations_not_assignment_inputs") != [
        "mlp_id",
        "weights_sha256",
    ]:
        raise AssertionError("assignment/attestation boundary changed")
    if _assignment_hash(0) != ASSIGNMENT_TEST_VECTOR_SEED_ZERO:
        raise AssertionError("assignment-law test vector failed")

    ranked: list[dict[str, Any]] = []
    for split_name, expected_count in (
        ("train", 700),
        ("calibration", 100),
        ("validation", 100),
        ("test", 100),
    ):
        rows = roster["splits"][split_name]
        if len(rows) != expected_count:
            raise AssertionError(f"wrong frozen primary split size: {split_name}")
        ranked.extend(rows)
    if [int(row["global_hash_rank"]) for row in ranked] != list(range(1000)):
        raise AssertionError("frozen global ranks changed")

    ids: list[int] = []
    seeds: list[int] = []
    digests: list[str] = []
    role_by_id: dict[int, str] = {}
    role_ids: dict[str, list[int]] = {role: [] for role in EXPECTED_ROLE_COUNTS}
    for rank, row in enumerate(ranked):
        mlp_id = int(row["mlp_id"])
        seed = int(row["mlp_seed"])
        digest = _assignment_hash(seed)
        role = _role_for_rank(rank)
        if row["assignment_sha256"] != digest or row["observer_role"] != role:
            raise AssertionError("frozen assignment digest or observer role changed")
        if mlp_id in role_by_id:
            raise AssertionError("duplicate mlp_id in frozen roster")
        ids.append(mlp_id)
        seeds.append(seed)
        digests.append(digest)
        role_by_id[mlp_id] = role
        role_ids[role].append(mlp_id)
    if len(set(ids)) != 1000 or len(set(seeds)) != 1000 or len(set(digests)) != 1000:
        raise AssertionError("roster IDs, seeds, or assignment digests are not unique")
    if digests != sorted(digests):
        raise AssertionError("frozen roster is not digest-ascending")
    if {role: len(values) for role, values in role_ids.items()} != EXPECTED_ROLE_COUNTS:
        raise AssertionError("frozen role counts changed")
    for role, expected_ids in role_ids.items():
        duplicate_index = [
            int(row["mlp_id"]) for row in roster["observer_roles"][role]
        ]
        if duplicate_index != expected_ids:
            raise AssertionError(f"duplicated role index changed: {role}")
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
        MINI_READER_PATH: EXPECTED_MINI_READER_SHA256,
        ARTIFACT_CODEC_PATH: EXPECTED_ARTIFACT_CODEC_SHA256,
    }
    for path, expected in expected_hashes.items():
        if _sha256_file(path) != expected:
            raise AssertionError(f"frozen v3 custodian input changed: {path.name}")
    authority = json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))
    extension = json.loads(EXTENSION_PATH.read_text(encoding="utf-8"))
    if authority.get("pinned_commit_sha") != EXPECTED_COMMIT:
        raise AssertionError("authority commit changed")
    if extension["parent_authority"]["sha256"] != EXPECTED_AUTHORITY_SHA256:
        raise AssertionError("extension parent-authority binding changed")
    if extension["full"]["manifest_sha256"] != EXPECTED_FULL_MANIFEST_SHA256:
        raise AssertionError("extension full-manifest binding changed")
    addressability = json.loads(ADDRESSABILITY_PATH.read_text(encoding="utf-8"))
    if addressability.get("status") != (
        "EXTERNAL_CONSTRAINT__RAW_PARQUET_CANNOT_PRESERVE_ROLE_TARGET_SEALS"
    ):
        raise AssertionError("raw physical-seal constraint changed")
    if addressability.get("raw_role_projection_physically_safe") is not False:
        raise AssertionError("raw roles unexpectedly marked physically safe")
    cache_audit = json.loads(CACHE_AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        cache_audit.get("status") != "PASS_FULL_CACHE_PAYLOAD_AND_SCHEMA"
        or cache_audit.get("payload_hashes_verified") is not True
        or cache_audit.get("target_values_decoded") != 0
    ):
        raise AssertionError("full cache-admission receipt changed")
    return _validate_roster()


def _validate_authorization(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PermissionError("authorization must be a regular non-symlink file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != AUTHORIZATION_SCHEMA or payload.get("authorized") is not True:
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


def _validate_runtime_capabilities_preclaim() -> dict[str, Any]:
    """Prove the exact interpreter and all used Arrow APIs without file I/O."""

    if Path(sys.executable).resolve() != EXPECTED_EXECUTABLE.resolve(strict=True):
        raise RuntimeError("v3 must run under data_gate/.venv/bin/python")
    if tuple(sys.version_info[:3]) != EXPECTED_PYTHON:
        raise RuntimeError("v3 Python patch version changed")
    if pa.__version__ != EXPECTED_PYARROW:
        raise RuntimeError("v3 PyArrow version changed")
    if RAW_TARGET_MODULE_NAME in sys.modules:
        raise AssertionError("raw target module was imported before claim")

    primitive = pa.array(np.arange(4, dtype=np.float32), type=pa.float32())
    inner = pa.FixedSizeListArray.from_arrays(primitive, 2)
    outer = pa.FixedSizeListArray.from_arrays(inner, 2)
    batch = pa.RecordBatch.from_arrays(
        [pa.array([7], type=pa.int32()), outer],
        names=["mlp_id", "all_layer_means"],
    )
    if tuple(batch.schema.names) != RAW_TARGET_COLUMNS:
        raise AssertionError("RecordBatch.schema.names capability changed")
    if batch.schema.get_field_index("all_layer_means") != 1:
        raise AssertionError("Schema.get_field_index capability changed")
    storage = batch.column(1).flatten().flatten()
    view = storage.to_numpy(zero_copy_only=True)
    if view.dtype != np.float32 or view.tolist() != [0.0, 1.0, 2.0, 3.0]:
        raise AssertionError("nested flatten/to_numpy capability changed")
    required_callables = {
        "ParquetFile": pq.ParquetFile,
        "ParquetFile.iter_batches": pq.ParquetFile.iter_batches,
        "write_table": pq.write_table,
        "Table.from_arrays": pa.Table.from_arrays,
        "FixedSizeListArray.from_arrays": pa.FixedSizeListArray.from_arrays,
    }
    if not all(callable(value) for value in required_callables.values()):
        raise AssertionError("required PyArrow API is absent")
    return {
        "python": ".".join(str(part) for part in EXPECTED_PYTHON),
        "pyarrow": EXPECTED_PYARROW,
        "executable": str(EXPECTED_EXECUTABLE),
        "required_apis": sorted(required_callables),
        "batch_name_api": "batch.schema.names",
        "raw_target_module_imported": False,
        "target_values_decoded": 0,
    }


def _load_artifact_codec_preclaim() -> Any:
    """Load only the already-hash-checked non-reader artifact serializer."""

    if _sha256_file(ARTIFACT_CODEC_PATH) != EXPECTED_ARTIFACT_CODEC_SHA256:
        raise AssertionError("artifact codec changed immediately before import")
    artifact_codec = __import__(ARTIFACT_CODEC_MODULE_NAME)
    if Path(artifact_codec.__file__).resolve() != ARTIFACT_CODEC_PATH.resolve():
        raise AssertionError("artifact codec resolved outside the data gate")
    if artifact_codec.TARGET_COLUMNS != RAW_TARGET_COLUMNS:
        raise AssertionError("artifact codec target columns changed")
    if artifact_codec.TARGET_SHAPE != TARGET_SHAPE:
        raise AssertionError("artifact codec target shape changed")
    if artifact_codec.PERSISTED_ROLES != PERSISTED_ROLES:
        raise AssertionError("artifact codec role allow-list changed")
    return artifact_codec


def _assert_execution_environment_is_fresh() -> None:
    if os.environ.get(BOUNDARY_ENVIRONMENT_VARIABLE) != EXPECTED_BOUNDARY_VALUE:
        raise PermissionError("isolated v3 custodian boundary is not asserted")
    paths = (*EARLIER_RUNTIME_PATHS, FIXED_CLAIM_PATH, FIXED_STAGING_PATH, FIXED_OUTPUT_PATH)
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError("a custodian claim, staging path, or release already exists")


def _consume_fixed_claim(authorization_path: Path, authorization: dict[str, Any]) -> str:
    claim = {
        "schema": "whest-phase2-training-target-sealer-claim-v3",
        "status": "CONSUMED_BEFORE_RAW_READER_IMPORT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_nonce": authorization["nonce"],
        "authorization_sha256": _sha256_file(authorization_path),
        "sealer_source_sha256": _sha256_file(Path(__file__).resolve()),
        "raw_reader_sha256": EXPECTED_RAW_READER_SHA256,
        "training_reader_sha256": EXPECTED_TRAINING_READER_SHA256,
        "artifact_codec_sha256": EXPECTED_ARTIFACT_CODEC_SHA256,
        "python": ".".join(str(part) for part in EXPECTED_PYTHON),
        "pyarrow": EXPECTED_PYARROW,
        "fixed_output_path": FIXED_OUTPUT_PATH.name,
        "crash_policy": "claim_is_never_removed; any crash consumes the attempt; no retry",
        "target_values_decoded_before_claim": 0,
    }
    _write_exclusive_fsynced(FIXED_CLAIM_PATH, _canonical_json(claim), 0o400)
    _fsync_directory(HERE)
    return _sha256_file(FIXED_CLAIM_PATH)


def _validate_target_array(values: np.ndarray) -> None:
    if values.shape != TARGET_SHAPE or values.dtype != TARGET_DTYPE:
        raise AssertionError("raw target shape/dtype changed")
    if not values.flags.c_contiguous or not np.isfinite(values).all():
        raise AssertionError("raw target layout or finiteness changed")


def _stack_role(ids: list[int], retained: dict[int, np.ndarray]) -> np.ndarray:
    if set(retained) != set(ids) or len(retained) != len(ids):
        raise AssertionError("retained role target IDs differ from frozen roster")
    values = np.stack([retained[mlp_id] for mlp_id in ids], axis=0)
    if values.shape != (len(ids), *TARGET_SHAPE):
        raise AssertionError("stacked role target geometry changed")
    if values.dtype != TARGET_DTYPE or not values.flags.c_contiguous:
        raise AssertionError("stacked role target dtype/layout changed")
    return values


def execute(authorization_path: Path) -> dict[str, Any]:
    role_by_id, role_ids = _validate_frozen_inputs()
    authorization = _validate_authorization(authorization_path)
    authorization_sha256 = _sha256_file(authorization_path)
    _assert_execution_environment_is_fresh()
    runtime = _validate_runtime_capabilities_preclaim()
    artifact_codec = _load_artifact_codec_preclaim()

    # Irreversible one-use boundary. Nothing below may remove this claim.
    claim_sha256 = _consume_fixed_claim(authorization_path, authorization)

    # STRICT V3 RAW BOUNDARY: claim -> raw import -> factory -> iterator.
    import importlib
    raw_target_module = importlib.import_module(RAW_TARGET_MODULE_NAME)
    raw_iterator_factory = getattr(raw_target_module, RAW_TARGET_FACTORY_NAME)
    raw_iterator = raw_iterator_factory(
        authorization_path=authorization_path,
        split_manifest_path=ROSTER_PATH,
    )

    if (
        getattr(raw_iterator_factory, "__module__", None) != RAW_TARGET_MODULE_NAME
        or getattr(raw_iterator_factory, "__name__", None) != RAW_TARGET_FACTORY_NAME
        or Path(raw_target_module.__file__).resolve() != RAW_READER_PATH.resolve()
    ):
        raise AssertionError("resolved raw reader/factory identity changed")
    FIXED_STAGING_PATH.mkdir(mode=0o700, exist_ok=False)
    _fsync_directory(HERE)

    retained: dict[str, dict[int, np.ndarray]] = {
        "observer_fit": {},
        "observer_selection": {},
    }
    decoded_counts: Counter[str] = Counter()
    seen_ids: set[int] = set()
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
        if role in PERSISTED_ROLES:
            retained[role][raw_row.mlp_id] = np.array(
                raw_row.all_layer_means, dtype=TARGET_DTYPE, order="C", copy=True
            )
        elif role not in WITHHELD_ROLES:
            raise AssertionError(f"unknown observer role: {role}")
        # Withheld arrays are deliberately neither copied nor appended.
    del raw_iterator, raw_row

    if (
        _sha256_file(authorization_path) != authorization_sha256
        or _sha256_file(ROSTER_PATH) != EXPECTED_ROSTER_SHA256
        or _sha256_file(RAW_READER_PATH) != EXPECTED_RAW_READER_SHA256
    ):
        raise AssertionError("authorization, roster, or reader changed during traversal")
    if seen_ids != set(role_by_id) or len(seen_ids) != 1000:
        raise AssertionError("raw traversal did not exhaust the frozen roster")
    if dict(decoded_counts) != EXPECTED_ROLE_COUNTS:
        raise AssertionError("honest physical-decode accounting changed")

    artifacts: dict[str, dict[str, Any]] = {}
    for role in ("observer_fit", "observer_selection"):
        values = _stack_role(role_ids[role], retained[role])
        artifacts[role] = artifact_codec.write_role_artifact(
            staging_path=FIXED_STAGING_PATH,
            role=role,
            ids=role_ids[role],
            values=values,
        )
        del values
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
        "raw_reader_sha256": EXPECTED_RAW_READER_SHA256,
        "training_reader_sha256": EXPECTED_TRAINING_READER_SHA256,
        "artifact_codec_sha256": EXPECTED_ARTIFACT_CODEC_SHA256,
        "execution_runtime": runtime,
        "target_columns": list(RAW_TARGET_COLUMNS),
        "raw_full_dataset_included": False,
        "withheld_roles_included": False,
        "artifacts": artifacts,
        "custodian_decode_accounting": {
            "raw_rows_physically_decoded": 1000,
            "raw_role_counts": dict(sorted(decoded_counts.items())),
            "persisted_training_rows": persisted_rows,
            "persisted_roles": sorted(PERSISTED_ROLES),
            "transient_withheld_rows_decoded_due_to_raw_row_group_mixing": withheld_decoded_rows,
            "transient_withheld_role_counts": {
                role: decoded_counts[role] for role in sorted(WITHHELD_ROLES)
            },
            "withheld_rows_written": 0,
        },
        "fit_or_score_performed": False,
        "host_features_generated": False,
        "crash_retry_allowed": False,
    }
    artifact_codec.write_manifest_atomic(FIXED_STAGING_PATH, manifest)
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
        "status": "PUBLISHED_SEALED_FIT_AND_SELECTION_TARGET_BUNDLE_V3",
        "bundle_path": str(FIXED_OUTPUT_PATH),
        "bundle_manifest_sha256": _sha256_file(FIXED_OUTPUT_PATH / "manifest.json"),
        "claim_path": str(FIXED_CLAIM_PATH),
        "claim_sha256": claim_sha256,
        "persisted_rows": persisted_rows,
        "transient_withheld_rows_decoded": withheld_decoded_rows,
        "withheld_rows_written": 0,
        "fit_or_score_performed": False,
    }


def _preclaim_runtime_proof() -> dict[str, Any]:
    runtime = _validate_runtime_capabilities_preclaim()
    paths = (*EARLIER_RUNTIME_PATHS, FIXED_CLAIM_PATH, FIXED_STAGING_PATH, FIXED_OUTPUT_PATH)
    if any(path.exists() or path.is_symlink() for path in paths):
        raise AssertionError("production path exists during preclaim proof")
    return {
        "status": "PASS_V3_RUNTIME_AND_PRECLAIM_RAW_IMPORT_PROOF",
        "runtime": runtime,
        "raw_target_module": RAW_TARGET_MODULE_NAME,
        "raw_target_module_imported": False,
        "production_paths_absent": True,
        "claim_created": False,
        "target_values_decoded": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--acknowledge-crash-consumes", action="store_true")
    parser.add_argument("--prove-runtime-preclaim-capabilities", action="store_true")
    args = parser.parse_args()
    if args.prove_runtime_preclaim_capabilities:
        if args.execute or args.authorization is not None:
            raise SystemExit("refusing mixed proof/execution arguments")
        print(json.dumps(_preclaim_runtime_proof(), indent=2, sort_keys=True))
        return
    if not args.execute:
        raise SystemExit("refusing: this sealer has no executable preflight mode")
    if args.authorization is None:
        raise SystemExit("refusing: --authorization is required for execution")
    if not args.acknowledge_crash_consumes:
        raise SystemExit("refusing: --acknowledge-crash-consumes is required")
    print(json.dumps(execute(args.authorization), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
