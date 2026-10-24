#!/usr/bin/env python3
"""Version-neutral role-target artifact codec shared by sealer v3 and tests.

This module has no raw-data reader, production path, authorization, fitting,
or scoring capability. It serializes already supplied fit/selection arrays to
the exact two-column Parquet contract consumed by sealed_role_targets_v2.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TARGET_COLUMNS = ("mlp_id", "all_layer_means")
TARGET_SHAPE = (16, 1024)
TARGET_DTYPE = np.dtype("float32")
PERSISTED_ROLES = frozenset(("observer_fit", "observer_selection"))
ARRAY2D_FIELD_METADATA = {
    b"ARROW:extension:name": b"datasets.features.features.Array2DExtensionType",
    b"ARROW:extension:metadata": b'[[16, 1024], "float32"]',
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(ids: list[int]) -> str:
    encoded = "".join(f"{mlp_id}\n" for mlp_id in ids).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
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
                raise OSError("short write while creating artifact file")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_staging(staging_path: Path) -> Path:
    resolved = staging_path.resolve(strict=True)
    if not resolved.is_dir() or staging_path.is_symlink():
        raise AssertionError("artifact staging must be a regular directory")
    return resolved


def _validate_target_array(values: np.ndarray, expected_rows: int) -> None:
    if values.shape != (expected_rows, *TARGET_SHAPE):
        raise AssertionError(f"role target shape changed: {values.shape}")
    if values.dtype != TARGET_DTYPE or not values.flags.c_contiguous:
        raise AssertionError("role target dtype/layout changed")
    if not np.isfinite(values).all():
        raise AssertionError("role target contains non-finite values")


def _array2d_storage(values: np.ndarray) -> pa.FixedSizeListArray:
    primitive = pa.array(values.reshape(-1), type=pa.float32(), from_pandas=False)
    inner = pa.FixedSizeListArray.from_arrays(primitive, TARGET_SHAPE[1])
    return pa.FixedSizeListArray.from_arrays(inner, TARGET_SHAPE[0])


def validate_written_parquet(path: Path, expected_rows: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if schema.names != list(TARGET_COLUMNS):
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


def write_role_artifact(
    *,
    staging_path: Path,
    role: str,
    ids: list[int],
    values: np.ndarray,
) -> dict[str, Any]:
    """Atomically serialize exactly one allowed role into an existing stage."""

    staging = _validate_staging(staging_path)
    if role not in PERSISTED_ROLES:
        raise AssertionError(f"refusing to persist withheld role: {role}")
    if not ids or len(ids) != len(set(ids)):
        raise AssertionError("role IDs are empty or duplicated")
    if any(mlp_id < -(1 << 31) or mlp_id >= (1 << 31) for mlp_id in ids):
        raise AssertionError("role mlp_id does not fit int32")
    _validate_target_array(values, len(ids))

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
    final_path = staging / final_name
    temporary_path = staging / f".{final_name}.tmp"
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
    physical = validate_written_parquet(temporary_path, len(ids))
    os.replace(temporary_path, final_path)
    fsync_directory(staging)
    return {
        "path": final_name,
        "sha256": sha256_file(final_path),
        "bytes": final_path.stat().st_size,
        "rows": len(ids),
        "columns": list(TARGET_COLUMNS),
        "mlp_id_dtype": "int32",
        "ordered_mlp_ids_sha256": ordered_ids_sha256(ids),
        "all_layer_means_shape_per_row": list(TARGET_SHAPE),
        "all_layer_means_dtype": "float32",
        "all_layer_means_layout": "C",
        "all_layer_means_c_order_sha256": hashlib.sha256(
            memoryview(values).cast("B")
        ).hexdigest(),
        **physical,
    }


def write_manifest_atomic(staging_path: Path, payload: dict[str, Any]) -> Path:
    staging = _validate_staging(staging_path)
    final_path = staging / "manifest.json"
    temporary_path = staging / ".manifest.json.tmp"
    if final_path.exists() or temporary_path.exists():
        raise FileExistsError("bundle manifest path already exists")
    _write_exclusive_fsynced(temporary_path, canonical_json(payload), 0o400)
    os.replace(temporary_path, final_path)
    fsync_directory(staging)
    return final_path
