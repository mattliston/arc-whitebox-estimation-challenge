#!/usr/bin/env python3
"""PyArrow 11/25-compatible target-blind mini100 iterator v2.

The iterator uses Parquet physical-column projection.  It never constructs a
Hugging Face row and never asks PyArrow to read a forbidden column chunk.
Schema/footer metadata may describe the complete file, but the only decoded
values are ``mlp_id``, ``mlp_name``, ``mlp_seed``, and ``weights``.  Projected
batch names are obtained from the stable ``batch.schema.names`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
AUTHORITY_PATH = HERE / "authority.json"
AUTHORITY = json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TargetBlindMiniRow:
    """One source row whose values are restricted to the sealed allow-list."""

    ordinal: int
    shard_index: int
    row_in_shard: int
    mlp_id: int
    mlp_name: str
    mlp_seed: int
    weights: np.ndarray
    source_read_wall_s: float
    decoded_columns: tuple[str, ...]


def pinned_snapshot_path() -> Path:
    """Return the exact local snapshot path pinned by ``authority.json``."""

    return (
        HERE
        / ".hf-cache"
        / "hub"
        / "datasets--aicrowd--arc-whestbench-public-2026"
        / "snapshots"
        / AUTHORITY["pinned_commit_sha"]
    )


def _extension_shape(field: pa.Field) -> list[object]:
    metadata = field.metadata or {}
    encoded = metadata.get(b"ARROW:extension:metadata")
    if encoded is not None:
        return json.loads(encoded)
    # Importing ``datasets`` (transitively through whestbench) registers the
    # extension and PyArrow then deserializes the same metadata into a concrete
    # Array3DExtensionType instead of retaining the field key/value pair.
    shape = getattr(field.type, "shape", None)
    value_type = getattr(field.type, "value_type", None)
    if shape is None or value_type is None:
        raise AssertionError("weights field lacks fixed-shape extension metadata")
    return [list(shape), str(value_type)]


def _attest_shard_pointer(shard: Path, expected: dict[str, object]) -> Path:
    """Attest the content-addressed HF cache pointer without rereading payloads.

    The seven payload hashes were already verified byte-for-byte at cache
    admission.  Runtime validation checks that every snapshot link still names
    that exact content-addressed blob and that its size is unchanged.  It does
    not hash the whole Parquet file again, which would read opaque bytes from
    forbidden column chunks even though it would not decode them.
    """

    if not shard.is_symlink():
        raise AssertionError(f"snapshot shard is not an HF cache symlink: {shard}")
    expected_digest = str(expected["lfs_sha256"])
    link_target = Path(os.readlink(shard))
    if link_target.name != expected_digest:
        raise AssertionError(
            f"cache pointer digest mismatch for {shard.name}: "
            f"{link_target.name} != {expected_digest}"
        )
    resolved = shard.resolve(strict=True)
    if resolved.name != expected_digest:
        raise AssertionError(f"resolved blob name mismatch for {shard.name}")
    if resolved.stat().st_size != int(expected["size"]):
        raise AssertionError(f"cached blob size mismatch for {shard.name}")
    return resolved


def validate_pinned_mini_snapshot() -> dict[str, object]:
    """Validate snapshot coordinates, physical schema, and exact row count.

    Parquet footers are opened for schema and row-count metadata only.  No
    record batch and therefore no column value is decoded by this function.
    """

    snapshot = pinned_snapshot_path()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing pinned mini snapshot: {snapshot}")

    expected_features = list(AUTHORITY["features"])
    mini = AUTHORITY["splits"]["mini"]
    rows = 0
    pointers: list[dict[str, object]] = []
    for shard_index, expected in enumerate(mini["shards"]):
        shard = snapshot / str(expected["path"])
        resolved = _attest_shard_pointer(shard, expected)
        parquet = pq.ParquetFile(resolved)
        if parquet.schema_arrow.names != expected_features:
            raise AssertionError(f"unexpected Parquet schema in {shard.name}")
        weights_field = parquet.schema_arrow.field("weights")
        if _extension_shape(weights_field) != [[16, 1024, 1024], "float32"]:
            raise AssertionError(f"unexpected weights extension in {shard.name}")
        shard_rows = parquet.metadata.num_rows
        rows += shard_rows
        pointers.append(
            {
                "shard_index": shard_index,
                "snapshot_path": str(shard),
                "blob_sha256": resolved.name,
                "bytes": resolved.stat().st_size,
                "rows": shard_rows,
            }
        )

    if rows != int(mini["n_mlps"]):
        raise AssertionError(f"mini row count mismatch: {rows} != {mini['n_mlps']}")
    return {
        "repository": AUTHORITY["repository"],
        "revision": AUTHORITY["revision"],
        "commit": AUTHORITY["pinned_commit_sha"],
        "snapshot": str(snapshot),
        "rows": rows,
        "allowed_value_columns": AUTHORITY["target_seal"]["allowed_value_columns"],
        "target_values_decoded": 0,
        "shards": pointers,
    }


def _weights_view(batch: pa.RecordBatch) -> np.ndarray:
    storage = batch.column(batch.schema.get_field_index("weights"))
    # See _extension_shape: with datasets registered this is an ExtensionArray;
    # without it, PyArrow returns the same nested ListArray storage directly.
    storage = getattr(storage, "storage", storage)
    for _ in range(3):
        storage = storage.flatten()
    weights = storage.to_numpy(zero_copy_only=True).reshape(16, 1024, 1024)
    if weights.dtype != np.float32:
        raise AssertionError(f"unexpected weights dtype: {weights.dtype}")
    if not weights.flags.c_contiguous:
        raise AssertionError("projected weights are not C-contiguous")
    # Bound peak temporary memory to one layer rather than scanning the entire
    # 64 MiB tensor into a same-size boolean array.
    for layer in weights:
        if not np.isfinite(layer).all():
            raise AssertionError("non-finite source weight")
    weights.setflags(write=False)
    return weights


def iter_target_blind_mini() -> Iterator[TargetBlindMiniRow]:
    """Yield all mini100 rows in manifest/shard order under the target seal."""

    snapshot = pinned_snapshot_path()
    allowed = tuple(AUTHORITY["target_seal"]["allowed_value_columns"])
    forbidden = set(AUTHORITY["target_seal"]["forbidden_value_columns"])
    if allowed != ("mlp_id", "mlp_name", "mlp_seed", "weights"):
        raise AssertionError(f"unexpected target-seal allow-list: {allowed}")
    if forbidden.intersection(allowed):
        raise AssertionError("target-seal allow-list intersects its deny-list")

    ordinal = 0
    seen_ids: set[int] = set()
    for shard_index, expected in enumerate(AUTHORITY["splits"]["mini"]["shards"]):
        shard = snapshot / str(expected["path"])
        resolved = _attest_shard_pointer(shard, expected)
        parquet = pq.ParquetFile(resolved)
        batches = parquet.iter_batches(
            batch_size=1,
            columns=list(allowed),
            use_threads=False,
        )
        row_in_shard = 0
        while True:
            started = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                break
            source_read_wall_s = time.perf_counter() - started
            if batch.num_rows != 1 or tuple(batch.schema.names) != allowed:
                raise AssertionError("Parquet projection violated the one-row contract")
            if forbidden.intersection(batch.schema.names):
                raise AssertionError("forbidden value column reached a record batch")

            mlp_id = int(batch.column(batch.schema.get_field_index("mlp_id"))[0].as_py())
            if mlp_id in seen_ids:
                raise AssertionError(f"duplicate mini mlp_id: {mlp_id}")
            seen_ids.add(mlp_id)
            yield TargetBlindMiniRow(
                ordinal=ordinal,
                shard_index=shard_index,
                row_in_shard=row_in_shard,
                mlp_id=mlp_id,
                mlp_name=str(
                    batch.column(batch.schema.get_field_index("mlp_name"))[0].as_py()
                ),
                mlp_seed=int(
                    batch.column(batch.schema.get_field_index("mlp_seed"))[0].as_py()
                ),
                weights=_weights_view(batch),
                source_read_wall_s=source_read_wall_s,
                decoded_columns=allowed,
            )
            ordinal += 1
            row_in_shard += 1

    expected_rows = int(AUTHORITY["splits"]["mini"]["n_mlps"])
    if ordinal != expected_rows:
        raise AssertionError(f"mini iterator yielded {ordinal} rows, expected {expected_rows}")


if __name__ == "__main__":
    print(json.dumps(validate_pinned_mini_snapshot(), indent=2, sort_keys=True))
