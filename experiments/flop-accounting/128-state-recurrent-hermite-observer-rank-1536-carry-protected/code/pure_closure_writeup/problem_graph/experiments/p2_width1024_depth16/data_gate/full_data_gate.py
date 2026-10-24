#!/usr/bin/env python3
"""Pinned Phase 2 full-split cache verifier and target-blind iterators.

This module never asks PyArrow for a target column.  The weights path is a
physical Parquet projection of exactly ``mlp_id``, ``mlp_name``, ``mlp_seed``,
and ``weights``; the lightweight identity path projects the first three only.
Hash verification reads opaque file bytes but never decodes Parquet values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
AUTHORITY_PATH = HERE / "authority.json"
AUTHORITY_BYTES = AUTHORITY_PATH.read_bytes()
AUTHORITY = json.loads(AUTHORITY_BYTES)
EXTENSION_PATH = HERE / "FULL_SPLIT_AUTHORITY_EXTENSION.json"
EXTENSION = json.loads(EXTENSION_PATH.read_text(encoding="utf-8"))
if hashlib.sha256(AUTHORITY_BYTES).hexdigest() != EXTENSION["parent_authority"]["sha256"]:
    raise AssertionError("frozen parent authority hash differs from full extension")
if EXTENSION["parent_authority"]["path"] != AUTHORITY_PATH.name:
    raise AssertionError("full extension names the wrong parent authority")
if (
    EXTENSION["repository"] != AUTHORITY["repository"]
    or EXTENSION["revision"] != AUTHORITY["revision"]
    or EXTENSION["pinned_commit_sha"] != AUTHORITY["pinned_commit_sha"]
):
    raise AssertionError("full extension coordinates differ from parent authority")
FULL = EXTENSION["full"]
for _key in (
    "hub_config",
    "metadata_declared_config",
    "split",
    "n_mlps",
    "shard_count",
    "download_bytes",
    "download_gib",
    "arrow_bytes_reported",
    "manifest_sha256",
):
    if FULL[_key] != AUTHORITY["splits"]["full"][_key]:
        raise AssertionError(f"full extension disagrees with parent authority: {_key}")
ALLOWED_VALUE_COLUMNS = tuple(AUTHORITY["target_seal"]["allowed_value_columns"])
FORBIDDEN_VALUE_COLUMNS = frozenset(
    AUTHORITY["target_seal"]["forbidden_value_columns"]
)
IDENTITY_COLUMNS = ("mlp_id", "mlp_name", "mlp_seed")
WEIGHT_COLUMNS = (*IDENTITY_COLUMNS, "weights")


@dataclass(frozen=True)
class FullIdentity:
    ordinal: int
    shard_index: int
    row_in_shard: int
    mlp_id: int
    mlp_name: str
    mlp_seed: int
    decoded_columns: tuple[str, ...]


@dataclass(frozen=True)
class TargetBlindFullRow:
    ordinal: int
    shard_index: int
    row_in_shard: int
    mlp_id: int
    mlp_name: str
    mlp_seed: int
    weights: np.ndarray
    source_read_wall_s: float
    decoded_columns: tuple[str, ...]


def _sha256_bytes(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def load_full_manifest() -> list[dict[str, object]]:
    """Load and self-authenticate the exact 63-path checked-in manifest."""

    path = HERE / str(FULL["shard_manifest_path"])
    raw = path.read_bytes()
    actual_file_hash = hashlib.sha256(raw).hexdigest()
    if actual_file_hash != FULL["shard_manifest_file_sha256"]:
        raise AssertionError("full shard TSV hash differs from authority")

    rows: list[dict[str, object]] = []
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        parts = line.split("\t")
        if len(parts) != 3:
            raise AssertionError(f"full shard row {index} is not three-field TSV")
        shard_path, size_text, digest = parts
        expected_path = f"data/full-{index:05d}-of-00067.parquet"
        if shard_path != expected_path:
            raise AssertionError(f"unexpected full shard path: {shard_path}")
        if len(digest) != 64 or not set(digest) <= set("0123456789abcdef"):
            raise AssertionError(f"invalid LFS SHA-256 for {shard_path}")
        rows.append(
            {"path": shard_path, "size": int(size_text), "lfs_sha256": digest}
        )

    canonical = "".join(
        f"{row['path']}\0{row['size']}\0{row['lfs_sha256']}\n"
        for row in sorted(rows, key=lambda row: str(row["path"]))
    ).encode()
    if len(rows) != int(FULL["shard_count"]):
        raise AssertionError("full shard count differs from authority")
    if sum(int(row["size"]) for row in rows) != int(FULL["download_bytes"]):
        raise AssertionError("full shard bytes differ from authority")
    if hashlib.sha256(canonical).hexdigest() != FULL["manifest_sha256"]:
        raise AssertionError("canonical full shard fingerprint differs from authority")
    if FULL["filename_declared_shard_count"] != 67:
        raise AssertionError("unexpected upstream filename denominator")
    if FULL["available_shard_indices"] != [0, 62]:
        raise AssertionError("unexpected available full shard index range")
    return rows


def pinned_snapshot_path() -> Path:
    return HERE / FULL["cache_repository_root"] / "snapshots" / AUTHORITY[
        "pinned_commit_sha"
    ]


def _extension_shape(field: pa.Field) -> list[object]:
    metadata = field.metadata or {}
    encoded = metadata.get(b"ARROW:extension:metadata")
    if encoded is not None:
        return json.loads(encoded)
    shape = getattr(field.type, "shape", None)
    value_type = getattr(field.type, "value_type", None)
    if shape is None or value_type is None:
        raise AssertionError("weights field lacks fixed-shape extension metadata")
    return [list(shape), str(value_type)]


def _attest_shard_pointer(shard: Path, expected: dict[str, object]) -> Path:
    if not shard.is_symlink():
        raise AssertionError(f"snapshot shard is not an HF cache symlink: {shard}")
    expected_digest = str(expected["lfs_sha256"])
    link_target = Path(os.readlink(shard))
    if link_target.name != expected_digest:
        raise AssertionError(f"cache pointer digest mismatch for {shard.name}")
    resolved = shard.resolve(strict=True)
    if resolved.name != expected_digest:
        raise AssertionError(f"resolved blob digest mismatch for {shard.name}")
    if resolved.stat().st_size != int(expected["size"]):
        raise AssertionError(f"cached blob size mismatch for {shard.name}")
    return resolved


def validate_pinned_full_snapshot(*, hash_payloads: bool) -> dict[str, object]:
    """Verify all pointers, sizes, schemas, row counts, and optionally bytes.

    Parquet is opened only for footer/schema metadata.  ``hash_payloads``
    performs a byte-for-byte SHA-256 admission check without decoding values.
    """

    snapshot = pinned_snapshot_path()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing pinned full snapshot: {snapshot}")

    rows = 0
    shards: list[dict[str, object]] = []
    expected_features = list(AUTHORITY["features"])
    for shard_index, expected in enumerate(load_full_manifest()):
        shard = snapshot / str(expected["path"])
        resolved = _attest_shard_pointer(shard, expected)
        payload_hash = None
        if hash_payloads:
            payload_hash = _sha256_bytes(resolved)
            if payload_hash != expected["lfs_sha256"]:
                raise AssertionError(f"payload SHA-256 mismatch for {shard.name}")

        parquet = pq.ParquetFile(resolved)
        if parquet.schema_arrow.names != expected_features:
            raise AssertionError(f"unexpected Parquet schema in {shard.name}")
        if _extension_shape(parquet.schema_arrow.field("weights")) != [
            [16, 1024, 1024],
            "float32",
        ]:
            raise AssertionError(f"unexpected weights geometry in {shard.name}")
        shard_rows = int(parquet.metadata.num_rows)
        rows += shard_rows
        shards.append(
            {
                "shard_index": shard_index,
                "path": str(expected["path"]),
                "bytes": resolved.stat().st_size,
                "expected_sha256": expected["lfs_sha256"],
                "verified_sha256": payload_hash,
                "rows": shard_rows,
                "row_groups": int(parquet.metadata.num_row_groups),
            }
        )

    if rows != int(FULL["n_mlps"]):
        raise AssertionError(f"full row count mismatch: {rows} != {FULL['n_mlps']}")
    return {
        "schema": "whest-phase2-full-cache-verification-v1",
        "status": "PASS_FULL_CACHE_PAYLOAD_AND_SCHEMA"
        if hash_payloads
        else "PASS_FULL_CACHE_POINTER_AND_SCHEMA",
        "repository": AUTHORITY["repository"],
        "revision": AUTHORITY["revision"],
        "commit": AUTHORITY["pinned_commit_sha"],
        "snapshot": str(snapshot),
        "manifest_sha256": FULL["manifest_sha256"],
        "manifest_file_sha256": FULL["shard_manifest_file_sha256"],
        "shard_count": len(shards),
        "download_bytes": sum(int(row["bytes"]) for row in shards),
        "rows": rows,
        "payload_hashes_verified": hash_payloads,
        "decoded_value_columns": [],
        "target_values_decoded": 0,
        "shards": shards,
    }


def _iter_batches(columns: Sequence[str]) -> Iterator[tuple[int, int, float, pa.RecordBatch]]:
    projected = tuple(columns)
    if not projected or len(set(projected)) != len(projected):
        raise AssertionError("projection must contain unique columns")
    if not set(projected) <= set(ALLOWED_VALUE_COLUMNS):
        raise AssertionError(f"projection exceeds target-blind allow-list: {projected}")
    if set(projected) & FORBIDDEN_VALUE_COLUMNS:
        raise AssertionError("target column entered target-blind projection")

    snapshot = pinned_snapshot_path()
    ordinal = 0
    for shard_index, expected in enumerate(load_full_manifest()):
        resolved = _attest_shard_pointer(snapshot / str(expected["path"]), expected)
        batches = pq.ParquetFile(resolved).iter_batches(
            batch_size=1,
            columns=list(projected),
            use_threads=False,
        )
        row_in_shard = 0
        while True:
            started = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                break
            wall_s = time.perf_counter() - started
            if batch.num_rows != 1 or tuple(batch.column_names) != projected:
                raise AssertionError("physical projection violated one-row contract")
            if set(batch.column_names) & FORBIDDEN_VALUE_COLUMNS:
                raise AssertionError("forbidden column reached a target-blind batch")
            yield ordinal, shard_index, row_in_shard, wall_s, batch
            ordinal += 1
            row_in_shard += 1
    if ordinal != int(FULL["n_mlps"]):
        raise AssertionError(f"full iterator yielded {ordinal} rows")


def _scalar(batch: pa.RecordBatch, name: str) -> object:
    return batch.column(batch.schema.get_field_index(name))[0].as_py()


def iter_full_identities() -> Iterator[FullIdentity]:
    """Decode only IDs, names, and seeds for split construction."""

    seen_ids: set[int] = set()
    for ordinal, shard_index, row_in_shard, _wall_s, batch in _iter_batches(
        IDENTITY_COLUMNS
    ):
        mlp_id = int(_scalar(batch, "mlp_id"))
        if mlp_id in seen_ids:
            raise AssertionError(f"duplicate full mlp_id: {mlp_id}")
        seen_ids.add(mlp_id)
        yield FullIdentity(
            ordinal=ordinal,
            shard_index=shard_index,
            row_in_shard=row_in_shard,
            mlp_id=mlp_id,
            mlp_name=str(_scalar(batch, "mlp_name")),
            mlp_seed=int(_scalar(batch, "mlp_seed")),
            decoded_columns=IDENTITY_COLUMNS,
        )


def _weights_view(batch: pa.RecordBatch) -> np.ndarray:
    storage = batch.column(batch.schema.get_field_index("weights"))
    storage = getattr(storage, "storage", storage)
    for _ in range(3):
        storage = storage.flatten()
    weights = storage.to_numpy(zero_copy_only=True).reshape(16, 1024, 1024)
    if weights.dtype != np.float32 or not weights.flags.c_contiguous:
        raise AssertionError("weights projection has unexpected dtype/layout")
    for layer in weights:
        if not np.isfinite(layer).all():
            raise AssertionError("non-finite full-split weight")
    weights.setflags(write=False)
    return weights


def iter_target_blind_full() -> Iterator[TargetBlindFullRow]:
    """Yield full rows through the exact four-column physical projection."""

    if WEIGHT_COLUMNS != ALLOWED_VALUE_COLUMNS:
        raise AssertionError("authority allow-list changed")
    seen_ids: set[int] = set()
    for ordinal, shard_index, row_in_shard, wall_s, batch in _iter_batches(
        WEIGHT_COLUMNS
    ):
        mlp_id = int(_scalar(batch, "mlp_id"))
        if mlp_id in seen_ids:
            raise AssertionError(f"duplicate full mlp_id: {mlp_id}")
        seen_ids.add(mlp_id)
        yield TargetBlindFullRow(
            ordinal=ordinal,
            shard_index=shard_index,
            row_in_shard=row_in_shard,
            mlp_id=mlp_id,
            mlp_name=str(_scalar(batch, "mlp_name")),
            mlp_seed=int(_scalar(batch, "mlp_seed")),
            weights=_weights_view(batch),
            source_read_wall_s=wall_s,
            decoded_columns=WEIGHT_COLUMNS,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-payloads", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    report = validate_pinned_full_snapshot(hash_payloads=args.hash_payloads)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, args.receipt)
    print(encoded, end="")


if __name__ == "__main__":
    main()
