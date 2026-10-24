#!/usr/bin/env python3
"""Read-only V4 provider for a finalized target-blind feature store.

The provider has no data-reader dependency.  It serves only the exact sealed
weights and host tapes, reconstructing Hermite edge matrices deterministically
from the frozen weights and checking their per-row V4 array hash.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

_SOURCE_DIR = Path(__file__).resolve().parent
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

import contract as _bound_contract
if Path(_bound_contract.__file__).resolve() != _SOURCE_DIR / "contract.py":
    raise ImportError("sealed provider resolved a different contract module")

from contract import (
    CAPS,
    DEPTH,
    FINAL_SCHEMA,
    FROZEN_BINDINGS,
    HEX64,
    LOCAL_FEATURES,
    ROLE_BINDINGS,
    ROLE_ORDER,
    SOURCE_MANIFEST_PATH,
    WIDTH,
    array_payload_sha256,
    expected_row,
    row_relative_path,
    sha256_file,
    validate_row_directory,
)


class SealedFeatureStore:
    def __init__(self, manifest_path: Path, manifest: dict[str, Any]):
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = manifest
        self._verified = False

    def _entry(self, role: str, role_rank: int) -> tuple[Path, dict[str, Any]]:
        if role not in ROLE_ORDER:
            raise PermissionError("only observer_fit/observer_selection features are released")
        if role_rank < 0 or role_rank >= ROLE_BINDINGS[role]["rows"]:
            raise IndexError("feature role rank out of range")
        entry = self.manifest["roles"][role]["row_manifests"][role_rank]
        expected_path = row_relative_path(role, role_rank) / "row_manifest.json"
        if entry.get("role_rank") != role_rank or entry.get("path") != str(expected_path):
            raise AssertionError("feature manifest row address changed")
        row_dir = self.root / expected_path.parent
        return row_dir, entry

    def verify_all_artifact_hashes_after_claim(self) -> dict[str, Any]:
        """Hash and schema-check every stored row; no target artifact is involved."""

        rows = 0
        bytes_hashed = 0
        for role in ROLE_ORDER:
            for role_rank in range(ROLE_BINDINGS[role]["rows"]):
                row_dir, entry = self._entry(role, role_rank)
                manifest_path = row_dir / "row_manifest.json"
                if sha256_file(manifest_path) != entry["sha256"]:
                    raise AssertionError("row manifest hash differs from final store")
                row_manifest = validate_row_directory(row_dir, role=role, role_rank=role_rank)
                bytes_hashed += manifest_path.stat().st_size
                bytes_hashed += sum(int(record["bytes"]) for record in row_manifest["files"].values())
                rows += 1
        if rows != 776:
            raise AssertionError("sealed provider did not verify exactly 776 rows")
        self._verified = True
        return {
            "status": "PASS_ALL_TARGET_BLIND_FEATURE_ARTIFACT_HASHES_AFTER_CLAIM",
            "rows_verified": rows,
            "bytes_hashed": bytes_hashed,
            "role_target_artifacts_opened_or_hashed": 0,
            "target_values_decoded": 0,
        }

    @staticmethod
    def _load(path: Path, shape: tuple[int, ...]) -> np.ndarray:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype != np.dtype("<f4") or tuple(value.shape) != shape:
            raise AssertionError(f"sealed feature array schema changed: {path.name}")
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise AssertionError(f"sealed feature array invalid: {path.name}")
        return value

    @staticmethod
    def _edge_matrices(weights: np.ndarray) -> np.ndarray:
        """Exact NumPy spelling of frozen V3 `estimator._edge_matrices`."""

        result = np.empty((DEPTH, 4, WIDTH, WIDTH), dtype=np.float32)
        for layer0 in range(DEPTH):
            weights32 = np.asarray(weights[layer0], dtype=np.float32)
            square = weights32 * weights32
            result[layer0, 0] = np.asarray(
                weights32 * (1.0 / math.sqrt(2.0)), dtype=np.float32
            )
            result[layer0, 1] = np.asarray(
                (square - (2.0 / WIDTH)) * math.sqrt(WIDTH / 8.0), dtype=np.float32
            )
            result[layer0, 2] = np.asarray(
                (square * weights32) * (WIDTH / math.sqrt(120.0)), dtype=np.float32
            )
            result[layer0, 3] = np.asarray(
                (square * square - (12.0 / (WIDTH * WIDTH)))
                * (WIDTH**1.5 / math.sqrt(1536.0)),
                dtype=np.float32,
            )
        if not result.flags.c_contiguous or not np.isfinite(result).all():
            raise FloatingPointError("reconstructed edge tape is invalid")
        return result

    def get_normalization_row(self, cap: int, role_rank: int) -> dict[str, Any]:
        if not self._verified:
            raise PermissionError("artifact hashes must be verified after the training claim")
        if cap not in CAPS:
            raise PermissionError("unknown host cap")
        role = "observer_fit"
        row_dir, _ = self._entry(role, role_rank)
        expected = expected_row(role, role_rank)
        raw = self._load(
            row_dir / f"cp{cap}_raw_locals.npy", (DEPTH, WIDTH, LOCAL_FEATURES)
        )
        return {"mlp_id": int(expected["mlp_id"]), "raw_locals": raw}

    def get_row(self, cap: int, role: str, role_rank: int) -> dict[str, Any]:
        if not self._verified:
            raise PermissionError("artifact hashes must be verified after the training claim")
        if cap not in CAPS:
            raise PermissionError("unknown host cap")
        row_dir, _ = self._entry(role, role_rank)
        expected = expected_row(role, role_rank)
        row_manifest = json.loads((row_dir / "row_manifest.json").read_text(encoding="utf-8"))
        weights = self._load(row_dir / "weights.npy", (DEPTH, WIDTH, WIDTH))
        edges = self._edge_matrices(weights)
        expected_edge_hash = row_manifest["hosts"][str(cap)]["edges_sha256"]
        if not HEX64.fullmatch(expected_edge_hash) or array_payload_sha256(edges) != expected_edge_hash:
            raise AssertionError("reconstructed edge tape differs from frozen V4 host semantics")
        raw = self._load(
            row_dir / f"cp{cap}_raw_locals.npy", (DEPTH, WIDTH, LOCAL_FEATURES)
        )
        host = self._load(row_dir / f"cp{cap}_host_rows.npy", (DEPTH, WIDTH))
        sigma = self._load(row_dir / f"cp{cap}_sigma.npy", (DEPTH, WIDTH))
        if np.any(host < 0) or np.any(sigma <= 0):
            raise FloatingPointError("sealed host/sigma sign contract failed")
        return {
            "mlp_id": int(expected["mlp_id"]),
            "raw_locals": raw,
            "edge_matrices": edges,
            "host_rows": host,
            "sigma": sigma,
            "entering_cp_state_hashes": tuple(
                row_manifest["hosts"][str(cap)]["entering_cp_state_hashes"]
            ),
            "host_latch": (True,) * DEPTH,
        }


def _validate_final_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FINAL_SCHEMA or manifest.get("status") != "SEALED_TARGET_BLIND_FEATURE_STORE":
        raise PermissionError("feature store is not final sealed")
    if manifest.get("provider_source_sha256") != sha256_file(Path(__file__)):
        raise PermissionError("feature manifest binds another provider source")
    if manifest.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise PermissionError("feature manifest binds another source freeze")
    if manifest.get("host_caps") != list(CAPS):
        raise PermissionError("feature host caps changed")
    if manifest.get("fields") != [
        "mlp_id", "raw_locals", "edge_matrices", "host_rows", "sigma",
        "entering_cp_state_hashes", "host_latch",
    ]:
        raise PermissionError("feature interface fields changed")
    if manifest.get("target_columns_decoded") != 0:
        raise PermissionError("feature manifest claims target decoding")
    if manifest.get("raw_full_dataset_target_projection") != "forbidden":
        raise PermissionError("feature manifest permits a target projection")
    if manifest.get("target_blind_full_source_sha256") != FROZEN_BINDINGS["target_blind_full_reader"]["sha256"]:
        raise PermissionError("feature manifest used another full reader")
    if manifest.get("full_cache_audit_sha256") != FROZEN_BINDINGS["full_cache_audit"]["sha256"]:
        raise PermissionError("feature manifest used another full cache")
    if manifest.get("independent_feature_store_audit_status") != "PASS":
        raise PermissionError("feature store lacks independent artifact PASS")
    if not HEX64.fullmatch(str(manifest.get("independent_feature_store_audit_sha256", ""))):
        raise PermissionError("feature store audit digest is malformed")
    roles = manifest.get("roles", {})
    if set(roles) != set(ROLE_ORDER):
        raise PermissionError("feature roles changed")
    for role in ROLE_ORDER:
        expected = ROLE_BINDINGS[role]
        record = roles[role]
        if record.get("rows") != expected["rows"]:
            raise PermissionError("feature role count changed")
        if record.get("ordered_mlp_ids_sha256") != expected["ordered_mlp_ids_sha256"]:
            raise PermissionError("feature role order changed")
        if len(record.get("row_manifests", [])) != expected["rows"]:
            raise PermissionError("feature row manifest membership changed")
    return manifest


def open_feature_store(manifest_path: Path) -> SealedFeatureStore:
    """Open metadata only; artifact hashing is explicit and post-claim."""

    path = Path(manifest_path)
    if path.is_symlink() or not path.is_file():
        raise PermissionError("feature manifest must be a regular file")
    manifest = _validate_final_manifest(path)
    return SealedFeatureStore(path, manifest)
