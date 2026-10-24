#!/usr/bin/env python3
"""Lazy, identity-safe join for the full public residual experiment.

The observer-fit host trajectory is the accumulating 688-row exact-combo
scan.  Rows whose old exact-floor trigger visibly blew up are replaced by the
confirmed near-floor trajectory.  The replacement roster is intentionally
derived from the old scan's numerical result, rather than a hand-maintained
list, so newly discovered tails become mandatory as soon as their old result
and repaired trajectory exist.

No array in the 51 GiB feature store is copied by this loader.  Targets are
small enough to keep resident; host/local/sigma/weight arrays stay memory
mapped per network.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
FEATURE_ROOT = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows"
)
TARGET_ROOT = P2 / "data_gate/sealed_role_target_bundle"

DEFAULT_FIT_HOST = (
    P2
    / "public_observer_fit_candidate_truth_score_v1/runs/manual_rho_hybrid_floor"
)
DEFAULT_FIT_PATCH = P2 / "public_covariance_trigger_probe_v1/runs/near_floor"
DEFAULT_FIT_PATCH_MIN = (
    P2
    / "public_observer_fit_candidate_truth_score_v1/runs/manual_rho_hybrid_near_floor_min"
)

DEPTH = 16
WIDTH = 1024
LOCAL_FEATURES = 20
ROLE_ROWS = {"observer_fit": 688, "observer_selection": 88}
TAIL_MSE_THRESHOLD = 5.0e-7
OLD_ESTIMATOR_SHA256 = (
    "5e0fccfe8569b43aa85f77997c1dd890c5dd1173f114167e6fb50f7e1d85e5b5"
)
PATCH_ESTIMATOR_SHA256 = (
    "96a19116bee4a85f14f826c0a1634fe93bf39a21d78100eb46cd7f31add1e52b"
)
PATCH_MIN_ESTIMATOR_SHA256 = (
    "26aca6ed1fb52ec431132f87739fe729ec6855cc2fa2c80642fadd1c4c786e5d"
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_arrays(role: str) -> tuple[np.ndarray, np.ndarray]:
    expected = ROLE_ROWS[role]
    table = pq.read_table(
        TARGET_ROOT / f"{role}.parquet", columns=["mlp_id", "all_layer_means"]
    )
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(expected, DEPTH, WIDTH)
    if ids.dtype != np.int32 or targets.dtype != np.float32:
        raise AssertionError(f"{role}: target schema changed")
    return np.ascontiguousarray(ids), np.ascontiguousarray(targets)


def _metric(receipt: dict[str, object], name: str) -> float:
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict) or name not in metrics:
        raise AssertionError(f"result receipt lacks metrics.{name}")
    return float(metrics[name])


@dataclass(frozen=True)
class JoinedRow:
    role: str
    role_rank: int
    mlp_id: int
    host: np.ndarray
    raw_locals: np.ndarray
    sigma: np.ndarray
    target: np.ndarray
    weights_path: Path
    host_source: str
    old_final_mse: float
    old_max_layer_mse: float


class PublicResidualDataset:
    """Role-local, rank-indexed dataset with data-driven tail replacement."""

    def __init__(
        self,
        role: str = "observer_fit",
        *,
        host_root: Path | None = None,
        patch_root: Path | None = None,
        tail_threshold: float = TAIL_MSE_THRESHOLD,
        expected_host_sha256: str | None = OLD_ESTIMATOR_SHA256,
        expected_patch_sha256: str | None = PATCH_ESTIMATOR_SHA256,
    ) -> None:
        if role not in ROLE_ROWS:
            raise ValueError(f"unknown role {role!r}")
        if host_root is None and role == "observer_fit":
            host_root = DEFAULT_FIT_HOST
        if host_root is None:
            raise ValueError(f"{role} needs an explicit host_root")
        self.role = role
        self.total = ROLE_ROWS[role]
        self.host_root = Path(host_root)
        self.tail_threshold = float(tail_threshold)
        self.expected_host_sha256 = expected_host_sha256
        if role == "observer_fit" and patch_root is None:
            # Prefer the tighter final near-floor trigger when it has been run
            # for a tail, then fall back to the earlier confirmed repair.
            self.patch_sources = (
                (
                    DEFAULT_FIT_PATCH_MIN,
                    PATCH_MIN_ESTIMATOR_SHA256,
                    "near-floor-min-repair",
                ),
                (DEFAULT_FIT_PATCH, PATCH_ESTIMATOR_SHA256, "near-floor-repair"),
            )
        elif patch_root is not None:
            self.patch_sources = (
                (Path(patch_root), expected_patch_sha256, "explicit-repair"),
            )
        else:
            self.patch_sources = ()
        self.mlp_ids, self.targets = _target_arrays(role)

    def feature_dir(self, rank: int) -> Path:
        return FEATURE_ROOT / self.role / f"{rank:04d}"

    def _run_dir(self, root: Path, rank: int) -> Path:
        return root / f"row_{rank:04d}"

    def base_complete(self, rank: int) -> bool:
        row = self._run_dir(self.host_root, rank)
        return (row / "result.json").is_file() and (row / "q24_output.npy").is_file()

    def available_ranks(self) -> tuple[int, ...]:
        return tuple(rank for rank in range(self.total) if self.base_complete(rank))

    def missing_ranks(self) -> tuple[int, ...]:
        available = set(self.available_ranks())
        return tuple(rank for rank in range(self.total) if rank not in available)

    def old_receipt(self, rank: int) -> dict[str, object]:
        return _json(self._run_dir(self.host_root, rank) / "result.json")

    def is_tail(self, rank: int) -> bool:
        receipt = self.old_receipt(rank)
        return max(
            _metric(receipt, "final_layer_mse"),
            _metric(receipt, "max_layer_mse"),
        ) > self.tail_threshold

    def tail_roster(self) -> tuple[int, ...]:
        """All currently completed old-trigger rows above the fixed threshold."""
        return tuple(rank for rank in self.available_ranks() if self.is_tail(rank))

    def patch_complete(self, rank: int) -> bool:
        return self._patch_choice(rank) is not None

    def _patch_choice(
        self, rank: int
    ) -> tuple[Path, str | None, str] | None:
        for root, expected_sha256, label in self.patch_sources:
            row = self._run_dir(root, rank)
            if (row / "result.json").is_file() and (row / "q24_output.npy").is_file():
                return root, expected_sha256, label
        return None

    def unpatched_tails(self) -> tuple[int, ...]:
        return tuple(rank for rank in self.tail_roster() if not self.patch_complete(rank))

    def _validate_receipt(
        self,
        rank: int,
        receipt: dict[str, object],
        *,
        expected_sha256: str | None,
    ) -> None:
        expected_id = int(self.mlp_ids[rank])
        if receipt.get("role") != self.role:
            raise AssertionError(f"rank {rank}: result role mismatch")
        if int(receipt.get("role_rank", -1)) != rank:
            raise AssertionError(f"rank {rank}: result role_rank mismatch")
        if int(receipt.get("mlp_id", -1)) != expected_id:
            raise AssertionError(f"rank {rank}: result mlp_id mismatch")
        if expected_sha256 is not None and receipt.get("estimator_sha256") != expected_sha256:
            raise AssertionError(f"rank {rank}: estimator hash mismatch")
        if receipt.get("exception_type") is not None:
            raise AssertionError(f"rank {rank}: estimator failed")

    def _host_path(self, rank: int) -> tuple[Path, str]:
        old = self.old_receipt(rank)
        self._validate_receipt(
            rank, old, expected_sha256=self.expected_host_sha256
        )
        if not self.is_tail(rank):
            return self._run_dir(self.host_root, rank) / "q24_output.npy", "old-trigger"
        choice = self._patch_choice(rank)
        if choice is None:
            raise FileNotFoundError(
                f"rank {rank} is a data-derived tail but lacks a repaired trajectory"
            )
        patch_root, expected_sha256, label = choice
        patch_dir = self._run_dir(patch_root, rank)
        repaired = _json(patch_dir / "result.json")
        self._validate_receipt(
            rank, repaired, expected_sha256=expected_sha256
        )
        if _metric(repaired, "max_layer_mse") >= _metric(old, "max_layer_mse"):
            raise AssertionError(f"rank {rank}: proposed repair did not improve the tail")
        return patch_dir / "q24_output.npy", label

    def load(self, rank: int) -> JoinedRow:
        if rank < 0 or rank >= self.total:
            raise IndexError(rank)
        if not self.base_complete(rank):
            raise FileNotFoundError(f"rank {rank}: old-trigger host is incomplete")
        feature_dir = self.feature_dir(rank)
        manifest = _json(feature_dir / "row_manifest.json")
        expected_id = int(self.mlp_ids[rank])
        if manifest.get("role") != self.role:
            raise AssertionError(f"rank {rank}: feature role mismatch")
        if int(manifest.get("role_rank", -1)) != rank:
            raise AssertionError(f"rank {rank}: feature role_rank mismatch")
        if int(manifest.get("mlp_id", -1)) != expected_id:
            raise AssertionError(f"rank {rank}: feature mlp_id mismatch")

        host_path, source = self._host_path(rank)
        host = np.load(host_path, mmap_mode="r", allow_pickle=False)
        raw = np.load(
            feature_dir / "cp1792_raw_locals.npy", mmap_mode="r", allow_pickle=False
        )
        sigma = np.load(
            feature_dir / "cp1792_sigma.npy", mmap_mode="r", allow_pickle=False
        )
        weights_path = feature_dir / "weights.npy"
        if host.dtype != np.float32 or host.shape != (DEPTH, WIDTH):
            raise AssertionError(f"rank {rank}: host schema changed")
        if raw.dtype != np.float32 or raw.shape != (DEPTH, WIDTH, LOCAL_FEATURES):
            raise AssertionError(f"rank {rank}: raw-local schema changed")
        if sigma.dtype != np.float32 or sigma.shape != (DEPTH, WIDTH):
            raise AssertionError(f"rank {rank}: sigma schema changed")
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        old = self.old_receipt(rank)
        return JoinedRow(
            role=self.role,
            role_rank=rank,
            mlp_id=expected_id,
            host=host,
            raw_locals=raw,
            sigma=sigma,
            target=self.targets[rank],
            weights_path=weights_path,
            host_source=source,
            old_final_mse=_metric(old, "final_layer_mse"),
            old_max_layer_mse=_metric(old, "max_layer_mse"),
        )

    def iter_rows(self, ranks: Iterator[int] | tuple[int, ...]) -> Iterator[JoinedRow]:
        for rank in ranks:
            yield self.load(int(rank))

    def readiness(self) -> dict[str, object]:
        available = self.available_ranks()
        tails = self.tail_roster()
        unpatched = self.unpatched_tails()
        return {
            "role": self.role,
            "expected": self.total,
            "available": len(available),
            "missing": list(self.missing_ranks()),
            "tail_threshold": self.tail_threshold,
            "data_derived_tail_ranks": list(tails),
            "repaired_tail_ranks": [rank for rank in tails if self.patch_complete(rank)],
            "unpatched_tail_ranks": list(unpatched),
            "ready": len(available) == self.total and not unpatched,
        }
