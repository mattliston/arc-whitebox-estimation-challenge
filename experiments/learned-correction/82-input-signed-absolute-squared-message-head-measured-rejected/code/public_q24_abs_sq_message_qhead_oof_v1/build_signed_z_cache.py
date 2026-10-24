#!/usr/bin/env python3
"""Materialize the already-tested signed raw20 recurrence without targets.

The cache is a fixed feature transform of native raw20 rows and weights.  Its
normalization uses the all-688 raw sufficient statistics only; no target,
host, prediction, fold, or score is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PILOT_ROWS = P2 / "public_q24_native_transport_energy_pilot_v1/runs/native_rows_v1"
FINAL_ROWS = P2 / "public_q24_native_transport_energy_final_v1/runs/native_rows_remaining_v1"
WEIGHT_ROWS = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
RAW_STATS = P2 / "public_full688_dynamic_multistate_residual_cv_v1/cache/raw_sufficient_stats.npz"
OUTPUT_ROWS = HERE / "cache" / "signed_rows"

ROWS = 688
DEPTH = 16
WIDTH = 1024
RAW_FEATURES = 20
LAYERS = tuple(range(10, 16))


def sha256_file(path: Path, block_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def native_row_dir(rank: int) -> Path:
    name = f"row_{rank:04d}"
    hits = [path for path in (PILOT_ROWS / name, FINAL_ROWS / name) if (path / "raw20.npy").is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(f"rank {rank}: native raw20 row is ambiguous: {hits}")
    return hits[0]


def global_raw_moments() -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(RAW_STATS, allow_pickle=False)
    sums = payload["sum"]
    sums2 = payload["sum2"]
    if sums.dtype != np.float64 or sums.shape != (ROWS, DEPTH, RAW_FEATURES - 1):
        raise ValueError(f"raw sufficient-stat schema changed: {sums.shape} {sums.dtype}")
    if sums2.dtype != np.float64 or sums2.shape != sums.shape:
        raise ValueError("raw square-stat schema changed")
    count = float(ROWS * WIDTH)
    centers = sums.sum(axis=0, dtype=np.float64) / count
    second = sums2.sum(axis=0, dtype=np.float64) / count
    variance = np.maximum(second - centers * centers, (2.0**-20) ** 2)
    return centers, np.sqrt(variance)


def feature_map(
    raw20: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    if raw20.shape != (DEPTH, WIDTH, RAW_FEATURES):
        raise ValueError(f"unexpected raw shape {raw20.shape}")
    if weights.shape != (DEPTH, WIDTH, WIDTH):
        raise ValueError(f"unexpected weight shape {weights.shape}")
    tail = np.clip(
        (raw20[..., 1:].astype(np.float64) - centers[:, None, :]) / scales[:, None, :],
        -8.0,
        8.0,
    )
    standard = np.concatenate((raw20[..., :1].astype(np.float64), tail), axis=-1)
    state = standard[2].copy()
    output = np.empty((len(LAYERS), WIDTH, RAW_FEATURES), dtype=np.float32)
    for layer in range(3, DEPTH):
        state = (
            raw20[layer, :, 5, None].astype(np.float64)
            * (weights[layer].T.astype(np.float64) @ state)
            + standard[layer]
        )
        if layer in LAYERS:
            output[LAYERS.index(layer)] = state.astype(np.float32)
    if not np.isfinite(output).all():
        raise FloatingPointError("nonfinite signed-message output")
    return output


def self_test(seed: int = 20260831) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    n, p = 13, 5
    x = rng.normal(size=(n, p))
    w = rng.normal(size=(n, n))
    phi = rng.normal(size=n)
    source_perm = rng.permutation(n)
    target_perm = rng.permutation(n)
    base = phi[:, None] * (w.T @ x)
    transformed = phi[target_perm, None] * (
        w[source_perm][:, target_perm].T @ x[source_perm]
    )
    error = float(np.max(np.abs(transformed - base[target_perm])))
    return {"permutation_equivariance_error": error, "pass": error <= 2.0e-14}


def build_rank(rank: int, centers: np.ndarray, scales: np.ndarray, overwrite: bool) -> dict[str, object]:
    raw_path = native_row_dir(rank) / "raw20.npy"
    weight_path = WEIGHT_ROWS / f"{rank:04d}" / "weights.npy"
    output_dir = OUTPUT_ROWS / f"{rank:04d}"
    output_path = output_dir / "signed_z.npy"
    receipt_path = output_dir / "receipt.json"
    if output_path.is_file() and receipt_path.is_file() and not overwrite:
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    raw20 = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    weights = np.load(weight_path, mmap_mode="r", allow_pickle=False)
    output = feature_map(raw20, weights, centers, scales)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output, allow_pickle=False)
    receipt: dict[str, object] = {
        "schema": "q24-target-blind-signed-raw20-message-row-v1",
        "rank": rank,
        "target_accessed": False,
        "recurrence": "Z_l=raw20[l,:,5]*(W_l.T@Z_(l-1))+standardized_raw20_l",
        "layers_zero_based": list(LAYERS),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": bool(np.isfinite(output).all()),
        "input_raw20_sha256": sha256_file(raw_path),
        "input_weights_sha256": sha256_file(weight_path),
        "output_bytes_sha256": sha256_array(output),
        "output_file_sha256": sha256_file(output_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_ranks(spec: str) -> list[int]:
    result: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_text, hi_text = token.split("-", 1)
            result.update(range(int(lo_text), int(hi_text) + 1))
        else:
            result.add(int(token))
    ranks = sorted(result)
    if not ranks or ranks[0] < 0 or ranks[-1] >= ROWS:
        raise ValueError("ranks must be a nonempty subset of 0..687")
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", default="35")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    test = self_test()
    if not test["pass"]:
        raise AssertionError(test)
    if args.self_test_only:
        print(json.dumps(test, indent=2, sort_keys=True))
        return
    ranks = list(range(ROWS)) if args.all else parse_ranks(args.ranks)
    centers, scales = global_raw_moments()
    receipts = [build_rank(rank, centers, scales, args.overwrite) for rank in ranks]
    summary = {
        "schema": "q24-target-blind-signed-raw20-message-cache-v1",
        "target_accessed": False,
        "ranks": ranks,
        "row_count": len(receipts),
        "self_test": test,
        "shape_per_row": [len(LAYERS), WIDTH, RAW_FEATURES],
        "raw_stats_sha256": sha256_file(RAW_STATS),
        "receipts": receipts,
    }
    summary_path = HERE / "SIGNED_CACHE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "receipts"}, indent=2))


if __name__ == "__main__":
    main()
