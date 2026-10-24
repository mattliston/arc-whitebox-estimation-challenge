#!/Users/matt/anaconda3/bin/python3
"""Extract the already-computed Q24 covariance state for one target-free row.

The Q24 closure computes every full pre/post covariance internally.  This
wrapper records those values without changing its numerical path, and checks
that the returned native host tape matches the previously sealed row.  It is
an observer-fit diagnostic artifact, not an estimator or a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py"
FEATURE_ROOT = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
NATIVE_ROOT = (
    P2
    / "public_q24_native_transport_energy_final_v1"
    / "runs/native_rows_remaining_v1"
)
OUTPUT_ROOT = HERE / "runs"
DEPTH = 16
WIDTH = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module():
    name = f"q24_covariance_capture_{os.getpid()}"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    if specification is None or specification.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    arguments = parser.parse_args()
    rank = int(arguments.rank)
    feature_dir = FEATURE_ROOT / f"{rank:04d}"
    native_dir = NATIVE_ROOT / f"row_{rank:04d}"
    output_dir = OUTPUT_ROOT / f"row_{rank:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "NATIVE_Q24_COVARIANCE_TAPE.npz"
    receipt_path = output_dir / "NATIVE_Q24_COVARIANCE_RECEIPT.json"
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("native Q24 covariance tape already exists")

    manifest = json.loads((feature_dir / "row_manifest.json").read_text())
    weights_path = feature_dir / "weights.npy"
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.float32 or weights.shape != (DEPTH, WIDTH, WIDTH):
        raise AssertionError("weight schema changed")
    mlp = MLP(
        width=WIDTH,
        depth=DEPTH,
        weights=[weights[layer] for layer in range(DEPTH)],
        seed=int(derive_estimator_seed(int(manifest["mlp_seed"]))),
        name=f"target-free-q24-covariance-rank-{rank:04d}",
    )
    mlp.validate()
    module = load_module()
    pre_covariances: list[np.ndarray] = []
    post_covariances: list[np.ndarray] = []

    original_initialize = module._initialize_h1
    original_reset = module._full_relu_reset

    def capture_initialize(*args, **kwargs):
        pre_covariances.append(np.array(args[0], dtype=np.float64, copy=True))
        result = original_initialize(*args, **kwargs)
        post_covariances.append(np.array(result[1], dtype=np.float64, copy=True))
        return result

    def capture_reset(*args, **kwargs):
        pre_covariances.append(np.array(args[1], dtype=np.float64, copy=True))
        result = original_reset(*args, **kwargs)
        post_covariances.append(np.array(result[1], dtype=np.float64, copy=True))
        return result

    module._initialize_h1 = capture_initialize
    module._full_relu_reset = capture_reset
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(
            flop_budget=2**41, wall_time_limit_s=120.0, quiet=True
        ) as context:
            host_value, raw_value, sigma_value = module._predict(
                mlp, return_native_tape=True
            )
    elapsed = time.perf_counter() - started
    host = np.ascontiguousarray(np.asarray(host_value), dtype=np.float32)
    raw = np.ascontiguousarray(np.asarray(raw_value), dtype=np.float32)
    sigma = np.ascontiguousarray(np.asarray(sigma_value), dtype=np.float32)
    if len(pre_covariances) != DEPTH - 1 or len(post_covariances) != DEPTH - 1:
        raise AssertionError((len(pre_covariances), len(post_covariances)))
    last_weight = np.asarray(weights[-1], dtype=np.float64)
    final_pre = last_weight.T @ post_covariances[-1] @ last_weight
    pre_covariances.append(np.asarray(final_pre, dtype=np.float64))
    pre_covariance = np.ascontiguousarray(np.stack(pre_covariances), dtype=np.float32)
    post_covariance = np.ascontiguousarray(np.stack(post_covariances), dtype=np.float32)
    sealed_host = np.load(native_dir / "host.npy", allow_pickle=False)
    sealed_raw = np.load(native_dir / "raw20.npy", allow_pickle=False)
    sealed_sigma = np.load(native_dir / "sigma.npy", allow_pickle=False)
    host_max = float(np.max(np.abs(host.astype(np.float64) - sealed_host)))
    raw_max = float(np.max(np.abs(raw.astype(np.float64) - sealed_raw)))
    sigma_max = float(np.max(np.abs(sigma.astype(np.float64) - sealed_sigma)))
    replay_tolerance = 5.0e-7
    if host_max > replay_tolerance or raw_max > replay_tolerance or sigma_max > replay_tolerance:
        raise AssertionError((host_max, raw_max, sigma_max))
    diagonal_sigma_error = float(
        np.max(
            np.abs(
                np.sqrt(
                    np.maximum(np.diagonal(pre_covariance, axis1=1, axis2=2), 0.0)
                ).astype(np.float64)
                - sigma.astype(np.float64)
            )
        )
    )
    np.savez_compressed(
        output_path,
        pre_covariance=pre_covariance,
        post_covariance=post_covariance,
    )
    receipt = {
        "schema": "native-q24-covariance-tape-v1",
        "role": "observer_fit",
        "role_rank": rank,
        "mlp_id": int(manifest["mlp_id"]),
        "target_columns_opened": [],
        "source_sha256": sha256(SOURCE),
        "weights_sha256": sha256(weights_path),
        "sealed_native_host_numerical_match": True,
        "sealed_native_host_replay_tolerance": replay_tolerance,
        "host_max_abs_difference": host_max,
        "raw20_max_abs_difference": raw_max,
        "sigma_max_abs_difference": sigma_max,
        "pre_covariance_shape": list(pre_covariance.shape),
        "post_covariance_shape": list(post_covariance.shape),
        "pre_covariance_diagonal_sigma_max_abs_error": diagonal_sigma_error,
        "flops_used_for_replay": int(context.flops_used),
        "wall_seconds": elapsed,
        "artifact": output_path.name,
        "artifact_sha256": sha256(output_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
