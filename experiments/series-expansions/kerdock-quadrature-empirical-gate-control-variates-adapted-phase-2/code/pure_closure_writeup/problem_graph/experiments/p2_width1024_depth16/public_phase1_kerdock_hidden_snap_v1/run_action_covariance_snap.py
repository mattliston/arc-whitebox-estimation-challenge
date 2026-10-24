#!/Users/matt/anaconda3/bin/python3
"""Rank-0 action-guided covariance-snap discriminator on the fixed MUB cloud.

Starting from the successful all-layer mean/diagonal-variance snap, this runner
aligns each block's pre-ReLU covariance in a frozen suffix-action subspace to
the covariance already computed by the target-free native Q24 closure.  It
screens nested ranks 16/32/64 and preserves split-block diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import run_discriminator as base


HERE = Path(__file__).resolve().parent
WIDTH = base.WIDTH
DEPTH = base.DEPTH
BLOCKS = base.BLOCKS
PATHS_PER_BLOCK = base.PATHS_PER_BLOCK
RANKS = (16, 32, 64)
ACTION_SEED = 0xC0A4_2026


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def action_bases(
    weights: np.ndarray, phi: np.ndarray, maximum_rank: int
) -> tuple[np.ndarray, dict[str, object]]:
    """One-pass randomized range bases for each preactivation suffix map.

    For row perturbations at preactivation layer l, the frozen suffix map is
    A_l=D_l W_{l+1} D_{l+1} ... W_15 D_15.  Starting from fixed Rademacher
    endpoint probes Omega, this computes A_l Omega backwards and takes its
    left singular vectors.  No target values enter the construction.
    """
    rng = np.random.default_rng(ACTION_SEED)
    probe_count = maximum_rank + 16
    propagated = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float64),
        size=(WIDTH, probe_count),
    ) / math.sqrt(float(probe_count))
    bases = np.empty((DEPTH, WIDTH, maximum_rank), dtype=np.float32)
    singular_values = np.empty((DEPTH, maximum_rank), dtype=np.float64)
    for layer in range(DEPTH - 1, -1, -1):
        propagated = phi[layer, :, None].astype(np.float64) * propagated
        left, values, _ = np.linalg.svd(propagated, full_matrices=False)
        bases[layer] = np.asarray(left[:, :maximum_rank], dtype=np.float32)
        singular_values[layer] = values[:maximum_rank]
        if layer > 0:
            propagated = np.asarray(weights[layer], dtype=np.float64) @ propagated
            scale = float(np.linalg.norm(propagated, ord="fro"))
            if not math.isfinite(scale) or scale <= 0.0:
                raise FloatingPointError((layer, scale))
            propagated /= scale
    return bases, {
        "definition": "left singular vectors of one-pass A_l*Omega, A_l=D_l W_(l+1)...W15 D15",
        "probe_distribution": "fixed Rademacher",
        "seed": ACTION_SEED,
        "probe_count": probe_count,
        "maximum_rank": maximum_rank,
        "singular_values": singular_values.tolist(),
        "target_columns_opened": [],
    }


def symmetric_root(matrix: torch.Tensor, inverse: bool) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(
        0.5 * (matrix.to(torch.float64) + matrix.to(torch.float64).T)
    )
    floor = max(float(torch.max(values)) * 1.0e-10, 1.0e-14)
    values = torch.clamp(values, min=floor)
    if inverse:
        values = torch.rsqrt(values)
    else:
        values = torch.sqrt(values)
    return (vectors * values[None, :]) @ vectors.T


def align_projected_covariance(
    values: torch.Tensor,
    target_mean: torch.Tensor,
    target_covariance: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Whiten/color the U projection and lift its correction to full space."""
    centered = values - target_mean
    projected = centered @ directions
    empirical = (projected.T @ projected) / float(values.shape[0])
    target = directions.T.to(torch.float64) @ (
        target_covariance.to(torch.float64) @ directions.to(torch.float64)
    )
    inverse_empirical_root = symmetric_root(empirical, inverse=True)
    target_root = symmetric_root(target, inverse=False)
    transform = inverse_empirical_root @ target_root
    recolored = projected.to(torch.float64) @ transform
    corrected = values + (
        (recolored.to(torch.float32) - projected) @ directions.T
    )
    corrected = corrected + (target_mean - torch.mean(corrected, dim=0))
    check_centered = corrected - target_mean
    check_projected = check_centered @ directions
    achieved = (check_projected.T @ check_projected) / float(values.shape[0])
    denominator = max(float(torch.linalg.norm(target)), 1.0e-30)
    return corrected, {
        "relative_projected_covariance_error_before": float(
            torch.linalg.norm(empirical.to(torch.float64) - target) / denominator
        ),
        "relative_projected_covariance_error_after": float(
            torch.linalg.norm(achieved.to(torch.float64) - target) / denominator
        ),
        "maximum_mean_error_after": float(
            torch.max(torch.abs(torch.mean(corrected, dim=0) - target_mean))
        ),
    }


def propagate_arm(
    points: np.ndarray,
    weights_array: np.ndarray,
    host: np.ndarray,
    sigma: np.ndarray,
    pre_covariance: np.ndarray,
    action: np.ndarray,
    rank: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    torch.set_num_threads(12)
    weights = torch.from_numpy(np.ascontiguousarray(weights_array, dtype=np.float32))
    values = torch.from_numpy(np.array(points, dtype=np.float32, copy=True))
    host_t = torch.from_numpy(np.ascontiguousarray(host, dtype=np.float32))
    sigma_t = torch.from_numpy(np.ascontiguousarray(sigma, dtype=np.float32))
    covariance_t = torch.from_numpy(
        np.ascontiguousarray(pre_covariance, dtype=np.float32)
    )
    directions_t = torch.from_numpy(
        np.ascontiguousarray(action[:, :, :rank], dtype=np.float32)
    )
    mean_blocks = torch.empty((BLOCKS, DEPTH, WIDTH), dtype=torch.float32)
    gate_blocks = torch.empty_like(mean_blocks)
    alignment_rows: list[dict[str, object]] = []
    radial_factor = base.expected_chi(WIDTH) ** 2 / WIDTH
    exact_mu1, exact_var2, exact_receipt = base.exact_h1_h2_snap(weights)
    started = time.perf_counter()
    with torch.no_grad():
        preactivation = values @ weights[0]
        gate = preactivation > 0.0
        values = torch.relu_(preactivation)
        for block in range(BLOCKS):
            block_slice = slice(
                block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK
            )
            raw_mean = torch.mean(values[block_slice], dim=0)
            values[block_slice].add_(exact_mu1 - raw_mean)
            mean_blocks[block, 0] = exact_mu1
            gate_blocks[block, 0] = torch.mean(
                gate[block_slice].to(torch.float32), dim=0
            )

        for layer in range(1, DEPTH):
            preactivation = values @ weights[layer]
            target_mean = (
                exact_mu1 @ weights[layer]
                if layer == 1
                else host_t[layer - 1] @ weights[layer]
            )
            target_variance = (
                exact_var2
                if layer == 1
                else torch.clamp(
                    radial_factor
                    * (sigma_t[layer] * sigma_t[layer] + target_mean * target_mean)
                    - target_mean * target_mean,
                    min=2.0**-40,
                )
            )
            target_cov = (
                radial_factor * covariance_t[layer].to(torch.float64)
                + (radial_factor - 1.0)
                * torch.outer(target_mean.to(torch.float64), target_mean.to(torch.float64))
            )
            layer_diagnostics: list[dict[str, float]] = []
            for block in range(BLOCKS):
                block_slice = slice(
                    block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK
                )
                block_pre = preactivation[block_slice]
                empirical_mean = torch.mean(block_pre, dim=0)
                centered = block_pre - empirical_mean
                empirical_variance = torch.mean(centered * centered, dim=0)
                block_pre = target_mean + centered * torch.sqrt(
                    target_variance / torch.clamp(empirical_variance, min=2.0**-40)
                )
                block_pre, diagnostics = align_projected_covariance(
                    block_pre,
                    target_mean,
                    target_cov,
                    directions_t[layer],
                )
                diagnostics["maximum_relative_diagonal_variance_error_after"] = float(
                    torch.max(
                        torch.abs(
                            torch.var(block_pre, dim=0, correction=0) / target_variance
                            - 1.0
                        )
                    )
                )
                layer_diagnostics.append(diagnostics)
                preactivation[block_slice] = block_pre
            gate = preactivation > 0.0
            values = torch.relu_(preactivation)
            for block in range(BLOCKS):
                block_slice = slice(
                    block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK
                )
                mean_blocks[block, layer] = torch.mean(values[block_slice], dim=0)
                gate_blocks[block, layer] = torch.mean(
                    gate[block_slice].to(torch.float32), dim=0
                )
            alignment_rows.append(
                {
                    "layer_zero_based": layer,
                    "blocks": layer_diagnostics,
                }
            )
    n = base.TOTAL_PATHS
    projection_reconstruction = 4 * n * WIDTH * rank * (DEPTH - 1)
    projected_covariance_ops = 4 * n * rank * rank * (DEPTH - 1)
    host_projection_ops = 4 * WIDTH * WIDTH * rank * (DEPTH - 1)
    return {
        "mean_blocks": mean_blocks.numpy(),
        "gate_blocks": gate_blocks.numpy(),
    }, {
        "rank": rank,
        "wall_seconds": time.perf_counter() - started,
        "exact_h1_h2": exact_receipt,
        "alignment": alignment_rows,
        "flop_upper_bound": {
            "path_projection_plus_reconstruction": projection_reconstruction,
            "path_projected_covariance_and_recolor": projected_covariance_ops,
            "host_covariance_projection": host_projection_ops,
            "total_excluding_small_r_cubic_eigendecompositions": (
                projection_reconstruction
                + projected_covariance_ops
                + host_projection_ops
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    arguments = parser.parse_args()
    role_rank = int(arguments.rank)
    if role_rank != 0:
        raise ValueError("this protected discriminator is frozen to observer_fit rank 0")
    feature_dir = base.FEATURE_ROOT / f"{role_rank:04d}"
    run_dir = base.RUN_ROOT / f"row_{role_rank:04d}"
    covariance_path = run_dir / "NATIVE_Q24_COVARIANCE_TAPE.npz"
    result_path = run_dir / "ACTION_COVARIANCE_SNAP_RESULT.json"
    arrays_path = run_dir / "ACTION_COVARIANCE_SNAP_ARRAYS.npz"
    if result_path.exists() or arrays_path.exists():
        raise FileExistsError("action covariance result already exists")
    weights_path = feature_dir / "weights.npy"
    weights = np.ascontiguousarray(
        np.load(weights_path, mmap_mode="r", allow_pickle=False), dtype=np.float32
    )
    native_dir = base.NATIVE_ROW_ROOT / f"row_{role_rank:04d}"
    host = np.ascontiguousarray(np.load(native_dir / "host.npy"), dtype=np.float32)
    raw = np.ascontiguousarray(np.load(native_dir / "raw20.npy"), dtype=np.float32)
    sigma = np.ascontiguousarray(np.load(native_dir / "sigma.npy"), dtype=np.float32)
    with np.load(covariance_path, allow_pickle=False) as loaded:
        pre_covariance = np.ascontiguousarray(
            loaded["pre_covariance"], dtype=np.float32
        )
    chirps = base.chirps()
    points, point_receipt = base.build_points(role_rank, chirps)
    action, action_receipt = action_bases(weights, raw[:, :, 5], max(RANKS))
    scalar = base.load_scalar_module()
    ids, target_prefix = scalar.target_prefix(role_rank + 1)
    manifest = json.loads((feature_dir / "row_manifest.json").read_text())
    if int(ids[role_rank]) != int(manifest["mlp_id"]):
        raise AssertionError("target rank/MLP mismatch")
    target = np.ascontiguousarray(target_prefix[role_rank], dtype=np.float32)
    result: dict[str, object] = {
        "schema": "action-guided-covariance-snap-discriminator-v1",
        "role": "observer_fit",
        "role_rank": role_rank,
        "mlp_id": int(manifest["mlp_id"]),
        "sampler": "fixed existing GF32 two-block 16-MUB N32768 cloud",
        "point_design": point_receipt,
        "action_basis": action_receipt,
        "native_covariance_tape_sha256": sha256(covariance_path),
        "weights_sha256": sha256(weights_path),
        "screened_ranks": list(RANKS),
        "arms": {},
    }
    arrays: dict[str, np.ndarray] = {
        "action_basis_max64": action,
    }
    for projected_rank in RANKS:
        arm_arrays, arm_receipt = propagate_arm(
            points,
            weights,
            host,
            sigma,
            pre_covariance,
            action,
            projected_rank,
        )
        native_result, native_arrays = base.native_j_suffix_diagnostics(
            arm_arrays["mean_blocks"],
            host,
            raw,
            sigma,
            weights,
            target,
        )
        arm_receipt["native_j_suffix_discriminator"] = native_result
        result["arms"][f"rank_{projected_rank}"] = arm_receipt
        for name, value in arm_arrays.items():
            arrays[f"rank_{projected_rank}_{name}"] = value
        for name, value in native_arrays.items():
            arrays[f"rank_{projected_rank}_{name}"] = value
    np.savez_compressed(arrays_path, **arrays)
    result["artifact"] = arrays_path.name
    result["artifact_sha256"] = sha256(arrays_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summaries = {
        key: value["native_j_suffix_discriminator"]
        for key, value in result["arms"].items()
    }
    print(json.dumps({"result": str(result_path), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
