#!/usr/bin/env python3
"""Build a target-blind rank-0 Gaussian mixed-D6 source/native-J replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
ESTIMATOR = P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py"
REPLAY_HELPER = P2 / "public_q24_cp3_nested_ht_quotient_discriminator_v1/build_native_replay.py"
QUARANTINE = P2 / "public_q24_cp3_full_snapshot_rank0_v1/runs/row_0000"
REFERENCE = P2 / "public_q24_cp3_age_retirement_discriminator_v1/runs/rank0000_full.npz"
WEIGHTS = (
    P2 / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0000/weights.npy"
)
WIDTH = 1024
DEPTH = 16
SOURCE_LAYERS = tuple(range(9, 15))
LEADER_ADJUSTED = 7.67408396767286e-9
R40_MAPPED_HOST = 893_495_871_479


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_{time.time_ns()}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def layer_array(layer: int, name: str) -> np.ndarray:
    return np.load(
        QUARANTINE / f"layer_{layer:02d}" / f"{name}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )


def mse(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.mean(value * value))


def shell_read(
    estimator, layer: int, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mean = np.asarray(layer_array(layer, "pre_mean"), dtype=np.float64)
    variance = np.asarray(layer_array(layer, "pre_variance"), dtype=np.float64)
    covariance = np.asarray(layer_array(layer, "pre_covariance"), dtype=np.float64)
    coefficient_table, _raw, sigma, _alpha, _pdf, _s2, _s3 = estimator._q38_coefficients(
        mean, variance, estimator._hermite_context()
    )
    c1 = np.asarray(coefficient_table[0, 1], dtype=np.float64)
    c2 = np.asarray(coefficient_table[0, 2], dtype=np.float64)
    c3 = np.asarray(coefficient_table[0, 3], dtype=np.float64)
    inv_sigma = 1.0 / np.asarray(sigma, dtype=np.float64)
    correlation = np.clip(
        covariance * inv_sigma[:, None] * inv_sigma[None, :], -1.0, 1.0
    )
    np.fill_diagonal(correlation, 1.0)
    r2 = correlation * correlation
    r3 = r2 * correlation
    w = np.asarray(weight, dtype=np.float64)
    b1 = c1[:, None] * w
    b2 = (0.5 * c2)[:, None] * w
    b3 = (c3 / 6.0)[:, None] * w
    x1 = correlation @ b1
    x2 = r2 @ b2
    x3 = r3 @ b3
    full = 36.0 * np.sum(b3 * x1 * x2, axis=0)
    # H@W without materializing H.  x2 and x3 use c2/2 and c3/6,
    # hence factors 2 and 6 in the second and third terms.
    hw = (
        (c2 * c3)[:, None] * x1
        + (2.0 * c1 * c3)[:, None] * x2
        + (6.0 * c1 * c2)[:, None] * x3
    )
    diagonal_h = 3.0 * c1 * c2 * c3
    lift = np.sum(3.0 * w * w * hw - 2.0 * diagonal_h[:, None] * w**3, axis=0)
    quotient = full - lift
    return full, quotient, {
        "full_rms": float(np.sqrt(mse(full))),
        "lift_rms": float(np.sqrt(mse(lift))),
        "quotient_rms": float(np.sqrt(mse(quotient))),
        "full_lift_correlation": float(np.corrcoef(full, lift)[0, 1]),
        "correlation_min": float(np.min(correlation)),
        "correlation_max": float(np.max(correlation)),
    }


def cost_receipt() -> dict[str, int | float]:
    n = WIDTH
    layers = len(SOURCE_LAYERS)
    # R, R^2 and R^3 are already present in Q24's rho-power bank.  Per source
    # layer this spelling performs three n-by-n GEMMs plus literal coefficient,
    # product, reduction, and quotient arithmetic.  Count every elementwise
    # primitive conservatively as one FLOP.
    gemm = layers * 3 * (2 * n**3)
    coefficient_scales = layers * 6 * n**2
    products_and_sums = layers * 16 * n**2
    vector_coefficients = layers * 10 * n
    extra = gemm + coefficient_scales + products_and_sums + vector_coefficients
    total = R40_MAPPED_HOST + extra
    raw_gate = LEADER_ADJUSTED * (2**41) / total
    return {
        "mapped_host_flops": R40_MAPPED_HOST,
        "gemm_flops": gemm,
        "literal_non_gemm_flops": coefficient_scales + products_and_sums + vector_coefficients,
        "extra_flops": extra,
        "combined_flops": total,
        "raw_mse_gate": raw_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-source-only", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    receipt_path = output.with_suffix(".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError((output, receipt_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    estimator = load_module(ESTIMATOR, "mixed_d6_estimator")
    replay = load_module(REPLAY_HELPER, "mixed_d6_replay_helper")
    weights = np.load(WEIGHTS, mmap_mode="r", allow_pickle=False)
    captured = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    for layer in range(1, DEPTH):
        captured[layer] = np.asarray(layer_array(layer, "k3"), dtype=np.float64)
    full_k3 = captured.copy()
    quotient_k3 = captured.copy()
    full_delta = np.zeros_like(captured)
    quotient_delta = np.zeros_like(captured)
    layer_diagnostics: list[dict[str, float | int]] = []
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        for source_layer in SOURCE_LAYERS:
            next_layer = source_layer + 1
            full, quotient, diagnostic = shell_read(
                estimator, source_layer, np.asarray(weights[next_layer])
            )
            full_delta[next_layer] = full
            quotient_delta[next_layer] = quotient
            full_k3[next_layer] += full
            quotient_k3[next_layer] += quotient
            layer_diagnostics.append(
                {"source_layer_zero_based": source_layer, "next_layer_zero_based": next_layer, **diagnostic}
            )

        with np.load(REFERENCE, allow_pickle=False) as reference:
            frozen_host = np.ascontiguousarray(reference["host"], dtype=np.float32)
            frozen_raw = np.ascontiguousarray(reference["native_raw"], dtype=np.float32)
            frozen_sigma = np.ascontiguousarray(reference["native_sigma"], dtype=np.float32)
            frozen_j = np.ascontiguousarray(reference["native_j_output"], dtype=np.float32)
        anchor_host, anchor_diag = replay.captured_anchor_host(
            estimator, frozen_host, frozen_raw, frozen_sigma
        )
        anchor_raw, anchor_sigma = replay.build_native_tape(
            estimator, weights, frozen_raw, frozen_sigma, anchor_host, captured
        )
        anchor_correction = np.asarray(
            estimator.transport_correction(anchor_raw, anchor_sigma, weights, estimator._TRANSPORT_PAYLOAD),
            dtype=np.float32,
        )
        anchor_j = np.asarray(np.maximum(anchor_host + anchor_correction, 0.0), dtype=np.float32)
        arrays: dict[str, np.ndarray] = {
            "captured_k3": captured.astype(np.float32),
            "full_d6_delta_k3": full_delta.astype(np.float32),
            "quotient_d6_delta_k3": quotient_delta.astype(np.float32),
            "anchor_host": anchor_host,
            "anchor_native_j": anchor_j,
        }
        candidate_diagnostics: dict[str, object] = {}
        for label, candidate_k3 in (("full_d6", full_k3), ("quotient_d6", quotient_k3)):
            host, host_diag = replay.candidate_host(estimator, anchor_host, captured, candidate_k3)
            raw, sigma = replay.build_native_tape(
                estimator, weights, frozen_raw, frozen_sigma, host, candidate_k3
            )
            correction = np.asarray(
                estimator.transport_correction(raw, sigma, weights, estimator._TRANSPORT_PAYLOAD),
                dtype=np.float32,
            )
            native_j = np.asarray(np.maximum(host + correction, 0.0), dtype=np.float32)
            arrays[f"{label}_k3"] = candidate_k3.astype(np.float32)
            arrays[f"{label}_host"] = host
            arrays[f"{label}_native_raw"] = raw
            arrays[f"{label}_native_sigma"] = sigma
            arrays[f"{label}_native_j"] = native_j
            candidate_diagnostics[label] = {
                "host_layers": host_diag,
                "final_host_vs_anchor_mse": mse(host[-1].astype(np.float64) - anchor_host[-1].astype(np.float64)),
                "final_j_vs_anchor_mse": mse(native_j[-1].astype(np.float64) - anchor_j[-1].astype(np.float64)),
                "final_j_vs_frozen_mse": mse(native_j[-1].astype(np.float64) - frozen_j[-1].astype(np.float64)),
            }
    np.savez_compressed(output, **arrays)
    receipt = {
        "schema": "q24-gaussian-mixed-d6-shell-source-replay-v1",
        "status": "COMPLETE_TARGET_BLIND_SOURCE_REPLAY",
        "formula": {
            "sector": "all six permutations of Hermite degrees (1,2,3)",
            "diagram_coefficient_with_factorial_normalized_B": 36,
            "existing_cp3_birth_sector": "(1,1,2), exactly disjoint",
            "quotient": "full shell minus exact repeated-slice cubic-star lift",
        },
        "source_layers_zero_based": list(SOURCE_LAYERS),
        "cost": cost_receipt(),
        "diagnostics": {
            "layers": layer_diagnostics,
            "anchor": {
                "construction": anchor_diag,
                "final_j_vs_frozen_mse": mse(anchor_j[-1].astype(np.float64) - frozen_j[-1].astype(np.float64)),
            },
            "candidates": candidate_diagnostics,
        },
        "bindings": {
            "preregister_sha256": sha256(HERE / "PREREGISTER.md"),
            "runner_sha256": sha256(Path(__file__)),
            "formula_verifier_sha256": sha256(HERE / "verify_formula.py"),
            "estimator_sha256": sha256(ESTIMATOR),
            "replay_helper_sha256": sha256(REPLAY_HELPER),
            "reference_sha256": sha256(REFERENCE),
            "weights_sha256": sha256(WEIGHTS),
            "quarantine_manifest_sha256": sha256(QUARANTINE / "QUARANTINE_MANIFEST.json"),
        },
        "access": {
            "target_or_error_arrays_opened": [],
            "reference_members_opened": ["host", "native_raw", "native_sigma", "native_j_output"],
            "training": False,
            "fitting": False,
            "sampling": False,
            "network": False,
        },
        "resource": {"wall_seconds": time.perf_counter() - started, "threads": args.threads},
        "output": {"path": str(output), "sha256": sha256(output)},
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

