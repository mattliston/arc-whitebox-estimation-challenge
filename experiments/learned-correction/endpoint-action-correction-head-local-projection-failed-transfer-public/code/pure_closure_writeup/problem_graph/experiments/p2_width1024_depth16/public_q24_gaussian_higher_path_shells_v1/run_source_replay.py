#!/usr/bin/env python3
"""Build one target-blind rank-0 higher Gaussian path-shell replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
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
D6_SOURCE = P2 / "public_q24_gaussian_mixed_d6_shell_v1/runs/rank0000_mixed_d6_source_replay_v1.npz"
WEIGHTS = (
    P2 / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0000/weights.npy"
)
WIDTH = 1024
DEPTH = 16
SOURCE_LAYERS = tuple(range(9, 15))
LEADER_ADJUSTED = 7.67408396767286e-9
R40_MAPPED_HOST = 893_495_871_479
CANONICAL_RANK0_RAW = 2.856783180041084e-8
SECTORS = {
    "d8_134": (1, 3, 4),
    "d8_224": (2, 2, 4),
    "d10_145": (1, 4, 5),
    "d10_235": (2, 3, 5),
}
CANDIDATES = (
    "d8_134",
    "d8_224",
    "d8_paths",
    "d10_145",
    "d10_235",
    "d10_paths",
    "d8_d10_paths",
    "d6_d8_d10_paths",
)


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


def path_quotient(
    degrees: tuple[int, int, int],
    coefficients: dict[int, np.ndarray],
    b: dict[int, np.ndarray],
    x: dict[int, np.ndarray],
    weight: np.ndarray,
    weight2: np.ndarray,
    weight3: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    a, leaf, high = degrees
    multiplicity = len(set(itertools.permutations(degrees)))
    full = (
        multiplicity
        * math.factorial(high)
        * np.sum(b[high] * x[a] * x[leaf], axis=0)
    )
    ca = coefficients[a]
    cb = coefficients[leaf]
    ch = coefficients[high]
    if a != leaf:
        scale = 2.0 / (math.factorial(a) * math.factorial(leaf))
        hw = scale * (
            math.factorial(a) * (cb * ch)[:, None] * x[a]
            + math.factorial(leaf) * (ca * ch)[:, None] * x[leaf]
            + math.factorial(high) * (ca * cb)[:, None] * x[high]
        )
        diagonal_h = 3.0 * scale * ca * cb * ch
    else:
        scale = 1.0 / math.factorial(a) ** 2
        hw = scale * (
            2.0 * math.factorial(a) * (ca * ch)[:, None] * x[a]
            + math.factorial(high) * (ca * ca)[:, None] * x[high]
        )
        diagonal_h = 3.0 * scale * ca * ca * ch
    lift = np.sum(
        3.0 * weight2 * hw - 2.0 * diagonal_h[:, None] * weight3,
        axis=0,
    )
    quotient = full - lift
    return quotient, {
        "full_rms": float(np.sqrt(mse(full))),
        "lift_rms": float(np.sqrt(mse(lift))),
        "quotient_rms": float(np.sqrt(mse(quotient))),
        "full_lift_correlation": float(np.corrcoef(full, lift)[0, 1]),
    }


def candidate_costs() -> dict[str, dict[str, int | float | list[int]]]:
    # Literal scalar-operation census for the formulas above.  R^{circ d}, W^2,
    # and W^3 already exist in the Q24/cubic-star host and are reused.  Each
    # X_d costs one conventional 2*n^3 dense GEMM.  B_d costs n^2+n.  A
    # distinct-leaf quotient costs 14*n^2+9*n; a repeated-leaf quotient costs
    # 12*n^2+7*n.  Summing s sector vectors costs (s-1)*n.
    specifications = {
        "d8_134": ({1, 3, 4}, 1, 0, 0),
        "d8_224": ({2, 4}, 0, 1, 0),
        "d8_paths": ({1, 2, 3, 4}, 1, 1, 1),
        "d10_145": ({1, 4, 5}, 1, 0, 0),
        "d10_235": ({2, 3, 5}, 1, 0, 0),
        "d10_paths": ({1, 2, 3, 4, 5}, 2, 0, 1),
        "d8_d10_paths": ({1, 2, 3, 4, 5}, 3, 1, 3),
        # D6 is another distinct-leaf path and shares X_1:X_3.
        "d6_d8_d10_paths": ({1, 2, 3, 4, 5}, 4, 1, 4),
    }
    rows: dict[str, dict[str, int | float | list[int]]] = {}
    n = WIDTH
    layers = len(SOURCE_LAYERS)
    for label, (degree_set, distinct, repeated, vector_adds) in specifications.items():
        transforms = len(degree_set)
        gemm = layers * transforms * 2 * n**3
        quadratic_ops = transforms + 14 * distinct + 12 * repeated
        linear_ops = transforms + 9 * distinct + 7 * repeated + vector_adds
        non_gemm = layers * (quadratic_ops * n**2 + linear_ops * n)
        extra = gemm + non_gemm
        total = R40_MAPPED_HOST + extra
        rows[label] = {
            "transform_degrees": sorted(degree_set),
            "dense_gemm_count": layers * transforms,
            "gemm_flops": gemm,
            "literal_non_gemm_flops": non_gemm,
            "extra_flops": extra,
            "combined_flops": total,
            "adjusted_positive_raw_below": CANONICAL_RANK0_RAW
            * R40_MAPPED_HOST
            / total,
            "leader_raw_mse_gate": LEADER_ADJUSTED * (2**41) / total,
        }
    return rows


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
    estimator = load_module(ESTIMATOR, "higher_path_estimator")
    replay = load_module(REPLAY_HELPER, "higher_path_replay_helper")
    weights = np.load(WEIGHTS, mmap_mode="r", allow_pickle=False)
    captured = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    for layer in range(1, DEPTH):
        captured[layer] = np.asarray(layer_array(layer, "k3"), dtype=np.float64)
    sector_delta = {
        label: np.zeros_like(captured) for label in SECTORS
    }
    layer_diagnostics: list[dict[str, object]] = []
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        for source_layer in SOURCE_LAYERS:
            next_layer = source_layer + 1
            mean = np.asarray(layer_array(source_layer, "pre_mean"), dtype=np.float64)
            variance = np.asarray(layer_array(source_layer, "pre_variance"), dtype=np.float64)
            covariance = np.asarray(layer_array(source_layer, "pre_covariance"), dtype=np.float64)
            table, _raw, sigma, _alpha, _pdf, _s2, _s3 = estimator._q38_coefficients(
                mean, variance, estimator._hermite_context()
            )
            coefficients = {
                degree: np.asarray(table[0, degree], dtype=np.float64)
                for degree in range(1, 6)
            }
            inv_sigma = 1.0 / np.asarray(sigma, dtype=np.float64)
            correlation = np.clip(
                covariance * inv_sigma[:, None] * inv_sigma[None, :], -1.0, 1.0
            )
            np.fill_diagonal(correlation, 1.0)
            correlation_powers = {1: correlation}
            for degree in range(2, 6):
                correlation_powers[degree] = correlation_powers[degree - 1] * correlation
            weight = np.asarray(weights[next_layer], dtype=np.float64)
            weight2 = weight * weight
            weight3 = weight2 * weight
            b = {
                degree: (
                    coefficients[degree] / math.factorial(degree)
                )[:, None]
                * weight
                for degree in range(1, 6)
            }
            x = {
                degree: correlation_powers[degree] @ b[degree]
                for degree in range(1, 6)
            }
            row: dict[str, object] = {
                "source_layer_zero_based": source_layer,
                "next_layer_zero_based": next_layer,
                "correlation_min": float(np.min(correlation)),
                "correlation_max": float(np.max(correlation)),
                "sectors": {},
            }
            for label, degrees in SECTORS.items():
                quotient, diagnostic = path_quotient(
                    degrees, coefficients, b, x, weight, weight2, weight3
                )
                sector_delta[label][next_layer] = quotient
                row["sectors"][label] = diagnostic
            layer_diagnostics.append(row)

        with np.load(D6_SOURCE, allow_pickle=False) as d6_source:
            d6_delta = np.ascontiguousarray(
                d6_source["quotient_d6_delta_k3"], dtype=np.float64
            )
        deltas = {
            **sector_delta,
            "d8_paths": sector_delta["d8_134"] + sector_delta["d8_224"],
            "d10_paths": sector_delta["d10_145"] + sector_delta["d10_235"],
        }
        deltas["d8_d10_paths"] = deltas["d8_paths"] + deltas["d10_paths"]
        deltas["d6_d8_d10_paths"] = d6_delta + deltas["d8_d10_paths"]

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
            estimator.transport_correction(
                anchor_raw, anchor_sigma, weights, estimator._TRANSPORT_PAYLOAD
            ),
            dtype=np.float32,
        )
        anchor_j = np.asarray(
            np.maximum(anchor_host + anchor_correction, 0.0), dtype=np.float32
        )
        arrays: dict[str, np.ndarray] = {
            "captured_k3": captured.astype(np.float32),
            "anchor_host": anchor_host,
            "anchor_native_j": anchor_j,
        }
        candidate_diagnostics: dict[str, object] = {}
        for label in CANDIDATES:
            delta = deltas[label]
            candidate_k3 = captured + delta
            host, host_diag = replay.candidate_host(
                estimator, anchor_host, captured, candidate_k3
            )
            raw, sigma = replay.build_native_tape(
                estimator, weights, frozen_raw, frozen_sigma, host, candidate_k3
            )
            correction = np.asarray(
                estimator.transport_correction(
                    raw, sigma, weights, estimator._TRANSPORT_PAYLOAD
                ),
                dtype=np.float32,
            )
            native_j = np.asarray(np.maximum(host + correction, 0.0), dtype=np.float32)
            arrays[f"{label}_delta_k3"] = delta.astype(np.float32)
            arrays[f"{label}_native_j"] = native_j
            candidate_diagnostics[label] = {
                "host_layers": host_diag,
                "final_j_vs_anchor_mse": mse(
                    native_j[-1].astype(np.float64) - anchor_j[-1].astype(np.float64)
                ),
                "final_j_vs_frozen_mse": mse(
                    native_j[-1].astype(np.float64) - frozen_j[-1].astype(np.float64)
                ),
            }
    np.savez_compressed(output, **arrays)
    receipt = {
        "schema": "q24-gaussian-higher-path-shells-source-replay-v1",
        "status": "COMPLETE_TARGET_BLIND_SOURCE_REPLAY",
        "formula": {
            "d8_connected": ["(1,3,4) path", "(2,2,4) path", "(2,3,3) triangle"],
            "d10_connected": [
                "(1,4,5) path",
                "(2,3,5) path",
                "(2,4,4) triangle",
                "(3,3,4) triangle",
            ],
            "executed": {label: list(degrees) for label, degrees in SECTORS.items()},
            "quotient": "full shell minus exact repeated-slice cubic-star lift",
            "triangle_cost_lower_bound_flops": len(SOURCE_LAYERS) * 2 * WIDTH**4,
            "triangles_executed": False,
        },
        "source_layers_zero_based": list(SOURCE_LAYERS),
        "cost_by_candidate": candidate_costs(),
        "diagnostics": {
            "layers": layer_diagnostics,
            "anchor": {
                "construction": anchor_diag,
                "final_j_vs_frozen_mse": mse(
                    anchor_j[-1].astype(np.float64) - frozen_j[-1].astype(np.float64)
                ),
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
            "d6_source_sha256": sha256(D6_SOURCE),
            "weights_sha256": sha256(WEIGHTS),
            "quarantine_manifest_sha256": sha256(QUARANTINE / "QUARANTINE_MANIFEST.json"),
        },
        "access": {
            "target_or_error_arrays_opened": [],
            "training": False,
            "fitting": False,
            "sampling": False,
            "network": False,
        },
        "resource": {
            "wall_seconds": time.perf_counter() - started,
            "threads": args.threads,
            "sole_process_owner": True,
        },
        "output": {"path": str(output), "sha256": sha256(output)},
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
