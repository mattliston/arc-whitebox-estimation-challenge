#!/usr/bin/env python3
"""Build target-blind native-Q24/J replays from a frozen nested-HT source tape."""

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
SOURCE = P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py"
QUARANTINE = P2 / "public_q24_cp3_full_snapshot_rank0_v1/runs/row_0000"
REFERENCE = (
    P2 / "public_q24_cp3_age_retirement_discriminator_v1/runs/rank0000_full.npz"
)
WEIGHTS = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0000/weights.npy"
)
PREREGISTER = HERE / "PREREGISTER.md"
STATE_SOURCE = HERE / "nested_ht_state.py"
SOURCE_RUNNER = HERE / "build_source_manifest.py"

WIDTH = 1024
DEPTH = 16
THREADS = 2
COMBINED_FLOPS = 551_371_243_519
RAW_MSE_GATE = 3.060640051205329e-8
EXPECTED_WEIGHTS_SHA256 = (
    "399488100ef907d6a76fa27e911285809bf5d53405bc59290c84396a7e4c8c97"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def mse(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.mean(array * array))


def load_module():
    spec = importlib.util.spec_from_file_location(
        f"q24_nested_ht_replay_{time.time_ns()}", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
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


def captured_context(module, layer: int) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:
    pre_mean = np.asarray(layer_array(layer, "pre_mean"), dtype=np.float64)
    pre_variance = np.asarray(layer_array(layer, "pre_variance"), dtype=np.float64)
    values = module._q38_coefficients(
        pre_mean, pre_variance, module._hermite_context()
    )
    return {
        "pre_mean": pre_mean,
        "pre_variance": pre_variance,
        "coefficients": np.asarray(values[0]),
        "gaussian_raw": tuple(np.asarray(item) for item in values[1]),
        "sigma": np.asarray(values[2]),
        "alpha": np.asarray(values[3]),
        "pdf": np.asarray(values[4]),
    }


def captured_anchor_host(
    module,
    frozen_host: np.ndarray,
    frozen_raw: np.ndarray,
    frozen_sigma: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    host = np.asarray(frozen_host, dtype=np.float64).copy()
    diagnostics: list[dict[str, float]] = []
    for layer in range(1, DEPTH):
        sigma = np.asarray(layer_array(layer, "sigma"), dtype=np.float64)
        k3 = np.asarray(layer_array(layer, "k3"), dtype=np.float64)
        rho3 = np.asarray(layer_array(layer, "rho3"), dtype=np.float64)
        pre_mean = np.asarray(layer_array(layer, "pre_mean"), dtype=np.float64)
        alpha = pre_mean / sigma
        pdf = np.exp(-0.5 * alpha * alpha) / np.sqrt(2.0 * np.pi)
        he4 = alpha**4 - 6.0 * alpha * alpha + 3.0
        c1_6 = sigma * he4 * pdf
        quadratic = c1_6 / (72.0 * sigma**6)
        frozen_k3 = (
            np.asarray(frozen_raw[layer, :, 10], dtype=np.float64)
            * np.asarray(frozen_sigma[layer], dtype=np.float64) ** 3
        )
        delta = k3 - frozen_k3
        estimate = np.maximum(
            np.asarray(frozen_host[layer], dtype=np.float64)
            + rho3 * delta
            - quadratic * delta * delta,
            0.0,
        )
        if layer == DEPTH - 1:
            actual = np.asarray(layer_array(layer, "reported_mean"), dtype=np.float64)
            diagnostics.append(
                {
                    "layer_zero_based": layer,
                    "estimated_vs_actual_rms": float(np.sqrt(mse(estimate - actual))),
                    "estimated_vs_actual_max_abs": float(
                        np.max(np.abs(estimate - actual))
                    ),
                }
            )
            estimate = actual
        host[layer] = estimate
    return np.asarray(host, dtype=np.float32), diagnostics


def candidate_host(
    module,
    anchor_host: np.ndarray,
    captured_k3: np.ndarray,
    candidate_k3: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    host = np.asarray(anchor_host, dtype=np.float64).copy()
    diagnostics: list[dict[str, float]] = []
    for layer in range(1, DEPTH):
        delta = np.asarray(candidate_k3[layer], dtype=np.float64) - np.asarray(
            captured_k3[layer], dtype=np.float64
        )
        rho3 = np.asarray(layer_array(layer, "rho3"), dtype=np.float64)
        context = captured_context(module, layer)
        coefficients = np.asarray(context["coefficients"])
        sigma = np.asarray(context["sigma"], dtype=np.float64)
        quadratic = np.asarray(coefficients[0, 6], dtype=np.float64) / (
            72.0 * sigma**6
        )
        reported = np.maximum(
            np.asarray(anchor_host[layer], dtype=np.float64)
            + rho3 * delta
            + quadratic * delta * delta,
            0.0,
        )
        host[layer] = reported
        diagnostics.append(
            {
                "layer_zero_based": layer,
                "k3_delta_rms": float(np.sqrt(mse(delta))),
                "k3_delta_max_abs": float(np.max(np.abs(delta))),
                "host_delta_rms": float(
                    np.sqrt(
                        mse(
                            reported
                            - np.asarray(anchor_host[layer], dtype=np.float64)
                        )
                    )
                ),
                "host_delta_max_abs": float(
                    np.max(
                        np.abs(
                            reported
                            - np.asarray(anchor_host[layer], dtype=np.float64)
                        )
                    )
                ),
            }
        )
    return np.asarray(host, dtype=np.float32), diagnostics


def native_row(
    module,
    weights: np.ndarray,
    layer: int,
    k3: np.ndarray,
    reported: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context = captured_context(module, layer)
    covariance = np.asarray(layer_array(layer, "pre_covariance"), dtype=np.float64)
    weight = np.asarray(weights[layer], dtype=np.float32)
    weight2 = weight * weight
    weight3 = weight2 * weight
    row = module._native_raw_features(
        layer,
        WIDTH,
        min(layer * WIDTH, 15 * WIDTH),
        context["coefficients"],
        context["gaussian_raw"],
        context["sigma"],
        context["alpha"],
        context["pdf"],
        covariance,
        context["pre_variance"],
        np.asarray(k3, dtype=np.float64),
        np.asarray(reported, dtype=np.float64),
        module._native_edge_summaries(weight, weight2, weight3),
    )
    return np.asarray(row, dtype=np.float32), np.asarray(
        context["sigma"], dtype=np.float32
    )


def build_native_tape(
    module,
    weights: np.ndarray,
    frozen_raw: np.ndarray,
    frozen_sigma: np.ndarray,
    host: np.ndarray,
    k3_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(frozen_raw, dtype=np.float32).copy()
    sigma = np.asarray(frozen_sigma, dtype=np.float32).copy()
    for layer in range(1, DEPTH):
        raw[layer], sigma[layer] = native_row(
            module, weights, layer, k3_rows[layer], host[layer]
        )
    return raw, sigma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-source-only", action="store_true", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()
    source_manifest_path = args.source_manifest.resolve()
    source_dir = source_manifest_path.parent
    output = args.output.resolve()
    receipt_path = output.with_suffix(".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError((output, receipt_path))
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("schema") != "q24-cp3-quotient-nested-ht-source-manifest-v1":
        raise RuntimeError(source_manifest.get("schema"))
    if source_manifest.get("status") != "COMPLETE_SOURCE_ONLY_NONPROMOTABLE_QUARANTINE":
        raise RuntimeError(source_manifest.get("status"))
    if source_manifest.get("r") != 128 or source_manifest.get("k") != 32:
        raise RuntimeError((source_manifest.get("r"), source_manifest.get("k")))
    bindings = source_manifest["bindings"]
    for key, path in (
        ("preregister_sha256", PREREGISTER),
        ("state_source_sha256", STATE_SOURCE),
        ("runner_sha256", SOURCE_RUNNER),
        ("weights_sha256", WEIGHTS),
    ):
        actual = sha256(path)
        if bindings.get(key) != actual:
            raise RuntimeError((key, bindings.get(key), actual))
    if sha256(WEIGHTS) != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError("weights hash mismatch")

    weights = np.load(WEIGHTS, mmap_mode="r", allow_pickle=False)
    quotient_k3 = np.load(source_dir / "quotient_k3.npy", allow_pickle=False)
    raw_control_k3 = np.load(source_dir / "raw_control_k3.npy", allow_pickle=False)
    captured_k3 = np.load(source_dir / "captured_full_k3.npy", allow_pickle=False)
    for value in (quotient_k3, raw_control_k3, captured_k3):
        if value.shape != (DEPTH, WIDTH):
            raise AssertionError(value.shape)

    with np.load(REFERENCE, allow_pickle=False) as reference:
        # Deliberately access only target-blind named members.
        frozen_host = np.ascontiguousarray(reference["host"], dtype=np.float32)
        frozen_raw = np.ascontiguousarray(reference["native_raw"], dtype=np.float32)
        frozen_sigma = np.ascontiguousarray(reference["native_sigma"], dtype=np.float32)
        frozen_native_j = np.ascontiguousarray(
            reference["native_j_output"], dtype=np.float32
        )

    module = load_module()
    started = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    with module.flops.BudgetContext(
        flop_budget=2**62,
        quiet=True,
        namespace="q24_nested_ht_source_replay",
    ) as diagnostic_budget, threadpool_limits(limits=args.threads):
        anchor_host, anchor_diagnostics = captured_anchor_host(
            module, frozen_host, frozen_raw, frozen_sigma
        )
        anchor_raw, anchor_sigma = build_native_tape(
            module, weights, frozen_raw, frozen_sigma, anchor_host, captured_k3
        )
        anchor_correction = np.asarray(
            module.transport_correction(
                anchor_raw, anchor_sigma, weights, module._TRANSPORT_PAYLOAD
            ),
            dtype=np.float32,
        )
        anchor_native_j = np.asarray(
            np.maximum(anchor_host + anchor_correction, 0.0), dtype=np.float32
        )
        arrays.update(
            {
                "captured_anchor_host": anchor_host,
                "captured_anchor_native_raw": anchor_raw,
                "captured_anchor_native_sigma": anchor_sigma,
                "captured_anchor_native_j": anchor_native_j,
            }
        )
        diagnostics["captured_anchor"] = {
            "host_vs_canonical_frozen_mse": mse(
                anchor_host[-1].astype(np.float64)
                - frozen_host[-1].astype(np.float64)
            ),
            "native_j_vs_canonical_frozen_mse": mse(
                anchor_native_j[-1].astype(np.float64)
                - frozen_native_j[-1].astype(np.float64)
            ),
            "construction": anchor_diagnostics,
        }
        for label, k3_rows in (
            ("quotient_r128_k32", quotient_k3),
            ("raw_control_r128_k32", raw_control_k3),
        ):
            host, host_diagnostics = candidate_host(
                module, anchor_host, captured_k3, k3_rows
            )
            raw, sigma = build_native_tape(
                module, weights, frozen_raw, frozen_sigma, host, k3_rows
            )
            correction = np.asarray(
                module.transport_correction(
                    raw, sigma, weights, module._TRANSPORT_PAYLOAD
                ),
                dtype=np.float32,
            )
            native_j = np.asarray(np.maximum(host + correction, 0.0), dtype=np.float32)
            arrays[f"{label}_host"] = host
            arrays[f"{label}_native_raw"] = raw
            arrays[f"{label}_native_sigma"] = sigma
            arrays[f"{label}_native_j"] = native_j
            arrays[f"{label}_k3"] = np.asarray(k3_rows, dtype=np.float32)
            diagnostics[label] = {
                "layers": host_diagnostics,
                "final_host_vs_anchor_mse": mse(
                    host[-1].astype(np.float64) - anchor_host[-1].astype(np.float64)
                ),
                "final_native_j_vs_anchor_mse": mse(
                    native_j[-1].astype(np.float64)
                    - anchor_native_j[-1].astype(np.float64)
                ),
                "final_host_vs_canonical_frozen_mse": mse(
                    host[-1].astype(np.float64) - frozen_host[-1].astype(np.float64)
                ),
                "final_native_j_vs_canonical_frozen_mse": mse(
                    native_j[-1].astype(np.float64)
                    - frozen_native_j[-1].astype(np.float64)
                ),
            }

    np.savez_compressed(output, **arrays)
    receipt = {
        "schema": "q24-cp3-quotient-nested-ht-native-replay-v1",
        "status": "COMPLETE_SOURCE_ONLY_NONPROMOTABLE_QUARANTINE_REPLAY",
        "score_gate": {
            "combined_flops": COMBINED_FLOPS,
            "raw_mse_at_most": RAW_MSE_GATE,
            "leader_adjusted": 7.67408396767286e-9,
        },
        "diagnostics": diagnostics,
        "bindings": {
            "preregister_sha256": sha256(PREREGISTER),
            "state_source_sha256": sha256(STATE_SOURCE),
            "source_runner_sha256": sha256(SOURCE_RUNNER),
            "replay_runner_sha256": sha256(Path(__file__)),
            "source_manifest_sha256": sha256(source_manifest_path),
            "source_estimator_sha256": sha256(SOURCE),
            "reference_sha256": sha256(REFERENCE),
            "weights_sha256": sha256(WEIGHTS),
        },
        "access": {
            "reference_npz_members_opened": [
                "host",
                "native_raw",
                "native_sigma",
                "native_j_output",
            ],
            "target_or_error_arrays_opened": [],
            "observer_selection": False,
            "network": False,
            "training": False,
            "fitting": False,
            "submission": False,
        },
        "resource": {
            "wall_seconds": time.perf_counter() - started,
            "threads": args.threads,
            "diagnostic_flopscope": diagnostic_budget.summary_dict(by_namespace=False),
            "note": "source replay only; no deployable FLOP claim",
        },
        "output": {
            "path": str(output),
            "sha256": sha256(output),
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "array_sha256": array_sha256(value),
                }
                for key, value in arrays.items()
            },
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
