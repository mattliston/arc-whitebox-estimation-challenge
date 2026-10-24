#!/usr/bin/env python3
"""Target-free full-Kerdock fourth-moment controls for retained basis readouts.

The full real Kerdock family, together with the coordinate basis, is an exact
real projective 2-design.  Consequently the population mean over all 512
Kerdock bases of the fourth moment of any fixed unit probe is known exactly.
This script uses the frozen mean-gate input-to-endpoint response as one probe
per endpoint coordinate and cross-fits a basis-level control coefficient
between the two retained eight-basis blocks.  Truth is opened only after all
coefficient laws and the split-disagreement selector have been evaluated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from run_discriminator import WIDTH, orientation, walsh_matrix
from standard_real_kerdock import chirp_signs, standard_kerdock_matrices


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/row_0000"
TAG = "FULL_KERDOCK_SEED20260826_ALL_LAYER_SNAP_HERMITE_CV_D12_DEGREES_BASIS_READOUTS"
DIAGNOSTICS = RUN / f"DIAGNOSTIC_ARRAYS_{TAG}.npz"
RESULT = RUN / f"RESULT_{TAG}.json"
FEATURES = RUN / "TERMINAL_TARGET_BLIND_FEATURES.npz"
WEIGHTS = (
    HERE.parent
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0000/weights.npy"
)
OUTPUT = RUN / "FULL_KERDOCK_BASIS_FOURTH_CV_RESULT.json"


def mse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.square(left.astype(np.float64) - right.astype(np.float64))))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def response_probes(weights: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Frozen mean-gate Jacobian from input to every h16 coordinate."""
    response = np.asarray(weights[0], dtype=np.float32) * phi[0][None, :]
    for layer in range(1, weights.shape[0]):
        response = np.asarray(response @ weights[layer], dtype=np.float32)
        response *= phi[layer][None, :]
    norms = np.linalg.norm(response.astype(np.float64), axis=0)
    if np.any(norms <= 1e-30):
        raise AssertionError("degenerate endpoint-response probe")
    return np.asarray(response / norms[None, :], dtype=np.float32)


def basis_fourth_controls(
    probes: np.ndarray, selected_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected-basis fourth moments and exact 512-family means."""
    matrices = standard_kerdock_matrices()
    signs = chirp_signs(matrices, selected_indices)
    hadamard = walsh_matrix()
    permutation, coordinate_sign, _ = orientation(0, 0)
    moments = np.empty((selected_indices.size, WIDTH), dtype=np.float64)
    for basis_index in range(selected_indices.size):
        basis = hadamard * signs[basis_index][None, :]
        basis = basis[:, permutation] * coordinate_sign[None, :]
        basis /= np.float32(np.sqrt(WIDTH))
        projections = np.asarray(basis @ probes, dtype=np.float32)
        moments[basis_index] = np.mean(
            np.square(np.square(projections.astype(np.float64))), axis=0
        )
    identity_moment = np.mean(np.square(np.square(probes.astype(np.float64))), axis=0)
    haar_moment = 3.0 / (WIDTH * (WIDTH + 2.0))
    population = (513.0 * haar_moment - identity_moment) / 512.0
    return moments, population


def crossfit(
    values: np.ndarray,
    centered_control: np.ndarray,
    ridge_fraction: float,
    coefficient_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    corrected_blocks = []
    coefficients = []
    for evaluation in (slice(0, 8), slice(8, 16)):
        training = slice(8, 16) if evaluation.start == 0 else slice(0, 8)
        train_c = centered_control[training]
        train_y = values[training]
        dc = train_c - np.mean(train_c, axis=0, keepdims=True)
        dy = train_y - np.mean(train_y, axis=0, keepdims=True)
        if coefficient_mode == "global":
            denominator = float(np.sum(np.square(dc)))
            beta = float(np.sum(dc * dy) / (denominator * (1.0 + ridge_fraction) + 1e-300))
        elif coefficient_mode == "coordinate":
            denominator = np.sum(np.square(dc), axis=0)
            floor = float(np.median(denominator[denominator > 0]))
            beta = np.sum(dc * dy, axis=0) / (
                denominator + ridge_fraction * floor + 1e-300
            )
        else:
            raise ValueError(coefficient_mode)
        corrected = np.mean(values[evaluation], axis=0) - beta * np.mean(
            centered_control[evaluation], axis=0
        )
        corrected_blocks.append(corrected)
        coefficients.append(np.asarray(beta))
    block_array = np.stack(corrected_blocks)
    return (
        np.mean(block_array, axis=0),
        block_array,
        {
            "coefficient_rms": float(
                np.sqrt(np.mean(np.square(np.stack(coefficients).astype(np.float64))))
            ),
            "split_disagreement_quarter_mse": float(
                np.mean(np.square(block_array[0] - block_array[1])) / 4.0
            ),
        },
    )


def main() -> None:
    result_payload = json.loads(RESULT.read_text(encoding="utf-8"))
    selected = np.asarray(result_payload["mub_proof"]["selected_indices"], dtype=np.int64)
    with np.load(FEATURES, allow_pickle=False) as loaded:
        phi = loaded["host_phi"]
    weights = np.load(WEIGHTS, mmap_mode="r", allow_pickle=False)
    probes = response_probes(weights, phi)
    moments, population = basis_fourth_controls(probes, selected)
    centered = moments - population[None, :]

    with np.load(DIAGNOSTICS, allow_pickle=False) as loaded:
        # These readouts and every coefficient below are target-blind.
        value_banks = {
            "raw": loaded["native_j_endpoint_basis_actions"].astype(np.float64),
            "d12": loaded["cv_d12_native_j_endpoint_basis_actions"].astype(np.float64),
        }
        bank = loaded["cv_d12_native_j_autonomous"][-1].astype(np.float64)
        truth_action = loaded["cv_d12_native_j_endpoint_truth"].astype(np.float64)
        truth_error = loaded["truth_error"][-1].astype(np.float64)

    candidates: list[dict[str, object]] = []
    ridge_grid = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    for source_name, values in value_banks.items():
        for mode in ("global", "coordinate"):
            for ridge_fraction in ridge_grid:
                estimate, blocks, diagnostics = crossfit(
                    values, centered, ridge_fraction, mode
                )
                candidates.append(
                    {
                        "source": source_name,
                        "coefficient_mode": mode,
                        "ridge_fraction": ridge_fraction,
                        **diagnostics,
                        "measurement_mse_revealed_after_selection": mse(
                            estimate, truth_action
                        ),
                        "final_mse_revealed_after_selection": mse(
                            bank + estimate, truth_error
                        ),
                        "measurement_correlation_revealed_after_selection": correlation(
                            estimate, truth_action
                        ),
                    }
                )

    # The selector is target-free: choose only by the two corrected blocks'
    # disagreement, then reveal its truth metrics from the already-built table.
    chosen = min(candidates, key=lambda item: item["split_disagreement_quarter_mse"])
    baseline_mean = np.mean(value_banks["d12"], axis=0)
    output = {
        "schema": "full-kerdock-basis-fourth-cv-rank0-v1",
        "authority": "observer_fit rank0 diagnostic only",
        "coefficient_selection": "minimum corrected two-block disagreement; no truth used",
        "exact_population_identity": "E_K m_s(v)=(513*3||v||^4/(n(n+2))-mean_i(v_i^4))/512",
        "selected_kerdock_indices": selected.tolist(),
        "control": {
            "basis_output_flat_correlation_raw": correlation(
                centered, value_banks["raw"] - np.mean(value_banks["raw"], axis=0)
            ),
            "basis_output_flat_correlation_d12": correlation(
                centered, value_banks["d12"] - np.mean(value_banks["d12"], axis=0)
            ),
            "population_center_residual_mean_rms": float(
                np.sqrt(np.mean(np.square(np.mean(centered, axis=0))))
            ),
            "selected_control_rms": float(np.sqrt(np.mean(np.square(centered)))),
        },
        "baseline": {
            "d12_measurement_mse": mse(baseline_mean, truth_action),
            "d12_final_mse_if_unshrunk": mse(bank + baseline_mean, truth_error),
            "d12_split_disagreement_quarter_mse": float(
                np.mean(
                    np.square(
                        np.mean(value_banks["d12"][:8], axis=0)
                        - np.mean(value_banks["d12"][8:], axis=0)
                    )
                )
                / 4.0
            ),
        },
        "target_free_choice_then_reveal": chosen,
        "candidates": candidates,
        "fail_fast": {
            "required_measurement_mse": 2.075e-9,
            "cap1024_required_final_raw": 8.38996e-9,
            "pass": bool(
                chosen["measurement_mse_revealed_after_selection"] < 2.075e-9
                and chosen["final_mse_revealed_after_selection"] < 8.38996e-9
            ),
        },
        "artifacts": {
            "diagnostics_sha256": sha256(DIAGNOSTICS),
            "features_sha256": sha256(FEATURES),
            "weights_sha256": sha256(WEIGHTS),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
