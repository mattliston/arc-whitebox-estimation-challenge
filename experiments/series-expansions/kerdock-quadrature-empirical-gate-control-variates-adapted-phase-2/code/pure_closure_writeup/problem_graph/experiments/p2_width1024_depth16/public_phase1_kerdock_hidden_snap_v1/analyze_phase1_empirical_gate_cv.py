#!/Users/matt/anaconda3/bin/python3
"""Literal #327723 empirical-gate deep-CV transplant on fixed Kerdock tapes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
ROW = HERE / "runs/row_0000"
FEATURE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0000"
)
NATIVE = (
    P2
    / "public_q24_native_transport_energy_final_v1"
    / "runs/native_rows_remaining_v1/row_0000"
)
BASE_TAPE = ROW / "KERDOCK_TAPE.npz"
SNAP_TAPE = ROW / "KERDOCK_TAPE_ALL_LAYER_SNAP_HERMITE_CV_D12_DEGREES.npz"
DIAGNOSTICS = ROW / "DIAGNOSTIC_ARRAYS_BASE_HERMITE_CV_D12_DEGREES.npz"
RESULT = ROW / "PHASE1_EMPIRICAL_GATE_CV_RESULT.json"
ARRAYS = ROW / "PHASE1_EMPIRICAL_GATE_CV_ARRAYS.npz"

DEPTH = 16
WIDTH = 1024
FIRST_Q = 10
FROZEN_PHASE1_SHRINK = 0.6
ANCHORS_ONE_BASED = tuple(range(8, 15))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.mean(difference * difference))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else 0.0


def physical_q_endpoint(
    observations: np.ndarray,
    bank_prior: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    outputs = np.empty((2, WIDTH), dtype=np.float64)
    for block in range(2):
        q = np.zeros((DEPTH, WIDTH), dtype=np.float64)
        for layer in range(FIRST_Q, DEPTH):
            q[layer] = observations[block, layer] - phi[layer] * (
                observations[block, layer - 1] @ weights[layer]
            )
        state = np.asarray(bank_prior, dtype=np.float64).copy()
        for layer in range(FIRST_Q, DEPTH):
            state = phi[layer] * (state @ weights[layer]) + q[layer]
        outputs[block] = state
    return outputs


def empirical_gate_controls(
    observations: np.ndarray,
    gates: np.ndarray,
    weights: np.ndarray,
    anchor_zero_based: int,
) -> np.ndarray:
    controls = np.empty((2, WIDTH), dtype=np.float64)
    for block in range(2):
        value = np.asarray(observations[block, anchor_zero_based], dtype=np.float64).copy()
        for layer in range(anchor_zero_based + 1, DEPTH):
            value = gates[block, layer] * (value @ weights[layer])
        controls[block] = value
    return controls


def split_fit_shrink(
    measurement: np.ndarray, control: np.ndarray, *, crossfit: bool
) -> tuple[np.ndarray, list[float]]:
    if not crossfit:
        difference_y = measurement[0] - measurement[1]
        difference_c = control[0] - control[1]
        coefficient = float(
            np.dot(difference_y, difference_c)
            / max(np.dot(difference_c, difference_c), np.finfo(np.float64).tiny)
        )
        coefficient = float(np.clip(coefficient, -2.0, 2.0))
        return measurement - coefficient * control, [coefficient]
    output = np.asarray(measurement, dtype=np.float64).copy()
    coefficients: list[float] = []
    for fit_parity, apply_parity in ((0, 1), (1, 0)):
        fit = np.arange(fit_parity, WIDTH, 2)
        apply = np.arange(apply_parity, WIDTH, 2)
        difference_y = measurement[0, fit] - measurement[1, fit]
        difference_c = control[0, fit] - control[1, fit]
        coefficient = float(
            np.dot(difference_y, difference_c)
            / max(np.dot(difference_c, difference_c), np.finfo(np.float64).tiny)
        )
        coefficient = float(np.clip(coefficient, -2.0, 2.0))
        output[:, apply] -= coefficient * control[:, apply]
        coefficients.append(coefficient)
    return output, coefficients


def metrics(
    corrected_error_blocks: np.ndarray,
    bank_final: np.ndarray,
    truth_final: np.ndarray,
    suffix_truth: np.ndarray,
) -> dict[str, float | bool]:
    correction_blocks = corrected_error_blocks - bank_final[None, :]
    average = correction_blocks.mean(axis=0)
    difference = correction_blocks[0] - correction_blocks[1]
    noise = float(np.mean(difference * difference) / 4.0)
    cross = float(np.mean(correction_blocks[0] * correction_blocks[1]))
    signal = max(cross, 0.0)
    beta = signal / (signal + noise + np.finfo(np.float64).tiny)
    desired_full = truth_final - bank_final
    truth_beta = float(
        np.clip(
            np.dot(average, desired_full)
            / max(np.dot(average, average), np.finfo(np.float64).tiny),
            0.0,
            1.0,
        )
    )
    return {
        "split_average_noise_mse": noise,
        "cross_signal_around_bank_mse": signal,
        "split_wiener_beta": beta,
        "closure_plus_cv_final_mse": mse(bank_final + beta * average, truth_final),
        "cv_only_final_mse": mse(corrected_error_blocks.mean(axis=0), truth_final),
        "full_missing_correction_mse": mse(average, desired_full),
        "full_missing_correction_correlation": correlation(average, desired_full),
        "q10_q15_endpoint_measurement_mse": mse(average, suffix_truth),
        "q10_q15_endpoint_correlation": correlation(average, suffix_truth),
        "truth_optimal_beta": truth_beta,
        "truth_optimal_final_mse": mse(bank_final + truth_beta * average, truth_final),
        "improves_bank_with_split_beta": mse(
            bank_final + beta * average, truth_final
        )
        < mse(bank_final, truth_final),
    }


def analyze_mode(
    mode: str,
    tape: dict[str, np.ndarray],
    host: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    bank_history: np.ndarray,
    truth_final: np.ndarray,
    suffix_truth: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    means = tape["mean_blocks"].astype(np.float64)
    gates = tape["gate_blocks"].astype(np.float64)
    observations = means - host[None, :, :]
    direct_measurement = observations[:, -1]
    q_measurement = physical_q_endpoint(
        observations, bank_history[FIRST_Q - 1], phi, weights
    )
    candidates: dict[str, object] = {}
    controls: list[np.ndarray] = []
    for anchor_one_based in ANCHORS_ONE_BASED:
        anchor = anchor_one_based - 1
        control = empirical_gate_controls(observations, gates, weights, anchor)
        controls.append(control)
        for measurement_name, measurement in (
            ("direct_final", direct_measurement),
            ("physical_q10_q15", q_measurement),
        ):
            frozen = measurement - FROZEN_PHASE1_SHRINK * control
            candidates[
                f"{measurement_name}_anchor_h{anchor_one_based}_frozen_0p6"
            ] = {
                "construction": {
                    "anchor_one_based": anchor_one_based,
                    "shrink": FROZEN_PHASE1_SHRINK,
                    "coefficient_source": "literal #327723 frozen control",
                },
                "metrics": metrics(
                    frozen, bank_history[-1], truth_final, suffix_truth
                ),
            }
            for crossfit in (False, True):
                corrected, coefficients = split_fit_shrink(
                    measurement, control, crossfit=crossfit
                )
                candidates[
                    f"{measurement_name}_anchor_h{anchor_one_based}_"
                    f"{'crossfit_' if crossfit else ''}split_fit"
                ] = {
                    "construction": {
                        "anchor_one_based": anchor_one_based,
                        "shrink_coefficients": coefficients,
                        "coefficient_source": (
                            "opposite output parity block disagreement"
                            if crossfit
                            else "all-coordinate block disagreement"
                        ),
                        "target_used": False,
                    },
                    "metrics": metrics(
                        corrected, bank_history[-1], truth_final, suffix_truth
                    ),
                }
    controls_array = np.stack(controls)
    # Literal anchor-family multivariate transplant.  Fit only block
    # disagreement on the opposite output parity, with the direct coefficient
    # fixed at one as required for a control variate.
    for measurement_name, measurement in (
        ("direct_final", direct_measurement),
        ("physical_q10_q15", q_measurement),
    ):
        for ridge in (0.001, 0.01, 0.1, 1.0, 10.0):
            corrected = measurement.copy()
            coefficient_rows: list[list[float]] = []
            for fit_parity, apply_parity in ((0, 1), (1, 0)):
                fit = np.arange(fit_parity, WIDTH, 2)
                apply = np.arange(apply_parity, WIDTH, 2)
                y = measurement[0, fit] - measurement[1, fit]
                design = (
                    controls_array[:, 0, fit] - controls_array[:, 1, fit]
                ).T
                scales = np.sqrt(np.mean(design * design, axis=0))
                normalized = design / np.maximum(scales, 1e-20)
                gram = normalized.T @ normalized
                gram += ridge * len(fit) * np.eye(len(ANCHORS_ONE_BASED))
                coefficient = np.linalg.solve(gram, normalized.T @ y) / np.maximum(
                    scales, 1e-20
                )
                coefficient = np.clip(coefficient, -2.0, 2.0)
                corrected[:, apply] -= np.einsum(
                    "a,abw->bw",
                    coefficient,
                    controls_array[:, :, apply],
                    optimize=True,
                )
                coefficient_rows.append(coefficient.tolist())
            candidates[f"{measurement_name}_multi_anchor_ridge_{ridge:g}"] = {
                "construction": {
                    "anchors_one_based": list(ANCHORS_ONE_BASED),
                    "opposite_parity_coefficients": coefficient_rows,
                    "ridge": ridge,
                    "target_used": False,
                },
                "metrics": metrics(
                    corrected, bank_history[-1], truth_final, suffix_truth
                ),
            }
    # A target-free decision may use only split noise and cross signal.  Freeze
    # the minimum predicted posterior variance candidate, then reveal truth.
    def predicted_posterior(candidate: dict[str, object]) -> float:
        row = candidate["metrics"]
        signal = float(row["cross_signal_around_bank_mse"])
        noise = float(row["split_average_noise_mse"])
        return signal * noise / max(signal + noise, np.finfo(np.float64).tiny)

    selected = min(candidates, key=lambda name: predicted_posterior(candidates[name]))
    summary: dict[str, object] = {
        "mode": mode,
        "candidate_count": len(candidates),
        "selection_used_truth": False,
        "target_free_selection_rule": "minimum split-estimated posterior variance",
        "selected_candidate": selected,
        "selected_postfreeze_truth_metrics": candidates[selected]["metrics"],
        "candidates": candidates,
    }
    arrays = {
        f"{mode}_controls_h8_h14": controls_array,
        f"{mode}_direct_final_measurement": direct_measurement,
        f"{mode}_physical_q_measurement": q_measurement,
    }
    return summary, arrays


def main() -> None:
    host = np.load(NATIVE / "host.npy", allow_pickle=False).astype(np.float64)
    raw = np.load(NATIVE / "raw20.npy", allow_pickle=False)
    phi = raw[:, :, 5].astype(np.float64)
    weights = np.load(FEATURE / "weights.npy", mmap_mode="r", allow_pickle=False).astype(
        np.float64
    )
    diagnostics = np.load(DIAGNOSTICS, allow_pickle=False)
    bank_history = diagnostics["native_j_autonomous"].astype(np.float64)
    truth_final = diagnostics["truth_error"][-1].astype(np.float64)
    suffix_truth = diagnostics["native_j_endpoint_truth"].astype(np.float64)
    result: dict[str, object] = {
        "schema": "phase1-327723-empirical-gate-deep-cv-transplant-v1",
        "status": "complete",
        "actual_phase1_identity": (
            "v=sample_anchor-host_anchor; repeat v=empirical_gate_l*(v@W_l); "
            "sample_final -= shrink*v"
        ),
        "target_use": (
            "No target used in shrink coefficients or candidate selection; rank0 truth "
            "is reported only after each construction is fixed."
        ),
        "inputs": {
            "base_tape_sha256": sha256(BASE_TAPE),
            "snap_tape_sha256": sha256(SNAP_TAPE),
            "diagnostics_sha256": sha256(DIAGNOSTICS),
        },
        "modes": {},
    }
    arrays: dict[str, np.ndarray] = {
        "bank_history": bank_history,
        "truth_final": truth_final,
        "suffix_truth": suffix_truth,
    }
    for mode, path in (("base", BASE_TAPE), ("all_layer_snap", SNAP_TAPE)):
        with np.load(path, allow_pickle=False) as payload:
            tape = {key: payload[key] for key in payload.files}
        summary, mode_arrays = analyze_mode(
            mode,
            tape,
            host,
            phi,
            weights,
            bank_history,
            truth_final,
            suffix_truth,
        )
        result["modes"][mode] = summary
        arrays.update(mode_arrays)
    np.savez_compressed(ARRAYS, **arrays)
    result["arrays"] = {"path": ARRAYS.name, "sha256": sha256(ARRAYS)}
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "selected": {
                    mode: result["modes"][mode]["selected_candidate"]
                    for mode in result["modes"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
