#!/Users/matt/anaconda3/bin/python3
"""Target-free degreewise endpoint control audit on the fixed rank-0 tapes."""

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
RESULT = ROW / "DEGREE_SHRINK_RESULT.json"
ARRAYS = ROW / "DEGREE_SHRINK_ARRAYS.npz"

DEPTH = 16
WIDTH = 1024
FIRST_Q = 10
LAST_Q = 15
REQUIRED_ENDPOINT_MSE = 2.075e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else 0.0


def mse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.mean(difference * difference))


def physical_q_blocks(
    means: np.ndarray, host: np.ndarray, phi: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    observation = np.asarray(means - host[None, :, :], dtype=np.float64)
    q = np.zeros_like(observation)
    q[:, 2] = observation[:, 2]
    for layer in range(3, DEPTH):
        for block in range(2):
            q[block, layer] = observation[block, layer] - phi[layer] * (
                observation[block, layer - 1] @ weights[layer]
            )
    return q


def endpoint_from_q(
    q: np.ndarray, bank_prior: np.ndarray, phi: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    outputs = np.empty((2, WIDTH), dtype=np.float64)
    for block in range(2):
        state = np.asarray(bank_prior, dtype=np.float64).copy()
        for layer in range(FIRST_Q, DEPTH):
            state = phi[layer] * (state @ weights[layer]) + q[block, layer]
        outputs[block] = state
    return outputs


def correction_endpoint_actions(
    corrections: np.ndarray, phi: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Map 13 per-degree post-mean corrections to final physical actions."""
    actions = np.empty((13, 2, WIDTH), dtype=np.float64)
    for degree in range(13):
        for block in range(2):
            q_correction = np.zeros((DEPTH, WIDTH), dtype=np.float64)
            for layer in range(FIRST_Q, DEPTH):
                q_correction[layer] = corrections[degree, block, layer] - phi[layer] * (
                    corrections[degree, block, layer - 1] @ weights[layer]
                )
            state = np.zeros(WIDTH, dtype=np.float64)
            for layer in range(FIRST_Q, DEPTH):
                state = phi[layer] * (state @ weights[layer]) + q_correction[layer]
            actions[degree, block] = state
    return actions


def candidate_metrics(
    endpoint_blocks: np.ndarray,
    endpoint_truth: np.ndarray,
    bank_final: np.ndarray,
    truth_final: np.ndarray,
) -> dict[str, float | bool]:
    blocks = np.asarray(endpoint_blocks, dtype=np.float64)
    average = blocks.mean(axis=0)
    difference = blocks[0] - blocks[1]
    noise = float(np.mean(difference * difference) / 4.0)
    cross = float(np.mean(blocks[0] * blocks[1]))
    signal = max(cross, 0.0)
    beta = signal / (signal + noise + np.finfo(np.float64).tiny)
    desired = truth_final - bank_final
    denominator = float(np.dot(average, average))
    truth_beta = float(
        np.clip(np.dot(average, desired) / max(denominator, np.finfo(np.float64).tiny), 0.0, 1.0)
    )
    endpoint_error = mse(average, endpoint_truth)
    truth_correlation = correlation(average, endpoint_truth)
    return {
        "split_average_noise_mse": noise,
        "cross_signal_around_bank_mse": signal,
        "split_wiener_beta": beta,
        "split_wiener_final_mse": mse(bank_final + beta * average, truth_final),
        "endpoint_measurement_error_mse": endpoint_error,
        "endpoint_truth_correlation": truth_correlation,
        "truth_optimal_beta": truth_beta,
        "truth_optimal_final_mse": mse(bank_final + truth_beta * average, truth_final),
        "right_sign": truth_correlation > 0.0,
        "split_noise_gate": noise < REQUIRED_ENDPOINT_MSE,
        "actual_endpoint_gate": endpoint_error < REQUIRED_ENDPOINT_MSE,
    }


def crossfit_scalar(
    base: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    output = np.asarray(base, dtype=np.float64).copy()
    coefficients: list[float] = []
    for fit_parity, apply_parity in ((0, 1), (1, 0)):
        fit = np.arange(fit_parity, WIDTH, 2)
        apply = np.arange(apply_parity, WIDTH, 2)
        base_difference = base[0, fit] - base[1, fit]
        control_difference = direction[0, fit] - direction[1, fit]
        denominator = float(np.dot(control_difference, control_difference))
        coefficient = -float(np.dot(base_difference, control_difference)) / max(
            denominator, np.finfo(np.float64).tiny
        )
        coefficient = float(np.clip(coefficient, -2.0, 2.0))
        output[:, apply] += coefficient * direction[:, apply]
        coefficients.append(coefficient)
    return output, coefficients


def crossfit_degreewise(
    base: np.ndarray,
    controls: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, list[list[float]]]:
    """Fit block-disagreement cancellation on the opposite output parity."""
    output = np.asarray(base, dtype=np.float64).copy()
    coefficient_rows: list[list[float]] = []
    for fit_parity, apply_parity in ((0, 1), (1, 0)):
        fit = np.arange(fit_parity, WIDTH, 2)
        apply = np.arange(apply_parity, WIDTH, 2)
        y = -(base[0, fit] - base[1, fit])
        design = (controls[:, 0, fit] - controls[:, 1, fit]).T
        scales = np.sqrt(np.mean(design * design, axis=0))
        active = scales > 1e-20
        normalized = design[:, active] / scales[active]
        gram = normalized.T @ normalized
        gram += float(ridge * len(fit)) * np.eye(gram.shape[0])
        normalized_coefficient = np.linalg.solve(gram, normalized.T @ y)
        coefficient = np.zeros(controls.shape[0], dtype=np.float64)
        coefficient[active] = normalized_coefficient / scales[active]
        coefficient = np.clip(coefficient, -2.0, 2.0)
        output[:, apply] += np.einsum(
            "d,dbw->bw", coefficient, controls[:, :, apply], optimize=True
        )
        coefficient_rows.append(coefficient.tolist())
    return output, coefficient_rows


def analyze_mode(
    name: str,
    tape: dict[str, np.ndarray],
    host: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    bank_history: np.ndarray,
    endpoint_truth: np.ndarray,
    truth_final: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    means = tape["mean_blocks"]
    corrections = tape["cv_degree_correction_blocks"].astype(np.float64)
    q = physical_q_blocks(means, host, phi, weights)
    endpoint_rows = endpoint_from_q(q, bank_history[FIRST_Q - 1], phi, weights)
    base = endpoint_rows - bank_history[-1][None, :]
    degree_actions = correction_endpoint_actions(corrections, phi, weights)
    candidates: dict[str, dict[str, object]] = {}

    def add(candidate_name: str, blocks: np.ndarray, construction: dict[str, object]) -> None:
        candidates[candidate_name] = {
            "construction": construction,
            "metrics": candidate_metrics(
                blocks, endpoint_truth, bank_history[-1], truth_final
            ),
        }

    add("base", base, {"base_weight": 1.0, "degree_weights": [0.0] * 13})
    for degree in range(1, 13):
        weights_row = np.zeros(13)
        weights_row[degree] = 1.0
        add(
            f"single_degree_{degree}",
            base + degree_actions[degree],
            {"base_weight": 1.0, "degree_weights": weights_row.tolist()},
        )
    for stop in (2, 4, 6, 8, 12):
        weights_row = np.zeros(13)
        weights_row[: stop + 1] = 1.0
        cumulative = np.sum(degree_actions[: stop + 1], axis=0)
        add(
            f"cumulative_D{stop}",
            base + cumulative,
            {"base_weight": 1.0, "degree_weights": weights_row.tolist()},
        )
        partial, coefficients = crossfit_scalar(base, cumulative)
        add(
            f"crossfit_partial_D{stop}",
            partial,
            {
                "base_weight": 1.0,
                "direction": f"cumulative_D{stop}-base",
                "opposite_parity_coefficients": coefficients,
                "fit_criterion": "block disagreement only; clipped [-2,2]",
            },
        )
    full = np.sum(degree_actions[:13], axis=0)
    for omitted in range(1, 13):
        weights_row = np.ones(13)
        weights_row[0] = 0.0
        weights_row[omitted] = 0.0
        add(
            f"D12_leave_degree_{omitted}_out",
            base + full - degree_actions[omitted],
            {"base_weight": 1.0, "degree_weights": weights_row.tolist()},
        )
    controls = degree_actions[1:13]
    for ridge in (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0):
        blocks, coefficients = crossfit_degreewise(base, controls, ridge)
        add(
            f"crossfit_degreewise_ridge_{ridge:g}",
            blocks,
            {
                "base_weight": 1.0,
                "degrees": list(range(1, 13)),
                "opposite_parity_coefficients": coefficients,
                "ridge": ridge,
                "fit_criterion": "block disagreement only; coefficients clipped [-2,2]",
            },
        )

    feature_actions = np.concatenate((base[None, :, :], degree_actions), axis=0)
    differences = feature_actions[:, 0] - feature_actions[:, 1]
    noise_covariance = differences @ differences.T / (4.0 * WIDTH)
    cross_covariance = (
        feature_actions[:, 0] @ feature_actions[:, 1].T
        + feature_actions[:, 1] @ feature_actions[:, 0].T
    ) / (2.0 * WIDTH)
    eligible = [
        candidate_name
        for candidate_name, candidate in candidates.items()
        if candidate["metrics"]["split_noise_gate"]
        and candidate["metrics"]["cross_signal_around_bank_mse"] > 0.0
    ]
    selected = min(
        eligible,
        key=lambda candidate_name: candidates[candidate_name]["metrics"][
            "split_average_noise_mse"
        ],
        default=None,
    )
    summary: dict[str, object] = {
        "mode": name,
        "selection_used_truth": False,
        "target_free_selection_rule": (
            "among positive cross-signal candidates passing split R<2.075e-9, "
            "choose minimum split R"
        ),
        "selected_candidate": selected,
        "selected_candidate_postfreeze_truth_metrics": (
            candidates[selected]["metrics"] if selected is not None else None
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    arrays = {
        f"{name}_endpoint_base_blocks": base,
        f"{name}_endpoint_degree_actions": degree_actions,
        f"{name}_endpoint_feature_noise_covariance": noise_covariance,
        f"{name}_endpoint_feature_cross_covariance": cross_covariance,
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
    endpoint_truth = diagnostics["native_j_endpoint_truth"].astype(np.float64)
    truth_final = diagnostics["truth_error"][-1].astype(np.float64)

    outputs: dict[str, np.ndarray] = {
        "endpoint_truth": endpoint_truth,
        "bank_history": bank_history,
        "truth_final": truth_final,
    }
    summaries: dict[str, object] = {}
    for mode, path in (("base", BASE_TAPE), ("all_layer_snap", SNAP_TAPE)):
        with np.load(path, allow_pickle=False) as payload:
            tape = {key: payload[key] for key in payload.files}
        summary, arrays = analyze_mode(
            mode,
            tape,
            host,
            phi,
            weights,
            bank_history,
            endpoint_truth,
            truth_final,
        )
        summaries[mode] = summary
        outputs.update(arrays)

    # Joint target-free action basis: base sample plus its degree controls,
    # the physical-snap delta, and the snap degree controls.
    base_actions = outputs["base_endpoint_base_blocks"]
    base_controls = outputs["base_endpoint_degree_actions"][1:]
    snap_base = outputs["all_layer_snap_endpoint_base_blocks"]
    snap_controls = outputs["all_layer_snap_endpoint_degree_actions"][1:]
    joint_controls = np.concatenate(
        ((snap_base - base_actions)[None, :, :], base_controls, snap_controls), axis=0
    )
    joint_candidates: dict[str, object] = {}
    for ridge in (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0):
        blocks, coefficients = crossfit_degreewise(base_actions, joint_controls, ridge)
        joint_candidates[f"ridge_{ridge:g}"] = {
            "construction": {
                "controls": ["physical_snap_delta"]
                + [f"base_degree_{degree}" for degree in range(1, 13)]
                + [f"snap_degree_{degree}" for degree in range(1, 13)],
                "opposite_parity_coefficients": coefficients,
                "ridge": ridge,
                "fit_criterion": "block disagreement only; coefficients clipped [-2,2]",
            },
            "metrics": candidate_metrics(
                blocks, endpoint_truth, bank_history[-1], truth_final
            ),
        }
    summaries["joint"] = {
        "selection_used_truth": False,
        "candidates": joint_candidates,
    }
    outputs["joint_controls"] = joint_controls

    payload = {
        "schema": "phase1-kerdock-degreewise-endpoint-shrink-v1",
        "status": "complete",
        "required_endpoint_mse": REQUIRED_ENDPOINT_MSE,
        "target_use": (
            "No target used in coefficients or candidate selection; rank0 truth opened "
            "only after each construction is fixed to report sign/error."
        ),
        "inputs": {
            "base_tape": {"path": BASE_TAPE.name, "sha256": sha256(BASE_TAPE)},
            "snap_tape": {"path": SNAP_TAPE.name, "sha256": sha256(SNAP_TAPE)},
            "diagnostics": {"path": DIAGNOSTICS.name, "sha256": sha256(DIAGNOSTICS)},
        },
        "modes": summaries,
    }
    np.savez_compressed(ARRAYS, **outputs)
    payload["arrays"] = {"path": ARRAYS.name, "sha256": sha256(ARRAYS)}
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(RESULT), "selected": {
        name: summary.get("selected_candidate") for name, summary in summaries.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
