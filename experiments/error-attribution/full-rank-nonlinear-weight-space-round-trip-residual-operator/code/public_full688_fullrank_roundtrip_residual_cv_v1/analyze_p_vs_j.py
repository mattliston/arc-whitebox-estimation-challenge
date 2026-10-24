#!/Users/matt/anaconda3/bin/python3
"""Read-only P2R0F2P-vs-J OOF causal decomposition and emitted-direction oracle.

No training, no model writes, no graph edits, no successor implementation.
Truth is used for analysis only. Writes P_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json
once; raises on any inconsistency with the verified CV artifacts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "p2r0f2p_frozen_runner", HERE / "run_experiment.py"
)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise ImportError("cannot load frozen P runner")
R = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(R)

OUTPUT = HERE / "P_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json"
HARD_ROWS = [201, 310, 599, 643]
NEEDED_CAPTURE = 0.69076  # fraction of J endpoint error removal required for 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    R.configure()
    seal = R.verify_seal()
    R.verify_release(seal)
    cv = read_json(R.CV_RESULT)
    verification = read_json(HERE / "VERIFICATION.json")
    if verification.get("status") != "PASS_INDEPENDENT_REPLAY":
        raise AssertionError("verification has not passed")
    if cv.get("status") != "complete":
        raise AssertionError("CV incomplete")

    dataset = R.PublicResidualDataset()
    R.readiness_or_raise(dataset)
    error, sums, sums2, folds = R.load_shared_cache(dataset)

    width = R.WIDTH
    j_error_final = np.zeros((688, width), dtype=np.float64)
    delta_final = np.zeros((688, width), dtype=np.float64)
    j_layer_sq = np.zeros((688, R.DEPTH), dtype=np.float64)
    p_layer_sq = np.zeros((688, R.DEPTH), dtype=np.float64)
    cv_order_weighted_final = 0.0

    for fold in range(R.FOLDS):
        record = read_json(HERE / "folds" / f"fold_{fold}.json")
        R.verify_completed_fold_record(record, fold, seal)
        artifact = torch.load(
            HERE / "folds" / f"fold_{fold}.pt", map_location="cpu", weights_only=False
        )
        state = artifact["state_dict"]
        observer = R.FullRankRoundTripResidualObserver(
            state["centers"].cpu().numpy(),
            state["scales"].cpu().numpy(),
            state["layer_rms"].cpu().numpy(),
        )
        observer.load_state_dict(state, strict=True)
        observer.eval()
        j_model = R.load_outer_j_model(fold, seal)
        valid = np.flatnonzero(folds == fold)
        fold_final_sse = 0.0
        with torch.no_grad():
            for offset in range(0, len(valid), 2):
                group = valid[offset : offset + 2]
                raw, sigma, weights, truth = R.load_batch(dataset, group, error)
                j_prediction, j_states, j_energies = j_model(
                    raw, sigma, weights, exposure=1.0, return_states=True
                )
                delta = observer(raw, sigma, weights, j_states, j_energies)
                j_err = (j_prediction - truth).double().numpy()
                d = delta.double().numpy()
                squared = ((j_prediction + delta) - truth).double().square()
                fold_final_sse += float(squared[:, -1].sum())
                for position, rank in enumerate(group):
                    j_error_final[rank] = j_err[position, -1]
                    delta_final[rank] = d[position, -1]
                    j_layer_sq[rank] = np.mean(j_err[position] ** 2, axis=-1)
                    p_layer_sq[rank] = np.mean(
                        (j_err[position] + d[position]) ** 2, axis=-1
                    )
        cv_order_weighted_final += len(valid) * (
            fold_final_sse / (len(valid) * width)
        )

    cv_order_final = cv_order_weighted_final / 688.0
    j_row = j_layer_sq[:, -1]
    p_row = p_layer_sq[:, -1]
    j_final = float(j_row.mean())
    p_final = float(p_row.mean())
    cv_reported = float(cv["oof"]["autonomous_final_mse"])
    if abs(cv_order_final - cv_reported) > 1e-15 * cv_reported:
        raise AssertionError(
            f"CV-order replay {cv_order_final} != CV {cv_reported}"
        )
    if abs(p_final - cv_reported) > 1e-8 * cv_reported:
        raise AssertionError(
            f"per-row replay {p_final} is inconsistent with CV {cv_reported}"
        )

    row_delta = p_row - j_row
    improvement = -row_delta[row_delta < 0].sum()
    regression = row_delta[row_delta > 0].sum()
    gross = float(np.abs(row_delta).sum())
    cancellation = float(min(improvement, regression) * 2.0 / gross) if gross else 0.0

    # Exact per-row MSE identity: p - j = 2<e_J, d>/W + |d|^2/W.
    cross_term = 2.0 * np.einsum("nw,nw->n", j_error_final, delta_final) / width
    quad_term = np.einsum("nw,nw->n", delta_final, delta_final) / width
    identity_residual = float(np.abs(row_delta - (cross_term + quad_term)).max())

    delta_norm = np.sqrt(np.einsum("nw,nw->n", delta_final, delta_final))
    error_norm = np.sqrt(np.einsum("nw,nw->n", j_error_final, j_error_final))
    inner = np.einsum("nw,nw->n", delta_final, j_error_final)
    safe = (delta_norm > 0) & (error_norm > 0)
    cosine_to_negative_error = np.zeros(688)
    cosine_to_negative_error[safe] = -inner[safe] / (
        delta_norm[safe] * error_norm[safe]
    )
    gain = -row_delta

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # Emitted-direction oracle: per-row truth-optimal scalar on the actual
    # emitted final-layer correction direction (analysis only, 0 parameters).
    captured_sq = np.zeros(688)
    captured_sq[safe] = (inner[safe] / delta_norm[safe]) ** 2
    oracle_row = (error_norm**2 - captured_sq) / width
    oracle_final = float(oracle_row.mean())
    captured_fraction = float(1.0 - oracle_final / j_final)

    layer_j = j_layer_sq.mean(axis=0)
    layer_p = p_layer_sq.mean(axis=0)
    layer_gain = [
        {
            "layer": layer,
            "j_mse": float(layer_j[layer]),
            "p_mse": float(layer_p[layer]),
            "fractional_improvement": float(
                (layer_j[layer] - layer_p[layer]) / layer_j[layer]
            )
            if layer_j[layer] > 0
            else 0.0,
        }
        for layer in range(R.FIRST_CORRECTED_LAYER, R.DEPTH)
    ]

    fold_rows = []
    for fold in range(R.FOLDS):
        valid = np.flatnonzero(folds == fold)
        fold_rows.append(
            {
                "fold": fold,
                "networks": int(len(valid)),
                "p_minus_j": float(row_delta[valid].mean()),
                "fractional_improvement": float(
                    -row_delta[valid].mean() / j_row[valid].mean()
                ),
                "wins": int((row_delta[valid] < 0).sum()),
                "losses": int((row_delta[valid] > 0).sum()),
            }
        )

    scalars = {
        str(fold): read_json(HERE / "folds" / f"fold_{fold}.json")["learned_scalars"]
        for fold in range(R.FOLDS)
    }

    payload = {
        "schema": "p2r0f2p-vs-j-oof-decomposition-and-closure-v1",
        "status": "PASS_CLOSED" if not cv["decision_gate"]["conjunctive_pass"] else "PASS_GATE",
        "node": "P2R0F2P",
        "scope": (
            "Graph-ready read-only observer_fit OOF causal decomposition and one "
            "parameter-free emitted-direction oracle; no candidate evaluation, no "
            "training, no retune"
        ),
        "bindings": {
            "analysis_program_sha256": sha256(Path(__file__)),
            "cv_results_sha256": sha256(R.CV_RESULT),
            "verification_sha256": sha256(HERE / "VERIFICATION.json"),
            "seal_sha256": sha256(R.SEAL),
            "audit_receipt_sha256": seal["audit_receipt_sha256"],
        },
        "firewalls": {
            "role": "observer_fit",
            "observer_fit_rows_replayed": 688,
            "observer_selection_targets_opened": 0,
            "public_name_joins": 0,
            "network_access": 0,
            "training_steps": 0,
            "model_writes": 0,
            "graph_edits": 0,
            "successor_implementations": 0,
            "operator_oracles_tested": 1,
        },
        "paired_final": {
            "j_oof_final_mse": j_final,
            "p_oof_final_mse": p_final,
            "p_minus_j": float(row_delta.mean()),
            "p_fractional_improvement": float(-row_delta.mean() / j_final),
            "wins": int((row_delta < 0).sum()),
            "losses": int((row_delta > 0).sum()),
            "ties": int((row_delta == 0).sum()),
            "negative_improvement_sum": float(-improvement),
            "positive_regression_sum": float(regression),
            "gross_absolute_sum": gross,
            "cancellation_fraction": cancellation,
            "row_delta_p10": float(np.percentile(row_delta, 10)),
            "row_delta_median": float(np.percentile(row_delta, 50)),
            "row_delta_p90": float(np.percentile(row_delta, 90)),
            "exact_mse_identity_max_row_residual": identity_residual,
        },
        "folds": fold_rows,
        "layerwise": layer_gain,
        "hard_rows": {
            "declared_role_ranks": HARD_ROWS,
            "j_mean": float(j_row[HARD_ROWS].mean()),
            "p_mean": float(p_row[HARD_ROWS].mean()),
            "row_p_minus_j": {
                str(rank): float(row_delta[rank]) for rank in HARD_ROWS
            },
            "breaches_above_1e_7_after_p": [
                int(rank) for rank in range(688) if p_row[rank] > 1e-7
            ],
        },
        "amplitude_and_alignment": {
            "final_delta_rms": float(np.sqrt(quad_term.mean())),
            "mean_absolute_delta_to_error_norm_ratio": float(
                (delta_norm[safe] / error_norm[safe]).mean()
            ),
            "mean_delta_negative_error_cosine": float(
                cosine_to_negative_error.mean()
            ),
            "gain_pearson_with_delta_negative_error_cosine": corr(
                gain, cosine_to_negative_error
            ),
            "gain_pearson_with_delta_rms": corr(gain, delta_norm),
            "learned_scalars_by_fold": scalars,
        },
        "emitted_direction_oracle": {
            "operator": (
                "Per row, truth-optimal scalar on the emitted final-layer round-trip "
                "direction d: oracle error = e - (<e,d>/|d|^2) d."
            ),
            "free_parameters": 0,
            "j_final_mse": j_final,
            "oracle_final_mse": oracle_final,
            "captured_error_fraction": captured_fraction,
            "needed_error_fraction_for_1e_8": NEEDED_CAPTURE,
            "factor_above_1e_8": oracle_final / 1e-8,
            "reaches_1e_8": bool(oracle_final < 1e-8),
        },
        "decision_gate_recorded": cv["decision_gate"],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(
        json.dumps(
            {
                "p_final": p_final,
                "j_final": j_final,
                "fractional_improvement": payload["paired_final"][
                    "p_fractional_improvement"
                ],
                "cancellation": cancellation,
                "oracle_final": oracle_final,
                "captured_fraction": captured_fraction,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
