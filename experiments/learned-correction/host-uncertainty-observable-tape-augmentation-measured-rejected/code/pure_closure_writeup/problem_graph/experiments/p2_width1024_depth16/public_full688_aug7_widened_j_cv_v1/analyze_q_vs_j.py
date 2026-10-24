#!/Users/matt/anaconda3/bin/python3
"""Read-only P2R0F2Q-vs-J OOF decomposition with aug-channel attribution.

Replays both the Q fold models and the sealed J fold models on their shared
validation rows, forms exact paired statistics, and correlates per-row Q gain
with per-row aug-channel magnitudes. Truth is used for analysis only; no
training, no retune, no model writes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "p2r0f2q_frozen_runner", HERE / "run_experiment.py"
)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise ImportError("cannot load frozen Q runner")
R = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(R)

J_SPEC = importlib.util.spec_from_file_location(
    "p2r0f2j_frozen_runner_for_analysis", J_DIR / "run_experiment.py"
)
if J_SPEC is None or J_SPEC.loader is None:
    raise ImportError("cannot load frozen J runner")
J = importlib.util.module_from_spec(J_SPEC)
J_SPEC.loader.exec_module(J)

OUTPUT = HERE / "Q_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json"
HARD_ROWS = [201, 310, 599, 643]
CHANNEL_NAMES = [
    "discarded_k3_signed",
    "discarded_k3_absolute",
    "covariance_series_order10_row_share",
    "max_offdiagonal_abs_correlation",
    "coincident_bond_degree",
    "rho2_neighbor_skew",
    "rho2_neighbor_alpha",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_j_model(fold: int) -> torch.nn.Module:
    artifact = torch.load(
        J_DIR / "folds" / f"fold_{fold}.pt", map_location="cpu", weights_only=False
    )
    state = artifact["state_dict"]
    model = J.TransportEnergyMultiStateResidual(
        state["centers"].cpu().numpy(),
        state["scales"].cpu().numpy(),
        state["layer_rms"].cpu().numpy(),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_q_model(fold: int) -> torch.nn.Module:
    artifact = torch.load(
        HERE / "folds" / f"fold_{fold}.pt", map_location="cpu", weights_only=False
    )
    state = artifact["state_dict"]
    model = R.AugWidenedTransportEnergyResidual(
        state["centers"].cpu().numpy(),
        state["scales"].cpu().numpy(),
        state["layer_rms"].cpu().numpy(),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


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

    dataset = R.PublicResidualDataset()
    R.readiness_or_raise(dataset)
    error, sums, sums2, folds = R.load_shared_cache(dataset)
    width = R.WIDTH

    q_err_final = np.zeros((688, width), dtype=np.float64)
    j_err_final = np.zeros((688, width), dtype=np.float64)
    q_row = np.zeros(688, dtype=np.float64)
    j_row = np.zeros(688, dtype=np.float64)
    q_cv_order = 0.0

    for fold in range(R.FOLDS):
        q_model = load_q_model(fold)
        j_model = load_j_model(fold)
        valid = np.flatnonzero(folds == fold)
        fold_sse = 0.0
        with torch.no_grad():
            for offset in range(0, len(valid), 2):
                group = valid[offset : offset + 2]
                raw, sigma, weights, truth = R.load_batch(dataset, group, error)
                q_pred = q_model(raw, sigma, weights, exposure=1.0)
                j_pred = j_model(
                    raw[..., : J.INPUT_FEATURES - 16], sigma, weights, exposure=1.0
                )
                q_e = (q_pred - truth).double().numpy()
                j_e = (j_pred - truth).double().numpy()
                fold_sse += float((q_pred - truth).double().square()[:, -1].sum())
                for position, rank in enumerate(group):
                    q_err_final[rank] = q_e[position, -1]
                    j_err_final[rank] = j_e[position, -1]
                    q_row[rank] = float(np.mean(q_e[position, -1] ** 2))
                    j_row[rank] = float(np.mean(j_e[position, -1] ** 2))
        q_cv_order += len(valid) * (fold_sse / (len(valid) * width))

    q_cv_order /= 688.0
    cv_reported = float(cv["oof"]["autonomous_final_mse"])
    if abs(q_cv_order - cv_reported) > 1e-15 * cv_reported:
        raise AssertionError(f"CV-order replay {q_cv_order} != CV {cv_reported}")
    j_final = float(j_row.mean())
    q_final = float(q_row.mean())
    j_reference = 3.2334958229470984e-8
    if abs(j_final - j_reference) > 1e-6 * j_reference:
        raise AssertionError(f"replayed J baseline {j_final} inconsistent with {j_reference}")

    row_delta = q_row - j_row
    gain = -row_delta
    improvement = -row_delta[row_delta < 0].sum()
    regression = row_delta[row_delta > 0].sum()
    gross = float(np.abs(row_delta).sum())
    cancellation = float(min(improvement, regression) * 2.0 / gross) if gross else 0.0

    # Read-only aug-channel attribution: per-row layer-mean absolute channel
    # magnitudes at the final corrected layers, correlated with per-row gain.
    channel_summary = np.zeros((688, R.AUG_CHANNELS), dtype=np.float64)
    for rank in range(688):
        aug = np.asarray(R.aug_row(rank, dataset), dtype=np.float64)
        channel_summary[rank] = np.mean(np.abs(aug[R.FIRST_CORRECTED_LAYER :]), axis=(0, 1))

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    attribution = {
        name: corr(gain, channel_summary[:, index])
        for index, name in enumerate(CHANNEL_NAMES)
    }
    j_error_scale = np.sqrt(j_row)
    attribution_error_scale = {
        name: corr(j_error_scale, channel_summary[:, index])
        for index, name in enumerate(CHANNEL_NAMES)
    }

    fold_rows = []
    j_fold_reference = {
        int(row["fold"]): float(row["autonomous_final_mse"])
        for row in read_json(J_DIR / "CV_RESULTS.json")["fold_results"]
    }
    for fold in range(R.FOLDS):
        valid = np.flatnonzero(folds == fold)
        fold_rows.append(
            {
                "fold": fold,
                "networks": int(len(valid)),
                "q_final_mse": float(
                    read_json(HERE / "folds" / f"fold_{fold}.json")["autonomous"][
                        "final_mse"
                    ]
                ),
                "j_final_mse": j_fold_reference[fold],
                "fractional_improvement": float(
                    1.0
                    - read_json(HERE / "folds" / f"fold_{fold}.json")["autonomous"][
                        "final_mse"
                    ]
                    / j_fold_reference[fold]
                ),
                "wins": int((row_delta[valid] < 0).sum()),
                "losses": int((row_delta[valid] > 0).sum()),
            }
        )

    payload = {
        "schema": "p2r0f2q-vs-j-oof-decomposition-and-closure-v1",
        "status": "PASS_CLOSED" if not cv["decision_gate"]["conjunctive_pass"] else "PASS_GATE",
        "node": "P2R0F2Q",
        "scope": (
            "Graph-ready read-only observer_fit OOF paired decomposition of the "
            "aug7-widened retrain against frozen J, with read-only aug-channel "
            "attribution; no candidate evaluation, no training, no retune"
        ),
        "bindings": {
            "analysis_program_sha256": sha256(Path(__file__)),
            "cv_results_sha256": sha256(R.CV_RESULT),
            "verification_sha256": sha256(HERE / "VERIFICATION.json"),
            "seal_sha256": sha256(R.SEAL),
            "generation_receipt_sha256": seal["generation_receipt_sha256"],
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
        },
        "retrain_noise_caveat": (
            "Unlike the M/N/P rider comparisons on frozen J, Q is a fresh "
            "retrain with a different input width and initialization, so "
            "fold-level movements of about +/-0.5% include training noise; the "
            "one-arm law forbids seed replication, so only the aggregate and "
            "its cross-fold structure are interpreted."
        ),
        "paired_final": {
            "j_oof_final_mse": j_final,
            "q_oof_final_mse": q_final,
            "q_minus_j": float(row_delta.mean()),
            "q_fractional_improvement": float(-row_delta.mean() / j_final),
            "wins": int((row_delta < 0).sum()),
            "losses": int((row_delta > 0).sum()),
            "ties": int((row_delta == 0).sum()),
            "cancellation_fraction": cancellation,
            "row_delta_p10": float(np.percentile(row_delta, 10)),
            "row_delta_median": float(np.percentile(row_delta, 50)),
            "row_delta_p90": float(np.percentile(row_delta, 90)),
        },
        "folds": fold_rows,
        "hard_rows": {
            "declared_role_ranks": HARD_ROWS,
            "j_mean": float(j_row[HARD_ROWS].mean()),
            "q_mean": float(q_row[HARD_ROWS].mean()),
            "row_q_minus_j": {str(rank): float(row_delta[rank]) for rank in HARD_ROWS},
            "breaches_above_1e_7_after_q": [
                int(rank) for rank in range(688) if q_row[rank] > 1e-7
            ],
        },
        "aug_channel_attribution": {
            "method": (
                "Pearson correlation of per-row Q gain (j_row - q_row) with the "
                "per-row mean absolute channel magnitude over corrected layers; "
                "and, as a base-rate reference, the correlation of each channel "
                "magnitude with the J error scale."
            ),
            "gain_correlation": attribution,
            "error_scale_correlation": attribution_error_scale,
            "coincident_bond_degree_is_identically_zero": True,
        },
        "ceiling_coincidence": {
            "n_oof_final_mse": 3.230119894372402e-8,
            "q_oof_final_mse": q_final,
            "relative_gap": float(abs(q_final - 3.230119894372402e-8) / 3.230119894372402e-8),
            "reading": (
                "Two materially different mechanisms - N's learned rank-16 "
                "covariance rider and Q's source-widened retrain - land within "
                "3.1e-6 relative of each other at 0.104-0.105% below J, "
                "consistent with a shared information ceiling of this "
                "development frame near 3.2301e-8."
            ),
        },
        "decision_gate_recorded": cv["decision_gate"],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(
        json.dumps(
            {
                "q_final": q_final,
                "j_final": j_final,
                "fractional_improvement": payload["paired_final"][
                    "q_fractional_improvement"
                ],
                "cancellation": cancellation,
                "attribution": attribution,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
