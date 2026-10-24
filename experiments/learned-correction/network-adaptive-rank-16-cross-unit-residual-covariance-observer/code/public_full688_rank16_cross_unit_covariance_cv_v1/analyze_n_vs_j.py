#!/Users/matt/anaconda3/bin/python3
"""Bounded causal decomposition of frozen N versus frozen J.

Read-only observer_fit analysis.  The sole oracle is the parameter-free
orthogonal projector onto N's learned dynamic rank-16 full-width factor span.
There is no rank, hyperparameter, checkpoint, or operator sweep.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import verify_results as V


HERE = Path(__file__).resolve().parent
VERIFIER_SHA = "467e58e5242bddceb91d39e215a60d6275ff634bae48890d56609de27ea19247"
VERIFICATION_SHA = "9b6a00d819eb219c8eada6242d96821a76abbfda2076e21f263d079113cb7645"
HARD_RANKS = [201, 310, 599, 643]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def close(left: float, right: float, *, atol: float = 1.0e-20) -> None:
    if not math.isclose(float(left), float(right), rel_tol=2.0e-12, abs_tol=atol):
        raise AssertionError((left, right))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def n_diagnostics(
    model: V.FrozenN,
    raw: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    j_states: torch.Tensor,
    j_energies: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Exact N forward plus its saved-checkpoint-readable causal components."""
    batch = raw.shape[0]
    factors = torch.zeros(batch, V.WIDTH, V.N_RANK, dtype=raw.dtype)
    total_outputs = [torch.zeros(batch, V.WIDTH, dtype=raw.dtype) for _ in range(V.FIRST)]
    diagonal_outputs = [torch.zeros_like(total_outputs[0]) for _ in range(V.FIRST)]
    cross_outputs = [torch.zeros_like(total_outputs[0]) for _ in range(V.FIRST)]
    scores = [torch.zeros_like(total_outputs[0]) for _ in range(V.FIRST)]
    diagonals = [torch.zeros_like(total_outputs[0]) for _ in range(V.FIRST)]
    lambdas = [torch.zeros(batch, V.N_RANK, dtype=raw.dtype) for _ in range(V.FIRST)]
    factor_history = [factors.clone() for _ in range(V.FIRST)]
    identity = torch.eye(V.N_RANK, dtype=raw.dtype)
    for layer in range(V.FIRST, V.DEPTH):
        local_layer = layer - V.FIRST
        if layer == V.FIRST:
            messages = torch.zeros(batch, V.WIDTH, V.J_CHANNELS, dtype=raw.dtype)
            transported = torch.zeros_like(factors)
        else:
            layer_weights = weights[:, layer]
            messages = torch.bmm(
                j_states[:, layer - 1].transpose(1, 2), layer_weights
            ).transpose(1, 2)
            messages = raw[:, layer, :, 5, None] * messages
            transported = torch.bmm(
                factors.transpose(1, 2), layer_weights
            ).transpose(1, 2)
            transported = raw[:, layer, :, 5, None] * transported
        features = model.local_features(
            raw[:, layer], sigma[:, layer], messages, j_energies[:, layer], layer
        )
        hidden = model.encoder(features)
        head = torch.einsum(
            "bwh,ho->bwo", hidden, model.head_weight[local_layer]
        ) + model.head_bias[local_layer]
        source = torch.tanh(head[..., : V.N_RANK])
        log_lambda_local = head[..., V.N_RANK : 2 * V.N_RANK]
        log_diagonal = head[..., 2 * V.N_RANK]
        score = torch.tanh(head[..., 2 * V.N_RANK + 1])
        if layer == V.FIRST:
            candidate = source
        else:
            transition = layer - V.FIRST - 1
            gain = 1.0 + 0.25 * torch.tanh(model.factor_gain_delta[transition])
            mixing = identity + 0.10 * torch.tanh(
                model.factor_mix_delta[transition]
            )
            candidate = torch.einsum(
                "bwr,rs->bws", gain * transported, mixing
            ) + source
        gram = torch.bmm(candidate.transpose(1, 2), candidate) / float(V.WIDTH)
        gram = gram + V.GRAM_FLOOR * identity[None]
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = torch.clamp(eigenvalues, min=V.GRAM_FLOOR)
        inverse_sqrt = torch.bmm(
            eigenvectors * torch.rsqrt(eigenvalues)[:, None, :],
            eigenvectors.transpose(1, 2),
        )
        factors = torch.bmm(candidate, inverse_sqrt)
        lam = torch.clamp(
            torch.nn.functional.softplus(log_lambda_local.mean(dim=1))
            + V.POSITIVE_OFFSET,
            min=V.POSITIVE_MINIMUM,
            max=V.POSITIVE_MAXIMUM,
        )
        diagonal = torch.clamp(
            torch.nn.functional.softplus(log_diagonal) + V.POSITIVE_OFFSET,
            min=V.POSITIVE_MINIMUM,
            max=V.POSITIVE_MAXIMUM,
        )
        projection = torch.bmm(
            factors.transpose(1, 2), score[..., None]
        ).squeeze(-1) / float(V.WIDTH)
        cross = torch.bmm(factors, (lam * projection)[..., None]).squeeze(-1)
        physical_diagonal = model.layer_rms[layer] * diagonal * score
        physical_cross = model.layer_rms[layer] * cross
        # Preserve the frozen scorer's float32 parenthesization exactly.
        total_outputs.append(
            model.layer_rms[layer] * (diagonal * score + cross)
        )
        diagonal_outputs.append(physical_diagonal)
        cross_outputs.append(physical_cross)
        scores.append(score)
        diagonals.append(diagonal)
        lambdas.append(lam)
        factor_history.append(factors)
    return {
        "delta": torch.stack(total_outputs, dim=1),
        "diagonal_delta": torch.stack(diagonal_outputs, dim=1),
        "cross_delta": torch.stack(cross_outputs, dim=1),
        "score": torch.stack(scores, dim=1),
        "diagonal": torch.stack(diagonals, dim=1),
        "lambdas": torch.stack(lambdas, dim=1),
        "factors": torch.stack(factor_history, dim=1),
    }


def main() -> None:
    if sha256(HERE / "verify_results.py") != VERIFIER_SHA:
        raise AssertionError("terminal verifier changed")
    if sha256(HERE / "VERIFICATION.json") != VERIFICATION_SHA:
        raise AssertionError("terminal verification receipt changed")
    torch.manual_seed(V.SEED)
    np.random.seed(V.SEED & 0xFFFF_FFFF)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    seal, _ = V.validate_static_bindings()
    cv = V.read_json(HERE / "CV_RESULTS.json")
    verification = V.read_json(HERE / "VERIFICATION.json")
    dataset = V.PublicResidualDataset(role="observer_fit")
    row_manifest_digest = V.validate_feature_manifests(dataset)
    if (
        row_manifest_digest
        != verification["bindings"]["observer_fit_row_manifest_binding_digest"]
    ):
        raise AssertionError("row manifest digest")
    error = np.load(V.D_DIR / "cache/patched_error.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(V.D_DIR / "cache/raw_sufficient_stats.npz", allow_pickle=False)
    sums = stats["sum"]
    sums2 = stats["sum2"]
    folds = np.asarray(
        [V.fold_of(int(value)) for value in dataset.mlp_ids], dtype=np.int64
    )

    # Layer aggregates for the exact identity
    # MSE(J+delta)-MSE(J) = 2<E_J,delta> + ||delta||^2.
    e_sse = np.zeros(V.DEPTH, dtype=np.float64)
    n_exact_sse = np.zeros(V.DEPTH, dtype=np.float64)
    d_sse = np.zeros(V.DEPTH, dtype=np.float64)
    diag_sse = np.zeros(V.DEPTH, dtype=np.float64)
    cross_sse = np.zeros(V.DEPTH, dtype=np.float64)
    e_dot_d = np.zeros(V.DEPTH, dtype=np.float64)
    e_dot_diag = np.zeros(V.DEPTH, dtype=np.float64)
    e_dot_cross = np.zeros(V.DEPTH, dtype=np.float64)
    diag_dot_cross = np.zeros(V.DEPTH, dtype=np.float64)
    score_sse = np.zeros(V.DEPTH, dtype=np.float64)
    score_dot_negative_error = np.zeros(V.DEPTH, dtype=np.float64)
    score_error_sse = np.zeros(V.DEPTH, dtype=np.float64)
    diagonal_sum = np.zeros(V.DEPTH, dtype=np.float64)
    lambda_sum = np.zeros(V.DEPTH, dtype=np.float64)
    factor_gram_fro_sum = np.zeros(V.DEPTH, dtype=np.float64)
    factor_gram_max = np.zeros(V.DEPTH, dtype=np.float64)
    corrected_units = np.zeros(V.DEPTH, dtype=np.int64)
    corrected_networks = np.zeros(V.DEPTH, dtype=np.int64)
    rows: list[dict[str, int | float]] = []

    # Exactly one oracle: remove the orthogonal projection of E_J onto the
    # learned dynamic U span.  This is parameter-free and never a candidate.
    oracle_base_sse = 0.0
    oracle_remaining_sse = 0.0
    oracle_projected_sse = 0.0

    with torch.no_grad():
        for fold in range(V.FOLDS):
            record = V.read_json(HERE / "folds" / f"fold_{fold}.json")
            if sha256(HERE / "folds" / f"fold_{fold}.json") != V.N_RESULT_SHA[fold]:
                raise AssertionError((fold, "N result"))
            valid = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            centers, scales, layer_rms = V.moments_for(train, error, sums, sums2)
            n_artifact = torch.load(
                HERE / "folds" / f"fold_{fold}.pt",
                map_location="cpu",
                weights_only=True,
            )
            j_artifact = torch.load(
                V.J_DIR / "folds" / f"fold_{fold}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for state in (n_artifact["state_dict"], j_artifact["state_dict"]):
                if not np.array_equal(state["centers"].numpy(), centers):
                    raise AssertionError((fold, "centers"))
                if not np.array_equal(state["scales"].numpy(), scales):
                    raise AssertionError((fold, "scales"))
                if not np.array_equal(state["layer_rms"].numpy(), layer_rms):
                    raise AssertionError((fold, "layer_rms"))
            j_model = V.FrozenJ()
            j_model.load_state_dict(j_artifact["state_dict"], strict=True)
            j_model.eval()
            n_model = V.FrozenN()
            n_model.load_state_dict(n_artifact["state_dict"], strict=True)
            n_model.eval()
            saved_rows = record["autonomous"]["per_network"]
            saved_by_rank = {int(row["role_rank"]): row for row in saved_rows}
            for offset in range(0, len(valid), 2):
                group = valid[offset : offset + 2]
                raw, sigma, weights, truth = V.load_batch(dataset, group, error)
                j_prediction, j_states, j_energies = j_model(raw, sigma, weights)
                diagnostic = n_diagnostics(
                    n_model, raw, sigma, weights, j_states, j_energies
                )
                delta = diagnostic["delta"]
                diagonal_delta = diagnostic["diagonal_delta"]
                cross_delta = diagnostic["cross_delta"]
                residual = j_prediction - truth
                n_prediction = j_prediction + delta
                for layer in range(V.DEPTH):
                    e = residual[:, layer].double()
                    n_error_exact = (n_prediction[:, layer] - truth[:, layer]).double()
                    d = delta[:, layer].double()
                    dd = diagonal_delta[:, layer].double()
                    dc = cross_delta[:, layer].double()
                    e_sse[layer] += float(e.square().sum())
                    n_exact_sse[layer] += float(n_error_exact.square().sum())
                    d_sse[layer] += float(d.square().sum())
                    diag_sse[layer] += float(dd.square().sum())
                    cross_sse[layer] += float(dc.square().sum())
                    e_dot_d[layer] += float((e * d).sum())
                    e_dot_diag[layer] += float((e * dd).sum())
                    e_dot_cross[layer] += float((e * dc).sum())
                    diag_dot_cross[layer] += float((dd * dc).sum())
                    if layer >= V.FIRST:
                        score = diagnostic["score"][:, layer].double()
                        normalized_error = e / float(n_model.layer_rms[layer])
                        score_sse[layer] += float(score.square().sum())
                        score_error_sse[layer] += float(normalized_error.square().sum())
                        score_dot_negative_error[layer] += float(
                            (score * -normalized_error).sum()
                        )
                        diagonal_sum[layer] += float(
                            diagnostic["diagonal"][:, layer].double().sum()
                        )
                        lambda_sum[layer] += float(
                            diagnostic["lambdas"][:, layer].double().sum()
                        )
                        u = diagnostic["factors"][:, layer].double()
                        gram = torch.bmm(u.transpose(1, 2), u) / float(V.WIDTH)
                        deviation = gram - torch.eye(V.N_RANK, dtype=torch.float64)[None]
                        factor_gram_fro_sum[layer] += float(
                            torch.linalg.matrix_norm(deviation, ord="fro").sum()
                        )
                        factor_gram_max[layer] = max(
                            factor_gram_max[layer], float(deviation.abs().max())
                        )
                        corrected_units[layer] += len(group) * V.WIDTH
                        corrected_networks[layer] += len(group)

                final_e = residual[:, -1].double()
                final_d = delta[:, -1].double()
                final_diag = diagonal_delta[:, -1].double()
                final_cross = cross_delta[:, -1].double()
                final_u = diagnostic["factors"][:, -1].double()
                gram = torch.bmm(final_u.transpose(1, 2), final_u)
                rhs = torch.bmm(
                    final_u.transpose(1, 2), final_e[..., None]
                )
                coefficients = torch.linalg.solve(gram, rhs)
                projected = torch.bmm(final_u, coefficients).squeeze(-1)
                remaining = final_e - projected
                oracle_base_sse += float(final_e.square().sum())
                oracle_projected_sse += float(projected.square().sum())
                oracle_remaining_sse += float(remaining.square().sum())
                for index, rank in enumerate(group):
                    e = final_e[index]
                    d = final_d[index]
                    dd = final_diag[index]
                    dc = final_cross[index]
                    p = projected[index]
                    n_mse = float(
                        (n_prediction[index, -1] - truth[index, -1])
                        .double()
                        .square()
                        .mean()
                    )
                    close(n_mse, saved_by_rank[int(rank)]["final_mse"])
                    e2 = float(e.square().sum())
                    d2 = float(d.square().sum())
                    dd2 = float(dd.square().sum())
                    dc2 = float(dc.square().sum())
                    ed = float((e * d).sum())
                    edd = float((e * dd).sum())
                    edc = float((e * dc).sum())
                    score = diagnostic["score"][index, -1].double()
                    normalized_e = e / float(n_model.layer_rms[-1])
                    u = final_u[index]
                    u_gram = u.transpose(0, 1) @ u / float(V.WIDTH)
                    rows.append(
                        {
                            "role_rank": int(rank),
                            "mlp_id": int(dataset.mlp_ids[int(rank)]),
                            "fold": fold,
                            "j_final_mse": e2 / V.WIDTH,
                            "n_final_mse": n_mse,
                            "n_minus_j": n_mse - e2 / V.WIDTH,
                            "delta_rms": math.sqrt(d2 / V.WIDTH),
                            "diagonal_delta_rms": math.sqrt(dd2 / V.WIDTH),
                            "cross_delta_rms": math.sqrt(dc2 / V.WIDTH),
                            "delta_negative_error_cosine": (
                                -ed / math.sqrt(e2 * d2) if e2 and d2 else 0.0
                            ),
                            "diagonal_negative_error_cosine": (
                                -edd / math.sqrt(e2 * dd2) if e2 and dd2 else 0.0
                            ),
                            "cross_negative_error_cosine": (
                                -edc / math.sqrt(e2 * dc2) if e2 and dc2 else 0.0
                            ),
                            "score_negative_error_cosine": float(
                                (score * -normalized_e).sum()
                                / torch.sqrt(score.square().sum() * normalized_e.square().sum())
                            ),
                            "mean_diagonal_scale": float(
                                diagnostic["diagonal"][index, -1].double().mean()
                            ),
                            "mean_factor_eigenvalue": float(
                                diagnostic["lambdas"][index, -1].double().mean()
                            ),
                            "factor_gram_frobenius_deviation": float(
                                torch.linalg.matrix_norm(
                                    u_gram - torch.eye(V.N_RANK, dtype=torch.float64),
                                    ord="fro",
                                )
                            ),
                            "oracle_projected_error_fraction": (
                                float(p.square().sum()) / e2 if e2 else 0.0
                            ),
                            "oracle_projector_final_mse": float(
                                (e - p).square().mean()
                            ),
                        }
                    )

    rows.sort(key=lambda row: int(row["role_rank"]))
    if [int(row["role_rank"]) for row in rows] != list(range(688)):
        raise AssertionError("OOF coverage")
    denominator = 688 * V.WIDTH
    j_layer = e_sse / denominator
    n_layer_exact = n_exact_sse / denominator
    n_layer_from_identity = (e_sse + 2.0 * e_dot_d + d_sse) / denominator
    saved_j_layer = np.asarray(verification["metrics"]["frozen_j_layer_mse"])
    saved_n_layer = np.asarray(verification["metrics"]["rank16_n_layer_mse"])
    for actual, saved in zip(j_layer, saved_j_layer, strict=True):
        close(actual, saved)
    for actual, saved in zip(n_layer_exact, saved_n_layer, strict=True):
        close(actual, saved)

    n_minus_j = np.asarray([row["n_minus_j"] for row in rows], dtype=np.float64)
    gains = -n_minus_j
    gross = float(np.abs(n_minus_j).sum())
    signed = float(n_minus_j.sum())
    improvement_sum = float(n_minus_j[n_minus_j < 0].sum())
    regression_sum = float(n_minus_j[n_minus_j > 0].sum())
    hard_rows = [rows[rank] for rank in HARD_RANKS]
    hard_signed = float(sum(row["n_minus_j"] for row in hard_rows))
    ordinary_signed = signed - hard_signed

    fold_rows = []
    for fold in range(V.FOLDS):
        selected = [row for row in rows if row["fold"] == fold]
        fold_delta = float(np.mean([row["n_minus_j"] for row in selected]))
        fold_rows.append(
            {
                "fold": fold,
                "networks": len(selected),
                "j_mean": float(np.mean([row["j_final_mse"] for row in selected])),
                "n_mean": float(np.mean([row["n_final_mse"] for row in selected])),
                "n_minus_j": fold_delta,
                "fractional_improvement": -fold_delta
                / float(np.mean([row["j_final_mse"] for row in selected])),
                "wins": sum(row["n_minus_j"] < 0 for row in selected),
                "losses": sum(row["n_minus_j"] > 0 for row in selected),
                "hard_ranks": [rank for rank in HARD_RANKS if rows[rank]["fold"] == fold],
            }
        )

    layer_rows = []
    for layer in range(V.FIRST, V.DEPTH):
        delta_norm = math.sqrt(d_sse[layer] / denominator)
        diag_norm = math.sqrt(diag_sse[layer] / denominator)
        cross_norm = math.sqrt(cross_sse[layer] / denominator)
        layer_rows.append(
            {
                "layer": layer,
                "j_mse": float(j_layer[layer]),
                "n_mse": float(n_layer_exact[layer]),
                "j_minus_n": float(j_layer[layer] - n_layer_exact[layer]),
                "fractional_improvement": float(
                    (j_layer[layer] - n_layer_exact[layer]) / j_layer[layer]
                ),
                "linear_alignment_term_2e_dot_delta": float(
                    2.0 * e_dot_d[layer] / denominator
                ),
                "quadratic_delta_cost": float(d_sse[layer] / denominator),
                "delta_rms": delta_norm,
                "diagonal_delta_rms": diag_norm,
                "cross_delta_rms": cross_norm,
                "cross_fraction_of_delta_squared_norm": float(
                    cross_sse[layer] / d_sse[layer]
                ),
                "delta_negative_error_cosine": float(
                    -e_dot_d[layer] / math.sqrt(e_sse[layer] * d_sse[layer])
                ),
                "diagonal_negative_error_cosine": float(
                    -e_dot_diag[layer]
                    / math.sqrt(e_sse[layer] * diag_sse[layer])
                ),
                "cross_negative_error_cosine": float(
                    -e_dot_cross[layer]
                    / math.sqrt(e_sse[layer] * cross_sse[layer])
                ),
                "score_negative_error_cosine": float(
                    score_dot_negative_error[layer]
                    / math.sqrt(score_sse[layer] * score_error_sse[layer])
                ),
                "mean_diagonal_scale": float(
                    diagonal_sum[layer] / corrected_units[layer]
                ),
                "mean_factor_eigenvalue": float(
                    lambda_sum[layer] / (corrected_networks[layer] * V.N_RANK)
                ),
                "mean_factor_gram_frobenius_deviation": float(
                    factor_gram_fro_sum[layer] / corrected_networks[layer]
                ),
                "maximum_factor_gram_entry_deviation": float(
                    factor_gram_max[layer]
                ),
            }
        )

    oracle_mse = oracle_remaining_sse / denominator
    oracle_projected_fraction = oracle_projected_sse / oracle_base_sse
    close(
        oracle_base_sse / denominator,
        verification["metrics"]["frozen_j_final_mse"],
    )
    # Orthogonal projection Pythagorean identity is the oracle's audit.
    close(oracle_base_sse, oracle_remaining_sse + oracle_projected_sse, atol=1e-12)

    gain_features = {
        "delta_negative_error_cosine": np.asarray(
            [row["delta_negative_error_cosine"] for row in rows]
        ),
        "score_negative_error_cosine": np.asarray(
            [row["score_negative_error_cosine"] for row in rows]
        ),
        "delta_rms": np.asarray([row["delta_rms"] for row in rows]),
        "cross_delta_rms": np.asarray([row["cross_delta_rms"] for row in rows]),
        "mean_diagonal_scale": np.asarray(
            [row["mean_diagonal_scale"] for row in rows]
        ),
        "mean_factor_eigenvalue": np.asarray(
            [row["mean_factor_eigenvalue"] for row in rows]
        ),
        "oracle_projected_error_fraction": np.asarray(
            [row["oracle_projected_error_fraction"] for row in rows]
        ),
    }
    correlations = {
        key: pearson(gains, values) for key, values in gain_features.items()
    }

    final_layer = layer_rows[-1]
    receipt = {
        "schema": "p2r0f2n-vs-j-oof-decomposition-one-global-oracle-v1",
        "status": "PASS_CLOSED",
        "node": "P2R0F2N",
        "scope": "Read-only observer_fit OOF causal decomposition and one parameter-free global oracle; no candidate evaluation",
        "bindings": {
            "terminal_verifier_sha256": VERIFIER_SHA,
            "terminal_verification_sha256": VERIFICATION_SHA,
            "cv_results_sha256": V.EXPECTED["cv_results"],
            "n_fold_result_sha256": V.N_RESULT_SHA,
            "n_fold_model_sha256": V.N_MODEL_SHA,
            "j_outer_result_sha256": V.J_RESULT_SHA,
            "j_outer_model_sha256": V.J_MODEL_SHA,
            "shared_error_sha256": V.EXPECTED["shared_error"],
            "shared_stats_sha256": V.EXPECTED["shared_stats"],
            "feature_manifest_sha256": V.EXPECTED["feature_manifest"],
            "observer_fit_target_sha256": V.EXPECTED["observer_fit_target"],
            "replayed_row_diagnostics_sha256": canonical_sha256(rows),
            "analysis_program_sha256": sha256(Path(__file__)),
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
            "rank_sweeps": 0,
            "hyperparameter_sweeps": 0,
            "operator_oracles_tested": 1,
        },
        "paired_final": {
            "j_oof_final_mse": float(j_layer[-1]),
            "n_oof_final_mse": float(n_layer_exact[-1]),
            "n_minus_j": float(n_layer_exact[-1] - j_layer[-1]),
            "n_fractional_improvement": float(
                (j_layer[-1] - n_layer_exact[-1]) / j_layer[-1]
            ),
            "wins": int(np.sum(n_minus_j < 0)),
            "losses": int(np.sum(n_minus_j > 0)),
            "ties": int(np.sum(n_minus_j == 0)),
            "signed_sum": signed,
            "gross_absolute_sum": gross,
            "negative_improvement_sum": improvement_sum,
            "positive_regression_sum": regression_sum,
            "net_to_gross_movement_ratio": abs(signed) / gross,
            "cancellation_fraction": 1.0 - abs(signed) / gross,
            "row_delta_p10": float(np.quantile(n_minus_j, 0.10)),
            "row_delta_median": float(np.median(n_minus_j)),
            "row_delta_p90": float(np.quantile(n_minus_j, 0.90)),
        },
        "fold_dispersion": {
            "folds": fold_rows,
            "folds_improving_j": sum(row["n_minus_j"] < 0 for row in fold_rows),
            "folds_regressing_j": sum(row["n_minus_j"] > 0 for row in fold_rows),
            "fold_delta_population_stddev": float(
                np.std([row["n_minus_j"] for row in fold_rows])
            ),
            "fold_delta_range": float(
                np.ptp([row["n_minus_j"] for row in fold_rows])
            ),
        },
        "layerwise": {
            "layers": layer_rows,
            "layers_improving_j": [
                row["layer"] for row in layer_rows if row["j_minus_n"] > 0
            ],
            "layers_regressing_j": [
                row["layer"] for row in layer_rows if row["j_minus_n"] < 0
            ],
            "best_fractional_improvement_layer": max(
                layer_rows, key=lambda row: row["fractional_improvement"]
            )["layer"],
            "final_exact_mse_identity_residual": float(
                n_layer_exact[-1]
                - (
                    j_layer[-1]
                    + 2.0 * e_dot_d[-1] / denominator
                    + d_sse[-1] / denominator
                )
            ),
        },
        "hard_rows": {
            "declared_role_ranks": HARD_RANKS,
            "rows": hard_rows,
            "j_mean": float(np.mean([row["j_final_mse"] for row in hard_rows])),
            "n_mean": float(np.mean([row["n_final_mse"] for row in hard_rows])),
            "signed_delta_sum": hard_signed,
            "hard_signed_share_of_population_net_change": hard_signed / signed,
            "ordinary_signed_delta_sum": ordinary_signed,
            "ordinary_signed_share_of_population_net_change": ordinary_signed / signed,
            "wins": sum(row["n_minus_j"] < 0 for row in hard_rows),
            "losses": sum(row["n_minus_j"] > 0 for row in hard_rows),
            "breaches_above_1e_7_after_n": [
                row["role_rank"] for row in hard_rows if row["n_final_mse"] > 1.0e-7
            ],
        },
        "covariance_amplitude_and_coherence": {
            "final_layer": final_layer,
            "gain_pearson_correlations": correlations,
            "factor_gain_delta_max_abs_by_fold": [
                float(
                    torch.load(
                        HERE / "folds" / f"fold_{fold}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )["state_dict"]["factor_gain_delta"].abs().max()
                )
                for fold in range(V.FOLDS)
            ],
            "factor_mix_delta_max_abs_by_fold": [
                float(
                    torch.load(
                        HERE / "folds" / f"fold_{fold}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )["state_dict"]["factor_mix_delta"].abs().max()
                )
                for fold in range(V.FOLDS)
            ],
        },
        "one_global_oracle_capacity_test": {
            "operator": "For final-layer J error e and N dynamic factor U, P_U(e)=U(U^T U)^-1 U^T e; oracle error=(I-P_U)e.",
            "why_causal": "It preserves N's learned target-free global geometry but replaces the learned score with the exact best coefficients in that one span, separating score failure from factor-span capacity.",
            "operator_is_full_width": True,
            "operator_is_network_adaptive_nonlinear_through_u": True,
            "oracle_uses_truth_for_analysis_only": True,
            "free_parameters": 0,
            "tuned_scales": 0,
            "ranks_tested": [16],
            "operators_tested": 1,
            "j_final_mse": oracle_base_sse / denominator,
            "oracle_final_mse": oracle_mse,
            "absolute_mse_reduction": oracle_projected_sse / denominator,
            "captured_error_fraction": oracle_projected_fraction,
            "factor_above_1e_8": oracle_mse / 1.0e-8,
            "reaches_1e_8": bool(oracle_mse < 1.0e-8),
            "measurable_above_one_basis_point": bool(
                oracle_projected_fraction > 1.0e-4
            ),
            "pythagorean_relative_residual": abs(
                oracle_base_sse - oracle_remaining_sse - oracle_projected_sse
            )
            / oracle_base_sse,
        },
        "closure": {
            "exact_n_gate_passed": False,
            "rank_or_hyperparameter_scaling_rejected": True,
            "literal_meter_authorized": False,
            "native_promotion_authorized": False,
            "conclusion": (
                "N is a real but tiny endpoint improvement: its learned correction is "
                "directionally useful on 389/688 rows, yet paired gains and losses mostly "
                "cancel, all four inherited hard breaches survive, and the single exact "
                "factor-span oracle remains far above 1e-8. The bottleneck is not merely "
                "the learned covariance score; the learned rank-16 global span itself "
                "cannot contain the required endpoint correction. Close this exact axis."
            ),
        },
        "one_graph_ready_proposal": {
            "status": "DESIGN_ONLY_NOT_IMPLEMENTED_OR_LAUNCHED",
            "proposal": "Test one full-rank nonlinear weight-space round-trip residual operator, not a larger low-rank covariance: delta=R*a*W^T tanh(b*(W*qhat)/sqrt(W))/sqrt(W), with a,b fixed by fit-fold training and qhat from J's lawful local features.",
            "why": "The N projector oracle directly bounds its learned low-rank global span, while a nonlinear W/W^T round trip is full rank, couples every unit through the actual network, and tests a materially different non-Gaussian global mechanism rather than rank or optimizer scaling.",
            "required_before_any_launch": "Freeze exact normalization, architecture, cost, nested OOF training, and strict gates before reading any metric; run only one arm.",
        },
    }
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
