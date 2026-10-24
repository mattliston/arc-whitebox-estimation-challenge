#!/Users/matt/anaconda3/bin/python3
"""Deterministic, JSON-only P2R0F2M-vs-J OOF decomposition.

The program reads completed observer_fit receipts.  It does not import an
estimator, replay a checkpoint, train, open observer_selection, meter, touch
the network, or edit the canonical problem graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"
K_DIR = P2 / "public_full688_full_gram_pooled_context_cv_v1"
I_DIR = P2 / "public_full688_typed_diagonal_error_variance_cv_v1"
GAUSSIAN_DIR = P2 / "public_gaussian_secant_transport_discriminator_v1"
OUTPUT = HERE / "M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json"
HARD_RANKS = (201, 310, 599, 643)
HARD_THRESHOLD = 1.0e-7
NETWORKS = 688
FOLDS = 4
WIDTH = 1024
CORRECTED_LAYERS = 14
TRANSITIONS = 13
COVARIANCE_RANK = 16
NATIVE_HOST_TAPE_BASELINE_FLOPS = 2_131_692_137_721
EXACT_SCORING_BUDGET = 2**41
PHASE2_BUDGET = 2_200_000_000_000


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saved_rows(directory: Path) -> dict[int, dict[str, int | float]]:
    rows: dict[int, dict[str, int | float]] = {}
    for fold in range(FOLDS):
        payload = read_json(directory / "folds" / f"fold_{fold}.json")
        if int(payload["fold"]) != fold:
            raise AssertionError("fold label changed")
        for row in payload["autonomous"]["per_network"]:
            rank = int(row["role_rank"])
            if rank in rows:
                raise AssertionError(f"duplicate role rank {rank}")
            rows[rank] = {
                "fold": fold,
                "mlp_id": int(row["mlp_id"]),
                "final_mse": float(row["final_mse"]),
            }
    if sorted(rows) != list(range(NETWORKS)):
        raise AssertionError("OOF coverage is not exactly ranks 0:687")
    return rows


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise AssertionError("pearson inputs changed")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else float("nan")


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def row_record(
    rank: int,
    j_rows: dict[int, dict[str, int | float]],
    m_rows: dict[int, dict[str, int | float]],
    deltas: list[float],
    gross_absolute_change: float,
    signed_total_change: float,
) -> dict[str, object]:
    j_row = j_rows[rank]
    m_row = m_rows[rank]
    delta = deltas[rank]
    return {
        "role_rank": rank,
        "mlp_id": int(j_row["mlp_id"]),
        "fold": int(j_row["fold"]),
        "hard": rank in HARD_RANKS,
        "j_final_mse": float(j_row["final_mse"]),
        "m_final_mse": float(m_row["final_mse"]),
        "m_minus_j": delta,
        "m_fractional_change_vs_j": delta / float(j_row["final_mse"]),
        "absolute_share_of_gross_paired_change": abs(delta)
        / gross_absolute_change,
        "signed_share_of_net_paired_change": delta / signed_total_change,
    }


def successor_cost() -> dict[str, int | str | bool]:
    factor_transport = (
        TRANSITIONS * COVARIANCE_RANK * (2 * WIDTH * WIDTH - WIDTH)
    )

    # The successor reuses J's 36 lawful local features. Its shared 36->128->128
    # encoder emits 16 factor innovations, 16 log eigenvalues, a log diagonal,
    # and a local score: 34 scalars per unit.
    trunk_per_unit = (2 * 36 - 1) * 128 + (2 * 128 - 1) * 128
    head_outputs = 2 * COVARIANCE_RANK + 2
    heads_per_unit = (2 * 128 - 1) * head_outputs
    local_encoder_and_heads = (
        CORRECTED_LAYERS * WIDTH * (trunk_per_unit + heads_per_unit)
    )
    gram_whitening_and_small_eigensolve = CORRECTED_LAYERS * (
        2 * WIDTH * COVARIANCE_RANK**2 + 20 * COVARIANCE_RANK**3
    )
    woodbury_apply_and_diagonal = CORRECTED_LAYERS * (
        4 * WIDTH * COVARIANCE_RANK + 20 * WIDTH
    )
    factor_source_gate_reserve = (
        CORRECTED_LAYERS * WIDTH * COVARIANCE_RANK * 256
    )
    output_addition = CORRECTED_LAYERS * WIDTH
    j_correction_upper = 1_051_656_192
    added_covariance_observer_upper = (
        factor_transport
        + local_encoder_and_heads
        + gram_whitening_and_small_eigensolve
        + woodbury_apply_and_diagonal
        + factor_source_gate_reserve
        + output_addition
    )
    correction_upper = j_correction_upper + added_covariance_observer_upper
    native_total = NATIVE_HOST_TAPE_BASELINE_FLOPS + correction_upper
    parameter_count_added = (
        (36 * 128 + 128)
        + (128 * 128 + 128)
        + CORRECTED_LAYERS * (128 * head_outputs + head_outputs)
        + TRANSITIONS
        * (COVARIANCE_RANK + COVARIANCE_RANK * COVARIANCE_RANK)
    )
    if parameter_count_added != 86_188:
        raise AssertionError(parameter_count_added)
    if factor_transport != 435_994_624:
        raise AssertionError(factor_transport)
    if correction_upper != 2_278_582_272:
        raise AssertionError(correction_upper)
    return {
        "authority": (
            "conservative source-structure arithmetic; literal meter required "
            "only after the frozen accuracy gate passes"
        ),
        "frozen_j_correction_upper": j_correction_upper,
        "rank16_factor_transport_upper": factor_transport,
        "local_encoder_and_heads_upper": local_encoder_and_heads,
        "gram_whitening_and_small_eigensolve_upper": (
            gram_whitening_and_small_eigensolve
        ),
        "woodbury_apply_and_diagonal_upper": woodbury_apply_and_diagonal,
        "factor_source_gate_reserve": factor_source_gate_reserve,
        "output_addition": output_addition,
        "added_covariance_observer_upper": added_covariance_observer_upper,
        "j_plus_covariance_observer_correction_upper": correction_upper,
        "native_host_tape_plus_j_plus_covariance_observer_upper": native_total,
        "headroom_below_exact_2^41": EXACT_SCORING_BUDGET - native_total,
        "headroom_below_phase2_cap": PHASE2_BUDGET - native_total,
        "within_exact_2^41": native_total < EXACT_SCORING_BUDGET,
        "within_phase2_cap": native_total < PHASE2_BUDGET,
        "new_parameters": parameter_count_added,
        "total_parameters_including_frozen_j": parameter_count_added + 25_600,
    }


def build_payload() -> dict[str, object]:
    j_cv = read_json(J_DIR / "CV_RESULTS.json")
    k_cv = read_json(K_DIR / "CV_RESULTS.json")
    i_cv = read_json(I_DIR / "CV_RESULTS.json")
    gaussian_result = read_json(GAUSSIAN_DIR / "RESULT.json")
    m_cv = read_json(HERE / "CV_RESULTS.json")
    m_verification = read_json(HERE / "VERIFICATION.json")
    if m_cv.get("status") != "complete" or m_verification.get("status") != "PASS":
        raise AssertionError("M terminal receipts are not complete/PASS")

    j_rows = saved_rows(J_DIR)
    k_rows = saved_rows(K_DIR)
    m_rows = saved_rows(HERE)
    for rank in range(NETWORKS):
        key_j = (int(j_rows[rank]["fold"]), int(j_rows[rank]["mlp_id"]))
        key_k = (int(k_rows[rank]["fold"]), int(k_rows[rank]["mlp_id"]))
        key_m = (int(m_rows[rank]["fold"]), int(m_rows[rank]["mlp_id"]))
        if key_j != key_k or key_j != key_m:
            raise AssertionError(f"paired identity/fold changed at rank {rank}")

    j_final = float(j_cv["oof"]["autonomous_final_mse"])
    m_final = float(m_cv["oof"]["autonomous_final_mse"])
    deltas = [
        float(m_rows[rank]["final_mse"]) - float(j_rows[rank]["final_mse"])
        for rank in range(NETWORKS)
    ]
    k_deltas = [
        float(k_rows[rank]["final_mse"]) - float(j_rows[rank]["final_mse"])
        for rank in range(NETWORKS)
    ]
    signed_total = sum(deltas)
    positive_total = sum(value for value in deltas if value > 0.0)
    negative_total = sum(value for value in deltas if value < 0.0)
    gross_absolute = sum(abs(value) for value in deltas)
    if not math.isclose(mean(deltas), m_final - j_final, rel_tol=1e-12):
        raise AssertionError("paired mean does not reproduce CV delta")

    j_breaches = {
        rank
        for rank in range(NETWORKS)
        if float(j_rows[rank]["final_mse"]) >= HARD_THRESHOLD
    }
    m_breaches = {
        rank
        for rank in range(NETWORKS)
        if float(m_rows[rank]["final_mse"]) >= HARD_THRESHOLD
    }
    k_breaches = {
        rank
        for rank in range(NETWORKS)
        if float(k_rows[rank]["final_mse"]) >= HARD_THRESHOLD
    }
    declared_hard = set(HARD_RANKS)
    if j_breaches != declared_hard or m_breaches != declared_hard:
        raise AssertionError("hard breach set changed")

    hard_delta = sum(deltas[rank] for rank in HARD_RANKS)
    ordinary_ranks = [rank for rank in range(NETWORKS) if rank not in declared_hard]
    ordinary_delta = sum(deltas[rank] for rank in ordinary_ranks)
    hard_j_sum = sum(float(j_rows[rank]["final_mse"]) for rank in HARD_RANKS)
    hard_m_sum = sum(float(m_rows[rank]["final_mse"]) for rank in HARD_RANKS)
    hard_gross = sum(abs(deltas[rank]) for rank in HARD_RANKS)

    folds: list[dict[str, object]] = []
    fold_deltas: list[float] = []
    for fold in range(FOLDS):
        ranks = [
            rank for rank in range(NETWORKS) if int(j_rows[rank]["fold"]) == fold
        ]
        fold_delta_values = [deltas[rank] for rank in ranks]
        fold_delta = mean(fold_delta_values)
        fold_deltas.append(fold_delta)
        hard_here = [rank for rank in ranks if rank in declared_hard]
        folds.append(
            {
                "fold": fold,
                "networks": len(ranks),
                "j_mean": mean([float(j_rows[rank]["final_mse"]) for rank in ranks]),
                "m_mean": mean([float(m_rows[rank]["final_mse"]) for rank in ranks]),
                "m_minus_j": fold_delta,
                "m_fractional_change_vs_j": fold_delta
                / mean([float(j_rows[rank]["final_mse"]) for rank in ranks]),
                "m_wins": sum(value < 0.0 for value in fold_delta_values),
                "m_losses": sum(value > 0.0 for value in fold_delta_values),
                "m_minus_j_median": statistics.median(fold_delta_values),
                "hard_ranks": hard_here,
                "hard_signed_delta_sum": sum(deltas[rank] for rank in hard_here),
            }
        )

    j_layers = [float(value) for value in j_cv["oof"]["autonomous_layer_mse"]]
    m_layers = [float(value) for value in m_cv["oof"]["autonomous_layer_mse"]]
    layer_delta = [m - j for j, m in zip(j_layers, m_layers, strict=True)]
    layers = []
    for layer, (j_value, m_value, delta) in enumerate(
        zip(j_layers, m_layers, layer_delta, strict=True)
    ):
        prior = layer_delta[layer - 1] if layer else 0.0
        layers.append(
            {
                "layer": layer,
                "j_mse": j_value,
                "m_mse": m_value,
                "m_minus_j": delta,
                "increment_in_m_minus_j_from_prior_layer": delta - prior,
                "m_fractional_change_vs_j": delta / j_value,
            }
        )

    m_wins = {rank for rank in range(NETWORKS) if deltas[rank] < 0.0}
    k_wins = {rank for rank in range(NETWORKS) if k_deltas[rank] < 0.0}
    same_sign = sum(
        (deltas[rank] < 0.0) == (k_deltas[rank] < 0.0)
        for rank in range(NETWORKS)
    )
    intersection = m_wins & k_wins
    union = m_wins | k_wins

    dominant_order = sorted(
        range(NETWORKS), key=lambda rank: (-abs(deltas[rank]), rank)
    )
    improvement_order = sorted(range(NETWORKS), key=lambda rank: (deltas[rank], rank))
    regression_order = sorted(
        range(NETWORKS), key=lambda rank: (-deltas[rank], rank)
    )

    peak_layer = max(range(len(layer_delta)), key=lambda layer: layer_delta[layer])
    top20 = dominant_order[:20]
    top20_signed = sum(deltas[rank] for rank in top20)

    payload: dict[str, object] = {
        "schema": "p2r0f2m-vs-j-json-only-oof-decomposition-one-successor-v1",
        "status": "complete-read-only-decomposition-one-successor-frozen-unlaunched",
        "node": "P2R0F2M",
        "scope": {
            "role": "observer_fit",
            "networks": NETWORKS,
            "observer_selection_targets_opened": 0,
            "new_training_performed": False,
            "new_rollouts_performed": False,
            "saved_checkpoints_loaded": False,
            "network_access": False,
            "canonical_graph_edited": False,
        },
        "bindings": {
            "analysis_program_sha256": sha256(Path(__file__)),
            "j_cv_results_sha256": sha256(J_DIR / "CV_RESULTS.json"),
            "j_verification_sha256": sha256(J_DIR / "VERIFICATION.json"),
            "k_cv_results_sha256": sha256(K_DIR / "CV_RESULTS.json"),
            "k_verification_sha256": sha256(K_DIR / "VERIFICATION.json"),
            "k_vs_j_decomposition_sha256": sha256(
                K_DIR / "K_VS_J_OOF_DECOMPOSITION_AND_SUCCESSOR.json"
            ),
            "i_cv_results_sha256": sha256(I_DIR / "CV_RESULTS.json"),
            "i_verification_sha256": sha256(I_DIR / "VERIFICATION.json"),
            "gaussian_secant_result_sha256": sha256(GAUSSIAN_DIR / "RESULT.json"),
            "gaussian_secant_preregistration_sha256": sha256(
                GAUSSIAN_DIR / "PREREGISTRATION.json"
            ),
            "m_cv_results_sha256": sha256(HERE / "CV_RESULTS.json"),
            "m_verification_sha256": sha256(HERE / "VERIFICATION.json"),
            "m_preregistration_sha256": sha256(HERE / "PREREGISTRATION.json"),
            "m_seal_sha256": sha256(HERE / "PREREGISTRATION_SEAL.json"),
            "m_training_release_sha256": sha256(HERE / "TRAINING_RELEASE.json"),
            "j_fold_result_sha256": [
                sha256(J_DIR / "folds" / f"fold_{fold}.json")
                for fold in range(FOLDS)
            ],
            "k_fold_result_sha256": [
                sha256(K_DIR / "folds" / f"fold_{fold}.json")
                for fold in range(FOLDS)
            ],
            "m_fold_result_sha256": [
                sha256(HERE / "folds" / f"fold_{fold}.json")
                for fold in range(FOLDS)
            ],
        },
        "headline": {
            "j_oof_final_mse": j_final,
            "m_oof_final_mse": m_final,
            "m_minus_j": m_final - j_final,
            "m_regression_fraction_vs_j": m_final / j_final - 1.0,
            "m_wins": len(m_wins),
            "m_losses": NETWORKS - len(m_wins),
            "m_minus_j_median": statistics.median(deltas),
            "m_minus_j_p10": quantile(deltas, 0.10),
            "m_minus_j_p90": quantile(deltas, 0.90),
            "raw_factor_above_1e_8": m_final / 1.0e-8,
            "fractional_reduction_required_to_reach_1e_8": 1.0 - 1.0e-8 / m_final,
        },
        "paired_cancellation": {
            "positive_regression_sum": positive_total,
            "negative_improvement_sum": negative_total,
            "signed_sum": signed_total,
            "gross_absolute_sum": gross_absolute,
            "signed_to_gross_absolute_ratio": signed_total / gross_absolute,
            "improvement_magnitude_to_regression_magnitude_ratio": (
                -negative_total / positive_total
            ),
            "interpretation": (
                "The branch is not numerically inert: large gains and losses cancel "
                "98.699% in the paired aggregate, leaving only 0.655% of gross "
                "movement as net regression."
            ),
        },
        "layers": layers,
        "layerwise_summary": {
            "first_nonzero_layer": 2,
            "layers_improving_j": [
                layer for layer, value in enumerate(layer_delta) if value < 0.0
            ],
            "layers_regressing_j": [
                layer for layer, value in enumerate(layer_delta) if value > 0.0
            ],
            "peak_regression_layer": peak_layer,
            "peak_regression_m_minus_j": layer_delta[peak_layer],
            "fraction_of_peak_regression_cancelled_by_l15": (
                1.0 - layer_delta[-1] / layer_delta[peak_layer]
            ),
            "interpretation": (
                "M is microscopically better only at L2, then worse through L14. "
                "The L15 transition cancels most of the accumulated mean regression, "
                "so the nearly tied endpoint is a late cancellation rather than a "
                "depth-wide improvement."
            ),
        },
        "fold_dispersion": {
            "folds": folds,
            "fold_delta_unweighted_mean": mean(fold_deltas),
            "fold_delta_population_stddev": statistics.pstdev(fold_deltas),
            "fold_delta_min": min(fold_deltas),
            "fold_delta_max": max(fold_deltas),
            "fold_delta_range": max(fold_deltas) - min(fold_deltas),
            "fold_stddev_to_absolute_global_delta": statistics.pstdev(fold_deltas)
            / abs(m_final - j_final),
            "folds_improving_j": sum(value < 0.0 for value in fold_deltas),
            "folds_regressing_j": sum(value > 0.0 for value in fold_deltas),
            "interpretation": (
                "Fold direction splits 2-2 and fold dispersion is 55.74x the tiny "
                "global delta; the backward feature has no stable cross-fold sign."
            ),
        },
        "hard_rows": {
            "threshold": HARD_THRESHOLD,
            "declared_role_ranks": list(HARD_RANKS),
            "j_breach_ranks": sorted(j_breaches),
            "m_breach_ranks": sorted(m_breaches),
            "k_breach_ranks": sorted(k_breaches),
            "j_m_breach_intersection": sorted(j_breaches & m_breaches),
            "j_m_breach_union": sorted(j_breaches | m_breaches),
            "j_m_breach_jaccard": len(j_breaches & m_breaches)
            / len(j_breaches | m_breaches),
            "population_fraction": len(HARD_RANKS) / NETWORKS,
            "j_total_error_share": hard_j_sum
            / sum(float(j_rows[rank]["final_mse"]) for rank in range(NETWORKS)),
            "m_total_error_share": hard_m_sum
            / sum(float(m_rows[rank]["final_mse"]) for rank in range(NETWORKS)),
            "j_mean": hard_j_sum / len(HARD_RANKS),
            "m_mean": hard_m_sum / len(HARD_RANKS),
            "m_minus_j_mean": hard_delta / len(HARD_RANKS),
            "m_fractional_change_vs_j": hard_m_sum / hard_j_sum - 1.0,
            "m_wins": sum(deltas[rank] < 0.0 for rank in HARD_RANKS),
            "m_losses": sum(deltas[rank] > 0.0 for rank in HARD_RANKS),
            "signed_delta_sum": hard_delta,
            "gross_absolute_delta_sum": hard_gross,
            "directional_coherence": -hard_delta / hard_gross,
            "hard_signed_fraction_of_net_regression": hard_delta / signed_total,
            "ordinary_networks": len(ordinary_ranks),
            "ordinary_m_minus_j_mean": ordinary_delta / len(ordinary_ranks),
            "ordinary_signed_delta_sum": ordinary_delta,
            "ordinary_signed_fraction_of_net_regression": ordinary_delta
            / signed_total,
            "rows": [
                row_record(
                    rank,
                    j_rows,
                    m_rows,
                    deltas,
                    gross_absolute,
                    signed_total,
                )
                for rank in HARD_RANKS
            ],
            "interpretation": (
                "M improves three hard rows and the hard mean by 0.431%, but all "
                "four remain >1e-7. Their net gain offsets 253.3% of the final net "
                "regression; the ordinary 684 rows regress enough to reverse it."
            ),
        },
        "overlap_and_coherence_with_k": {
            "purpose": (
                "K is used only as an already completed comparator to test whether "
                "M found a new stable population direction."
            ),
            "pearson_m_minus_j_with_k_minus_j": pearson(deltas, k_deltas),
            "same_improvement_loss_sign_rows": same_sign,
            "same_sign_fraction": same_sign / NETWORKS,
            "m_wins": len(m_wins),
            "k_wins": len(k_wins),
            "both_win": len(intersection),
            "win_union": len(union),
            "win_jaccard": len(intersection) / len(union),
            "both_win_fraction_of_m_wins": len(intersection) / len(m_wins),
            "both_win_fraction_of_k_wins": len(intersection) / len(k_wins),
            "hard_same_sign": sum(
                (deltas[rank] < 0.0) == (k_deltas[rank] < 0.0)
                for rank in HARD_RANKS
            ),
            "hard_both_win": sum(
                deltas[rank] < 0.0 and k_deltas[rank] < 0.0
                for rank in HARD_RANKS
            ),
            "interpretation": (
                "The 0.603 paired-delta correlation and 71.1% sign agreement with "
                "K indicate substantial reuse of the same generic coadaptation, "
                "not a new coherent residual mode."
            ),
        },
        "dominant_rows": {
            "top_20_absolute": [
                row_record(
                    rank,
                    j_rows,
                    m_rows,
                    deltas,
                    gross_absolute,
                    signed_total,
                )
                for rank in top20
            ],
            "top_10_improvements": [
                row_record(
                    rank,
                    j_rows,
                    m_rows,
                    deltas,
                    gross_absolute,
                    signed_total,
                )
                for rank in improvement_order[:10]
            ],
            "top_10_regressions": [
                row_record(
                    rank,
                    j_rows,
                    m_rows,
                    deltas,
                    gross_absolute,
                    signed_total,
                )
                for rank in regression_order[:10]
            ],
            "top_20_absolute_share_of_gross_change": sum(
                abs(deltas[rank]) for rank in top20
            )
            / gross_absolute,
            "top_20_signed_delta_sum": top20_signed,
            "top_20_signed_fraction_of_net": top20_signed / signed_total,
        },
        "already_closed_axes": {
            "gaussian_secant_mean_curvature": {
                "status": gaussian_result["status"],
                "causal_stop": gaussian_result["causal_stop"],
                "secant_ratio_to_tangent": gaussian_result["metrics"][
                    "secant_ratio_to_tangent"
                ],
                "conclusion": (
                    "Exact Gaussian secant transport was causally closed and is not "
                    "reintroduced by the successor."
                ),
            },
            "typed_diagonal_variance_i": {
                "oof_final_mse": float(i_cv["oof"]["autonomous_final_mse"]),
                "conclusion": (
                    "The diagonal-variance child regressed J and is not repeated."
                ),
            },
            "full_gram_pooled_context_k": {
                "oof_final_mse": float(k_cv["oof"]["autonomous_final_mse"]),
                "conclusion": (
                    "K's 4x4 channel Gram and pooled network context regressed J; N's "
                    "W x W cross-unit factor covariance is a different object."
                ),
            },
            "fixed_low_rank_endpoint_modes": {
                "ordinary_top8_enrichment": 9.42352316081469,
                "hard_top8_enrichment": 3.770347721960925,
                "conclusion": (
                    "The earlier fixed-mode diagnostic under-covered hard rows. N "
                    "uses target-free per-network, per-layer transported factors and "
                    "does not reuse fixed endpoint modes."
                ),
            },
        },
        "mechanistic_diagnosis": [
            (
                "The exact transpose W@(Phi*b) is only a transport operator. Its "
                "state is seeded by target-free learned local sources and is trained "
                "only indirectly through the forward predictor; it is not an adjoint "
                "of the unknown endpoint residual and imposes no global consistency "
                "condition on the correction trajectory."
            ),
            (
                "M therefore adds a second expressive feature path rather than "
                "solving a smoothing problem. The gross paired movement is large but "
                "98.699% cancels, the fold sign is 2-2, and ordinary rows reverse the "
                "gain on the four persistent failures."
            ),
            (
                "Its endpoint tie is especially misleading: M regresses J at every "
                "layer L3-L14, peaks at L13, then cancels 95.27% of that mean "
                "regression in L14-L15. This is not a trajectory-wide reduction."
            ),
            (
                "All four >1e-7 breaches survive exactly. M is 3.234x above the "
                "1e-8 raw target, so scaling this free backward latent path is not a "
                "credible route to the required 69.1% reduction."
            ),
        ],
        "one_frozen_successor": {
            "node": "P2R0F2N",
            "name": "network-adaptive rank-16 cross-unit residual covariance observer",
            "status": "frozen-design-only-unlaunched",
            "material_difference": (
                "I modeled only diagonal variance, K modeled a 4x4 channel Gram at "
                "each unit plus pooled scalars, and M transported a free mean/tangent "
                "state backward. N instead represents the network-specific W x W "
                "residual covariance as diagonal plus a learned rank-16 factor and "
                "uses that covariance operator to couple correction scores across "
                "units. The factors are dynamic per network, not fixed endpoint modes."
            ),
            "base_and_role": {
                "base": "frozen P2R0F2J outer-fold predictor",
                "role": "observer_fit only",
                "networks": NETWORKS,
                "observer_selection_targets_opened": 0,
                "deployment_truth_inputs": 0,
                "network_identity_inputs": 0,
                "public_name_joins": 0,
            },
            "native_dependencies": {
                "terminal_native_transfer_result_sha256": (
                    "7ecdc3438ddd4934853134ef0002b517e8111a2ecbf3a0e9abdafd0efb7e89a7"
                ),
                "terminal_native_transfer_model_sha256": (
                    "76daf05e69022df67eb546b7099c47a3a1538c849fe3c8512c9c8dfe1ce0cdf9"
                ),
                "required_fields": (
                    "the frozen native raw20/sigma/Phi tape plus the public MLP W; "
                    "J states/messages/energies are deterministically derived"
                ),
                "additional_native_target_fields": 0,
            },
            "architecture": {
                "covariance_rank": COVARIANCE_RANK,
                "lawful_local_features": (
                    "Reuse J's 36 target-free raw20/sigma, four signed-message, "
                    "four energy, and four coherence-derived local features."
                ),
                "local_encoder": "shared 36->128->128 SiLU",
                "layer_heads": (
                    "For L2:L15, a layer-specific 34-vector head emits 16 factor "
                    "innovations s, 16 log factor eigenvalues, one log diagonal "
                    "variance, and one signed local dimensionless score qhat per unit."
                ),
                "physical_scale": (
                    "R_l is the outer-fit-only layer RMS already used by J. U, s, "
                    "d, lambda, and qhat are dimensionless; the physical covariance "
                    "is Sigma_l=R_l^2[diag(d_l)+U_l diag(lambda_l)U_l^T/W], "
                    "the physical score is q_l=qhat_l/R_l, and delta_l=Sigma_l q_l."
                ),
                "factor_transport": (
                    "P_l=Phi_l*(U_(l-1)@W_l) for l>2; P_2=0. This is one exact "
                    "rank-16 full-width forward transport per transition."
                ),
                "factor_update": (
                    "s_l=tanh(factor_head_l) is dimensionless and "
                    "V_l=(gain_l*P_l)@M_l+s_l, with gain in [0.75,1.25] and "
                    "M_l=I+0.10*tanh(mix_delta_l)."
                ),
                "deterministic_whitening": (
                    "G_l=V_l^T V_l/W+2^-10 I; U_l=V_l G_l^-1/2 using a "
                    "symmetric 16x16 eigendecomposition with eigenvalues floored at "
                    "2^-10. No randomized sketch, pivot, tolerance, or sign choice."
                ),
                "covariance_operator": (
                    "Let ahat_(i,k) be the 16 per-unit log-eigenvalue heads and "
                    "abar_k=mean_i ahat_(i,k), a deterministic width mean. Then "
                    "d_i=clamp(softplus(logd_i)+2^-12,2^-10,2^6), "
                    "lambda_k=clamp(softplus(abar_k)+2^-12,2^-10,2^6), and "
                    "C_l=diag(d_l)+U_l diag(lambda_l)U_l^T/W. C is never "
                    "materialized as W x W."
                ),
                "global_cross_unit_update": (
                    "qhat_l=tanh(score_head_l) and delta_l=R_l*C_l*qhat_l = "
                    "R_l[d_l*qhat_l+U_l{lambda_l*(U_l^T qhat_l/W)}]. This is "
                    "the only new reported residual path."
                ),
                "reported_correction": (
                    "J correction plus delta_l at every layer; no mixture, selector, "
                    "hard-row gate, or post-metric fusion."
                ),
                "new_parameters": 86_188,
                "total_parameters_including_frozen_j": 111_788,
            },
            "training": {
                "outer_folds": 4,
                "outer_fold_seed": 1_753_358_438,
                "outer_fold_sizes": [147, 172, 171, 198],
                "stacking_firewall": (
                    "Within each outer training complement, generate J residual "
                    "targets with fixed three-way inner cross-fitting. Train N only "
                    "on those inner-OOF residuals; score once on the untouched outer "
                    "fold using the J model fitted on the complete outer complement."
                ),
                "inner_fold_rule": (
                    "mlp_id grouped; deterministic seed outer_seed XOR "
                    "(outer_fold+1)*0x9E3779B1; no metric stratification."
                ),
                "epochs": 8,
                "batch_size_networks": 3,
                "optimizer": "AdamW",
                "learning_rate": 8.0e-4,
                "minimum_learning_rate": 8.0e-5,
                "schedule": "CosineAnnealingLR over exactly all eight epochs",
                "weight_decay": 2.0e-5,
                "gradient_clip_norm": 5.0,
                "loss": (
                    "Fit-fold-normalized MSE of J+delta against truth on L2:L15 "
                    "with fixed layer weights 0.5..1.5 plus equal final-layer focus, "
                    "plus coefficient 0.05 times the exact diagonal-plus-low-rank "
                    "Gaussian score of the remaining residual divided by outer-fit "
                    "R_l (Woodbury inverse and matrix-determinant lemma on C_l), "
                    "plus 1e-5 log-scale penalty."
                ),
                "checkpoint": (
                    "fixed epoch 8; no early stopping, checkpoint selection, "
                    "architecture retuning, or second arm after any OOF metric"
                ),
            },
            "frozen_conjunctive_gate": {
                "all_688_outer_oof_rollouts_finite": True,
                "observer_selection_targets_opened": 0,
                "oof_final_mse_strictly_below": 1.0e-8,
                "every_outer_fold_final_mse_strictly_below": 1.1e-8,
                "every_network_final_mse_strictly_below": 1.0e-7,
                "all_conditions_required": True,
                "on_failure": (
                    "Close this exact rank-16 covariance observer measured; no retune, second "
                    "checkpoint, meter, or deployment."
                ),
            },
            "cost": successor_cost(),
            "why_this_is_the_one_next_mechanism": (
                "M established that merely exposing a learned reverse feature path "
                "moves errors but does not control their sign. The remaining untested "
                "structural axis is network-adaptive cross-unit covariance: N exposes "
                "it directly while avoiding the already closed Gaussian-secant, "
                "diagonal-variance, local-Gram/context, scalar, and backward-tangent axes."
            ),
            "launch_performed": False,
        },
    }
    return payload


def main() -> None:
    payload = build_payload()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if "--write" in sys.argv:
        if OUTPUT.exists():
            if OUTPUT.read_text(encoding="utf-8") != serialized:
                raise FileExistsError(f"refusing to overwrite changed {OUTPUT}")
        else:
            temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(OUTPUT)
    else:
        sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
