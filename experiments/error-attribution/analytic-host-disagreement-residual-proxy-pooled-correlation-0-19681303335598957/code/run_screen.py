#!/usr/bin/env python3
"""Can analytic-host disagreement replace the unavailable Q24 residual?

This is a saved-array screen on observer-fit role ranks 0:199.  Ranks 0:159
fit the proxy and ranks 160:199 evaluate it.  The primary endpoint is the
final-layer Q24 error after replacing the teacher-forced previous-layer error
with a target-blind disagreement proxy in the physical Phi * W transport.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesRegressor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET = ROOT / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ROWS = (
    ROOT
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
Q24 = (
    ROOT
    / "public_observer_fit_candidate_truth_score_v1/runs/manual_rho_hybrid_floor"
)
CAP_BUNDLE = (
    ROOT
    / "cp3_full_birth_cap_ladder/mini100_seal_v3/receipts/bundles"
    / "mini100-cp3caps-v3-3eb980d14b36e443614427f3c1d781763cca1c38f88a2231b5559bdac32fe4c0"
)
OUT = HERE / "RESULTS.json"

DEPTH = 16
WIDTH = 1024
N = 200
FIT = np.arange(0, 160)
VALID = np.arange(160, 200)
RIDGES = (0.0, 1e-6, 1e-4, 1e-2, 1.0)
FIT_OUTLIER_MSE = 5e-7


def row_file(rank: int, name: str) -> Path:
    return ROWS / f"{rank:04d}" / name


def load_targets() -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(TARGET, columns=["mlp_id", "all_layer_means"])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    target = storage.to_numpy(zero_copy_only=False).reshape(-1, DEPTH, WIDTH)
    return ids[:N].astype(np.int64), np.ascontiguousarray(target[:N], dtype=np.float32)


def load_panel() -> dict[str, np.ndarray]:
    ids, truth = load_targets()
    q24 = np.stack(
        [np.load(Q24 / f"row_{rank:04d}/q24_output.npy") for rank in range(N)]
    ).astype(np.float32)
    cp1792 = np.stack(
        [np.load(row_file(rank, "cp1792_host_rows.npy")) for rank in range(N)]
    ).astype(np.float32)
    cp1536 = np.stack(
        [np.load(row_file(rank, "cp1536_host_rows.npy")) for rank in range(N)]
    ).astype(np.float32)
    return {
        "ids": ids,
        "truth": truth,
        "q24": q24,
        "cp1792": cp1792,
        "cp1536": cp1536,
    }


def pooled_corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64).reshape(-1)
    yy = y.astype(np.float64).reshape(-1)
    xx -= np.mean(xx)
    yy -= np.mean(yy)
    denom = np.sqrt(np.dot(xx, xx) * np.dot(yy, yy))
    return float(np.dot(xx, yy) / max(denom, 1e-300))


def mean_network_cosine(x: np.ndarray, y: np.ndarray) -> float:
    xx = x.astype(np.float64)
    yy = y.astype(np.float64)
    numerator = np.sum(xx * yy, axis=1)
    denominator = np.sqrt(np.sum(xx * xx, axis=1) * np.sum(yy * yy, axis=1))
    return float(np.mean(numerator / np.maximum(denominator, 1e-300)))


def proxy_design(
    d1: np.ndarray, d2: np.ndarray, mode: str
) -> np.ndarray:
    """Return [net, coordinate, feature] without using targets."""
    if mode == "cp1792":
        return d1[..., None]
    if mode == "cp1536":
        return d2[..., None]
    if mode == "pair":
        return np.stack((d1, d2), axis=-1)
    if mode == "pair_context":
        mean1 = np.broadcast_to(np.mean(d1, axis=1)[:, None], d1.shape)
        mean2 = np.broadcast_to(np.mean(d2, axis=1)[:, None], d2.shape)
        rms1 = np.broadcast_to(
            np.sqrt(np.mean(d1.astype(np.float64) ** 2, axis=1))[:, None], d1.shape
        )
        rms2 = np.broadcast_to(
            np.sqrt(np.mean(d2.astype(np.float64) ** 2, axis=1))[:, None], d2.shape
        )
        return np.stack((d1, d2, mean1, mean2, rms1, rms2), axis=-1)
    raise ValueError(mode)


def fit_linear(
    design: np.ndarray, response: np.ndarray, ranks: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = design[ranks].reshape(-1, design.shape[-1]).astype(np.float64)
    y = response[ranks].reshape(-1).astype(np.float64)
    center = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-12)
    z = (x - center) / scale
    z = np.concatenate((np.ones((len(z), 1)), z), axis=1)
    gram = (z.T @ z) / len(z)
    gram[1:, 1:] += ridge * np.eye(z.shape[1] - 1)
    rhs = (z.T @ y) / len(z)
    beta = np.linalg.lstsq(gram, rhs, rcond=1e-12)[0]
    return beta, center, scale


def apply_linear(
    design: np.ndarray, beta: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    z = (design.astype(np.float64) - center) / scale
    return (beta[0] + np.einsum("...f,f->...", z, beta[1:])).astype(np.float32)


def proxy_screen(panel: dict[str, np.ndarray]) -> tuple[dict, np.ndarray, str]:
    error = panel["truth"] - panel["q24"]
    d1 = panel["cp1792"] - panel["q24"]
    d2 = panel["cp1536"] - panel["q24"]
    modes = ("cp1792", "cp1536", "pair", "pair_context")
    candidates: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    # Select one frozen proxy family by mean fit-layer normalized MSE.  Report
    # validation for every family; the final transport uses the fit-selected one.
    fit_scores: dict[str, float] = {}
    for mode in modes:
        predicted = np.empty_like(error)
        layer_records = []
        selected_ridges = []
        for layer in range(DEPTH):
            design = proxy_design(d1[:, layer], d2[:, layer], mode)
            ridge_records = []
            # Group holdout inside FIT: choose ridge on ranks 128:159, never
            # individual neurons.  Refit the chosen ridge on all 160 networks.
            per_network_mse = np.mean(
                error[:, layer].astype(np.float64) ** 2, axis=1
            )
            stable = per_network_mse <= FIT_OUTLIER_MSE
            subfit = np.arange(0, 128)[stable[:128]]
            subvalid = np.arange(128, 160)[stable[128:160]]
            for ridge in RIDGES:
                beta, center, scale = fit_linear(design, error[:, layer], subfit, ridge)
                p = apply_linear(design[subvalid], beta, center, scale)
                ridge_records.append(
                    (float(np.mean((p - error[subvalid, layer]) ** 2, dtype=np.float64)), ridge)
                )
            ridge = min(ridge_records)[1]
            stable_fit = FIT[stable[FIT]]
            beta, center, scale = fit_linear(
                design, error[:, layer], stable_fit, ridge
            )
            predicted[:, layer] = apply_linear(design, beta, center, scale)
            selected_ridges.append(ridge)
            true_valid = error[VALID, layer]
            proxy_valid = predicted[VALID, layer]
            layer_records.append(
                {
                    "layer": layer,
                    "ridge": ridge,
                    "validation_pooled_correlation": pooled_corr(proxy_valid, true_valid),
                    "validation_mean_network_cosine": mean_network_cosine(proxy_valid, true_valid),
                    "validation_true_mse": float(np.mean(true_valid ** 2, dtype=np.float64)),
                    "validation_proxy_residual_mse": float(
                        np.mean((proxy_valid - true_valid) ** 2, dtype=np.float64)
                    ),
                }
            )
        key = mode
        predictions[key] = predicted
        fit_mse = []
        true_fit_mse = []
        for layer in range(DEPTH):
            stable = np.mean(error[FIT, layer].astype(np.float64) ** 2, axis=1) <= FIT_OUTLIER_MSE
            fit_mse.append(
                np.mean(
                    (predicted[FIT[stable], layer] - error[FIT[stable], layer]) ** 2,
                    dtype=np.float64,
                )
            )
            true_fit_mse.append(
                np.mean(error[FIT[stable], layer] ** 2, dtype=np.float64)
            )
        fit_mse = np.asarray(fit_mse)
        true_fit_mse = np.asarray(true_fit_mse)
        fit_score = float(np.mean(fit_mse[2:] / true_fit_mse[2:]))
        fit_scores[key] = fit_score
        candidates.append(
            {
                "mode": mode,
                "fit_selected_score": fit_score,
                "selected_ridges": selected_ridges,
                "layers": layer_records,
                "validation_final_proxy_residual_mse": layer_records[-1][
                    "validation_proxy_residual_mse"
                ],
                "validation_final_pooled_correlation": layer_records[-1][
                    "validation_pooled_correlation"
                ],
            }
        )
    winner = min(fit_scores, key=fit_scores.get)
    return {"candidates": candidates, "fit_selected_mode": winner}, predictions[winner], winner


def final_local_features(
    panel: dict[str, np.ndarray], ranks: np.ndarray, centers: np.ndarray, scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    layer = DEPTH - 1
    out = np.empty((len(ranks), WIDTH, 60), dtype=np.float32)
    sigma_out = np.empty((len(ranks), WIDTH), dtype=np.float64)
    for position, rank in enumerate(ranks):
        raw = np.load(row_file(int(rank), "cp1792_raw_locals.npy"), mmap_mode="r")[
            layer
        ].astype(np.float64)
        sigma = np.load(row_file(int(rank), "cp1792_sigma.npy"), mmap_mode="r")[
            layer
        ].astype(np.float64)
        host = panel["cp1792"][rank, layer].astype(np.float64)
        z = np.clip((raw[:, 1:] - centers) / scales, -8.0, 8.0)
        context = np.concatenate(
            (
                np.mean(z, axis=0, dtype=np.float64),
                np.sqrt(np.mean(z * z, axis=0, dtype=np.float64)) - 1.0,
            )
        )
        out[position] = np.concatenate(
            (
                z,
                np.broadcast_to(context, (WIDTH, 38)),
                (panel["q24"][rank, layer] / sigma)[:, None],
                (host / sigma)[:, None],
                ((panel["q24"][rank, layer] - host) / sigma)[:, None],
            ),
            axis=1,
        )
        sigma_out[position] = sigma
    return out, sigma_out


def fit_raw_centers_scales(ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Only final layer is needed.  Pool sufficient sums instead of retaining
    # the 160 x 1024 x 19 raw tensor.
    total = np.zeros(19, dtype=np.float64)
    total2 = np.zeros(19, dtype=np.float64)
    for rank in ranks:
        raw = np.load(row_file(int(rank), "cp1792_raw_locals.npy"), mmap_mode="r")[
            -1, :, 1:
        ].astype(np.float64)
        total += np.sum(raw, axis=0)
        total2 += np.sum(raw * raw, axis=0)
    count = float(len(ranks) * WIDTH)
    center = total / count
    scale = np.sqrt(np.maximum(total2 / count - center * center, 1e-16))
    return center, scale


def transported(
    residual: np.ndarray, ranks: np.ndarray, layer: int = DEPTH - 1
) -> np.ndarray:
    out = np.empty((len(ranks), WIDTH), dtype=np.float64)
    for position, rank in enumerate(ranks):
        weights = np.load(row_file(int(rank), "weights.npy"), mmap_mode="r")
        raw = np.load(row_file(int(rank), "cp1792_raw_locals.npy"), mmap_mode="r")
        out[position] = raw[layer, :, 5].astype(np.float64) * (
            residual[rank, layer - 1].astype(np.float64) @ weights[layer].astype(np.float64)
        )
    return out


def final_endpoint(
    panel: dict[str, np.ndarray], proxy: np.ndarray, proxy_mode: str, trees: int,
    jobs: int, min_leaf: int,
) -> dict:
    error = (panel["truth"] - panel["q24"]).astype(np.float64)
    fit_final_mse = np.mean(error[FIT, -1] ** 2, axis=1, dtype=np.float64)
    stable_fit = FIT[fit_final_mse <= FIT_OUTLIER_MSE]
    true_fit_t = transported(error, stable_fit)
    true_valid_t = transported(error, VALID)
    proxy_valid_t = transported(proxy, VALID)
    x = true_fit_t.reshape(-1)
    y = error[stable_fit, -1].reshape(-1)
    a = float(np.dot(x, y) / max(np.dot(x, x), 1e-300))
    innovation_fit = error[stable_fit, -1] - a * true_fit_t

    centers, scales = fit_raw_centers_scales(stable_fit)
    fit_features, fit_sigma = final_local_features(panel, stable_fit, centers, scales)
    valid_features, valid_sigma = final_local_features(panel, VALID, centers, scales)
    target_standard = (innovation_fit / fit_sigma).reshape(-1)
    model = ExtraTreesRegressor(
        n_estimators=trees,
        criterion="squared_error",
        max_features=0.75,
        min_samples_leaf=min_leaf,
        bootstrap=False,
        n_jobs=jobs,
        random_state=20260824,
    )
    fit_started = time.perf_counter()
    model.fit(fit_features.reshape(-1, 60), target_standard)
    fit_seconds = time.perf_counter() - fit_started
    local_valid = valid_sigma * model.predict(valid_features.reshape(-1, 60)).reshape(
        len(VALID), WIDTH
    )
    teacher_correction = a * true_valid_t + local_valid
    proxy_correction = a * proxy_valid_t + local_valid
    direct_local = local_valid
    baseline_error = error[VALID, -1]

    # Deliberately validation-oracle upper bounds: even these are not deployable,
    # but they show whether scalar recalibration or adding the same-layer proxy
    # could rescue a weak proxy family.
    oracle_transport_scale = float(
        np.dot(proxy_valid_t.reshape(-1), (baseline_error - local_valid).reshape(-1))
        / max(np.dot(proxy_valid_t.reshape(-1), proxy_valid_t.reshape(-1)), 1e-300)
    )
    oracle_span = np.stack(
        (local_valid, proxy_valid_t, proxy[VALID, -1].astype(np.float64)), axis=-1
    ).reshape(-1, 3)
    oracle_beta = np.linalg.lstsq(oracle_span, baseline_error.reshape(-1), rcond=1e-12)[0]
    oracle_span_residual = baseline_error.reshape(-1) - oracle_span @ oracle_beta

    def mse(residual: np.ndarray) -> float:
        return float(np.mean(residual * residual, dtype=np.float64))

    return {
        "proxy_mode": proxy_mode,
        "transport_scale": a,
        "trees": trees,
        "jobs": jobs,
        "min_samples_leaf": min_leaf,
        "fit_networks": int(len(stable_fit)),
        "excluded_fit_ranks": [int(rank) for rank in FIT if rank not in set(stable_fit)],
        "fit_seconds": fit_seconds,
        "validation_previous_residual_pooled_correlation": pooled_corr(
            proxy[VALID, -2], error[VALID, -2]
        ),
        "validation_previous_residual_mean_network_cosine": mean_network_cosine(
            proxy[VALID, -2], error[VALID, -2]
        ),
        "validation_transported_proxy_true_correlation": pooled_corr(
            proxy_valid_t, true_valid_t
        ),
        "validation_baseline_final_mse": mse(baseline_error),
        "validation_direct_local_final_mse": mse(baseline_error - direct_local),
        "validation_teacher_forced_final_mse": mse(
            baseline_error - teacher_correction
        ),
        "validation_proxy_transport_final_mse": mse(baseline_error - proxy_correction),
        "validation_oracle_scaled_proxy_plus_local_final_mse": mse(
            baseline_error - local_valid - oracle_transport_scale * proxy_valid_t
        ),
        "validation_oracle_transport_scale": oracle_transport_scale,
        "validation_oracle_local_proxy_transport_same_layer_span_final_mse": mse(
            oracle_span_residual
        ),
        "validation_oracle_span_coefficients": oracle_beta.tolist(),
        "validation_true_transport_only_final_mse": mse(
            baseline_error - a * true_valid_t
        ),
        "validation_proxy_transport_only_final_mse": mse(
            baseline_error - a * proxy_valid_t
        ),
        "validation_per_network_proxy_transport_final_mse": np.mean(
            (baseline_error - proxy_correction) ** 2, axis=1, dtype=np.float64
        ).tolist(),
    }


def inventory(ids: np.ndarray) -> dict:
    q24_count = sum(
        (Q24 / f"row_{rank:04d}/q24_output.npy").is_file() for rank in range(N)
    )
    q32_root = ROOT / "public_observer_fit_direct_instability_scan_v1/runs"
    q32_ranks = [
        rank
        for rank in range(N)
        if (q32_root / f"row_{rank:04d}/q32_output.npy").is_file()
    ]
    cap_names = (
        "predictions_cap16384.npy",
        "predictions_cap16384_layer2_mean_carry.npy",
        "predictions_cap16384_recursive_mean_carry.npy",
    )
    matched_fit = [int(rank) for rank in FIT if 0 <= ids[rank] < 100]
    matched_valid = [int(rank) for rank in VALID if 0 <= ids[rank] < 100]
    return {
        "q24_combo_rows_on_fixed_panel": q24_count,
        "cp1792_host_rows_on_fixed_panel": N,
        "cp1536_host_rows_on_fixed_panel": N,
        "q32_saved_ranks_on_fixed_panel": q32_ranks,
        "cap16384_saved_files": [name for name in cap_names if (CAP_BUNDLE / name).is_file()],
        "cap16384_joinable_fit_ranks": matched_fit,
        "cap16384_joinable_validation_ranks": matched_valid,
        "cap16384_note": "mini100 only; 17/160 fit and 8/40 validation rows join by mlp_id, so it is not a complete fixed-panel proxy",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--min-leaf", type=int, default=16)
    parser.add_argument("--proxy-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    panel = load_panel()
    proxy_results, proxy, winner = proxy_screen(panel)
    result = {
        "schema": "public-observable-q24-residual-proxy-v1",
        "split": {"fit": [0, 160], "validation": [160, 200]},
        "inventory": inventory(panel["ids"]),
        "proxy_screen": proxy_results,
        "final_endpoint": None,
    }
    if not args.proxy_only:
        result["final_endpoint"] = final_endpoint(
            panel, proxy, winner, args.trees, args.jobs, args.min_leaf
        )
    result["wall_seconds"] = time.perf_counter() - started
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
