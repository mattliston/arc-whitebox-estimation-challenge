#!/Users/matt/anaconda3/bin/python3
"""Fast fixed-split screen for accuracy versus cheap distillation.

The only fit rows are observer-fit ranks 0:160.  Ranks 160:200 are loaded
only after every model and hyperparameter below is fixed; they are never used
for checkpointing or selection.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesRegressor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROWS = (
    ROOT / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
Q24 = ROOT / "public_observer_fit_candidate_truth_score_v1/runs/manual_rho_hybrid_floor"
TARGET = ROOT / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
OUT = HERE / "RESULTS.json"

N, FIT_STOP, DEPTH, WIDTH = 200, 160, 16, 1024
RIDGE = 3.0e-2  # fixed from the earlier 0:20/20:25 pilot
Q24_FLOPS = 2_126_977_473_483
CP1792_FLOPS = 214_122_341_378
BUDGET = 2**41


def load_truth() -> np.ndarray:
    values = pq.read_table(TARGET, columns=["all_layer_means"]).column(0).combine_chunks()
    values = getattr(values, "storage", values)
    while not pa.types.is_primitive(values.type):
        values = values.flatten()
    return np.ascontiguousarray(
        values.to_numpy(zero_copy_only=False).reshape(-1, DEPTH, WIDTH)[:N],
        dtype=np.float64,
    )


def row_file(rank: int, name: str) -> Path:
    return ROWS / f"{rank:04d}" / name


def load_panel() -> dict[str, np.ndarray]:
    return {
        "q24": np.stack(
            [np.load(Q24 / f"row_{r:04d}/q24_output.npy") for r in range(N)]
        ).astype(np.float64),
        "cp": np.stack(
            [np.load(row_file(r, "cp1792_host_rows.npy")) for r in range(N)]
        ).astype(np.float64),
        "sigma": np.stack(
            [np.load(row_file(r, "cp1792_sigma.npy")) for r in range(N)]
        ).astype(np.float64),
        "raw": np.stack(
            [np.load(row_file(r, "cp1792_raw_locals.npy")) for r in range(N)]
        ).astype(np.float64),
    }


def standardized_features(panel: dict[str, np.ndarray]) -> np.ndarray:
    raw = panel["raw"]
    tail = raw[:FIT_STOP, ..., 1:]
    center = tail.mean(axis=(0, 2), dtype=np.float64)
    scale = np.maximum(tail.std(axis=(0, 2), dtype=np.float64), 2.0**-20)
    z = np.clip((raw[..., 1:] - center[None, :, None]) / scale[None, :, None], -8.0, 8.0)
    context = np.concatenate(
        (z.mean(axis=2), np.sqrt(np.mean(z * z, axis=2)) - 1.0), axis=-1
    )
    context = np.broadcast_to(context[:, :, None, :], (N, DEPTH, WIDTH, 38))
    sigma = np.maximum(panel["sigma"], np.finfo(np.float64).tiny)
    return np.ascontiguousarray(
        np.concatenate(
            (z, context, (panel["q24"] / sigma)[..., None],
             (panel["cp"] / sigma)[..., None],
             ((panel["q24"] - panel["cp"]) / sigma)[..., None]),
            axis=-1,
        ),
        dtype=np.float32,
    )


def true_transport(error: np.ndarray, gate: np.ndarray) -> np.ndarray:
    result = np.zeros_like(error)
    for rank in range(N):
        weights = np.load(row_file(rank, "weights.npy"), mmap_mode="r")
        for layer in range(1, DEPTH):
            result[rank, layer] = gate[rank, layer] * (
                weights[layer].T @ error[rank, layer - 1]
            )
        if (rank + 1) % 40 == 0:
            print(f"transport {rank + 1}/{N}", flush=True)
    return result


def robust_fit_mask(error: np.ndarray, layer: int) -> np.ndarray:
    """Use all fit ranks except the top 2.5% response-energy contamination."""
    per_network = np.mean(error[:FIT_STOP, layer] ** 2, axis=1)
    cutoff = np.quantile(per_network, 0.975)
    return np.flatnonzero(per_network <= cutoff)


def fit_accuracy_lane(
    truth: np.ndarray,
    panel: dict[str, np.ndarray],
    features: np.ndarray,
) -> dict[str, object]:
    error = truth - panel["q24"]
    gate = panel["raw"][..., 5]
    transport = true_transport(error, gate)
    betas: list[np.ndarray] = []
    gains = np.zeros(DEPTH, dtype=np.float64)
    local_validation = np.empty((N - FIT_STOP, DEPTH, WIDTH), dtype=np.float64)
    teacher_validation = np.empty_like(local_validation)
    for layer in range(DEPTH):
        keep = robust_fit_mask(error, layer)
        t = transport[keep, layer].reshape(-1)
        e = error[keep, layer].reshape(-1)
        gains[layer] = 0.0 if layer == 0 else np.dot(t, e) / max(np.dot(t, t), 1e-300)
        innovation = error[keep, layer] - gains[layer] * transport[keep, layer]
        x = features[keep, layer].reshape(-1, features.shape[-1]).astype(np.float64)
        y = innovation.reshape(-1)
        column_scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-12)
        xn = x / column_scale
        gram = (xn.T @ xn) / len(y)
        rhs = (xn.T @ y) / len(y)
        beta = np.linalg.solve(gram + RIDGE * np.eye(gram.shape[0]), rhs) / column_scale
        betas.append(beta)
        local_validation[:, layer] = (
            features[FIT_STOP:, layer].astype(np.float64) @ beta
        )
        teacher_validation[:, layer] = (
            gains[layer] * transport[FIT_STOP:, layer] + local_validation[:, layer]
        )

    correction = np.zeros_like(local_validation)
    for position, rank in enumerate(range(FIT_STOP, N)):
        weights = np.load(row_file(rank, "weights.npy"), mmap_mode="r")
        correction[position, 0] = local_validation[position, 0]
        for layer in range(1, DEPTH):
            inherited = gate[rank, layer] * (
                weights[layer].T @ correction[position, layer - 1]
            )
            correction[position, layer] = gains[layer] * inherited + local_validation[position, layer]
    autonomous = np.maximum(panel["q24"][FIT_STOP:] + correction, 0.0)
    teacher = np.maximum(panel["q24"][FIT_STOP:] + teacher_validation, 0.0)
    truth_val = truth[FIT_STOP:]
    layer_auto = np.mean((autonomous - truth_val) ** 2, axis=(0, 2))
    layer_teacher = np.mean((teacher - truth_val) ** 2, axis=(0, 2))
    baseline = np.mean((panel["q24"][FIT_STOP:] - truth_val) ** 2, axis=(0, 2))
    head_flops = DEPTH * WIDTH * (2 * features.shape[-1])
    transport_flops = (DEPTH - 1) * (2 * WIDTH * WIDTH - WIDTH + 2 * WIDTH)
    total = Q24_FLOPS + head_flops + transport_flops
    return {
        "model": "robust ridge innovations plus autonomous scalar physical transport",
        "fit_ranks": [0, FIT_STOP],
        "validation_ranks_opened_once": [FIT_STOP, N],
        "ridge_fixed_from_prior_pilot": RIDGE,
        "fit_response_energy_trim_fraction": 0.025,
        "feature_count": int(features.shape[-1]),
        "transport_gains": gains.tolist(),
        "validation_baseline_final_raw_mse": float(baseline[-1]),
        "validation_teacher_forced_final_raw_mse": float(layer_teacher[-1]),
        "validation_autonomous_final_raw_mse": float(layer_auto[-1]),
        "validation_autonomous_layer_mse": layer_auto.tolist(),
        "estimated_flops": {
            "q24_host": Q24_FLOPS,
            "local_linear_head": int(head_flops),
            "physical_transport": int(transport_flops),
            "total": int(total),
            "fraction_of_budget": float(total / BUDGET),
        },
        "projected_adjusted_score": float(layer_auto[-1] * max(0.1, total / BUDGET)),
        "projects_below_1e_minus_8_adjusted": bool(
            layer_auto[-1] * max(0.1, total / BUDGET) < 1e-8
        ),
    }


def cheap_layer_features(
    features: np.ndarray, previous: np.ndarray | None, sigma: np.ndarray, layer: int
) -> np.ndarray:
    # CP-only deployment columns: raw/context plus CP/sigma.  Four invariant
    # summaries of the student's previous correction make the student a cheap
    # autonomous network-level recurrence without another dense W transport.
    base = np.concatenate((features[:, layer, :, :57], features[:, layer, :, 58:59]), axis=-1)
    if previous is None:
        summary = np.zeros((N, 4), dtype=np.float32)
    else:
        normalized = previous[:, layer - 1] / np.maximum(sigma[:, layer - 1], 1e-12)
        summary = np.stack(
            (normalized.mean(1), normalized.std(1),
             np.mean(normalized**3, axis=1), np.max(np.abs(normalized), axis=1)),
            axis=1,
        ).astype(np.float32)
    return np.concatenate(
        (base, np.broadcast_to(summary[:, None, :], (N, WIDTH, 4))), axis=-1
    )


def fit_cheap_student(
    label: str,
    teacher: np.ndarray,
    truth: np.ndarray,
    panel: dict[str, np.ndarray],
    features: np.ndarray,
) -> dict[str, object]:
    prediction = np.empty_like(panel["cp"])
    correction = np.empty_like(panel["cp"])
    for layer in range(DEPTH):
        x = cheap_layer_features(features, None if layer == 0 else correction, panel["sigma"], layer)
        y = ((teacher[:, layer] - panel["cp"][:, layer]) / np.maximum(panel["sigma"][:, layer], 1e-12))
        model = ExtraTreesRegressor(
            n_estimators=32,
            criterion="squared_error",
            max_features=0.75,
            min_samples_leaf=128,
            bootstrap=False,
            n_jobs=2,
            random_state=0xD157_1100 + layer,
        )
        model.fit(x[:FIT_STOP].reshape(-1, x.shape[-1]), y[:FIT_STOP].reshape(-1))
        # Populate train and validation recursively.  Validation has no role in fitting.
        correction[:, layer] = panel["sigma"][:, layer] * model.predict(
            x.reshape(-1, x.shape[-1])
        ).reshape(N, WIDTH)
        prediction[:, layer] = np.maximum(panel["cp"][:, layer] + correction[:, layer], 0.0)
        if (layer + 1) % 4 == 0:
            print(f"cheap-{label} layers {layer + 1}/{DEPTH}", flush=True)
    val = slice(FIT_STOP, N)
    raw_layer = np.mean((prediction[val] - truth[val]) ** 2, axis=(0, 2))
    q24_layer = np.mean((prediction[val] - panel["q24"][val]) ** 2, axis=(0, 2))
    # Conservative comparison count: each tree takes at most 16 decisions;
    # feature/recurrent summary arithmetic is separately bounded at 128 FLOPs/neuron/layer.
    tree_flops = DEPTH * WIDTH * 32 * 16
    feature_flops = DEPTH * WIDTH * 128
    total = CP1792_FLOPS + tree_flops + feature_flops
    return {
        "training_teacher": label,
        "model": "32-tree/layer CP-local student with four-scalar autonomous recurrent summary",
        "fit_ranks": [0, FIT_STOP],
        "validation_ranks_opened_once": [FIT_STOP, N],
        "validation_truth_final_raw_mse": float(raw_layer[-1]),
        "validation_q24_distillation_final_mse": float(q24_layer[-1]),
        "estimated_flops": {
            "cp1792_host": CP1792_FLOPS,
            "student_upper_bound": int(tree_flops + feature_flops),
            "total": int(total),
            "fraction_of_budget": float(total / BUDGET),
        },
        "projected_adjusted_score": float(raw_layer[-1] * max(0.1, total / BUDGET)),
        "projects_below_1e_minus_8_adjusted": bool(
            raw_layer[-1] * max(0.1, total / BUDGET) < 1e-8
        ),
    }


def main() -> None:
    started = time.perf_counter()
    truth = load_truth()
    panel = load_panel()
    features = standardized_features(panel)
    fit_baselines = {
        "q24_train_final_raw_mse": float(np.mean((panel["q24"][:FIT_STOP, -1] - truth[:FIT_STOP, -1]) ** 2)),
        "q24_validation_final_raw_mse": float(np.mean((panel["q24"][FIT_STOP:, -1] - truth[FIT_STOP:, -1]) ** 2)),
        "cp1792_validation_final_raw_mse": float(np.mean((panel["cp"][FIT_STOP:, -1] - truth[FIT_STOP:, -1]) ** 2)),
    }
    accuracy = fit_accuracy_lane(truth, panel, features)
    cheap_q24 = fit_cheap_student("current_q24", panel["q24"], truth, panel, features)
    cheap_truth = fit_cheap_student("public_truth", truth, truth, panel, features)
    result = {
        "schema": "public-dual-distill-fast-screen-v1",
        "split": {"fit": [0, FIT_STOP], "validation_opened_once": [FIT_STOP, N]},
        "baselines": fit_baselines,
        "accuracy_lane": accuracy,
        "cheap_lane_q24_distillation": cheap_q24,
        "cheap_lane_truth_supervision": cheap_truth,
        "wall_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
