#!/usr/bin/env python3
"""Low-memory layerwise rich ridge and autonomous tangent rollout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
ROWS = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit"
PRED = P2 / "public_observer_fit_candidate_truth_score_v1/runs/manual_rho_hybrid_floor"
TARGET = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
RICH = HERE / "rich_rows"
OUT = HERE / "RICH_RIDGE_RESULTS.json"
DEPTH, WIDTH, N = 16, 1024, 35
DEV, VALID, HELD = np.arange(20), np.arange(20, 25), np.arange(25, 35)
LAMBDAS = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def row_file(rank: int, name: str) -> Path:
    return ROWS / f"{rank:04d}" / name


def targets() -> np.ndarray:
    values = pq.read_table(TARGET, columns=["all_layer_means"]).column(0).combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    return np.ascontiguousarray(
        storage.to_numpy(zero_copy_only=False).reshape(-1, DEPTH, WIDTH)[:N], dtype=np.float32
    )


def layer_design(layer: int) -> np.ndarray:
    raw = np.stack(
        [np.load(row_file(rank, "cp1792_raw_locals.npy"), mmap_mode="r")[layer, :, 1:] for rank in range(N)]
    ).astype(np.float32)
    rich = np.stack(
        [np.load(RICH / f"row_{rank:04d}/rich_features.npy", mmap_mode="r")[layer] for rank in range(N)]
    ).astype(np.float32)
    local = np.concatenate((raw, rich), axis=-1).astype(np.float64)
    center = np.median(local[DEV], axis=(0, 1))
    mad = np.median(np.abs(local[DEV] - center[None, None, :]), axis=(0, 1)) / 0.6744897501960817
    fallback = np.std(local[DEV], axis=(0, 1))
    scale = np.maximum(np.where(mad > 1e-8, mad, fallback), 1e-8)
    z = np.clip((local - center[None, None, :]) / scale[None, None, :], -8.0, 8.0)
    context = np.concatenate((np.mean(z, axis=1), np.sqrt(np.mean(z * z, axis=1))), axis=-1)
    context = np.broadcast_to(context[:, None, :], (N, WIDTH, context.shape[-1]))
    design = np.concatenate(
        (
            np.ones((N, WIDTH, 1), dtype=np.float64),
            z,
            z * z - 1.0,
            context,
        ),
        axis=-1,
    )
    return np.ascontiguousarray(design, dtype=np.float32)


def ridge_select(design: np.ndarray, response: np.ndarray):
    xdev = design[DEV].reshape(-1, design.shape[-1]).astype(np.float64)
    ydev = response[DEV].reshape(-1).astype(np.float64)
    column = np.sqrt(np.mean(xdev * xdev, axis=0))
    column = np.maximum(column, 1e-10)
    xnorm = xdev / column
    gram = (xnorm.T @ xnorm) / len(ydev)
    rhs = (xnorm.T @ ydev) / len(ydev)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected = eigenvectors.T @ rhs
    xvalid = design[VALID].reshape(-1, design.shape[-1]).astype(np.float64)
    yvalid = response[VALID].reshape(-1).astype(np.float64)
    best = None
    for ridge in LAMBDAS:
        beta_norm = eigenvectors @ (projected / (eigenvalues + ridge))
        beta = beta_norm / column
        mse = float(np.mean((xvalid @ beta - yvalid) ** 2))
        if best is None or mse < best[0]:
            best = (mse, ridge, beta)
    assert best is not None
    prediction = np.einsum("nwf,f->nw", design.astype(np.float64), best[2], optimize=True)
    return best[0], best[1], best[2], prediction.astype(np.float32)


def main() -> None:
    truth = targets()
    pred = np.stack([np.load(PRED / f"row_{rank:04d}/q24_output.npy") for rank in range(N)]).astype(np.float32)
    error = truth - pred
    gate = np.stack(
        [np.load(RICH / f"row_{rank:04d}/rich_features.npy", mmap_mode="r")[..., 38] for rank in range(N)]
    ).astype(np.float32)
    transport = np.zeros_like(error)
    for rank in range(N):
        weights = np.load(row_file(rank, "weights.npy"), mmap_mode="r")
        for layer in range(1, DEPTH):
            transport[rank, layer] = gate[rank, layer] * (error[rank, layer - 1] @ weights[layer])
    a = np.zeros(DEPTH, dtype=np.float64)
    for layer in range(1, DEPTH):
        xx = transport[DEV, layer].astype(np.float64).reshape(-1)
        yy = error[DEV, layer].astype(np.float64).reshape(-1)
        a[layer] = np.dot(xx, yy) / max(np.dot(xx, xx), 1e-300)
    local = error - a.astype(np.float32)[None, :, None] * transport

    predicted_local = np.empty_like(local)
    layer_records = []
    betas = []
    for layer in range(DEPTH):
        design = layer_design(layer)
        valid_mse, ridge, beta, prediction = ridge_select(design, local[:, layer])
        predicted_local[:, layer] = prediction
        betas.append(beta)
        held_mse = float(np.mean((prediction[HELD] - local[HELD, layer]) ** 2, dtype=np.float64))
        zero_mse = float(np.mean(local[HELD, layer].astype(np.float64) ** 2))
        layer_records.append(
            {"layer": layer, "ridge": ridge, "valid_local_mse": valid_mse, "held_local_mse": held_mse, "held_zero_mse": zero_mse}
        )
        print(f"layer={layer:02d} ridge={ridge:.1e} valid={valid_mse:.9e} held={held_mse:.9e}", flush=True)
        del design

    # Select a causal local-innovation shrink at each layer on validation.
    correction = np.zeros_like(error)
    local_gain = np.ones(DEPTH, dtype=np.float64)
    for layer in range(DEPTH):
        inherited = np.zeros((N, WIDTH), dtype=np.float32)
        if layer:
            for rank in range(N):
                weights = np.load(row_file(rank, "weights.npy"), mmap_mode="r")
                inherited[rank] = a[layer] * gate[rank, layer] * (correction[rank, layer - 1] @ weights[layer])
        x = predicted_local[VALID, layer].astype(np.float64).reshape(-1)
        y = (error[VALID, layer] - inherited[VALID]).astype(np.float64).reshape(-1)
        local_gain[layer] = np.clip(np.dot(x, y) / max(np.dot(x, x), 1e-300), 0.0, 2.0)
        correction[:, layer] = inherited + np.float32(local_gain[layer]) * predicted_local[:, layer]

    corrected = np.maximum(pred + correction, 0.0)
    autonomous = np.mean(
        (corrected.astype(np.float64) - truth.astype(np.float64)) ** 2, axis=(0, 2)
    )
    baseline = np.mean((pred.astype(np.float64) - truth.astype(np.float64)) ** 2, axis=(0, 2))
    result = {
        "split": {"dev": [0, 20], "valid": [20, 25], "held": [25, 35]},
        "basis": "[1,z,z^2-1,group_mean(z),group_rms(z)] over raw19+rich40",
        "transport_scale": a.tolist(),
        "local_gain_selected_on_validation": local_gain.tolist(),
        "layers": layer_records,
        "held_final_local_mse": layer_records[-1]["held_local_mse"],
        "held_mean_local_mse": float(np.mean([row["held_local_mse"] for row in layer_records])),
        "held_baseline_final_mse": float(baseline[HELD, -1].mean()) if baseline.ndim > 1 else float(np.mean((pred[HELD, -1].astype(np.float64)-truth[HELD, -1].astype(np.float64))**2)),
        "held_autonomous_layer_mse": np.mean((corrected[HELD].astype(np.float64) - truth[HELD].astype(np.float64)) ** 2, axis=(0, 2)).tolist(),
        "held_autonomous_final_mse": float(np.mean((corrected[HELD, -1].astype(np.float64) - truth[HELD, -1].astype(np.float64)) ** 2)),
    }
    # The preceding baseline expression is kept simple and made explicit here.
    result["held_baseline_final_mse"] = float(
        np.mean((pred[HELD, -1].astype(np.float64) - truth[HELD, -1].astype(np.float64)) ** 2)
    )
    result["held_autonomous_ratio"] = result["held_autonomous_final_mse"] / result["held_baseline_final_mse"]
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("held_final_local_mse", "held_autonomous_final_mse", "held_autonomous_ratio")}, indent=2))


if __name__ == "__main__":
    main()
