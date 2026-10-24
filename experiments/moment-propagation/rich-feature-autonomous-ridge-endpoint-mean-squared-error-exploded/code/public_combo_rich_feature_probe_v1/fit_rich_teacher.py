#!/Users/matt/anaconda3/bin/python3
"""Low-memory first discriminator for rich teacher-forced innovations."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
ROWS = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit"
PRED = P2 / "public_observer_fit_candidate_truth_score_v1/runs/manual_rho_hybrid_floor"
TARGET = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
RICH = HERE / "rich_rows"
OUT = HERE / "RICH_TEACHER_RESULTS.json"
DEPTH, WIDTH, N = 16, 1024, 35
HIDDEN = 96


def target_panel() -> np.ndarray:
    values = pq.read_table(TARGET, columns=["all_layer_means"]).column(0).combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    return np.ascontiguousarray(
        storage.to_numpy(zero_copy_only=False).reshape(-1, DEPTH, WIDTH)[:N], dtype=np.float32
    )


def path(rank: int, name: str) -> Path:
    return ROWS / f"{rank:04d}" / name


def robust_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values64 = values.astype(np.float64)
    center = np.median(values64, axis=(0, 2))
    scale = np.median(np.abs(values64 - center[None, :, None, :]), axis=(0, 2)) / 0.6744897501960817
    fallback = np.std(values64, axis=(0, 2))
    return center, np.maximum(np.where(scale > 1e-8, scale, fallback), 1e-8)


def make_features(raw, rich, rc, rs, qc, qs):
    rz = np.clip((raw[..., 1:].astype(np.float64) - rc[None, :, None, :]) / rs[None, :, None, :], -8, 8)
    qz = np.clip((rich.astype(np.float64) - qc[None, :, None, :]) / qs[None, :, None, :], -8, 8)
    local = np.concatenate((rz, qz), axis=-1)
    context = np.concatenate(
        (np.mean(local, axis=2), np.sqrt(np.mean(local * local, axis=2))), axis=-1
    )
    context = np.broadcast_to(context[:, :, None, :], (*local.shape[:3], context.shape[-1]))
    depth = np.broadcast_to(
        np.linspace(-1, 1, DEPTH)[None, :, None, None], (*local.shape[:3], 1)
    )
    return np.ascontiguousarray(np.concatenate((depth, local, context), axis=-1), dtype=np.float32)


class Teacher(nn.Module):
    def __init__(self, features: int, scales: np.ndarray):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(features, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU()
        )
        self.readout = nn.Parameter(torch.empty(DEPTH, HIDDEN))
        self.bias = nn.Parameter(torch.zeros(DEPTH))
        nn.init.normal_(self.readout, std=0.01)
        self.register_buffer("scales", torch.from_numpy(scales))

    def forward(self, x):
        batch = x.shape[0]
        h = self.trunk(x.reshape(-1, x.shape[-1])).reshape(batch, DEPTH, WIDTH, HIDDEN)
        return self.scales[None, :, None] * (
            torch.einsum("blwh,lh->blw", h, self.readout) + self.bias[None, :, None]
        )


def metric(model, x, target):
    model.eval()
    with torch.no_grad():
        mse = torch.mean((model(x) - target).double().square(), dim=(0, 2)).cpu().numpy()
    return mse


def main():
    torch.manual_seed(0xC04B0)
    np.random.seed(0xC04B0)
    torch.set_num_threads(12)
    truth = target_panel()
    pred = np.stack([np.load(PRED / f"row_{r:04d}/q24_output.npy") for r in range(N)]).astype(np.float32)
    raw = np.stack([np.load(path(r, "cp1792_raw_locals.npy")) for r in range(N)]).astype(np.float32)
    rich = np.stack([np.load(RICH / f"row_{r:04d}/rich_features.npy") for r in range(N)]).astype(np.float32)
    error = truth - pred
    gate = rich[..., 38]
    transport = np.zeros_like(error)
    for rank in range(N):
        weights = np.load(path(rank, "weights.npy"), mmap_mode="r")
        for layer in range(1, DEPTH):
            transport[rank, layer] = gate[rank, layer] * (
                error[rank, layer - 1].astype(np.float64) @ weights[layer].astype(np.float64)
            )
    a = np.zeros(DEPTH, dtype=np.float64)
    for layer in range(1, DEPTH):
        xx = transport[:20, layer].astype(np.float64).reshape(-1)
        yy = error[:20, layer].astype(np.float64).reshape(-1)
        a[layer] = np.dot(xx, yy) / max(np.dot(xx, xx), 1e-300)
    local = error - a.astype(np.float32)[None, :, None] * transport

    rc, rs = robust_fit(raw[:20, ..., 1:])
    qc, qs = robust_fit(rich[:20])
    x = make_features(raw, rich, rc, rs, qc, qs)
    scale = np.maximum(np.sqrt(np.mean(local[:20].astype(np.float64) ** 2, axis=(0, 2))), 1e-7).astype(np.float32)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(np.ascontiguousarray(local))
    scale_t = torch.from_numpy(scale)[None, :, None]
    model = Teacher(x.shape[-1], scale).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=8e-5)
    best = math.inf
    state = None
    best_epoch = -1
    stale = 0
    trace = []
    for epoch in range(161):
        if epoch % 5 == 0:
            val = metric(model, xt[20:25], yt[20:25])
            score = float(val[-1])
            trace.append({"epoch": epoch, "valid_final": score, "valid_mean": float(np.mean(val))})
            print(f"epoch={epoch:03d} valid_final={score:.9e} valid_mean={np.mean(val):.9e}", flush=True)
            if score < best:
                best, best_epoch, state, stale = score, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= 10:
                break
        if epoch == 160:
            break
        order = torch.randperm(20)
        for start in range(0, 20, 5):
            take = order[start : start + 5]
            optimizer.zero_grad(set_to_none=True)
            normalized = (model(xt[take]) - yt[take]) / scale_t
            loss = torch.mean(normalized.square())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    if state is None:
        raise RuntimeError("no checkpoint")
    model.load_state_dict(state)
    dev = metric(model, xt[:20], yt[:20])
    val = metric(model, xt[20:25], yt[20:25])
    held = metric(model, xt[25:35], yt[25:35])
    zero = np.mean(local[25:35].astype(np.float64) ** 2, axis=(0, 2))
    result = {
        "split": {"dev": [0, 20], "valid": [20, 25], "held": [25, 35]},
        "features": int(x.shape[-1]),
        "hidden": HIDDEN,
        "selected_epoch": best_epoch,
        "transport_scale": a.tolist(),
        "dev_layer_mse": dev.tolist(),
        "valid_layer_mse": val.tolist(),
        "held_layer_mse": held.tolist(),
        "held_final_local_mse": float(held[-1]),
        "held_mean_local_mse": float(np.mean(held)),
        "held_zero_predictor_layer_mse": zero.tolist(),
        "held_final_fraction_of_zero": float(held[-1] / zero[-1]),
        "trace": trace,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("held_final_local_mse", "held_mean_local_mse", "held_final_fraction_of_zero")}, indent=2))


if __name__ == "__main__":
    main()
