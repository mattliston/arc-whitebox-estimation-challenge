#!/Users/matt/anaconda3/bin/python3
"""Quick structured residual pilot using the exact combo's rich state tape."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

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
OUT = HERE / "RICH_STRUCTURED_RESULTS.json"

DEPTH = 16
WIDTH = 1024
DEV = tuple(range(20))
VALID = tuple(range(20, 25))
HELD = tuple(range(25, 35))
HIDDEN = 96
SEED = 0xC04B0


def targets(stop: int) -> np.ndarray:
    table = pq.read_table(TARGET, columns=["all_layer_means"])
    values = table.column(0).combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    return np.ascontiguousarray(
        storage.to_numpy(zero_copy_only=False).reshape(-1, DEPTH, WIDTH)[:stop],
        dtype=np.float32,
    )


def row_file(rank: int, name: str) -> Path:
    return ROWS / f"{rank:04d}" / name


def load_arrays(stop: int, include_weights: bool) -> dict[str, np.ndarray]:
    payload = {
        "pred": np.stack([np.load(PRED / f"row_{r:04d}/q24_output.npy") for r in range(stop)]),
        "raw": np.stack([np.load(row_file(r, "cp1792_raw_locals.npy")) for r in range(stop)]),
        "sigma": np.stack([np.load(row_file(r, "cp1792_sigma.npy")) for r in range(stop)]),
        "rich": np.stack([np.load(RICH / f"row_{r:04d}/rich_features.npy") for r in range(stop)]),
    }
    if include_weights:
        payload["weights"] = np.stack(
            [np.load(row_file(r, "weights.npy"), mmap_mode="r") for r in range(stop)]
        )
    return {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in payload.items()}


def robust_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values.astype(np.float64), axis=(0, 2))
    deviation = np.abs(values.astype(np.float64) - center[None, :, None, :])
    scale = np.median(deviation, axis=(0, 2)) / 0.6744897501960817
    fallback = np.std(values.astype(np.float64), axis=(0, 2))
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.maximum(scale, 1e-8)
    return center, scale


def features(
    raw: np.ndarray,
    rich: np.ndarray,
    raw_center: np.ndarray,
    raw_scale: np.ndarray,
    rich_center: np.ndarray,
    rich_scale: np.ndarray,
) -> np.ndarray:
    raw_z = np.clip(
        (raw[..., 1:].astype(np.float64) - raw_center[None, :, None, :])
        / raw_scale[None, :, None, :],
        -8.0,
        8.0,
    )
    rich_z = np.clip(
        (rich.astype(np.float64) - rich_center[None, :, None, :])
        / rich_scale[None, :, None, :],
        -8.0,
        8.0,
    )
    local = np.concatenate((raw_z, rich_z), axis=-1)
    context_mean = np.mean(local, axis=2, dtype=np.float64)
    context_rms = np.sqrt(np.mean(local * local, axis=2, dtype=np.float64))
    context = np.concatenate((context_mean, context_rms), axis=-1)
    context = np.broadcast_to(context[:, :, None, :], (*local.shape[:3], context.shape[-1]))
    depth = np.linspace(-1.0, 1.0, DEPTH, dtype=np.float64)[None, :, None, None]
    depth = np.broadcast_to(depth, (*local.shape[:3], 1))
    return np.ascontiguousarray(np.concatenate((depth, local, context), axis=-1), dtype=np.float32)


def true_transport(error: np.ndarray, gate: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.zeros_like(error, dtype=np.float32)
    for rank in range(error.shape[0]):
        for layer in range(1, DEPTH):
            result[rank, layer] = gate[rank, layer] * (
                error[rank, layer - 1].astype(np.float64) @ weights[rank, layer].astype(np.float64)
            )
    return result


def fit_transport_scale(error: np.ndarray, transport: np.ndarray) -> np.ndarray:
    scale = np.zeros(DEPTH, dtype=np.float64)
    for layer in range(1, DEPTH):
        x = transport[:, layer].astype(np.float64).reshape(-1)
        y = error[:, layer].astype(np.float64).reshape(-1)
        scale[layer] = np.dot(x, y) / max(np.dot(x, x), 1e-300)
    return scale.astype(np.float32)


class RichStructured(nn.Module):
    def __init__(self, n_features: int, local_scale: np.ndarray, a: np.ndarray):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )
        self.readout = nn.Parameter(torch.empty(DEPTH, HIDDEN))
        self.bias = nn.Parameter(torch.zeros(DEPTH))
        nn.init.normal_(self.readout, std=0.01)
        self.transport_delta = nn.Parameter(torch.zeros(DEPTH))
        self.register_buffer("local_scale", torch.from_numpy(local_scale))
        self.register_buffer("base_a", torch.from_numpy(a))

    def local(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        hidden = self.trunk(x.reshape(-1, x.shape[-1])).reshape(batch, DEPTH, WIDTH, HIDDEN)
        standard = torch.einsum("blwh,lh->blw", hidden, self.readout)
        standard = standard + self.bias[None, :, None]
        return standard * self.local_scale[None, :, None]

    def a(self) -> torch.Tensor:
        return self.base_a + 0.15 * torch.tanh(self.transport_delta)

    def rollout(self, x: torch.Tensor, weights: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.local(x)
        state = [local[:, 0]]
        a = self.a()
        for layer in range(1, DEPTH):
            transported = torch.bmm(state[-1].unsqueeze(1), weights[:, layer]).squeeze(1)
            state.append(a[layer] * gate[:, layer] * transported + local[:, layer])
        return torch.stack(state, dim=1), local


def local_metric(model, x, target_local) -> tuple[float, list[float]]:
    model.eval()
    with torch.no_grad():
        delta = model.local(x) - target_local
        layer = torch.mean(delta.double().square(), dim=(0, 2)).cpu().numpy()
    return float(layer[-1]), layer.tolist()


def rollout_metric(model, x, weights, gate, error, pred, truth) -> tuple[float, list[float]]:
    model.eval()
    with torch.no_grad():
        correction, _ = model.rollout(x, weights, gate)
        corrected = torch.clamp(pred + correction, min=0.0)
        layer = torch.mean((corrected - truth).double().square(), dim=(0, 2)).cpu().numpy()
    return float(layer[-1]), layer.tolist()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(12)
    started = time.perf_counter()

    data25 = load_arrays(25, include_weights=True)
    truth25 = targets(25)
    error25 = truth25 - data25["pred"]
    gate25 = data25["rich"][..., 38]
    transport25 = true_transport(error25, gate25, data25["weights"])
    a = fit_transport_scale(error25[:20], transport25[:20])
    local25 = error25 - a[None, :, None] * transport25
    local_scale = np.maximum(
        np.sqrt(np.mean(local25[:20].astype(np.float64) ** 2, axis=(0, 2))),
        1e-7,
    ).astype(np.float32)

    raw_center, raw_scale = robust_fit(data25["raw"][:20, ..., 1:])
    rich_center, rich_scale = robust_fit(data25["rich"][:20])
    x25 = features(
        data25["raw"], data25["rich"], raw_center, raw_scale, rich_center, rich_scale
    )
    tensors = {
        "x": torch.from_numpy(x25),
        "weights": torch.from_numpy(data25["weights"]),
        "gate": torch.from_numpy(gate25),
        "error": torch.from_numpy(error25),
        "local": torch.from_numpy(local25),
        "pred": torch.from_numpy(data25["pred"]),
        "truth": torch.from_numpy(truth25),
    }
    model = RichStructured(x25.shape[-1], local_scale, a).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-5)
    scale_tensor = torch.from_numpy(local_scale)[None, :, None]

    teacher_trace = []
    best_teacher = math.inf
    best_teacher_state = None
    stale = 0
    for epoch in range(121):
        if epoch % 5 == 0:
            valid, _ = local_metric(model, tensors["x"][20:25], tensors["local"][20:25])
            teacher_trace.append({"epoch": epoch, "validation_final_local_mse": valid})
            print(f"teacher {epoch:03d} valid_local={valid:.9e}", flush=True)
            if valid < best_teacher:
                best_teacher = valid
                best_teacher_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 8:
                break
        if epoch == 120:
            break
        order = torch.randperm(20)
        for start in range(0, 20, 5):
            take = order[start : start + 5]
            optimizer.zero_grad(set_to_none=True)
            local = model.local(tensors["x"][take])
            normalized = (local - tensors["local"][take]) / scale_tensor
            loss = torch.mean(normalized.square())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    if best_teacher_state is None:
        raise RuntimeError("teacher checkpoint absent")
    model.load_state_dict(best_teacher_state)
    frozen_teacher_valid, teacher_valid_layers = local_metric(
        model, tensors["x"][20:25], tensors["local"][20:25]
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=5e-5)
    error_scale = np.maximum(
        np.sqrt(np.mean(error25[:20].astype(np.float64) ** 2, axis=(0, 2))), 1e-7
    ).astype(np.float32)
    error_scale_tensor = torch.from_numpy(error_scale)[None, :, None]
    best_valid = math.inf
    best_state = None
    best_epoch = -1
    rollout_trace = []
    stale = 0
    for epoch in range(121):
        if epoch % 4 == 0:
            valid, _ = rollout_metric(
                model,
                tensors["x"][20:25], tensors["weights"][20:25], tensors["gate"][20:25],
                tensors["error"][20:25], tensors["pred"][20:25], tensors["truth"][20:25],
            )
            rollout_trace.append({"epoch": epoch, "validation_final_mse": valid})
            print(f"rollout {epoch:03d} valid={valid:.9e} best={best_valid:.9e}@{best_epoch}", flush=True)
            if valid < best_valid:
                best_valid = valid
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 10:
                break
        if epoch == 120:
            break
        optimizer.zero_grad(set_to_none=True)
        correction, local = model.rollout(tensors["x"][:20], tensors["weights"][:20], tensors["gate"][:20])
        normalized = (correction - tensors["error"][:20]) / error_scale_tensor
        teacher_anchor = (local - tensors["local"][:20]) / scale_tensor
        loss = torch.mean(normalized.square()) + 0.15 * torch.mean(teacher_anchor.square())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    if best_state is None:
        raise RuntimeError("rollout checkpoint absent")
    model.load_state_dict(best_state)

    # The checkpoint is now frozen; open held ranks exactly once for reporting.
    data35 = load_arrays(35, include_weights=False)
    truth35 = targets(35)
    held_weights = np.stack(
        [np.load(row_file(r, "weights.npy"), mmap_mode="r") for r in HELD]
    ).astype(np.float32)
    xheld = features(
        data35["raw"][25:35], data35["rich"][25:35], raw_center, raw_scale, rich_center, rich_scale
    )
    error_held = truth35[25:35] - data35["pred"][25:35]
    gate_held = data35["rich"][25:35, ..., 38]
    transport_held = true_transport(error_held, gate_held, held_weights)
    local_held = error_held - a[None, :, None] * transport_held
    held = {
        "x": torch.from_numpy(xheld),
        "weights": torch.from_numpy(np.ascontiguousarray(held_weights)),
        "gate": torch.from_numpy(np.ascontiguousarray(gate_held)),
        "local": torch.from_numpy(np.ascontiguousarray(local_held)),
        "error": torch.from_numpy(np.ascontiguousarray(error_held)),
        "pred": torch.from_numpy(np.ascontiguousarray(data35["pred"][25:35])),
        "truth": torch.from_numpy(np.ascontiguousarray(truth35[25:35])),
    }
    held_local_final, held_local_layers = local_metric(model, held["x"], held["local"])
    held_final, held_layers = rollout_metric(
        model, held["x"], held["weights"], held["gate"], held["error"], held["pred"], held["truth"]
    )
    baseline = np.mean(
        (data35["pred"][25:35].astype(np.float64) - truth35[25:35].astype(np.float64)) ** 2,
        axis=(0, 2),
    )
    result = {
        "split": {"development": [0, 20], "validation": [20, 25], "held": [25, 35]},
        "feature_count": int(x25.shape[-1]),
        "rich_feature_count": int(data25["rich"].shape[-1]),
        "hidden": HIDDEN,
        "transport_scale_fit": a.tolist(),
        "teacher": {
            "validation_final_local_mse_before_rollout": frozen_teacher_valid,
            "validation_layer_local_mse_before_rollout": teacher_valid_layers,
            "held_final_local_mse_after_rollout_fit": held_local_final,
            "held_layer_local_mse_after_rollout_fit": held_local_layers,
            "trace": teacher_trace,
        },
        "autonomous": {
            "selected_epoch": best_epoch,
            "selected_validation_final_mse": best_valid,
            "held_final_mse": held_final,
            "held_layer_mse": held_layers,
            "held_baseline_final_mse": float(baseline[-1]),
            "held_ratio": held_final / float(baseline[-1]),
            "trace": rollout_trace,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["autonomous"], indent=2), flush=True)


if __name__ == "__main__":
    main()
