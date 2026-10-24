#!/usr/bin/env python3
"""Small teacher-local q10:q15 residual head on frozen continued native-J.

The head predicts only the six missing physical local innovations.  Its one
scalar residual state is propagated by the actual ``Phi * W`` suffix.  Model
selection uses the continuation experiment's inner32; its protected40 is
prepared and scored only after the checkpoint has been selected.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PILOT = P2 / "public_q24_native_transport_energy_pilot_v1"
CONT = P2 / "public_q24_native_j_autonomous_continuation_v1"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
FEATURE_STORE = (
    P2 / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ROWS = PILOT / "runs/native_rows_v1"
BASE_MODEL = CONT / "CONTINUED_MODEL.pt"
SPLIT_RESULT = CONT / "RESULT.json"
RESULT = HERE / "RESULT.json"
MODEL = HERE / "Q_SUFFIX_HEAD.pt"

ROSTER = tuple(range(35, 235))
SUFFIX = tuple(range(10, 16))
HIDDEN = 64
INPUTS = 22
EPOCHS = 14
SAMPLES_PER_LAYER = 256
NETWORKS_PER_STEP = 4
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 2.0e-5
SEED = 0x51313015


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


class NativeRow:
    def __init__(self, raw_locals: np.ndarray, sigma: np.ndarray, weights_path: Path):
        self.raw_locals = raw_locals
        self.sigma = sigma
        self.weights_path = weights_path


class Dataset:
    def __init__(self) -> None:
        receipts = [json.loads((ROWS / f"row_{rank:04d}/result.json").read_text()) for rank in ROSTER]
        self.mlp_ids = np.asarray([int(row["mlp_id"]) for row in receipts], dtype=np.int64)

    def load(self, local: int) -> NativeRow:
        rank = ROSTER[int(local)]
        row = ROWS / f"row_{rank:04d}"
        features = FEATURE_STORE / f"rows/observer_fit/{rank:04d}"
        return NativeRow(
            np.load(row / "raw20.npy", mmap_mode="r", allow_pickle=False),
            np.load(row / "sigma.npy", mmap_mode="r", allow_pickle=False),
            features / "weights.npy",
        )


def targets_and_host() -> tuple[np.ndarray, np.ndarray]:
    values = pq.read_table(TARGETS, columns=["all_layer_means"]).column(0).combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    target = np.ascontiguousarray(
        storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)[35:235],
        dtype=np.float32,
    )
    host = np.ascontiguousarray(
        np.stack([np.load(ROWS / f"row_{rank:04d}/host.npy") for rank in ROSTER]),
        dtype=np.float32,
    )
    return target, host


def split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = json.loads(SPLIT_RESULT.read_text())
    section = record["split"]
    groups = []
    for name in (
        "development_128_role_ranks",
        "inner_selection_32_role_ranks",
        "protected_once_40_role_ranks",
    ):
        groups.append(np.asarray([int(rank) - ROSTER[0] for rank in section[name]], dtype=np.int64))
    dev, inner, protected = groups
    if (len(dev), len(inner), len(protected)) != (128, 32, 40):
        raise AssertionError("continued split changed")
    return dev, inner, protected


def load_base(j):
    artifact = torch.load(BASE_MODEL, map_location="cpu", weights_only=False)
    state = artifact["state_dict"]
    model = j.TransportEnergyMultiStateResidual(
        state["centers"].cpu().numpy(), state["scales"].cpu().numpy(),
        state["layer_rms"].cpu().numpy(),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class QSuffixHead(nn.Module):
    """Shared target-blind local net with six layer-specific zero heads."""

    def __init__(self, centers: torch.Tensor, scales: torch.Tensor, layer_rms: torch.Tensor):
        super().__init__()
        self.register_buffer("centers", centers.detach().clone())
        self.register_buffer("scales", scales.detach().clone())
        self.register_buffer("layer_rms", layer_rms.detach().clone())
        self.trunk = nn.Sequential(
            nn.Linear(INPUTS, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
        )
        self.head_weight = nn.Parameter(torch.zeros(len(SUFFIX), HIDDEN))
        self.head_bias = nn.Parameter(torch.zeros(len(SUFFIX)))

    def source(
        self, raw: torch.Tensor, sigma: torch.Tensor,
        base_state: torch.Tensor, base_q: torch.Tensor, layer: int,
    ) -> torch.Tensor:
        tail = torch.clamp(
            (raw[..., 1:] - self.centers[layer]) / self.scales[layer], -8.0, 8.0
        )
        rms = self.layer_rms[layer]
        features = torch.cat(
            (
                raw[..., :1], tail,
                torch.clamp(base_state / rms, -8.0, 8.0)[..., None],
                torch.clamp(
                    base_q / torch.clamp(sigma * rms, min=2.0**-30), -8.0, 8.0
                )[..., None],
            ),
            dim=-1,
        )
        hidden = self.trunk(features)
        slot = SUFFIX.index(layer)
        standard = torch.einsum("...h,h->...", hidden, self.head_weight[slot])
        standard = standard + self.head_bias[slot]
        return sigma * rms * standard


def prepare(
    j, base, dataset: Dataset, locals_: np.ndarray, error: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(locals_)
    base_pred = np.empty((n, 16, 1024), dtype=np.float32)
    base_q = np.empty((n, len(SUFFIX), 1024), dtype=np.float32)
    delta_q = np.empty_like(base_q)
    residual_final = np.empty((n, 1024), dtype=np.float32)
    base.eval()
    with torch.no_grad():
        for position, local in enumerate(locals_):
            raw, sigma, weights_t, _ = j.load_batch(dataset, np.asarray([local]), error)
            prediction = base(raw, sigma, weights_t, exposure=1.0)
            assert isinstance(prediction, torch.Tensor)
            a = prediction[0].cpu().numpy().astype(np.float32, copy=False)
            weights = weights_t[0].cpu().numpy()
            row_raw = raw[0].cpu().numpy()
            residual = error[int(local)] - a
            base_pred[position] = a
            residual_final[position] = residual[-1]
            for slot, layer in enumerate(SUFFIX):
                phi = row_raw[layer, :, 5].astype(np.float64)
                a_message = phi * (a[layer - 1].astype(np.float64) @ weights[layer].astype(np.float64))
                r_message = phi * (
                    residual[layer - 1].astype(np.float64) @ weights[layer].astype(np.float64)
                )
                base_q[position, slot] = (a[layer].astype(np.float64) - a_message).astype(np.float32)
                delta_q[position, slot] = (residual[layer].astype(np.float64) - r_message).astype(np.float32)
            if (position + 1) % 16 == 0:
                print(f"prepare {position + 1}/{n}", flush=True)
    return {
        "locals": locals_.copy(), "base_pred": base_pred, "base_q": base_q,
        "delta_q": delta_q, "residual_final": residual_final,
    }


def sample_indices(local: int, layer: int, epoch: int) -> np.ndarray:
    value = SEED ^ (int(local) * 0x9E3779B1) ^ (layer * 0x85EBCA77) ^ (epoch * 0xC2B2AE3D)
    return np.random.default_rng(value & 0xFFFF_FFFF).choice(
        1024, size=SAMPLES_PER_LAYER, replace=False
    )


def autonomous_metrics(
    model: QSuffixHead, dataset: Dataset, prepared: dict[str, np.ndarray]
) -> dict[str, float]:
    sse = 0.0
    oracle_sse = 0.0
    local_sse = 0.0
    local_count = 0
    model.eval()
    with torch.no_grad():
        for position, local in enumerate(prepared["locals"]):
            row = dataset.load(int(local))
            raw_np = np.asarray(row.raw_locals)
            sigma_np = np.asarray(row.sigma)
            weights = np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
            state = np.zeros(1024, dtype=np.float32)
            oracle = np.zeros(1024, dtype=np.float32)
            for slot, layer in enumerate(SUFFIX):
                raw = torch.from_numpy(np.asarray(raw_np[layer]).copy())
                sigma = torch.from_numpy(np.asarray(sigma_np[layer]).copy())
                a = torch.from_numpy(prepared["base_pred"][position, layer])
                qj = torch.from_numpy(prepared["base_q"][position, slot])
                source = model.source(raw, sigma, a, qj, layer).cpu().numpy()
                phi = raw_np[layer, :, 5].astype(np.float64)
                state = (
                    phi * (state.astype(np.float64) @ weights[layer].astype(np.float64))
                    + source.astype(np.float64)
                ).astype(np.float32)
                oracle = (
                    phi * (oracle.astype(np.float64) @ weights[layer].astype(np.float64))
                    + prepared["delta_q"][position, slot].astype(np.float64)
                ).astype(np.float32)
                delta = source.astype(np.float64) - prepared["delta_q"][position, slot]
                local_sse += float(np.dot(delta, delta))
                local_count += 1024
            residual = prepared["residual_final"][position].astype(np.float64)
            candidate_error = residual - state.astype(np.float64)
            oracle_error = residual - oracle.astype(np.float64)
            sse += float(np.dot(candidate_error, candidate_error))
            oracle_sse += float(np.dot(oracle_error, oracle_error))
    count = len(prepared["locals"]) * 1024
    baseline = float(np.mean(prepared["residual_final"].astype(np.float64) ** 2))
    return {
        "baseline_final_mse": baseline,
        "candidate_final_mse": sse / count,
        "exact_q10_q15_oracle_final_mse": oracle_sse / count,
        "teacher_local_q_mse": local_sse / local_count,
    }


def main() -> None:
    if RESULT.exists() or MODEL.exists():
        raise FileExistsError("q-head output already exists")
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    j = load_module("native_j_q_suffix_exact_j", J_SOURCE)
    dataset = Dataset()
    target, host = targets_and_host()
    error = np.ascontiguousarray(target - host, dtype=np.float32)
    dev, inner, protected = split()
    base = load_base(j)

    # Protected rows are intentionally absent from both preparations here.
    prepared_dev = prepare(j, base, dataset, dev, error)
    prepared_inner = prepare(j, base, dataset, inner, error)
    q_rms = np.maximum(
        np.sqrt(np.mean(prepared_dev["delta_q"].astype(np.float64) ** 2, axis=(0, 2))),
        2.0**-20,
    ).astype(np.float32)
    model = QSuffixHead(base.centers, base.scales, base.layer_rms)
    initial_inner = autonomous_metrics(model, dataset, prepared_inner)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * math.ceil(len(dev) / NETWORKS_PER_STEP), eta_min=1.0e-4
    )
    order_rng = np.random.default_rng(SEED ^ 0xA5A5A5A5)
    trace: list[dict[str, float | int]] = []
    best_mse = initial_inner["candidate_final_mse"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    position_of = {int(local): pos for pos, local in enumerate(dev)}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = order_rng.permutation(dev)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        running = 0.0
        for offset, local in enumerate(order):
            position = position_of[int(local)]
            row = dataset.load(int(local))
            losses = []
            for slot, layer in enumerate(SUFFIX):
                index = sample_indices(int(local), layer, epoch)
                raw = torch.from_numpy(np.asarray(row.raw_locals[layer, index]).copy())
                sigma = torch.from_numpy(np.asarray(row.sigma[layer, index]).copy())
                a = torch.from_numpy(prepared_dev["base_pred"][position, layer, index])
                qj = torch.from_numpy(prepared_dev["base_q"][position, slot, index])
                truth_q = torch.from_numpy(prepared_dev["delta_q"][position, slot, index])
                source = model.source(raw, sigma, a, qj, layer)
                losses.append(torch.mean(((source - truth_q) / q_rms[slot]).square()))
            loss = torch.stack(losses).mean() / NETWORKS_PER_STEP
            loss.backward()
            running += float(loss.detach()) * NETWORKS_PER_STEP
            pending += 1
            if pending == NETWORKS_PER_STEP or offset + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        inner_metric = autonomous_metrics(model, dataset, prepared_inner)
        record = {
            "epoch": epoch,
            "sampled_teacher_objective": running / len(dev),
            "inner_autonomous_final_mse": inner_metric["candidate_final_mse"],
            "inner_teacher_local_q_mse": inner_metric["teacher_local_q_mse"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if inner_metric["candidate_final_mse"] < best_mse:
            best_mse = inner_metric["candidate_final_mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    # Selection is frozen.  Prepare and score protected40 exactly once now.
    prepared_protected = prepare(j, base, dataset, protected, error)
    protected_metric = autonomous_metrics(model, dataset, prepared_protected)
    artifact = {
        "schema": "q24-native-j-q10q15-residual-head-checkpoint-v1",
        "state_dict": model.state_dict(), "selected_epoch": best_epoch,
        "base_model_sha256": sha256(BASE_MODEL),
    }
    temporary = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary)
    temporary.replace(MODEL)

    dense = lambda i, o: (2 * i - 1) * o
    local_dense = len(SUFFIX) * 1024 * (dense(INPUTS, HIDDEN) + dense(HIDDEN, HIDDEN) + dense(HIDDEN, 1))
    transport = len(SUFFIX) * ((2 * 1024 - 1) * 1024 + 1024)
    feature_and_nonlinear_reserve = len(SUFFIX) * 1024 * (2 * HIDDEN * 64 + 64)
    added_upper = int(local_dense + transport + feature_and_nonlinear_reserve)
    result = {
        "schema": "q24-native-j-q10q15-residual-head-discriminator-v1",
        "status": "complete",
        "observer_fit_only": True,
        "observer_selection_targets_opened": 0,
        "board_submissions": 0,
        "split": json.loads(SPLIT_RESULT.read_text())["split"],
        "architecture": {
            "base": "continued native-J epoch9",
            "suffix_layers_zero_based": list(SUFFIX),
            "input_features": INPUTS,
            "hidden": HIDDEN,
            "head_initialization": "exact zero",
            "recurrence": "d_l=Phi_l*(d_(l-1)@W_l)+qhead_l(raw20,A_l,qJ_l)",
            "parameters": sum(value.numel() for value in model.parameters()),
        },
        "training": {
            "target": "physical delta-q_l=(truth-host-J)_l-Phi_l*((truth-host-J)_(l-1)@W_l)",
            "epochs": EPOCHS,
            "samples_per_layer_per_network": SAMPLES_PER_LAYER,
            "q_rms_fit_on_dev128": q_rms.tolist(),
            "trace": trace,
        },
        "selection": {
            "metric": "inner32 autonomous final-layer MSE",
            "selected_epoch": best_epoch,
            "initial_inner": initial_inner,
            "selected_inner_final_mse": best_mse,
            "no_tuning_after_protected_open": True,
        },
        "protected_once": protected_metric,
        "deployability": {
            "incremental_dense_flops": int(local_dense),
            "incremental_physical_transport_flops": int(transport),
            "feature_and_nonlinear_conservative_reserve": int(feature_and_nonlinear_reserve),
            "incremental_flops_upper": added_upper,
            "reuses_native_j_raw_standardization_and_primary_message": True,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "base_model_sha256": sha256(BASE_MODEL),
            "selected_model_sha256": sha256(MODEL),
            "j_source_sha256": sha256(J_SOURCE),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_epoch": best_epoch, "protected": protected_metric}, sort_keys=True))


if __name__ == "__main__":
    main()
