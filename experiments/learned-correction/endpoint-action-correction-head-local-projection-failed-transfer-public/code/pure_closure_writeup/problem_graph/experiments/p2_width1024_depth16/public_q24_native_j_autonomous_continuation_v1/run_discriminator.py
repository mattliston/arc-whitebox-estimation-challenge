#!/usr/bin/env python3
"""Bounded observer-fit discriminator for autonomous native-J continuation.

The existing 160/40 Q24-native pilot checkpoint is the immutable starting
point.  Its original 160 fit rows are deterministically split 128/32.  Only
the 128 development rows receive additional exposure=1 updates; the 32 inner
rows select one continuation epoch.  The original protected 40 are scored
once, after selection, and never influence the selected checkpoint.
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


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PILOT = P2 / "public_q24_native_transport_energy_pilot_v1"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ROWS = PILOT / "runs/native_rows_v1"
START_MODEL = PILOT / "PILOT_MODEL.pt"
PILOT_RESULT = PILOT / "PILOT_RESULT.json"
RESULT = HERE / "RESULT.json"
MODEL = HERE / "CONTINUED_MODEL.pt"

ROSTER = tuple(range(35, 235))
CONTINUATION_EPOCHS = 12
BATCH_SIZE = 3
LEARNING_RATE = 1.0e-4
MIN_LEARNING_RATE = 2.0e-5
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
SEED = 0x4A434F4E


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


def load_targets() -> np.ndarray:
    table = pq.read_table(TARGETS, columns=["all_layer_means"])
    values = table.column(0).combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    all_values = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)
    return np.ascontiguousarray(all_values[35:235], dtype=np.float32)


class NativeRow:
    def __init__(self, raw_locals: np.ndarray, sigma: np.ndarray, weights_path: Path):
        self.raw_locals = raw_locals
        self.sigma = sigma
        self.weights_path = weights_path


class Dataset:
    def __init__(self) -> None:
        receipts = [
            json.loads((ROWS / f"row_{rank:04d}/result.json").read_text())
            for rank in ROSTER
        ]
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


def split(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pilot = json.loads(PILOT_RESULT.read_text())
    original_train = np.asarray(
        [int(rank) - ROSTER[0] for rank in pilot["train_role_ranks"]], dtype=np.int64
    )
    protected = np.asarray(
        [int(rank) - ROSTER[0] for rank in pilot["validation_role_ranks"]], dtype=np.int64
    )
    keyed: list[tuple[bytes, int]] = []
    for local in original_train:
        mlp_id = int(dataset.mlp_ids[int(local)])
        key = hashlib.blake2s(
            f"native-j-autonomous-continuation-v1:{mlp_id}".encode("ascii"),
            digest_size=16,
        ).digest()
        keyed.append((key, int(local)))
    inner = np.asarray(sorted(local for _, local in sorted(keyed)[:32]), dtype=np.int64)
    dev = np.asarray(sorted(set(original_train.tolist()) - set(inner.tolist())), dtype=np.int64)
    if len(dev) != 128 or len(inner) != 32 or len(protected) != 40:
        raise AssertionError((len(dev), len(inner), len(protected)))
    if set(dev.tolist()) & set(inner.tolist()) or set(original_train.tolist()) & set(protected.tolist()):
        raise AssertionError("split overlap")
    return dev, inner, protected


def load_model(j):
    artifact = torch.load(START_MODEL, map_location="cpu", weights_only=False)
    state = artifact["state_dict"]
    model = j.TransportEnergyMultiStateResidual(
        state["centers"].cpu().numpy(),
        state["scales"].cpu().numpy(),
        state["layer_rms"].cpu().numpy(),
    )
    model.load_state_dict(state, strict=True)
    return model


def train_objective(j, model, dataset, ranks, error) -> float:
    model.eval()
    total = 0.0
    count = 0
    layer_weights = torch.linspace(0.5, 1.5, 14)
    with torch.no_grad():
        for offset in range(0, len(ranks), BATCH_SIZE):
            take = ranks[offset : offset + BATCH_SIZE]
            raw, sigma, weights, truth = j.load_batch(dataset, take, error)
            prediction, states, _ = model(
                raw, sigma, weights, truth=truth, exposure=1.0, return_states=True
            )
            normalized = (
                prediction[:, 2:] - truth[:, 2:]
            ) / model.layer_rms[2:][None, :, None]
            layer_mse = normalized.square().mean(dim=(0, 2))
            auxiliary = states[:, 2:, :, 1:] / model.layer_rms[2:][None, :, None, None]
            loss = (
                torch.mean(layer_weights * layer_mse)
                + layer_mse[-1]
                + j.AUX_STATE_PENALTY * auxiliary.square().mean()
                + j.IDENTITY_PENALTY
                * (model.transport_delta.square().mean() + model.mix_delta.square().mean())
            )
            total += float(loss) * len(take)
            count += len(take)
    return total / count


def main() -> None:
    if RESULT.exists() or MODEL.exists():
        raise FileExistsError("discriminator output already exists")
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xFFFF_FFFF)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    j = load_module("native_j_continuation_exact_j", J_SOURCE)
    dataset = Dataset()
    truth = load_targets()
    host = np.ascontiguousarray(
        np.stack([np.load(ROWS / f"row_{rank:04d}/host.npy") for rank in ROSTER]),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(truth - host, dtype=np.float32)
    dev, inner, protected = split(dataset)
    model = load_model(j)

    starting_inner = j.score(model, dataset, inner, error)
    starting_dev_objective = train_objective(j, model, dataset, dev, error)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = CONTINUATION_EPOCHS * math.ceil(len(dev) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=MIN_LEARNING_RATE
    )
    order_rng = np.random.default_rng(SEED ^ 0x9E3779B1)
    layer_weights = torch.linspace(0.5, 1.5, 14)
    trace: list[dict[str, object]] = []
    best_mse = float(starting_inner["final_mse"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()

    for epoch in range(1, CONTINUATION_EPOCHS + 1):
        model.train()
        order = order_rng.permutation(dev)
        total_loss = 0.0
        examples = 0
        for offset in range(0, len(order), BATCH_SIZE):
            take = order[offset : offset + BATCH_SIZE]
            raw, sigma, weights, batch_truth = j.load_batch(dataset, take, error)
            optimizer.zero_grad(set_to_none=True)
            prediction, states, _ = model(
                raw, sigma, weights, truth=batch_truth, exposure=1.0, return_states=True
            )
            normalized = (
                prediction[:, 2:] - batch_truth[:, 2:]
            ) / model.layer_rms[2:][None, :, None]
            layer_mse = normalized.square().mean(dim=(0, 2))
            auxiliary = states[:, 2:, :, 1:] / model.layer_rms[2:][None, :, None, None]
            loss = (
                torch.mean(layer_weights * layer_mse)
                + layer_mse[-1]
                + j.AUX_STATE_PENALTY * auxiliary.square().mean()
                + j.IDENTITY_PENALTY
                * (model.transport_delta.square().mean() + model.mix_delta.square().mean())
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * len(take)
            examples += len(take)

        inner_score = j.score(model, dataset, inner, error)
        dev_objective = train_objective(j, model, dataset, dev, error)
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "batch_training_objective": total_loss / examples,
            "same_seed_exposure1_dev_objective": dev_objective,
            "inner_autonomous_final_mse": float(inner_score["final_mse"]),
            "inner_delta_from_start": float(inner_score["final_mse"] - starting_inner["final_mse"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if float(inner_score["final_mse"]) < best_mse:
            best_mse = float(inner_score["final_mse"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    # First and only protected score, after the epoch is irreversibly selected.
    protected_score = j.score(model, dataset, protected, error)
    artifact = {
        "schema": "q24-native-j-autonomous-continuation-checkpoint-v1",
        "state_dict": model.state_dict(),
        "selected_epoch": best_epoch,
        "start_model_sha256": sha256(START_MODEL),
        "selection_role": "observer_fit inner32 only",
    }
    temporary = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary)
    temporary.replace(MODEL)

    def global_ranks(values: np.ndarray) -> list[int]:
        return [ROSTER[int(value)] for value in values]

    result = {
        "schema": "q24-native-j-autonomous-continuation-discriminator-v1",
        "status": "complete",
        "observer_fit_only": True,
        "observer_selection_targets_opened": 0,
        "board_submissions": 0,
        "split": {
            "development_128_role_ranks": global_ranks(dev),
            "inner_selection_32_role_ranks": global_ranks(inner),
            "protected_once_40_role_ranks": global_ranks(protected),
            "hash_law": "first32 by blake2s-128('native-j-autonomous-continuation-v1:'+mlp_id) within original pilot train160",
        },
        "starting_checkpoint": {
            "sha256": sha256(START_MODEL),
            "inner_autonomous_final_mse": float(starting_inner["final_mse"]),
            "same_seed_exposure1_dev_objective": starting_dev_objective,
            "published_protected_autonomous_final_mse": 3.107235383605394e-8,
        },
        "training": {
            "continuation_exposure": 1.0,
            "epochs_considered": CONTINUATION_EPOCHS,
            "optimizer": "fresh AdamW",
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size_networks": BATCH_SIZE,
            "seed": SEED,
            "trace": trace,
        },
        "selection": {
            "metric": "inner32 autonomous final-layer MSE",
            "selected_epoch": best_epoch,
            "selected_inner_autonomous_final_mse": best_mse,
            "no_tuning_after_protected_open": True,
        },
        "protected_once": protected_score,
        "deployability": {
            "state_schema_unchanged": True,
            "inference_exposure": 1.0,
            "incremental_flops": 0,
            "same_literal_native_j_inference": True,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "j_source_sha256": sha256(J_SOURCE),
            "pilot_result_sha256": sha256(PILOT_RESULT),
            "selected_model_sha256": sha256(MODEL),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_epoch": best_epoch, "protected": protected_score}, sort_keys=True))


if __name__ == "__main__":
    main()
