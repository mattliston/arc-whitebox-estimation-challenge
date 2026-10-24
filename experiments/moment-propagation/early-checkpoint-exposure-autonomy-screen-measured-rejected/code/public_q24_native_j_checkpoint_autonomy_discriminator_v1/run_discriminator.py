#!/usr/bin/env python3
"""Protected per-epoch autonomous screen of the exact current native-Q24 J model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PILOT_DIR = P2 / "public_q24_native_transport_energy_pilot_v1"
PILOT_SOURCE = PILOT_DIR / "run_transfer_pilot.py"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
PREREGISTER = HERE / "PREREGISTER.md"
RESULT = HERE / "RESULT.json"
CHECKPOINTS = HERE / "CHECKPOINTS.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(tag: str, path: Path):
    specification = importlib.util.spec_from_file_location(tag, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[tag] = module
    specification.loader.exec_module(module)
    return module


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def score_with_innovations(
    j,
    model: torch.nn.Module,
    dataset,
    ranks: np.ndarray,
    error: np.ndarray,
) -> dict[str, object]:
    layer_sse = np.zeros(j.DEPTH, dtype=np.float64)
    q_sse = np.zeros(j.DEPTH, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranks), 2):
            group = ranks[offset : offset + 2]
            raw, _sigma, weights, truth = j.load_batch(dataset, group, error)
            prediction = model(raw, _sigma, weights, exposure=1.0)
            if not isinstance(prediction, torch.Tensor):
                raise TypeError(type(prediction))
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("nonfinite autonomous rollout")
            layer_sse += (
                (prediction - truth).double().square().sum(dim=(0, 2)).cpu().numpy()
            )
            for layer in range(10, 16):
                phi = raw[:, layer, :, 5]
                pred_transport = phi * torch.bmm(
                    prediction[:, layer - 1, None, :], weights[:, layer]
                )[:, 0, :]
                truth_transport = phi * torch.bmm(
                    truth[:, layer - 1, None, :], weights[:, layer]
                )[:, 0, :]
                q_pred = prediction[:, layer] - pred_transport
                q_truth = truth[:, layer] - truth_transport
                q_sse[layer] += float((q_pred - q_truth).double().square().sum())
    denominator = len(ranks) * j.WIDTH
    layer_mse = layer_sse / denominator
    q_layer_mse = q_sse / denominator
    return {
        "autonomous_final_mse": float(layer_mse[-1]),
        "autonomous_layer_mse": layer_mse.tolist(),
        "q10_q15_layer_mse": q_layer_mse[10:16].tolist(),
        "q10_q15_mean_mse": float(np.mean(q_layer_mse[10:16], dtype=np.float64)),
    }


def main() -> None:
    if RESULT.exists() or CHECKPOINTS.exists():
        raise FileExistsError("discriminator output already exists")
    started = time.perf_counter()
    pilot = load_module("p2_native_j_checkpoint_pilot", PILOT_SOURCE)
    pilot.verify_seal()
    records, native_aggregate, native_storage = pilot.verify_native_rows()
    train, valid = pilot.protected_split(records)
    # First target read, after all target-free native-row receipts are verified.
    targets = pilot.load_roster_targets(records)
    hosts = np.ascontiguousarray(
        np.stack(
            [
                np.load(
                    pilot.NATIVE_ROWS / f"row_{rank:04d}/host.npy",
                    allow_pickle=False,
                )
                for rank in pilot.ROSTER
            ]
        ),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    dataset = pilot.NativePilotDataset(records)
    sums = np.zeros((200, 16, 19), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    for local, rank in enumerate(pilot.ROSTER):
        raw = np.load(
            pilot.NATIVE_ROWS / f"row_{rank:04d}/raw20.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        tail = np.asarray(raw[:, :, 1:], dtype=np.float64)
        sums[local] = tail.sum(axis=1, dtype=np.float64)
        sums2[local] = np.square(tail).sum(axis=1, dtype=np.float64)

    j = load_module("p2_native_j_checkpoint_source", J_SOURCE)
    j.configure()
    centers, scales, layer_rms = j.moments_for(train, error, sums, sums2)
    torch.manual_seed(pilot.MODEL_SEED)
    model = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    if j.parameter_count(model) != 25_600:
        raise AssertionError("exact J parameter count changed")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=j.LEARNING_RATE, weight_decay=j.WEIGHT_DECAY
    )
    total_steps = j.EPOCHS * math.ceil(len(train) / j.BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=j.MIN_LEARNING_RATE
    )
    layer_weights = torch.linspace(
        0.5, 1.5, j.DEPTH - j.FIRST_CORRECTED_LAYER
    )
    order_rng = np.random.default_rng(
        j.SEED ^ (pilot.SHUFFLE_STREAM_ID * 0x9E3779B1)
    )
    trace: list[dict[str, object]] = []
    checkpoint_states: dict[int, dict[str, torch.Tensor]] = {}
    identity = None
    for epoch, exposure in enumerate(j.EXPOSURE_SCHEDULE, start=1):
        order = order_rng.permutation(train)
        objective_sum = 0.0
        examples = 0
        model.train()
        for offset in range(0, len(order), j.BATCH_SIZE):
            ranks = order[offset : offset + j.BATCH_SIZE]
            raw, sigma, weights, truth = j.load_batch(dataset, ranks, error)
            optimizer.zero_grad(set_to_none=True)
            prediction, states, _ = model(
                raw,
                sigma,
                weights,
                truth=truth,
                exposure=exposure,
                return_states=True,
            )
            normalized = (
                prediction[:, j.FIRST_CORRECTED_LAYER :]
                - truth[:, j.FIRST_CORRECTED_LAYER :]
            ) / model.layer_rms[j.FIRST_CORRECTED_LAYER :][None, :, None]
            layer_mse = normalized.square().mean(dim=(0, 2))
            rollout = torch.mean(layer_weights * layer_mse)
            final_focus = layer_mse[-1]
            auxiliary = states[:, j.FIRST_CORRECTED_LAYER :, :, 1:]
            aux_normalized = auxiliary / model.layer_rms[
                j.FIRST_CORRECTED_LAYER :
            ][None, :, None, None]
            aux_penalty = j.AUX_STATE_PENALTY * aux_normalized.square().mean()
            identity_penalty = j.IDENTITY_PENALTY * (
                model.transport_delta.square().mean()
                + model.mix_delta.square().mean()
            )
            loss = rollout + final_focus + aux_penalty + identity_penalty
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"epoch {epoch}: nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), j.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach()) * len(ranks)
            examples += len(ranks)
        autonomous = score_with_innovations(j, model, dataset, valid, error)
        record = {
            "epoch": epoch,
            "training_exposure": float(exposure),
            "fit_objective": objective_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **autonomous,
        }
        trace.append(record)
        checkpoint_states[epoch] = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        print(json.dumps(record, sort_keys=True), flush=True)

    selected = min(
        trace, key=lambda row: (float(row["autonomous_final_mse"]), int(row["epoch"]))
    )
    selected_epoch = int(selected["epoch"])
    baseline = float(trace[-1]["autonomous_final_mse"])
    selected_mse = float(selected["autonomous_final_mse"])
    payload = {
        "schema": "p2-q24-native-j-checkpoint-autonomy-discriminator-v1",
        "state_dict_by_epoch": checkpoint_states,
        "selected_epoch": selected_epoch,
        "model_seed": pilot.MODEL_SEED,
        "shuffle_stream_id": pilot.SHUFFLE_STREAM_ID,
    }
    temporary = CHECKPOINTS.with_name(f".{CHECKPOINTS.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(CHECKPOINTS)
    result: dict[str, object] = {
        "schema": "p2-q24-native-j-checkpoint-autonomy-result-v1",
        "status": "complete",
        "role": "observer_fit",
        "observer_selection_opened": False,
        "name_or_board_join": False,
        "samples_used": 0,
        "train_networks": len(train),
        "validation_networks": len(valid),
        "trace": trace,
        "selection": {
            "rule": "earliest epoch attaining minimum autonomous validation final MSE",
            "selected_epoch": selected_epoch,
            "selected_training_exposure": selected["training_exposure"],
            "selected_autonomous_final_mse": selected_mse,
            "epoch6_autonomous_final_mse": baseline,
            "relative_mse_reduction_vs_epoch6": (baseline - selected_mse) / baseline,
            "licenses_all688_stopping_epoch_only": selected_epoch != j.EPOCHS,
        },
        "deployment": {
            "inference_exposure": 1.0,
            "truth_inputs": 0,
            "parameter_count": 25_600,
            "same_state_schema_and_flop_count_as_epoch6": True,
        },
        "resource": {
            "elapsed_seconds": time.perf_counter() - started,
            "native_row_storage_bytes": native_storage,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "preregister_sha256": sha256(PREREGISTER),
            "pilot_source_sha256": sha256(PILOT_SOURCE),
            "j_source_sha256": sha256(J_SOURCE),
            "native_row_aggregate_sha256": native_aggregate,
            "checkpoints_sha256": sha256(CHECKPOINTS),
        },
    }
    write_json_once(RESULT, result)
    print(json.dumps({"status": "complete", "selection": result["selection"]}), flush=True)


if __name__ == "__main__":
    main()
