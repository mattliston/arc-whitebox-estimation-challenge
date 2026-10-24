#!/usr/bin/env python3
"""Endpoint-action fine-tune of the frozen teacher-local q10:q15 head."""

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
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
Q_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
Q_MODEL = P2 / "public_q24_native_j_q10q15_residual_head_v1/Q_SUFFIX_HEAD.pt"
BASE_MODEL = P2 / "public_q24_native_j_checkpoint_interpolation_v1/INTERPOLATED_MODEL.pt"
BANK_RESULT = P2 / "public_q24_native_j_checkpoint_interpolation_v1/RESULT.json"
OUT = HERE / "RESULT.json"
MODEL = HERE / "ACTION_FINETUNED_Q_HEAD.pt"

SUFFIX = tuple(range(10, 16))
EPOCHS = 12
BATCH = 2
LEARNING_RATE = 3.0e-4
MIN_LEARNING_RATE = 3.0e-5
WEIGHT_DECAY = 2.0e-5
LOCAL_ANCHOR = 0.10
GATE_RATIO = 0.99
SEED = 0x41435451


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


def load_base(j):
    state = torch.load(BASE_MODEL, map_location="cpu", weights_only=False)["state_dict"]
    model = j.TransportEnergyMultiStateResidual(
        state["centers"].numpy(), state["scales"].numpy(), state["layer_rms"].numpy()
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_head(q, base):
    state = torch.load(Q_MODEL, map_location="cpu", weights_only=False)["state_dict"]
    model = q.QSuffixHead(base.centers, base.scales, base.layer_rms)
    model.load_state_dict(state, strict=True)
    return model


def batch_arrays(q, dataset, prepared, positions: np.ndarray):
    rows = [dataset.load(int(prepared["locals"][int(pos)])) for pos in positions]
    raw = torch.from_numpy(np.ascontiguousarray(np.stack([
        np.asarray(row.raw_locals)[list(SUFFIX)] for row in rows
    ]), dtype=np.float32))
    sigma = torch.from_numpy(np.ascontiguousarray(np.stack([
        np.asarray(row.sigma)[list(SUFFIX)] for row in rows
    ]), dtype=np.float32))
    weights = torch.from_numpy(np.ascontiguousarray(np.stack([
        np.load(row.weights_path, mmap_mode="r", allow_pickle=False)[list(SUFFIX)]
        for row in rows
    ]), dtype=np.float32))
    base_pred = torch.from_numpy(np.ascontiguousarray(
        prepared["base_pred"][positions][:, list(SUFFIX)], dtype=np.float32
    ))
    base_q = torch.from_numpy(np.ascontiguousarray(
        prepared["base_q"][positions], dtype=np.float32
    ))
    delta_q = torch.from_numpy(np.ascontiguousarray(
        prepared["delta_q"][positions], dtype=np.float32
    ))
    final = torch.from_numpy(np.ascontiguousarray(
        prepared["residual_final"][positions], dtype=np.float32
    ))
    return raw, sigma, weights, base_pred, base_q, delta_q, final


def training_loss(q, model, arrays, q_rms, endpoint_rms):
    raw, sigma, weights, base_pred, base_q, delta_q, final = arrays
    state = torch.zeros(raw.shape[0], 1024, dtype=raw.dtype)
    local_losses = []
    for slot, layer in enumerate(SUFFIX):
        source = model.source(
            raw[:, slot], sigma[:, slot], base_pred[:, slot], base_q[:, slot], layer
        )
        message = torch.bmm(state[:, None, :], weights[:, slot]).squeeze(1)
        state = raw[:, slot, :, 5] * message + source
        local_losses.append(torch.mean(((source - delta_q[:, slot]) / q_rms[slot]).square()))
    endpoint = torch.mean(((state - final) / endpoint_rms).square())
    local = torch.stack(local_losses).mean()
    return endpoint + LOCAL_ANCHOR * local, endpoint, local


def main() -> None:
    if OUT.exists() or MODEL.exists():
        raise FileExistsError("action fine-tune output already exists")
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    j = load_module("qhead_action_exact_j", J_SOURCE)
    q = load_module("qhead_action_q", Q_SOURCE)
    dataset = q.Dataset()
    target, host = q.targets_and_host()
    error = np.ascontiguousarray(target - host, dtype=np.float32)
    dev, inner, protected = q.split()
    base = load_base(j)
    model = load_head(q, base)
    prepared_dev = q.prepare(j, base, dataset, dev, error)
    prepared_inner = q.prepare(j, base, dataset, inner, error)
    q_rms = np.maximum(
        np.sqrt(np.mean(prepared_dev["delta_q"].astype(np.float64) ** 2, axis=(0, 2))),
        2.0**-20,
    ).astype(np.float32)
    endpoint_rms = float(max(
        np.sqrt(np.mean(prepared_dev["residual_final"].astype(np.float64) ** 2)),
        2.0**-20,
    ))
    initial_inner = q.autonomous_metrics(model, dataset, prepared_inner)
    threshold = GATE_RATIO * initial_inner["candidate_final_mse"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * math.ceil(len(dev) / BATCH)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=MIN_LEARNING_RATE
    )
    rng = np.random.default_rng(SEED ^ 0x9E3779B1)
    best_mse = initial_inner["candidate_final_mse"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    trace = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(dev))
        objective = endpoint_sum = local_sum = 0.0
        examples = 0
        for offset in range(0, len(order), BATCH):
            positions = order[offset : offset + BATCH]
            arrays = batch_arrays(q, dataset, prepared_dev, positions)
            optimizer.zero_grad(set_to_none=True)
            loss, endpoint, local = training_loss(
                q, model, arrays, torch.from_numpy(q_rms), endpoint_rms
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            count = len(positions)
            objective += float(loss.detach()) * count
            endpoint_sum += float(endpoint.detach()) * count
            local_sum += float(local.detach()) * count
            examples += count
        inner_metric = q.autonomous_metrics(model, dataset, prepared_inner)
        row = {
            "epoch": epoch,
            "dev_objective": objective / examples,
            "dev_endpoint_normalized": endpoint_sum / examples,
            "dev_local_anchor_normalized": local_sum / examples,
            "inner_autonomous_final_mse": inner_metric["candidate_final_mse"],
            "inner_teacher_local_q_mse": inner_metric["teacher_local_q_mse"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if inner_metric["candidate_final_mse"] < best_mse:
            best_mse = inner_metric["candidate_final_mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    gate = bool(best_mse <= threshold)
    protected_record = None
    if gate:
        prepared_protected = q.prepare(j, base, dataset, protected, error)
        protected_record = {
            "opened": True,
            **q.autonomous_metrics(model, dataset, prepared_protected),
        }
    else:
        protected_record = {
            "opened": False,
            "reason": "selected inner checkpoint did not reduce the frozen q-head inner MSE by at least 1%",
        }
    artifact = {
        "schema": "q24-native-j-qhead-action-finetune-checkpoint-v1",
        "state_dict": model.state_dict(),
        "selected_epoch": best_epoch,
        "base_model_sha256": sha256(BASE_MODEL),
        "initial_q_head_sha256": sha256(Q_MODEL),
    }
    temporary = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary)
    temporary.replace(MODEL)
    result = {
        "schema": "q24-native-j-qhead-action-finetune-discriminator-v1",
        "status": "complete",
        "observer_fit_only": True,
        "observer_selection_targets_opened": 0,
        "board_submissions": 0,
        "hypothesis": "teacher-local q fitting preserves local tensor energy rather than endpoint action; autonomous endpoint fine-tuning can rotate the same 6022 parameters toward downstream action",
        "split": json.loads(P2.joinpath("public_q24_native_j_autonomous_continuation_v1/RESULT.json").read_text())["split"],
        "training": {
            "epochs": EPOCHS,
            "endpoint_loss_weight": 1.0,
            "teacher_local_anchor_weight": LOCAL_ANCHOR,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "trace": trace,
        },
        "gate": {
            "initial_inner": initial_inner,
            "required_ratio": GATE_RATIO,
            "threshold_mse": threshold,
            "selected_inner_mse": best_mse,
            "selected_inner_ratio": best_mse / initial_inner["candidate_final_mse"],
            "pass": gate,
        },
        "selection": {
            "selected_epoch": best_epoch,
            "metric": "inner32 autonomous final-layer MSE",
            "no_tuning_after_protected_open": True,
        },
        "protected_once": protected_record,
        "deployability": {
            "incremental_inference_flops_over_frozen_q_head": 0,
            "parameters": sum(value.numel() for value in model.parameters()),
            "same_literal_recurrence_and_features": True,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "base_model_sha256": sha256(BASE_MODEL),
            "initial_q_head_sha256": sha256(Q_MODEL),
            "bank_result_sha256": sha256(BANK_RESULT),
            "selected_model_sha256": sha256(MODEL),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": result["gate"], "protected": protected_record}, sort_keys=True))


if __name__ == "__main__":
    main()
