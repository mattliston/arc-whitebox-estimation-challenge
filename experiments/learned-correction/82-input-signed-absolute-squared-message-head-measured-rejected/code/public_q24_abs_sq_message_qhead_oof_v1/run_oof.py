#!/Users/matt/anaconda3/bin/python3
"""One preregistered observer-fit OOF test of an 82-input q head.

There is exactly one candidate: fold-pure local22, the already-tested signed
raw20 recurrence (20), and target-blind absolute/squared-weight raw20 messages
(20+20).  No feature ablation, hyperparameter search, full refit, selection
target, board access, or estimator execution is permitted by this program.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREG = HERE / "PREREGISTRATION.json"
RESULT = HERE / "RESULT.json"
MODELS_OUT = HERE / "OOF_MODELS.pt"
CLAIM = HERE / "RUN_CLAIM.json"

J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
QREF_SOURCE = P2 / "public_q24_qhead_full688_refit_v1/run_refit.py"
QSAMPLE_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
NATIVE22_RUNNER = P2 / "public_q24_signed_pair_qinnovation_distillation_v1/native22_capacity_runner.py"
NATIVE22_BUNDLE = P2 / "public_q24_signed_pair_qinnovation_distillation_v1/NATIVE22_CAPACITY_OUTPUT_V1"
FOLD_MODELS = NATIVE22_BUNDLE / "MODELS.pt"
BASELINE_RESULT = NATIVE22_BUNDLE / "RESULT.json"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ABS_SQ_ROOT = P2 / "public_q24_raw_abs_sq_messages_v1"
ABS_SQ_SUMMARY = ABS_SQ_ROOT / "CACHE_SUMMARY.json"
SIGNED_SUMMARY = HERE / "SIGNED_CACHE_SUMMARY.json"

ROWS = 688
WIDTH = 1024
DEPTH = 16
FOLDS = 4
FIT = 516
HELD = 172
SUFFIX = tuple(range(10, 16))
LOCAL_INPUTS = 22
SIGNED_INPUTS = 20
ABS_INPUTS = 20
SQUARE_INPUTS = 20
INPUTS = LOCAL_INPUTS + SIGNED_INPUTS + ABS_INPUTS + SQUARE_INPUTS
HIDDEN = 256
HIDDEN_LAYERS = 3
MODEL_SEED = 1_362_178_069
EPOCHS = 13
NETWORK_BATCH = 4
LEARNING_RATE = 1.2e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
FOLD_ASSIGNMENT_SHA256 = "0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c"
SAVED_SHARED256_MSE = 2.985249910997228e-8
MATERIAL_RATIO = 0.99
MATERIAL_THRESHOLD = MATERIAL_RATIO * SAVED_SHARED256_MSE
CURRENT_BOARD_RAW = 2.9409071836e-8


def sha256(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def verify_cache_summary(
    path: Path,
    expected_schema: str,
    row_path: Any,
) -> dict[str, Any]:
    value = read_json(path)
    if (
        value.get("schema") != expected_schema
        or value.get("target_accessed") is not False
        or value.get("row_count") != ROWS
        or value.get("ranks") != list(range(ROWS))
    ):
        raise AssertionError(f"incomplete or target-contaminated cache: {path}")
    receipts = value.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != ROWS:
        raise AssertionError(f"cache receipts missing: {path}")
    for expected_rank, receipt in enumerate(receipts):
        if not isinstance(receipt, dict) or receipt.get("rank") != expected_rank:
            raise AssertionError(f"cache receipt order changed: {path} rank {expected_rank}")
        live_path = row_path(expected_rank)
        expected_hash = receipt.get("output_file_sha256")
        if not live_path.is_file() or not isinstance(expected_hash, str) or sha256(live_path) != expected_hash:
            raise AssertionError(f"live cache row changed: {live_path}")
    return value


def verify_pre_target() -> dict[str, Any]:
    prereg = read_json(PREREG)
    if prereg.get("schema") != "q24-abs-sq-message-qhead-preregistration-v1":
        raise AssertionError("wrong preregistration schema")
    if prereg.get("single_candidate_only") is not True:
        raise AssertionError("candidate multiplicity changed")
    architecture = prereg.get("architecture", {})
    expected_architecture = {
        "inputs": INPUTS,
        "local22": LOCAL_INPUTS,
        "signed_recurrent_raw20": SIGNED_INPUTS,
        "absolute_weight_raw20": ABS_INPUTS,
        "squared_weight_raw20": SQUARE_INPUTS,
        "hidden": HIDDEN,
        "hidden_layers": HIDDEN_LAYERS,
        "epochs": EPOCHS,
        "folds": FOLDS,
    }
    if architecture != expected_architecture:
        raise AssertionError("frozen architecture changed")
    expected_hashes = prereg.get("bindings", {})
    live = {
        "program_sha256": sha256(Path(__file__)),
        "j_source_sha256": sha256(J_SOURCE),
        "qref_source_sha256": sha256(QREF_SOURCE),
        "qsample_source_sha256": sha256(QSAMPLE_SOURCE),
        "native22_runner_sha256": sha256(NATIVE22_RUNNER),
        "fold_models_sha256": sha256(FOLD_MODELS),
        "baseline_result_sha256": sha256(BASELINE_RESULT),
        "target_bundle_sha256": sha256(TARGETS),
        "abs_sq_cache_summary_sha256": sha256(ABS_SQ_SUMMARY),
        "signed_cache_summary_sha256": sha256(SIGNED_SUMMARY),
    }
    if expected_hashes != live:
        mismatch = {key: (expected_hashes.get(key), value) for key, value in live.items() if expected_hashes.get(key) != value}
        raise AssertionError(f"frozen binding changed: {mismatch}")
    verify_cache_summary(
        ABS_SQ_SUMMARY,
        "q24-raw-abs-sq-message-cache-summary-v1",
        abs_sq_path,
    )
    verify_cache_summary(
        SIGNED_SUMMARY,
        "q24-target-blind-signed-raw20-message-cache-v1",
        signed_path,
    )
    if RESULT.exists() or MODELS_OUT.exists() or CLAIM.exists():
        raise FileExistsError("one OOF run already completed or partially published")
    return prereg


def write_claim(prereg: dict[str, Any], native_storage: int) -> None:
    payload = {
        "schema": "q24-abs-sq-message-qhead-one-use-claim-v1",
        "status": "CLAIMED_AFTER_ALL_STATIC_CHECKS_BEFORE_FIRST_SEMANTIC_TARGET_READ__NO_RETRY",
        "owner_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "program_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREG),
        "bindings": prereg["bindings"],
        "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        "native_storage_bytes": native_storage,
        "target_bundle_hashed_but_not_semantically_decoded_before_claim": True,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(CLAIM, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(HERE, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def abs_sq_path(rank: int) -> Path:
    return ABS_SQ_ROOT / "cache" / "rows" / f"{rank:04d}" / "raw_abs_sq_messages.npy"


def signed_path(rank: int) -> Path:
    return HERE / "cache" / "signed_rows" / f"{rank:04d}" / "signed_z.npy"


def load_extra_row(rank: int) -> np.ndarray:
    signed = np.load(signed_path(rank), mmap_mode="r", allow_pickle=False)
    abs_sq = np.load(abs_sq_path(rank), mmap_mode="r", allow_pickle=False)
    if signed.shape != (len(SUFFIX), WIDTH, SIGNED_INPUTS) or signed.dtype != np.float32:
        raise ValueError(f"signed cache schema changed at {rank}: {signed.shape} {signed.dtype}")
    if abs_sq.shape != (len(SUFFIX), WIDTH, ABS_INPUTS + SQUARE_INPUTS) or abs_sq.dtype != np.float32:
        raise ValueError(f"abs/sq cache schema changed at {rank}: {abs_sq.shape} {abs_sq.dtype}")
    return np.ascontiguousarray(np.concatenate((signed, abs_sq), axis=-1), dtype=np.float32)


def extra_stats(fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros((len(SUFFIX), INPUTS - LOCAL_INPUTS), dtype=np.float64)
    total2 = np.zeros_like(total)
    for rank_value in fit:
        value = load_extra_row(int(rank_value)).astype(np.float64)
        total += value.sum(axis=1, dtype=np.float64)
        total2 += np.square(value, dtype=np.float64).sum(axis=1, dtype=np.float64)
    count = float(len(fit) * WIDTH)
    center = total / count
    variance = np.maximum(total2 / count - center * center, (2.0**-20) ** 2)
    return center.astype(np.float32), np.sqrt(variance).astype(np.float32)


def load_normalized_extra(center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    output = np.empty((ROWS, len(SUFFIX), WIDTH, INPUTS - LOCAL_INPUTS), dtype=np.float32)
    clipped = 0
    maximum = 0.0
    for rank in range(ROWS):
        value = load_extra_row(rank).astype(np.float64)
        standard = (value - center[:, None, :]) / scale[:, None, :]
        clipped += int(np.sum(np.abs(standard) > 8.0))
        maximum = max(maximum, float(np.max(np.abs(standard))))
        output[rank] = np.clip(standard, -8.0, 8.0).astype(np.float32)
    return output, {
        "maximum_absolute_before_clip": maximum,
        "clip_fraction": clipped / output.size,
        "scale_minimum": float(np.min(scale)),
        "scale_maximum": float(np.max(scale)),
    }


@dataclass
class FoldData:
    prepared: Any
    extra: np.ndarray
    extra_center: np.ndarray
    extra_scale: np.ndarray


class MessageHead(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.register_buffer("centers", base.centers.detach().clone())
        self.register_buffer("scales", base.scales.detach().clone())
        self.register_buffer("layer_rms", base.layer_rms.detach().clone())
        layers: list[nn.Module] = [nn.Linear(INPUTS, HIDDEN), nn.SiLU()]
        for _ in range(HIDDEN_LAYERS - 1):
            layers.extend((nn.Linear(HIDDEN, HIDDEN), nn.SiLU()))
        self.trunk = nn.Sequential(*layers)
        self.head_weight = nn.Parameter(torch.zeros(len(SUFFIX), HIDDEN))
        self.head_bias = nn.Parameter(torch.zeros(len(SUFFIX)))

    def source(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        base_state: torch.Tensor,
        base_q: torch.Tensor,
        extra: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        tail = torch.clamp((raw[..., 1:] - self.centers[layer]) / self.scales[layer], -8.0, 8.0)
        rms = self.layer_rms[layer]
        local = torch.cat(
            (
                raw[..., :1],
                tail,
                torch.clamp(base_state / rms, -8.0, 8.0)[..., None],
                torch.clamp(base_q / torch.clamp(sigma * rms, min=2.0**-30), -8.0, 8.0)[..., None],
            ),
            dim=-1,
        )
        features = torch.cat((local, extra), dim=-1)
        if features.shape[-1] != INPUTS:
            raise AssertionError("message-head feature width changed")
        hidden = self.trunk(features)
        slot = SUFFIX.index(layer)
        standard = torch.einsum("...h,h->...", hidden, self.head_weight[slot]) + self.head_bias[slot]
        return sigma * rms * standard


def canonical_model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        array = value.detach().cpu().numpy()
        digest.update(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def q_rms_for(prepared: Any, fit: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sqrt(np.mean(prepared.delta_q[fit].astype(np.float64) ** 2, axis=(0, 2))),
        2.0**-20,
    ).astype(np.float32)


def train_fixed(
    base: nn.Module,
    fold_data: FoldData,
    fit: np.ndarray,
    fold: int,
    qsample: Any,
) -> tuple[MessageHead, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = MessageHead(base)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != 154_374:
        raise AssertionError(f"parameter count changed: {parameter_count}")
    initial_hash = canonical_model_hash(model)
    q_rms = q_rms_for(fold_data.prepared, fit)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1.0e-8, amsgrad=False,
    )
    steps_per_epoch = len(fit) // NETWORK_BATCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * steps_per_epoch, eta_min=MIN_LEARNING_RATE
    )
    order_seed = (MODEL_SEED ^ 0xA5A5A5A5 ^ (fold * 0x9E3779B1)) & 0xFFFF_FFFF
    rng = np.random.Generator(np.random.PCG64(order_seed))
    order_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    trace: list[dict[str, Any]] = []
    prepared = fold_data.prepared
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(fit).astype(np.int32, copy=False)
        order_digest.update(np.ascontiguousarray(order.astype("<i4", copy=False)).tobytes())
        objective = 0.0
        model.train()
        for offset in range(0, len(order), NETWORK_BATCH):
            ranks = order[offset : offset + NETWORK_BATCH]
            if len(ranks) != NETWORK_BATCH:
                raise AssertionError("fit fold no longer divides into batches of four")
            losses: list[torch.Tensor] = []
            for slot, layer in enumerate(SUFFIX):
                indices = [qsample.sample_indices(int(rank), layer, epoch) for rank in ranks]
                for index in indices:
                    sample_digest.update(np.ascontiguousarray(index.astype("<i4", copy=False)).tobytes())
                raw = torch.from_numpy(np.ascontiguousarray(np.concatenate([
                    prepared.raw_suffix[int(rank), slot, index]
                    for rank, index in zip(ranks, indices, strict=True)
                ])))
                sigma = torch.from_numpy(np.ascontiguousarray(np.concatenate([
                    prepared.sigma_suffix[int(rank), slot, index]
                    for rank, index in zip(ranks, indices, strict=True)
                ])))
                state = torch.from_numpy(np.ascontiguousarray(np.concatenate([
                    prepared.base_state[int(rank), slot, index]
                    for rank, index in zip(ranks, indices, strict=True)
                ])))
                base_q = torch.from_numpy(np.ascontiguousarray(np.concatenate([
                    prepared.base_q[int(rank), slot, index]
                    for rank, index in zip(ranks, indices, strict=True)
                ])))
                extra = torch.from_numpy(np.ascontiguousarray(np.concatenate([
                    fold_data.extra[int(rank), slot, index]
                    for rank, index in zip(ranks, indices, strict=True)
                ])))
                truth = torch.from_numpy(np.ascontiguousarray(np.concatenate([
                    prepared.delta_q[int(rank), slot, index]
                    for rank, index in zip(ranks, indices, strict=True)
                ])))
                source = model.source(raw, sigma, state, base_q, extra, layer)
                losses.append(torch.mean(((source - truth) / float(q_rms[slot])).square()))
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective += float(loss.detach()) * len(ranks)
        trace.append({
            "epoch": epoch,
            "full_node_normalized_q_loss": objective / len(fit),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "maximum_process_rss_bytes": rss_bytes(),
        })
        print(json.dumps({"stage": "train", "fold": fold, **trace[-1]}, sort_keys=True), flush=True)
    model.eval()
    return model, {
        "initial_state_sha256": initial_hash,
        "row_order_sha256": order_digest.hexdigest(),
        "sample_index_sha256": sample_digest.hexdigest(),
        "samples_per_layer_per_network_per_epoch": int(qsample.SAMPLES_PER_LAYER),
        "q_rms": q_rms.tolist(),
        "trace": trace,
    }


def score_fold(model: MessageHead, fold_data: FoldData, dataset: Any, held: np.ndarray) -> dict[str, Any]:
    prepared = fold_data.prepared
    rows: list[dict[str, Any]] = []
    local_sse = 0.0
    local_count = 0
    with torch.no_grad():
        for rank_value in held:
            rank = int(rank_value)
            weights = np.load(dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False)
            state = np.zeros(WIDTH, dtype=np.float32)
            oracle = np.zeros(WIDTH, dtype=np.float32)
            for slot, layer in enumerate(SUFFIX):
                source = model.source(
                    torch.from_numpy(np.ascontiguousarray(prepared.raw_suffix[rank, slot])),
                    torch.from_numpy(np.ascontiguousarray(prepared.sigma_suffix[rank, slot])),
                    torch.from_numpy(np.ascontiguousarray(prepared.base_state[rank, slot])),
                    torch.from_numpy(np.ascontiguousarray(prepared.base_q[rank, slot])),
                    torch.from_numpy(np.ascontiguousarray(fold_data.extra[rank, slot])),
                    layer,
                ).cpu().numpy()
                truth = prepared.delta_q[rank, slot]
                difference = source.astype(np.float64) - truth.astype(np.float64)
                local_sse += float(np.dot(difference, difference))
                local_count += WIDTH
                phi = prepared.raw_suffix[rank, slot, :, 5].astype(np.float64)
                state = (phi * (state.astype(np.float64) @ weights[layer].astype(np.float64)) + source).astype(np.float32)
                oracle = (phi * (oracle.astype(np.float64) @ weights[layer].astype(np.float64)) + truth).astype(np.float32)
            residual = prepared.residual_final[rank].astype(np.float64)
            rows.append({
                "role_rank": rank,
                "candidate_mse": float(np.mean((residual - state.astype(np.float64)) ** 2)),
                "foldJ_base_mse": float(np.mean(residual * residual)),
                "exact_q_oracle_mse": float(np.mean((residual - oracle.astype(np.float64)) ** 2)),
            })
    return {
        "rows": rows,
        "local_q_mse": local_sse / local_count,
        "endpoint_mse": float(np.mean([row["candidate_mse"] for row in rows])),
        "foldJ_base_mse": float(np.mean([row["foldJ_base_mse"] for row in rows])),
        "exact_q_oracle_mse": float(np.mean([row["exact_q_oracle_mse"] for row in rows])),
    }


def paired_z(improvement: np.ndarray) -> float:
    standard_error = float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
    if standard_error == 0.0:
        return float("inf") if float(np.mean(improvement)) > 0.0 else 0.0
    return float(np.mean(improvement) / standard_error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prereg = verify_pre_target()
    j = load_module("abs_sq_oof_j", J_SOURCE)
    qref = load_module("abs_sq_oof_qref", QREF_SOURCE)
    qsample = load_module("abs_sq_oof_qsample", QSAMPLE_SOURCE)
    if int(qsample.SAMPLES_PER_LAYER) != 256 or tuple(qsample.SUFFIX) != SUFFIX:
        raise AssertionError("frozen node sampler changed")
    records, native_storage = qref.verify_native_rows()
    folds = qref.balanced_folds(records)
    if hashlib.sha256(folds.tobytes()).hexdigest() != FOLD_ASSIGNMENT_SHA256:
        raise AssertionError("balanced fold assignment changed")
    if args.preflight_only:
        print(json.dumps({
            "status": "PASS_TARGET_BLIND_PREFLIGHT",
            "program_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREG),
            "native_storage_bytes": native_storage,
            "fold_sizes": [int(np.sum(folds == fold)) for fold in range(FOLDS)],
        }, sort_keys=True))
        return

    write_claim(prereg, native_storage)
    # This is the first semantic target read.  Every architecture, gate,
    # program byte, cache byte, and fold was frozen and checked above.
    targets = qref.load_targets(records)
    hosts = np.ascontiguousarray(
        np.stack([np.load(qref.row_dir(rank) / "host.npy", allow_pickle=False) for rank in range(ROWS)]),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    dataset = qref.All688Dataset(records)
    saved_folds = torch.load(FOLD_MODELS, map_location="cpu", weights_only=False)
    baseline = read_json(BASELINE_RESULT)
    if baseline.get("aggregate", {}).get("shared256_endpoint_mse") != SAVED_SHARED256_MSE:
        raise AssertionError("saved shared256 baseline changed")
    baseline_rows = baseline.get("per_network", {}).get("shared256")
    if not isinstance(baseline_rows, list) or len(baseline_rows) != ROWS:
        raise AssertionError("saved shared256 per-network baseline missing")
    baseline_by_rank = {int(row["role_rank"]): float(row["candidate_mse"]) for row in baseline_rows}

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    fold_results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    model_payload: dict[str, Any] = {"schema": "q24-abs-sq-message-qhead-oof-models-v1", "folds": {}}
    for fold in range(FOLDS):
        fit = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        if len(fit) != FIT or len(held) != HELD:
            raise AssertionError("fold size changed")
        entry = saved_folds["folds"][str(fold)]
        fold_j = j.TransportEnergyMultiStateResidual(entry["j_centers"], entry["j_scales"], entry["j_layer_rms"])
        fold_j.load_state_dict(entry["j_state_dict"], strict=True)
        fold_j.eval()
        prepared = qref.prepare_all(j, fold_j, dataset, error)
        center, scale = extra_stats(fit)
        extra, extra_diagnostic = load_normalized_extra(center, scale)
        fold_data = FoldData(prepared, extra, center, scale)
        model, training = train_fixed(fold_j, fold_data, fit, fold, qsample)
        score = score_fold(model, fold_data, dataset, held)
        candidate_fold = float(score["endpoint_mse"])
        baseline_fold = float(np.mean([baseline_by_rank[int(rank)] for rank in held]))
        fold_result = {
            "fold": fold,
            "fit_rows": len(fit),
            "held_rows": len(held),
            "candidate_endpoint_mse": candidate_fold,
            "saved_shared256_endpoint_mse": baseline_fold,
            "candidate_local_q_mse": float(score["local_q_mse"]),
            "foldJ_base_endpoint_mse": float(score["foldJ_base_mse"]),
            "exact_q_oracle_endpoint_mse": float(score["exact_q_oracle_mse"]),
            "candidate_beats_saved_shared256": candidate_fold < baseline_fold,
            "extra_diagnostic": extra_diagnostic,
            "training": training,
        }
        fold_results.append(fold_result)
        all_rows.extend(score["rows"])
        model_payload["folds"][str(fold)] = {
            "state_dict": copy.deepcopy(model.state_dict()),
            "extra_center": center,
            "extra_scale": scale,
            "training": training,
        }
        print(json.dumps({"stage": "fold_score", **{key: value for key, value in fold_result.items() if key not in ("training", "extra_diagnostic")}}, sort_keys=True), flush=True)
        del extra, fold_data, prepared, fold_j, model

    all_rows.sort(key=lambda row: int(row["role_rank"]))
    if [int(row["role_rank"]) for row in all_rows] != list(range(ROWS)):
        raise AssertionError("OOF coverage changed")
    candidate_values = np.asarray([float(row["candidate_mse"]) for row in all_rows], dtype=np.float64)
    baseline_values = np.asarray([baseline_by_rank[rank] for rank in range(ROWS)], dtype=np.float64)
    foldj_values = np.asarray([float(row["foldJ_base_mse"]) for row in all_rows], dtype=np.float64)
    oracle_values = np.asarray([float(row["exact_q_oracle_mse"]) for row in all_rows], dtype=np.float64)
    candidate_mse = float(np.mean(candidate_values))
    gain = baseline_values - candidate_values
    gates = {
        "aggregate_below_0_99_times_saved_shared256": candidate_mse < MATERIAL_THRESHOLD,
        "all_four_folds_beat_saved_shared256": all(bool(row["candidate_beats_saved_shared256"]) for row in fold_results),
        "finite_complete_688_oof": bool(np.all(np.isfinite(np.concatenate((candidate_values, baseline_values, foldj_values, oracle_values))))),
        "single_combined_candidate_only": prereg.get("single_candidate_only") is True,
        "observer_selection_targets_opened_zero": True,
        "full_refit_performed": False,
    }
    passed = bool(
        gates["aggregate_below_0_99_times_saved_shared256"]
        and gates["all_four_folds_beat_saved_shared256"]
        and gates["finite_complete_688_oof"]
        and gates["single_combined_candidate_only"]
        and gates["observer_selection_targets_opened_zero"]
        and not gates["full_refit_performed"]
    )
    torch.save(model_payload, MODELS_OUT)
    result = {
        "schema": "q24-abs-sq-message-qhead-oof-result-v1",
        "status": "PASS_MATERIAL_OOF__NO_FULL_REFIT_PERFORMED" if passed else "FAIL_MATERIAL_OOF__STOP_NO_REFIT",
        "access": {
            "role": "observer_fit",
            "target_columns": ["mlp_id", "all_layer_means"],
            "observer_selection_targets": [],
            "names": [],
            "scores": [],
            "board_accessed": False,
            "estimator_runs": 0,
            "full_refit_performed": False,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREG),
            "run_claim_sha256": sha256(CLAIM),
            "fold_models_sha256": sha256(FOLD_MODELS),
            "baseline_result_sha256": sha256(BASELINE_RESULT),
            "abs_sq_cache_summary_sha256": sha256(ABS_SQ_SUMMARY),
            "signed_cache_summary_sha256": sha256(SIGNED_SUMMARY),
            "oof_models_sha256": sha256(MODELS_OUT),
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        },
        "architecture": prereg["architecture"],
        "aggregate": {
            "networks": ROWS,
            "candidate_endpoint_mse": candidate_mse,
            "saved_shared256_endpoint_mse": float(np.mean(baseline_values)),
            "absolute_gain_vs_saved_shared256": float(np.mean(gain)),
            "relative_gain_vs_saved_shared256": float(np.mean(gain)) / float(np.mean(baseline_values)),
            "material_threshold_mse": MATERIAL_THRESHOLD,
            "current_board_raw_reference": CURRENT_BOARD_RAW,
            "candidate_minus_current_board_raw": candidate_mse - CURRENT_BOARD_RAW,
            "foldJ_base_endpoint_mse": float(np.mean(foldj_values)),
            "exact_q_oracle_endpoint_mse": float(np.mean(oracle_values)),
            "candidate_local_q_mse": float(np.mean([row["candidate_local_q_mse"] for row in fold_results])),
            "network_win_fraction_vs_saved_shared256": float(np.mean(gain > 0.0)),
            "paired_network_z_vs_saved_shared256": paired_z(gain),
        },
        "fold_results": fold_results,
        "gates": gates,
        "decision": "STOP_NO_REFIT" if not passed else "OOF_GATE_PASS__SEPARATE_AUTHORITY_REQUIRED_FOR_ANY_REFIT",
        "per_network": all_rows,
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
