#!/usr/bin/env python3
"""Four-fold OOF discriminator for a nonlinear transported-raw20 q head.

This is deliberately a single augmented-feature arm.  It reuses the frozen
22-input full688 refit's OOF scores as its comparator and never retrains that
base arm.  The only scientific change is twenty target-blind state channels

    Z_l = Phi_l * (W_l.T @ Z_(l-1)) + standardized_raw20_l.

The suffix head sees the original 22 inputs followed by standardized Z_l.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE_DIR = P2 / "public_q24_qhead_full688_refit_v1"
BASE_PROGRAM = BASE_DIR / "run_refit.py"
BASE_PREREGISTRATION = BASE_DIR / "PREREGISTRATION.json"
BASE_RESULT = BASE_DIR / "RESULT.json"
BASE_MODEL = BASE_DIR / "FULL688_Q_SUFFIX_HEAD.pt"
RIDGE_RESULT = P2 / "public_q24_native_j_transported_raw20_bank_v1/RESULT.json"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
Q_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUN_STATE = HERE / "RUN_STATE.json"
RESULT = HERE / "RESULT.json"
MODEL = HERE / "NONLINEAR_Z_HEAD.pt"

BASE_PROGRAM_SHA256 = "7ffaae1a3d03a7d1f3fcfe3e9026d1674892c52291f7899978ff379f8da78cd2"
BASE_PREREGISTRATION_SHA256 = "8014f6eb7368f5d161929549f7e4e0893f30f24418cfd57644a1ce7fea9e220c"
BASE_RESULT_SHA256 = "87d22be8d48d7c73821c02bb3562b3e8ad67955022df0e2185c2f94838e87a25"
BASE_MODEL_SHA256 = "d57407e0534e0be8b29a8b3f8bb80f617e83671584ddb32af757a3631e0fc58f"
RIDGE_RESULT_SHA256 = "3672e604cc4907bd6ab46f44ce55290a90f27def1dece6558c83490bd08f7e2f"
J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
Q_SOURCE_SHA256 = "1e4c906936f4685bd800f922ded6c582af8213ecad1fbe41da651faff82be9c3"
FOLD_ASSIGNMENT_SHA256 = "0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c"

ALL_RANKS = np.arange(688, dtype=np.int64)
SUFFIX = tuple(range(10, 16))
FOLDS = 4
RAW_INPUTS = 22
Z_INPUTS = 20
INPUTS = RAW_INPUTS + Z_INPUTS
HIDDEN = 64
PARAMETERS = 7_302
FIXED_EPOCH = 13
SCHEDULER_HORIZON_EPOCHS = 14
SAMPLES_PER_LAYER = 256
NETWORKS_PER_STEP = 4
LEARNING_RATE = 1.2e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 2.0e-5
MODEL_SEED = 0x51313015
GRADIENT_CLIP = 5.0
MIN_RELATIVE_OOF_GAIN = 0.0025
MIN_IMPROVING_FOLDS = 3
MIN_PAIRED_NETWORK_Z = 2.0
MIN_NETWORK_WIN_FRACTION = 0.55
MIN_FOLD_NETWORK_WIN_FRACTION = 0.50
MAX_RSS_BYTES = 8 * (1 << 30)


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


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def enforce_rss(where: str) -> int:
    value = resident_bytes()
    if value >= MAX_RSS_BYTES:
        raise MemoryError(f"RSS gate failed at {where}: {value}")
    return value


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def replace_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_static_bindings() -> dict[str, object]:
    expected = {
        "program_sha256": sha256(Path(__file__)),
        "base_program_sha256": sha256(BASE_PROGRAM),
        "base_preregistration_sha256": sha256(BASE_PREREGISTRATION),
        "base_result_sha256": sha256(BASE_RESULT),
        "base_model_sha256": sha256(BASE_MODEL),
        "ridge_result_sha256": sha256(RIDGE_RESULT),
        "j_source_sha256": sha256(J_SOURCE),
        "q_source_sha256": sha256(Q_SOURCE),
    }
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_ANY_NONLINEAR_Z_TARGET_OPEN":
        raise AssertionError("nonlinear-Z preregistration is not frozen")
    for key, observed in expected.items():
        if prereg.get("bindings", {}).get(key) != observed:
            raise AssertionError(f"preregistered binding changed: {key}")
    literals = {
        "base_program_sha256": BASE_PROGRAM_SHA256,
        "base_preregistration_sha256": BASE_PREREGISTRATION_SHA256,
        "base_result_sha256": BASE_RESULT_SHA256,
        "base_model_sha256": BASE_MODEL_SHA256,
        "ridge_result_sha256": RIDGE_RESULT_SHA256,
        "j_source_sha256": J_SOURCE_SHA256,
        "q_source_sha256": Q_SOURCE_SHA256,
    }
    for key, value in literals.items():
        if expected[key] != value:
            raise AssertionError(f"literal authority changed: {key}")
    return prereg


def load_base_oof_authority() -> tuple[dict[str, object], np.ndarray]:
    if not BASE_RESULT.is_file():
        raise FileNotFoundError("the frozen 22-input full688 OOF authority must finish first")
    result = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    if (
        result.get("schema") != "q24-qhead-full688-refit-discriminator-v1"
        or result.get("status") != "PROMOTED_FIXED_FULL688_MODEL"
        or result.get("bindings", {}).get("program_sha256") != BASE_PROGRAM_SHA256
        or result.get("fixed_rule", {}).get("fold_assignment_sha256") != FOLD_ASSIGNMENT_SHA256
        or result.get("access_declarations", {}).get("board_opened") is not False
        or result.get("access_declarations", {}).get("submission") is not False
    ):
        raise AssertionError("22-input OOF authority contract changed")
    rows = result.get("crossfit", {}).get("per_network", [])
    if len(rows) != 688:
        raise AssertionError("22-input OOF authority does not cover all 688 rows")
    values = np.empty(688, dtype=np.float64)
    seen: set[int] = set()
    for row in rows:
        rank = int(row["role_rank"])
        if rank in seen or not 0 <= rank < 688:
            raise AssertionError("invalid 22-input OOF role-rank coverage")
        seen.add(rank)
        values[rank] = float(row["candidate_final_mse"])
    recorded = float(result["crossfit"]["candidate_oof_final_mse"])
    if not np.isclose(float(np.mean(values)), recorded, rtol=0.0, atol=1e-20):
        raise AssertionError("22-input OOF mean is internally inconsistent")
    return result, values


@dataclass
class Prepared:
    base_state: np.ndarray
    base_q: np.ndarray
    delta_q: np.ndarray
    residual_final: np.ndarray
    raw_suffix: np.ndarray
    sigma_suffix: np.ndarray
    z_suffix: np.ndarray
    z_center: np.ndarray
    z_scale: np.ndarray
    target_blind_diagnostic: dict[str, object]


def standard_raw(raw: np.ndarray, base_model) -> np.ndarray:
    centers = base_model.centers.detach().cpu().numpy()
    scales = base_model.scales.detach().cpu().numpy()
    tail = np.clip(
        (raw[..., 1:].astype(np.float64) - centers[:, None, :]) / scales[:, None, :],
        -8.0,
        8.0,
    )
    return np.ascontiguousarray(np.concatenate((raw[..., :1], tail), axis=-1), dtype=np.float32)


def prepare_all_with_z(base_runner, j, base_model, dataset, error: np.ndarray) -> Prepared:
    shape = (688, len(SUFFIX), 1024)
    base_state = np.empty(shape, dtype=np.float32)
    base_q = np.empty(shape, dtype=np.float32)
    delta_q = np.empty(shape, dtype=np.float32)
    residual_final = np.empty((688, 1024), dtype=np.float32)
    raw_suffix = np.empty((*shape, 20), dtype=np.float32)
    sigma_suffix = np.empty(shape, dtype=np.float32)
    z_suffix = np.empty((*shape, 20), dtype=np.float32)
    z_sum = np.zeros((len(SUFFIX), 20), dtype=np.float64)
    z_square = np.zeros_like(z_sum)
    layer_sums = {
        name: np.zeros(16, dtype=np.float64)
        for name in ("previous_square", "message_square", "source_square", "combined_square")
    }
    layer_counts = np.zeros(16, dtype=np.int64)
    base_model.eval()
    with torch.no_grad():
        for rank in range(688):
            raw_t, sigma_t, weights_t, _ = j.load_batch(
                dataset, np.asarray([rank], dtype=np.int64), error
            )
            prediction = base_model(raw_t, sigma_t, weights_t, exposure=1.0)
            if not isinstance(prediction, torch.Tensor):
                raise TypeError("alpha=.75 J returned non-tensor")
            a = prediction[0].cpu().numpy().astype(np.float32, copy=False)
            raw = raw_t[0].cpu().numpy()
            sigma = sigma_t[0].cpu().numpy()
            weights = weights_t[0].cpu().numpy()
            residual = error[rank] - a
            standardized = standard_raw(raw, base_model)
            residual_final[rank] = residual[-1]
            raw_suffix[rank] = raw[list(SUFFIX)]
            sigma_suffix[rank] = sigma[list(SUFFIX)]
            for slot, layer in enumerate(SUFFIX):
                phi = raw[layer, :, 5].astype(np.float64)
                prior = np.stack((a[layer - 1], residual[layer - 1])).astype(np.float64)
                message = phi[None, :] * (prior @ weights[layer].astype(np.float64))
                base_state[rank, slot] = a[layer]
                base_q[rank, slot] = (a[layer].astype(np.float64) - message[0]).astype(np.float32)
                delta_q[rank, slot] = (residual[layer].astype(np.float64) - message[1]).astype(np.float32)
            z_state = standardized[2].copy()
            source64 = standardized[2].astype(np.float64)
            layer_sums["source_square"][2] += float(np.sum(source64 * source64))
            layer_sums["combined_square"][2] += float(np.sum(source64 * source64))
            layer_counts[2] += source64.size
            for layer in range(3, 16):
                previous64 = z_state.astype(np.float64)
                source64 = standardized[layer].astype(np.float64)
                transported = weights[layer].T.astype(np.float64) @ previous64
                message64 = raw[layer, :, 5, None].astype(np.float64) * transported
                combined64 = message64 + source64
                layer_sums["previous_square"][layer] += float(np.sum(previous64 * previous64))
                layer_sums["message_square"][layer] += float(np.sum(message64 * message64))
                layer_sums["source_square"][layer] += float(np.sum(source64 * source64))
                layer_sums["combined_square"][layer] += float(np.sum(combined64 * combined64))
                layer_counts[layer] += combined64.size
                z_state = combined64.astype(np.float32)
                if layer in SUFFIX:
                    slot = SUFFIX.index(layer)
                    z_suffix[rank, slot] = z_state
                    z_sum[slot] += np.sum(z_state.astype(np.float64), axis=0)
                    z_square[slot] += np.sum(z_state.astype(np.float64) ** 2, axis=0)
            if (rank + 1) % 32 == 0:
                print(json.dumps({"stage": "prepare_with_z", "rows": rank + 1}), flush=True)
            enforce_rss(f"prepare_with_z rank {rank}")
    count = float(688 * 1024)
    z_center = (z_sum / count).astype(np.float32)
    z_variance = np.maximum(z_square / count - (z_sum / count) ** 2, 1e-12)
    z_scale = np.sqrt(z_variance).astype(np.float32)
    clipped = 0
    total = z_suffix.size
    maximum_standardized_abs = 0.0
    for rank in range(688):
        standardized_z = (
            (z_suffix[rank].astype(np.float64) - z_center[:, None, :])
            / z_scale[:, None, :]
        )
        clipped += int(np.sum(np.abs(standardized_z) > 8.0))
        maximum_standardized_abs = max(maximum_standardized_abs, float(np.max(np.abs(standardized_z))))
    layers: list[dict[str, object]] = []
    for layer in range(2, 16):
        denominator = max(int(layer_counts[layer]), 1)
        rms = {
            name.removesuffix("_square") + "_rms": math.sqrt(float(values[layer]) / denominator)
            for name, values in layer_sums.items()
        }
        previous_rms = rms["previous_rms"]
        source_rms = rms["source_rms"]
        layers.append(
            {
                "layer_zero_based": layer,
                **rms,
                "transport_contraction_ratio": (
                    rms["message_rms"] / previous_rms if previous_rms > 0 else None
                ),
                "transport_to_source_rms_ratio": (
                    rms["message_rms"] / source_rms if source_rms > 0 else None
                ),
            }
        )
    diagnostic = {
        "depends_on_targets": False,
        "recurrence": "Z_l=Phi_l*(W_l.T@Z_(l-1))+standardized_raw20_l",
        "layer_rms_and_shrink": layers,
        "z_scale_minimum": float(np.min(z_scale)),
        "z_scale_maximum": float(np.max(z_scale)),
        "standardized_abs_maximum_before_clip": maximum_standardized_abs,
        "standardized_clip_fraction_at_8": clipped / total,
        "normalization_population": "all688 target-free native rows",
    }
    return Prepared(
        base_state, base_q, delta_q, residual_final, raw_suffix, sigma_suffix,
        z_suffix, z_center, z_scale, diagnostic,
    )


class NonlinearZHead(nn.Module):
    def __init__(self, base_model, z_center: np.ndarray, z_scale: np.ndarray):
        super().__init__()
        self.register_buffer("centers", base_model.centers.detach().clone())
        self.register_buffer("scales", base_model.scales.detach().clone())
        self.register_buffer("layer_rms", base_model.layer_rms.detach().clone())
        self.register_buffer("z_center", torch.from_numpy(z_center.copy()))
        self.register_buffer("z_scale", torch.from_numpy(z_scale.copy()))
        self.trunk = nn.Sequential(
            nn.Linear(INPUTS, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )
        self.head_weight = nn.Parameter(torch.zeros(len(SUFFIX), HIDDEN))
        self.head_bias = nn.Parameter(torch.zeros(len(SUFFIX)))

    def source(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        base_state: torch.Tensor,
        base_q: torch.Tensor,
        z_state: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        slot = SUFFIX.index(layer)
        tail = torch.clamp(
            (raw[..., 1:] - self.centers[layer]) / self.scales[layer], -8.0, 8.0
        )
        rms = self.layer_rms[layer]
        z_standard = torch.clamp(
            (z_state - self.z_center[slot]) / self.z_scale[slot], -8.0, 8.0
        )
        features = torch.cat(
            (
                raw[..., :1],
                tail,
                torch.clamp(base_state / rms, -8.0, 8.0)[..., None],
                torch.clamp(
                    base_q / torch.clamp(sigma * rms, min=2.0**-30), -8.0, 8.0
                )[..., None],
                z_standard,
            ),
            dim=-1,
        )
        hidden = self.trunk(features)
        standard = torch.einsum("...h,h->...", hidden, self.head_weight[slot])
        standard = standard + self.head_bias[slot]
        return sigma * rms * standard


def q_rms_for(prepared: Prepared, train: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sqrt(np.mean(prepared.delta_q[train].astype(np.float64) ** 2, axis=(0, 2))),
        2.0**-20,
    ).astype(np.float32)


def train_fixed(q, base_model, prepared: Prepared, train: np.ndarray, stream_id: int):
    torch.manual_seed(MODEL_SEED)
    model = NonlinearZHead(base_model, prepared.z_center, prepared.z_scale)
    if sum(value.numel() for value in model.parameters()) != PARAMETERS:
        raise AssertionError("nonlinear-Z parameter count changed")
    q_rms = q_rms_for(prepared, train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train) / NETWORKS_PER_STEP)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_HORIZON_EPOCHS * steps_per_epoch,
        eta_min=MIN_LEARNING_RATE,
    )
    order_rng = np.random.default_rng(
        (MODEL_SEED ^ 0xA5A5A5A5 ^ (int(stream_id) * 0x9E3779B1)) & 0xFFFF_FFFF
    )
    trace: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, FIXED_EPOCH + 1):
        model.train()
        order = order_rng.permutation(train)
        objective_sum = 0.0
        examples = 0
        for offset in range(0, len(order), NETWORKS_PER_STEP):
            ranks = order[offset : offset + NETWORKS_PER_STEP]
            if len(ranks) != NETWORKS_PER_STEP:
                raise AssertionError("frozen fold sizes must be divisible by network batch")
            losses = []
            for slot, layer in enumerate(SUFFIX):
                indices = [q.sample_indices(int(rank), layer, epoch) for rank in ranks]
                raw = np.concatenate(
                    [prepared.raw_suffix[int(rank), slot, index] for rank, index in zip(ranks, indices, strict=True)]
                )
                sigma = np.concatenate(
                    [prepared.sigma_suffix[int(rank), slot, index] for rank, index in zip(ranks, indices, strict=True)]
                )
                state = np.concatenate(
                    [prepared.base_state[int(rank), slot, index] for rank, index in zip(ranks, indices, strict=True)]
                )
                base_q = np.concatenate(
                    [prepared.base_q[int(rank), slot, index] for rank, index in zip(ranks, indices, strict=True)]
                )
                z_state = np.concatenate(
                    [prepared.z_suffix[int(rank), slot, index] for rank, index in zip(ranks, indices, strict=True)]
                )
                truth_q = np.concatenate(
                    [prepared.delta_q[int(rank), slot, index] for rank, index in zip(ranks, indices, strict=True)]
                )
                source = model.source(
                    torch.from_numpy(np.ascontiguousarray(raw)),
                    torch.from_numpy(np.ascontiguousarray(sigma)),
                    torch.from_numpy(np.ascontiguousarray(state)),
                    torch.from_numpy(np.ascontiguousarray(base_q)),
                    torch.from_numpy(np.ascontiguousarray(z_state)),
                    layer,
                )
                truth = torch.from_numpy(np.ascontiguousarray(truth_q))
                losses.append(torch.mean(((source - truth) / float(q_rms[slot])).square()))
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"stream {stream_id} epoch {epoch}: nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach()) * len(ranks)
            examples += len(ranks)
        record = {
            "stream_id": int(stream_id),
            "epoch": epoch,
            "sampled_teacher_objective": objective_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_process_rss_bytes": enforce_rss(f"stream {stream_id} epoch {epoch}"),
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    model.eval()
    return model, q_rms, trace


def score_fold(model, prepared: Prepared, dataset, held: np.ndarray) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    local_sse = 0.0
    model.eval()
    with torch.no_grad():
        for rank_value in held:
            rank = int(rank_value)
            weights = np.load(dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False)
            state = np.zeros(1024, dtype=np.float32)
            for slot, layer in enumerate(SUFFIX):
                source = model.source(
                    torch.from_numpy(prepared.raw_suffix[rank, slot]),
                    torch.from_numpy(prepared.sigma_suffix[rank, slot]),
                    torch.from_numpy(prepared.base_state[rank, slot]),
                    torch.from_numpy(prepared.base_q[rank, slot]),
                    torch.from_numpy(prepared.z_suffix[rank, slot]),
                    layer,
                ).cpu().numpy()
                local_sse += float(np.sum((source.astype(np.float64) - prepared.delta_q[rank, slot]) ** 2))
                phi = prepared.raw_suffix[rank, slot, :, 5].astype(np.float64)
                state = (
                    phi * (state.astype(np.float64) @ weights[layer].astype(np.float64))
                    + source.astype(np.float64)
                ).astype(np.float32)
            residual = prepared.residual_final[rank].astype(np.float64)
            rows.append(
                {
                    "role_rank": rank,
                    "mlp_id": int(dataset.mlp_ids[rank]),
                    "candidate_final_mse": float(np.mean((residual - state.astype(np.float64)) ** 2)),
                }
            )
    return {
        "networks": len(held),
        "candidate_final_mse": float(np.mean([row["candidate_final_mse"] for row in rows])),
        "candidate_teacher_local_q_mse": local_sse / (len(held) * len(SUFFIX) * 1024),
        "per_network": rows,
    }


def main() -> None:
    if RUN_STATE.exists() or RESULT.exists() or MODEL.exists():
        raise FileExistsError("nonlinear-Z run has already started or completed")
    prereg = verify_static_bindings()
    base_oof_result, base_oof = load_base_oof_authority()
    base_runner = load_module("nonlinear_z_base_runner", BASE_PROGRAM)
    records, native_storage = base_runner.verify_native_rows()
    folds = base_runner.balanced_folds(records)
    if hashlib.sha256(folds.tobytes()).hexdigest() != FOLD_ASSIGNMENT_SHA256:
        raise AssertionError("target-blind fold assignment changed")
    write_json_once(
        RUN_STATE,
        {
            "schema": "q24-nonlinear-transported-raw20-run-state-v1",
            "status": "RUNNING_ONE_SHOT",
            "pid": os.getpid(),
            "started_unix_seconds": time.time(),
            "program_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREGISTRATION),
            "base_oof_result_sha256": sha256(BASE_RESULT),
            "target_columns_opened": [],
        },
    )
    # First target read occurs only after static, native-row, fold, and base-OOF checks.
    targets = base_runner.load_targets(records)
    hosts = np.ascontiguousarray(
        np.stack([np.load(base_runner.row_dir(rank) / "host.npy", allow_pickle=False) for rank in range(688)]),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    dataset = base_runner.All688Dataset(records)
    j = load_module("nonlinear_z_exact_j", J_SOURCE)
    q = load_module("nonlinear_z_sampling_authority", Q_SOURCE)
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    base_model = base_runner.load_alpha_j(j)
    prepared = prepare_all_with_z(base_runner, j, base_model, dataset, error)
    del targets, hosts, error
    gc.collect()
    enforce_rss("post preparation")

    fold_results: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    q_rms_records: list[list[float]] = []
    per_network: list[dict[str, object] | None] = [None] * 688
    started = time.perf_counter()
    for fold in range(FOLDS):
        train = ALL_RANKS[folds != fold]
        held = ALL_RANKS[folds == fold]
        candidate, q_rms, trace = train_fixed(q, base_model, prepared, train, fold)
        scored = score_fold(candidate, prepared, dataset, held)
        for row in scored["per_network"]:
            per_network[int(row["role_rank"])] = row
        summary = {key: value for key, value in scored.items() if key != "per_network"}
        summary.update(
            {
                "fold": fold,
                "training_rows": len(train),
                "held_rows": held.tolist(),
                "base_22input_oof_final_mse": float(np.mean(base_oof[held])),
                "network_win_fraction_vs_base": float(np.mean(
                    np.asarray([row["candidate_final_mse"] for row in scored["per_network"]])
                    < base_oof[held]
                )),
            }
        )
        fold_results.append(summary)
        traces.extend(trace)
        q_rms_records.append(q_rms.tolist())
        print(json.dumps({"stage": "fold_score", **summary}, sort_keys=True), flush=True)
        del candidate
        gc.collect()
    if any(row is None for row in per_network):
        raise AssertionError("nonlinear-Z OOF coverage is incomplete")
    rows = [row for row in per_network if row is not None]
    candidate_values = np.asarray([float(row["candidate_final_mse"]) for row in rows])
    improvement = base_oof - candidate_values
    improvement_se = float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
    paired_z = float(np.mean(improvement) / improvement_se) if improvement_se > 0 else float("inf")
    candidate_mse = float(np.mean(candidate_values))
    base_mse = float(np.mean(base_oof))
    relative_gain = (base_mse - candidate_mse) / base_mse
    fold_candidate = [float(np.mean(candidate_values[folds == fold])) for fold in range(FOLDS)]
    fold_base = [float(np.mean(base_oof[folds == fold])) for fold in range(FOLDS)]
    fold_wins = [
        float(np.mean(improvement[folds == fold] > 0.0)) for fold in range(FOLDS)
    ]
    improving_folds = sum(c < b for c, b in zip(fold_candidate, fold_base, strict=True))
    network_win_fraction = float(np.mean(improvement > 0.0))
    protected = np.asarray(
        base_oof_result["crossfit"]["legacy_protected40_oof"]["role_ranks"], dtype=np.int64
    )
    protected_candidate = float(np.mean(candidate_values[protected]))
    protected_base = float(np.mean(base_oof[protected]))
    gates = {
        "all_688_have_exactly_one_oof_prediction": True,
        "candidate_is_finite": bool(np.isfinite(candidate_values).all()),
        "relative_oof_gain_at_least_0_25pct": relative_gain >= MIN_RELATIVE_OOF_GAIN,
        "at_least_three_of_four_folds_improve": improving_folds >= MIN_IMPROVING_FOLDS,
        "paired_network_z_at_least_2": paired_z >= MIN_PAIRED_NETWORK_Z,
        "at_least_55pct_of_networks_improve": network_win_fraction >= MIN_NETWORK_WIN_FRACTION,
        "every_fold_has_at_least_50pct_network_wins": min(fold_wins) >= MIN_FOLD_NETWORK_WIN_FRACTION,
        "legacy_protected40_oof_does_not_worsen": protected_candidate <= protected_base,
        "rss_below_8_gib": enforce_rss("post crossfit") < MAX_RSS_BYTES,
    }
    promote = all(gates.values())
    final_trace: list[dict[str, object]] = []
    final_q_rms: list[float] | None = None
    model_hash: str | None = None
    if promote:
        final_model, q_rms, final_trace = train_fixed(q, base_model, prepared, ALL_RANKS, FOLDS)
        artifact = {
            "schema": "q24-nonlinear-transported-raw20-head-checkpoint-v1",
            "state_dict": final_model.state_dict(),
            "selected_epoch": FIXED_EPOCH,
            "refit_kind": "all688-fixed-epoch13-after-four-fold-oof-gate",
            "base_oof_result_sha256": sha256(BASE_RESULT),
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        }
        temporary = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
        torch.save(artifact, temporary)
        temporary.replace(MODEL)
        model_hash = sha256(MODEL)
        final_q_rms = q_rms.tolist()

    n = 1024
    bank_transport = 13 * Z_INPUTS * (((2 * n - 1) * n) + n)
    bank_source_add = 14 * Z_INPUTS * n
    widened_first_layer = len(SUFFIX) * n * HIDDEN * (2 * Z_INPUTS)
    z_read_standardization_upper_bound = len(SUFFIX) * n * Z_INPUTS * 4
    result = {
        "schema": "q24-nonlinear-transported-raw20-head-oof-v1",
        "status": "PROMOTED_FIXED_FULL688_MODEL" if promote else "CLOSED_NO_MATERIAL_OOF_GAIN",
        "access_declarations": {
            "role": "observer_fit",
            "observer_fit_targets_opened": ["mlp_id", "all_layer_means"],
            "observer_selection_targets_opened": [],
            "names_opened": [],
            "board_opened": False,
            "submission": False,
        },
        "fixed_rule": {
            "architecture": "42->64->64 with six layer-specific scalar heads",
            "parameters": PARAMETERS,
            "features": "raw20+A+qJ+standardized exact transported raw20 Z",
            "recurrence": "Z_l=Phi_l*(W_l.T@Z_(l-1))+standardized_raw20_l",
            "folds": FOLDS,
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
            "fixed_checkpoint_epoch": FIXED_EPOCH,
            "checkpoint_averaging": "none",
            "base_22input_arm_retrained": False,
        },
        "crossfit": {
            "networks": 688,
            "fold_sizes": [172] * FOLDS,
            "candidate_oof_final_mse": candidate_mse,
            "base_22input_oof_final_mse": base_mse,
            "relative_oof_gain": relative_gain,
            "absolute_oof_gain": base_mse - candidate_mse,
            "paired_network_improvement_standard_error": improvement_se,
            "paired_network_z": paired_z,
            "network_win_fraction": network_win_fraction,
            "fold_candidate_mse": fold_candidate,
            "fold_base_22input_mse": fold_base,
            "fold_network_win_fractions": fold_wins,
            "improving_folds": improving_folds,
            "fold_results": fold_results,
            "legacy_protected40_oof": {
                "role_ranks": protected.tolist(),
                "candidate_final_mse": protected_candidate,
                "base_22input_final_mse": protected_base,
            },
            "per_network": rows,
        },
        "promotion_gate": {
            "minimum_relative_oof_gain": MIN_RELATIVE_OOF_GAIN,
            "minimum_improving_folds": MIN_IMPROVING_FOLDS,
            "minimum_paired_network_z": MIN_PAIRED_NETWORK_Z,
            "minimum_network_win_fraction": MIN_NETWORK_WIN_FRACTION,
            "minimum_each_fold_network_win_fraction": MIN_FOLD_NETWORK_WIN_FRACTION,
            "gates": gates,
            "conjunctive_pass": promote,
            "scientific_model_only_not_submission_authority": True,
        },
        "target_blind_z_diagnostic": prepared.target_blind_diagnostic,
        "training": {
            "crossfit_q_rms": q_rms_records,
            "crossfit_trace": traces,
            "final_fit_started_only_after_gate": promote,
            "final_fit_q_rms": final_q_rms,
            "final_fit_trace": final_trace,
        },
        "deployability": {
            "incremental_bank_transport_flops": bank_transport,
            "incremental_bank_source_add_flops": bank_source_add,
            "incremental_first_layer_widening_flops_vs_22input_head": widened_first_layer,
            "incremental_z_read_standardization_flops_upper_bound": z_read_standardization_upper_bound,
            "incremental_total_flops_vs_22input_head_upper_bound": (
                bank_transport + bank_source_add + widened_first_layer
                + z_read_standardization_upper_bound
            ),
            "incremental_live_z_state_bytes_float32": n * Z_INPUTS * 4,
            "six_layer_z_cache_bytes_if_materialized": len(SUFFIX) * n * Z_INPUTS * 4,
            "incremental_model_parameter_bytes_float32": (PARAMETERS - 6_022) * 4,
            "training_z_cache_bytes": prepared.z_suffix.nbytes,
            "maximum_process_rss_bytes": enforce_rss("result"),
            "native_row_storage_bytes": native_storage,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREGISTRATION),
            "base_program_sha256": BASE_PROGRAM_SHA256,
            "base_oof_result_sha256": sha256(BASE_RESULT),
            "ridge_result_sha256": RIDGE_RESULT_SHA256,
            "j_source_sha256": J_SOURCE_SHA256,
            "q_source_sha256": Q_SOURCE_SHA256,
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
            "promoted_model_sha256": model_hash,
        },
    }
    write_json_once(RESULT, result)
    replace_json(
        RUN_STATE,
        {
            "schema": "q24-nonlinear-transported-raw20-run-state-v1",
            "status": "COMPLETE",
            "result_sha256": sha256(RESULT),
            "promoted_model_sha256": model_hash,
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "crossfit": result["crossfit"],
                "model_sha256": model_hash,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
