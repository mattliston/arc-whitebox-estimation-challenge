#!/Users/matt/anaconda3/bin/python3
"""Frozen all-688 refit discriminator for the alpha=.75 q10:q15 head.

Four balanced target-blind folds provide one out-of-fold prediction for every
observer-fit network.  Neither the epoch nor any checkpoint is selected from
those predictions.  A single all-688 model is fit only if the preregistered
paired promotion gate passes.
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
import pyarrow as pa
import pyarrow.parquet as pq
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PILOT = P2 / "public_q24_native_transport_energy_pilot_v1"
FINAL = P2 / "public_q24_native_transport_energy_final_v1"
PILOT_ROWS = PILOT / "runs/native_rows_v1"
FINAL_ROWS = FINAL / "runs/native_rows_remaining_v1"
FEATURE_STORE = (
    P2 / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
FEATURE_MANIFEST = FEATURE_STORE / "manifest.json"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
Q_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
ALPHA_MODEL = P2 / "public_q24_native_j_checkpoint_interpolation_v1/INTERPOLATED_MODEL.pt"
INCUMBENT_Q_MODEL = P2 / "public_q24_native_j_q10q15_residual_head_v1/Q_SUFFIX_HEAD.pt"
CONTINUATION_RESULT = P2 / "public_q24_native_j_autonomous_continuation_v1/RESULT.json"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
SEAL = HERE / "RUN_SEAL.json"
RUN_STATE = HERE / "RUN_STATE.json"
RESULT = HERE / "RESULT.json"
MODEL = HERE / "FULL688_Q_SUFFIX_HEAD.pt"

J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
Q_SOURCE_SHA256 = "1e4c906936f4685bd800f922ded6c582af8213ecad1fbe41da651faff82be9c3"
ALPHA_MODEL_SHA256 = "fc65528720c7e1378ed7011b7b201f4086217a544b092de208ea1064656d9ef5"
INCUMBENT_Q_MODEL_SHA256 = "5fb57a510d933940f19ddd5ab11650c3a966c2039c6bd12f73988457c2ce91b8"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
TARGETS_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
PILOT_GENERATION_SEAL_SHA256 = "2310cfeacb3203a462492f0963568a8bfdff9fdaea3526658da9fd15ab226d4c"
FINAL_GENERATION_SEAL_SHA256 = "8959fd274df3120003b71229a5453b5ee2bc1c875c4e9a139d2e72d104135a6b"
CONTINUATION_RESULT_SHA256 = "f8397f8eda9c240d40b3a63f32a5b8c9634c5d11b3a428e506056a49c39936e9"
INSTRUMENTED_SHA256 = "f539889324fb82cc1ded35a41c8b5a359014d9ddac52eab62949257e8e58a0ef"
NATIVE_AGGREGATE_SHA256 = "204097b1e43387fdc2080eb690606d7dc34aeda7ca4794cb98382137ebfa3db3"
FOLD_ASSIGNMENT_SHA256 = "0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c"

ALL_RANKS = np.arange(688, dtype=np.int64)
FOLDS = 4
FOLD_SALT = "q24-qhead-full688-refit-v1"
SUFFIX = tuple(range(10, 16))
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
OBSERVED_G11B_BOARD_SHIFT_FACTOR = 1.0577849019072405
MAX_RSS_BYTES = 8 * (1 << 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def replace_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_seal() -> dict[str, object]:
    if not SEAL.is_file() or not PREREGISTRATION.is_file():
        raise FileNotFoundError("frozen preregistration and run seal are required")
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "FROZEN_BEFORE_ANY_REFIT_TARGET_OPEN":
        raise AssertionError("run seal status changed")
    expected = {
        "program_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "j_source_sha256": sha256(J_SOURCE),
        "q_source_sha256": sha256(Q_SOURCE),
        "alpha_model_sha256": sha256(ALPHA_MODEL),
        "incumbent_q_model_sha256": sha256(INCUMBENT_Q_MODEL),
        "feature_manifest_sha256": sha256(FEATURE_MANIFEST),
        "observer_fit_targets_sha256": sha256(TARGETS),
        "pilot_generation_seal_sha256": sha256(PILOT / "GENERATION_SEAL.json"),
        "final_generation_seal_sha256": sha256(FINAL / "GENERATION_SEAL.json"),
        "continuation_result_sha256": sha256(CONTINUATION_RESULT),
    }
    for key, observed in expected.items():
        if seal.get(key) != observed:
            raise AssertionError(f"sealed binding changed: {key}: {observed}")
    literals = {
        "j_source_sha256": J_SOURCE_SHA256,
        "q_source_sha256": Q_SOURCE_SHA256,
        "alpha_model_sha256": ALPHA_MODEL_SHA256,
        "incumbent_q_model_sha256": INCUMBENT_Q_MODEL_SHA256,
        "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
        "observer_fit_targets_sha256": TARGETS_SHA256,
        "pilot_generation_seal_sha256": PILOT_GENERATION_SEAL_SHA256,
        "final_generation_seal_sha256": FINAL_GENERATION_SEAL_SHA256,
        "continuation_result_sha256": CONTINUATION_RESULT_SHA256,
    }
    for key, value in literals.items():
        if expected[key] != value:
            raise AssertionError(f"literal authority changed: {key}")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_ANY_REFIT_TARGET_OPEN":
        raise AssertionError("preregistration status changed")
    rule = prereg.get("fixed_rule", {})
    required_rule = {
        "folds": FOLDS,
        "fixed_checkpoint_epoch": FIXED_EPOCH,
        "checkpoint_averaging": "none",
        "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
        "minimum_relative_oof_gain": MIN_RELATIVE_OOF_GAIN,
        "minimum_improving_folds": MIN_IMPROVING_FOLDS,
        "minimum_paired_network_z": MIN_PAIRED_NETWORK_Z,
        "minimum_network_win_fraction": MIN_NETWORK_WIN_FRACTION,
        "minimum_each_fold_network_win_fraction": MIN_FOLD_NETWORK_WIN_FRACTION,
        "legacy_protected40_must_not_worsen": True,
    }
    if rule != required_rule:
        raise AssertionError("preregistered fixed rule changed")
    return seal


def row_dir(rank: int) -> Path:
    root = PILOT_ROWS if 35 <= int(rank) < 235 else FINAL_ROWS
    return root / f"row_{int(rank):04d}"


@dataclass(frozen=True)
class NativeRow:
    raw_locals: np.ndarray
    sigma: np.ndarray
    weights_path: Path


class All688Dataset:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.mlp_ids = np.asarray([int(record["mlp_id"]) for record in records], dtype=np.int64)

    def load(self, rank: int) -> NativeRow:
        rank = int(rank)
        directory = row_dir(rank)
        return NativeRow(
            raw_locals=np.load(directory / "raw20.npy", mmap_mode="r", allow_pickle=False),
            sigma=np.load(directory / "sigma.npy", mmap_mode="r", allow_pickle=False),
            weights_path=FEATURE_STORE / f"rows/observer_fit/{rank:04d}/weights.npy",
        )


def verify_native_rows() -> tuple[list[dict[str, object]], int]:
    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    manifest_rows = manifest["roles"]["observer_fit"]["row_manifests"]
    if len(manifest_rows) != 688:
        raise AssertionError("feature manifest observer-fit roster changed")
    records: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    native_storage = 0
    for rank in range(688):
        directory = row_dir(rank)
        result_path = directory / "result.json"
        receipt = json.loads(result_path.read_text(encoding="utf-8"))
        expected_generation_seal = (
            PILOT_GENERATION_SEAL_SHA256 if 35 <= rank < 235 else FINAL_GENERATION_SEAL_SHA256
        )
        if (
            receipt.get("status") != "complete-target-free"
            or receipt.get("role") != "observer_fit"
            or int(receipt.get("role_rank", -1)) != rank
            or receipt.get("target_columns_opened") != []
            or receipt.get("observer_selection_opened") is not False
            or receipt.get("name_or_board_join") is not False
            or receipt.get("instrumented_source_sha256") != INSTRUMENTED_SHA256
            or receipt.get("generation_seal_sha256") != expected_generation_seal
        ):
            raise AssertionError(f"native receipt contract failed at rank {rank}")
        for name in ("host.npy", "raw20.npy", "sigma.npy"):
            path = directory / name
            binding = receipt["files"][name]
            if path.stat().st_size != int(binding["bytes"]) or sha256(path) != binding["sha256"]:
                raise AssertionError(f"native payload changed: rank {rank} {name}")
            native_storage += path.stat().st_size
        manifest_entry = manifest_rows[rank]
        row_manifest = FEATURE_STORE / str(manifest_entry["path"])
        if (
            int(manifest_entry["role_rank"]) != rank
            or int(manifest_entry["mlp_id"]) != int(receipt["mlp_id"])
            or manifest_entry["sha256"] != receipt["row_manifest_sha256"]
            or sha256(row_manifest) != manifest_entry["sha256"]
        ):
            raise AssertionError(f"feature row binding changed at rank {rank}")
        feature_receipt = json.loads(row_manifest.read_text(encoding="utf-8"))
        weights_binding = feature_receipt["files"]["weights.npy"]
        weights_path = FEATURE_STORE / f"rows/observer_fit/{rank:04d}/weights.npy"
        if (
            weights_binding["sha256"] != receipt["weights_file_sha256"]
            or weights_path.stat().st_size != int(weights_binding["bytes"])
        ):
            raise AssertionError(f"weight authority changed at rank {rank}")
        result_hash = sha256(result_path)
        aggregate.update(rank.to_bytes(4, "big"))
        aggregate.update(result_hash.encode("ascii"))
        records.append(receipt)
    if aggregate.hexdigest() != NATIVE_AGGREGATE_SHA256:
        raise AssertionError("native receipt aggregate changed")
    ids = [int(record["mlp_id"]) for record in records]
    if len(set(ids)) != 688:
        raise AssertionError("observer-fit mlp_id roster is not unique")
    return records, native_storage


def balanced_folds(records: list[dict[str, object]]) -> np.ndarray:
    ids = [int(record["mlp_id"]) for record in records]
    order = sorted(
        range(688),
        key=lambda rank: (
            hashlib.blake2s(
                f"{FOLD_SALT}:{ids[rank]}".encode("ascii"), digest_size=16
            ).digest(),
            ids[rank],
            rank,
        ),
    )
    folds = np.empty(688, dtype=np.uint8)
    for position, rank in enumerate(order):
        folds[rank] = position % FOLDS
    if [int(np.sum(folds == fold)) for fold in range(FOLDS)] != [172] * FOLDS:
        raise AssertionError("balanced fold sizes changed")
    if hashlib.sha256(folds.tobytes()).hexdigest() != FOLD_ASSIGNMENT_SHA256:
        raise AssertionError("balanced fold assignment changed")
    return folds


def load_targets(records: list[dict[str, object]]) -> np.ndarray:
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)
    expected_ids = np.asarray([int(record["mlp_id"]) for record in records], dtype=np.int32)
    if ids.dtype != np.int32 or targets.dtype != np.float32 or not np.array_equal(ids, expected_ids):
        raise AssertionError("target/native identity or schema changed")
    return np.ascontiguousarray(targets, dtype=np.float32)


def load_alpha_j(j):
    artifact = torch.load(ALPHA_MODEL, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "q24-native-j-interpolated-checkpoint-v1" or artifact.get("alpha") != 0.75:
        raise AssertionError("alpha=.75 authority changed")
    state = artifact["state_dict"]
    model = j.TransportEnergyMultiStateResidual(
        state["centers"].cpu().numpy(),
        state["scales"].cpu().numpy(),
        state["layer_rms"].cpu().numpy(),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@dataclass
class Prepared:
    base_state: np.ndarray
    base_q: np.ndarray
    delta_q: np.ndarray
    residual_final: np.ndarray
    raw_suffix: np.ndarray
    sigma_suffix: np.ndarray


def prepare_all(j, base, dataset: All688Dataset, error: np.ndarray) -> Prepared:
    base_state = np.empty((688, 6, 1024), dtype=np.float32)
    base_q = np.empty_like(base_state)
    delta_q = np.empty_like(base_state)
    residual_final = np.empty((688, 1024), dtype=np.float32)
    raw_suffix = np.empty((688, 6, 1024, 20), dtype=np.float32)
    sigma_suffix = np.empty((688, 6, 1024), dtype=np.float32)
    base.eval()
    with torch.no_grad():
        for rank in range(688):
            raw_t, sigma_t, weights_t, _ = j.load_batch(
                dataset, np.asarray([rank], dtype=np.int64), error
            )
            prediction = base(raw_t, sigma_t, weights_t, exposure=1.0)
            if not isinstance(prediction, torch.Tensor):
                raise TypeError("alpha J returned non-tensor")
            a = prediction[0].cpu().numpy().astype(np.float32, copy=False)
            raw = raw_t[0].cpu().numpy()
            sigma = sigma_t[0].cpu().numpy()
            weights = weights_t[0].cpu().numpy()
            residual = error[rank] - a
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
            if (rank + 1) % 32 == 0:
                print(json.dumps({"stage": "prepare", "rows": rank + 1}), flush=True)
            enforce_rss(f"prepare rank {rank}")
    return Prepared(base_state, base_q, delta_q, residual_final, raw_suffix, sigma_suffix)


def q_rms_for(prepared: Prepared, train: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sqrt(np.mean(prepared.delta_q[train].astype(np.float64) ** 2, axis=(0, 2))),
        2.0**-20,
    ).astype(np.float32)


def train_fixed(q, base, prepared: Prepared, train: np.ndarray, stream_id: int):
    torch.manual_seed(MODEL_SEED)
    model = q.QSuffixHead(base.centers, base.scales, base.layer_rms)
    if sum(value.numel() for value in model.parameters()) != 6_022:
        raise AssertionError("q-head parameter count changed")
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
                truth_q = np.concatenate(
                    [prepared.delta_q[int(rank), slot, index] for rank, index in zip(ranks, indices, strict=True)]
                )
                source = model.source(
                    torch.from_numpy(np.ascontiguousarray(raw)),
                    torch.from_numpy(np.ascontiguousarray(sigma)),
                    torch.from_numpy(np.ascontiguousarray(state)),
                    torch.from_numpy(np.ascontiguousarray(base_q)),
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


def load_incumbent(q, base):
    artifact = torch.load(INCUMBENT_Q_MODEL, map_location="cpu", weights_only=False)
    if (
        artifact.get("schema") != "q24-native-j-q10q15-residual-head-checkpoint-v1"
        or int(artifact.get("selected_epoch", -1)) != 13
    ):
        raise AssertionError("incumbent q-head authority changed")
    model = q.QSuffixHead(base.centers, base.scales, base.layer_rms)
    model.load_state_dict(artifact["state_dict"], strict=True)
    for name in ("centers", "scales", "layer_rms"):
        if not torch.equal(getattr(model, name), getattr(base, name)):
            raise AssertionError(f"q-head/J normalization mismatch: {name}")
    model.eval()
    return model


def score_fold(q, candidate, incumbent, prepared: Prepared, dataset: All688Dataset, held: np.ndarray):
    rows: list[dict[str, object]] = []
    local_candidate_sse = 0.0
    local_incumbent_sse = 0.0
    candidate.eval()
    incumbent.eval()
    with torch.no_grad():
        for rank_value in held:
            rank = int(rank_value)
            weights = np.load(dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False)
            states = np.zeros((3, 1024), dtype=np.float32)
            for slot, layer in enumerate(SUFFIX):
                raw = torch.from_numpy(prepared.raw_suffix[rank, slot])
                sigma = torch.from_numpy(prepared.sigma_suffix[rank, slot])
                base_state = torch.from_numpy(prepared.base_state[rank, slot])
                base_q = torch.from_numpy(prepared.base_q[rank, slot])
                source_candidate = candidate.source(raw, sigma, base_state, base_q, layer).cpu().numpy()
                source_incumbent = incumbent.source(raw, sigma, base_state, base_q, layer).cpu().numpy()
                truth_q = prepared.delta_q[rank, slot]
                local_candidate_sse += float(np.sum((source_candidate.astype(np.float64) - truth_q) ** 2))
                local_incumbent_sse += float(np.sum((source_incumbent.astype(np.float64) - truth_q) ** 2))
                sources = np.stack((source_candidate, source_incumbent, truth_q)).astype(np.float64)
                phi = prepared.raw_suffix[rank, slot, :, 5].astype(np.float64)
                messages = states.astype(np.float64) @ weights[layer].astype(np.float64)
                states = (phi[None, :] * messages + sources).astype(np.float32)
            residual = prepared.residual_final[rank].astype(np.float64)
            base_mse = float(np.mean(residual * residual))
            candidate_mse = float(np.mean((residual - states[0].astype(np.float64)) ** 2))
            incumbent_mse = float(np.mean((residual - states[1].astype(np.float64)) ** 2))
            oracle_mse = float(np.mean((residual - states[2].astype(np.float64)) ** 2))
            rows.append(
                {
                    "role_rank": rank,
                    "mlp_id": int(dataset.mlp_ids[rank]),
                    "base_alpha075_j_final_mse": base_mse,
                    "candidate_final_mse": candidate_mse,
                    "incumbent_final_mse": incumbent_mse,
                    "oracle_delta_q_final_mse": oracle_mse,
                }
            )
    count = len(held) * len(SUFFIX) * 1024
    return {
        "networks": len(held),
        "base_alpha075_j_final_mse": float(np.mean([row["base_alpha075_j_final_mse"] for row in rows])),
        "candidate_final_mse": float(np.mean([row["candidate_final_mse"] for row in rows])),
        "incumbent_final_mse": float(np.mean([row["incumbent_final_mse"] for row in rows])),
        "oracle_delta_q_final_mse": float(np.mean([row["oracle_delta_q_final_mse"] for row in rows])),
        "candidate_teacher_local_q_mse": local_candidate_sse / count,
        "incumbent_teacher_local_q_mse": local_incumbent_sse / count,
        "per_network": rows,
    }


def main() -> None:
    if RUN_STATE.exists() or RESULT.exists() or MODEL.exists():
        raise FileExistsError("full688 refit has already started or completed")
    seal = verify_seal()
    records, native_storage = verify_native_rows()
    folds = balanced_folds(records)
    continuation = json.loads(CONTINUATION_RESULT.read_text(encoding="utf-8"))
    legacy_protected = np.asarray(
        continuation["split"]["protected_once_40_role_ranks"], dtype=np.int64
    )
    if len(legacy_protected) != 40:
        raise AssertionError("legacy protected roster changed")
    write_json_once(
        RUN_STATE,
        {
            "schema": "q24-qhead-full688-refit-run-state-v1",
            "status": "RUNNING_ONE_SHOT",
            "pid": os.getpid(),
            "started_unix_seconds": time.time(),
            "seal_sha256": sha256(SEAL),
            "target_columns_opened": [],
        },
    )
    # First semantic target open occurs only after every target-free row and fold check passes.
    targets = load_targets(records)
    hosts = np.ascontiguousarray(
        np.stack([np.load(row_dir(rank) / "host.npy", allow_pickle=False) for rank in range(688)]),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    dataset = All688Dataset(records)
    j = load_module("qhead_full688_exact_j", J_SOURCE)
    q = load_module("qhead_full688_frozen_recipe", Q_SOURCE)
    if (
        tuple(q.SUFFIX) != SUFFIX
        or q.INPUTS != 22
        or q.HIDDEN != 64
        or q.SAMPLES_PER_LAYER != SAMPLES_PER_LAYER
        or q.NETWORKS_PER_STEP != NETWORKS_PER_STEP
        or q.LEARNING_RATE != LEARNING_RATE
        or q.WEIGHT_DECAY != WEIGHT_DECAY
        or q.SEED != MODEL_SEED
    ):
        raise AssertionError("frozen q-head recipe changed")
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    base = load_alpha_j(j)
    incumbent = load_incumbent(q, base)
    prepared = prepare_all(j, base, dataset, error)
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
        candidate, q_rms, trace = train_fixed(q, base, prepared, train, fold)
        scored = score_fold(q, candidate, incumbent, prepared, dataset, held)
        for row in scored["per_network"]:
            per_network[int(row["role_rank"])] = row
        scored_without_rows = {key: value for key, value in scored.items() if key != "per_network"}
        scored_without_rows["fold"] = fold
        scored_without_rows["training_rows"] = len(train)
        scored_without_rows["held_rows"] = held.tolist()
        fold_results.append(scored_without_rows)
        traces.extend(trace)
        q_rms_records.append(q_rms.tolist())
        print(json.dumps({"stage": "fold_score", **scored_without_rows}, sort_keys=True), flush=True)
        del candidate
        gc.collect()
    if any(row is None for row in per_network):
        raise AssertionError("OOF coverage is incomplete")
    rows = [row for row in per_network if row is not None]
    candidate_values = np.asarray([float(row["candidate_final_mse"]) for row in rows])
    incumbent_values = np.asarray([float(row["incumbent_final_mse"]) for row in rows])
    base_values = np.asarray([float(row["base_alpha075_j_final_mse"]) for row in rows])
    oracle_values = np.asarray([float(row["oracle_delta_q_final_mse"]) for row in rows])
    improvement = incumbent_values - candidate_values
    improvement_se = float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
    paired_z = float(np.mean(improvement) / improvement_se) if improvement_se > 0 else float("inf")
    candidate_mse = float(np.mean(candidate_values))
    incumbent_mse = float(np.mean(incumbent_values))
    relative_gain = (incumbent_mse - candidate_mse) / incumbent_mse
    improving_folds = sum(
        float(record["candidate_final_mse"]) < float(record["incumbent_final_mse"])
        for record in fold_results
    )
    network_win_fraction = float(np.mean(improvement > 0.0))
    fold_network_win_fractions = [
        float(np.mean(improvement[folds == fold] > 0.0)) for fold in range(FOLDS)
    ]
    protected_candidate = float(np.mean(candidate_values[legacy_protected]))
    protected_incumbent = float(np.mean(incumbent_values[legacy_protected]))
    gates = {
        "all_688_have_exactly_one_oof_prediction": True,
        "candidate_is_finite": bool(np.isfinite(candidate_values).all()),
        "relative_oof_gain_at_least_0_25pct": relative_gain >= MIN_RELATIVE_OOF_GAIN,
        "at_least_three_of_four_folds_improve": improving_folds >= MIN_IMPROVING_FOLDS,
        "paired_network_z_at_least_2": paired_z >= MIN_PAIRED_NETWORK_Z,
        "at_least_55pct_of_networks_improve": network_win_fraction >= MIN_NETWORK_WIN_FRACTION,
        "every_fold_has_at_least_50pct_network_wins": min(fold_network_win_fractions) >= MIN_FOLD_NETWORK_WIN_FRACTION,
        "legacy_protected40_oof_does_not_worsen": protected_candidate <= protected_incumbent,
        "rss_below_8_gib": enforce_rss("post crossfit") < MAX_RSS_BYTES,
    }
    promote = all(gates.values())
    final_trace: list[dict[str, object]] = []
    final_q_rms: list[float] | None = None
    model_hash: str | None = None
    if promote:
        final_model, q_rms, final_trace = train_fixed(q, base, prepared, ALL_RANKS, FOLDS)
        artifact = {
            "schema": "q24-native-j-q10q15-residual-head-checkpoint-v1",
            "state_dict": final_model.state_dict(),
            "selected_epoch": FIXED_EPOCH,
            "base_model_sha256": ALPHA_MODEL_SHA256,
            "refit_kind": "all688-fixed-epoch13-after-four-fold-oof-gate",
            "run_seal_sha256": sha256(SEAL),
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        }
        temporary_model = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
        torch.save(artifact, temporary_model)
        temporary_model.replace(MODEL)
        model_hash = sha256(MODEL)
        final_q_rms = q_rms.tolist()
    result = {
        "schema": "q24-qhead-full688-refit-discriminator-v1",
        "status": "PROMOTED_FIXED_FULL688_MODEL" if promote else "CLOSED_NO_MATERIAL_OOF_GAIN",
        "access_declarations": {
            "role": "observer_fit",
            "observer_fit_targets_opened": ["mlp_id", "all_layer_means"],
            "observer_selection_targets_opened": [],
            "names_opened": [],
            "board_opened": False,
            "scores_opened": [],
            "network": False,
            "submission": False,
        },
        "fixed_rule": {
            "architecture": "22->64->64 with six layer-specific scalar heads",
            "features_and_loss": "exact frozen q-head source authority",
            "base": "frozen alpha=.75 native J",
            "folds": FOLDS,
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
            "fixed_checkpoint_epoch": FIXED_EPOCH,
            "checkpoint_averaging": "none",
            "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
            "checkpoint_selection": False,
        },
        "crossfit": {
            "networks": 688,
            "fold_sizes": [172] * 4,
            "candidate_oof_final_mse": candidate_mse,
            "incumbent_final_mse_on_same_rows": incumbent_mse,
            "relative_oof_gain": relative_gain,
            "absolute_oof_gain": incumbent_mse - candidate_mse,
            "base_alpha075_j_final_mse": float(np.mean(base_values)),
            "oracle_delta_q_final_mse": float(np.mean(oracle_values)),
            "paired_network_improvement_standard_error": improvement_se,
            "paired_network_z": paired_z,
            "network_win_fraction": network_win_fraction,
            "fold_network_win_fractions": fold_network_win_fractions,
            "improving_folds": improving_folds,
            "fold_results": fold_results,
            "legacy_protected40_oof": {
                "role_ranks": legacy_protected.tolist(),
                "candidate_final_mse": protected_candidate,
                "incumbent_final_mse": protected_incumbent,
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
            "future_deployment_must_use_d1_or_validated_cap_base": True,
            "observed_g11b_board_shift_allowance_factor": OBSERVED_G11B_BOARD_SHIFT_FACTOR,
        },
        "training": {
            "crossfit_q_rms": q_rms_records,
            "crossfit_trace": traces,
            "final_fit_started_only_after_gate": promote,
            "final_fit_q_rms": final_q_rms,
            "final_fit_trace": final_trace,
        },
        "resource": {
            "native_row_storage_bytes": native_storage,
            "maximum_process_rss_bytes": enforce_rss("result"),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "bindings": {
            "run_seal_sha256": sha256(SEAL),
            "preregistration_sha256": seal["preregistration_sha256"],
            "program_sha256": seal["program_sha256"],
            "j_source_sha256": J_SOURCE_SHA256,
            "q_source_sha256": Q_SOURCE_SHA256,
            "alpha_model_sha256": ALPHA_MODEL_SHA256,
            "incumbent_q_model_sha256": INCUMBENT_Q_MODEL_SHA256,
            "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
            "observer_fit_targets_sha256": TARGETS_SHA256,
            "native_row_aggregate_sha256": NATIVE_AGGREGATE_SHA256,
            "promoted_model_sha256": model_hash,
        },
    }
    write_json_once(RESULT, result)
    replace_json(
        RUN_STATE,
        {
            "schema": "q24-qhead-full688-refit-run-state-v1",
            "status": "COMPLETE",
            "result_sha256": sha256(RESULT),
            "promoted_model_sha256": model_hash,
        },
    )
    print(json.dumps({"status": result["status"], "crossfit": result["crossfit"], "model": model_hash}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
