#!/usr/bin/env python3
"""Frozen 160/40 transfer test of exact J on Q24-native raw20/sigma."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
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
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"
J_SOURCE = J_DIR / "run_experiment.py"
J_CV_RESULT = J_DIR / "CV_RESULTS.json"
INSTRUMENTED = HERE / "instrumented_estimator.py"
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
FEATURE_MANIFEST = FEATURE_STORE / "manifest.json"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
NATIVE_ROWS = HERE / "runs/native_rows_v1"
GENERATION_SEAL = HERE / "GENERATION_SEAL.json"
PREREGISTRATION = HERE / "PILOT_PREREGISTRATION.json"
RUN_MANIFEST = HERE / "PILOT_RUN_MANIFEST.json"
SEAL = HERE / "PILOT_SEAL.json"
RESULT = HERE / "PILOT_RESULT.json"
MODEL = HERE / "PILOT_MODEL.pt"

J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
J_CV_RESULT_SHA256 = "70a0917258a2d656ab9593a67b3a04483af153d8ef59162be9570a6a62b9b5b4"
INSTRUMENTED_SHA256 = "f539889324fb82cc1ded35a41c8b5a359014d9ddac52eab62949257e8e58a0ef"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
TARGETS_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
GENERATION_SEAL_SHA256 = "2310cfeacb3203a462492f0963568a8bfdff9fdaea3526658da9fd15ab226d4c"
ROSTER = tuple(range(35, 235))
EXPECTED_VALID_GLOBAL = (
    37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101, 122, 130,
    132, 135, 139, 145, 146, 149, 150, 151, 155, 159, 163, 169, 173, 175,
    181, 192, 198, 199, 204, 219, 224, 227, 233, 234,
)
MODEL_SEED = 1_753_358_478
SHUFFLE_STREAM_ID = 40
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_STORAGE_BYTES = 1 << 30


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


def verify_seal() -> dict[str, object]:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "frozen-after-native-meter-before-pilot-target-open":
        raise AssertionError("pilot seal status changed")
    expected = {
        "preregistration_sha256": sha256(PREREGISTRATION),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "program_sha256": sha256(Path(__file__)),
        "j_source_sha256": sha256(J_SOURCE),
        "j_cv_result_sha256": sha256(J_CV_RESULT),
        "instrumented_source_sha256": sha256(INSTRUMENTED),
        "feature_manifest_sha256": sha256(FEATURE_MANIFEST),
        "target_bundle_sha256": sha256(TARGETS),
        "generation_seal_sha256": sha256(GENERATION_SEAL),
    }
    for key, observed in expected.items():
        if seal.get(key) != observed:
            raise AssertionError(f"pilot seal changed: {key}: {observed}")
    literals = {
        "j_source_sha256": J_SOURCE_SHA256,
        "j_cv_result_sha256": J_CV_RESULT_SHA256,
        "instrumented_source_sha256": INSTRUMENTED_SHA256,
        "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
        "target_bundle_sha256": TARGETS_SHA256,
        "generation_seal_sha256": GENERATION_SEAL_SHA256,
    }
    for key, expected_value in literals.items():
        if expected[key] != expected_value:
            raise AssertionError(f"pilot literal binding changed: {key}")
    return seal


def verify_native_rows() -> tuple[list[dict[str, object]], str, int]:
    records: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    storage = 0
    for rank in ROSTER:
        row_dir = NATIVE_ROWS / f"row_{rank:04d}"
        result_path = row_dir / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"native generation incomplete at rank {rank}")
        receipt = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "complete-target-free"
            or receipt.get("role") != "observer_fit"
            or int(receipt.get("role_rank", -1)) != rank
            or receipt.get("target_columns_opened") != []
            or receipt.get("observer_selection_opened") is not False
            or receipt.get("name_or_board_join") is not False
            or receipt.get("instrumented_source_sha256") != INSTRUMENTED_SHA256
            or receipt.get("generation_seal_sha256") != GENERATION_SEAL_SHA256
        ):
            raise AssertionError(f"native row receipt contract failed at rank {rank}")
        for name in ("host.npy", "raw20.npy", "sigma.npy"):
            path = row_dir / name
            record = receipt["files"][name]
            if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise AssertionError(f"native payload binding failed: rank {rank} {name}")
            storage += path.stat().st_size
        result_hash = sha256(result_path)
        aggregate.update(rank.to_bytes(4, "big"))
        aggregate.update(result_hash.encode("ascii"))
        records.append(receipt)
    if storage >= MAX_STORAGE_BYTES:
        raise OSError(f"native store exceeds pilot storage bound: {storage}")
    return records, aggregate.hexdigest(), storage


def protected_split(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    keyed = []
    for local, receipt in enumerate(records):
        mlp_id = int(receipt["mlp_id"])
        digest = hashlib.blake2s(
            f"p2r0f2j-q24-native-v1:{mlp_id}".encode("ascii"), digest_size=16
        ).digest()
        keyed.append((digest, local))
    valid = np.asarray(sorted(local for _, local in sorted(keyed)[:40]), dtype=np.int64)
    train = np.asarray(sorted(set(range(200)) - set(valid.tolist())), dtype=np.int64)
    valid_global = tuple(ROSTER[int(local)] for local in valid)
    if valid_global != EXPECTED_VALID_GLOBAL or len(train) != 160:
        raise AssertionError("protected split changed")
    return train, valid


def load_roster_targets(records: list[dict[str, object]]) -> np.ndarray:
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)
    if ids.dtype != np.int32 or targets.dtype != np.float32:
        raise AssertionError("observer_fit target schema changed")
    expected_ids = np.asarray([int(record["mlp_id"]) for record in records], dtype=np.int32)
    if not np.array_equal(ids[35:235], expected_ids):
        raise AssertionError("target/native identity alignment changed")
    return np.ascontiguousarray(targets[35:235], dtype=np.float32)


@dataclass(frozen=True)
class NativeJoinedRow:
    raw_locals: np.ndarray
    sigma: np.ndarray
    weights_path: Path


class NativePilotDataset:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.mlp_ids = np.asarray([int(record["mlp_id"]) for record in records], dtype=np.int64)

    def load(self, local: int) -> NativeJoinedRow:
        rank = ROSTER[int(local)]
        row_dir = NATIVE_ROWS / f"row_{rank:04d}"
        feature_dir = FEATURE_STORE / f"rows/observer_fit/{rank:04d}"
        return NativeJoinedRow(
            raw_locals=np.load(row_dir / "raw20.npy", mmap_mode="r", allow_pickle=False),
            sigma=np.load(row_dir / "sigma.npy", mmap_mode="r", allow_pickle=False),
            weights_path=feature_dir / "weights.npy",
        )


def main() -> None:
    seal = verify_seal()
    if RESULT.exists() or MODEL.exists():
        raise FileExistsError("pilot output already exists")
    records, native_aggregate, native_storage = verify_native_rows()
    train, valid = protected_split(records)
    # This is the first target read in the runner, after complete native-row coverage.
    targets = load_roster_targets(records)
    hosts = np.ascontiguousarray(
        np.stack(
            [
                np.load(NATIVE_ROWS / f"row_{rank:04d}/host.npy", allow_pickle=False)
                for rank in ROSTER
            ]
        ),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    dataset = NativePilotDataset(records)
    sums = np.zeros((200, 16, 19), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    for local, rank in enumerate(ROSTER):
        raw = np.load(
            NATIVE_ROWS / f"row_{rank:04d}/raw20.npy", mmap_mode="r", allow_pickle=False
        )
        tail = np.asarray(raw[:, :, 1:], dtype=np.float64)
        sums[local] = tail.sum(axis=1, dtype=np.float64)
        sums2[local] = np.square(tail).sum(axis=1, dtype=np.float64)

    j = load_module("p2r0f2j_q24_native_exact_j", J_SOURCE)
    j.configure()
    centers, scales, layer_rms = j.moments_for(train, error, sums, sums2)
    torch.manual_seed(MODEL_SEED)
    model = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    if j.parameter_count(model) != 25_600:
        raise AssertionError("exact J parameter count changed")
    started = time.perf_counter()
    trace, training_max_rss = j.fit_fold(
        SHUFFLE_STREAM_ID, model, dataset, train, error
    )
    autonomous = j.score(model, dataset, valid, error)
    host_squared = np.square(error[valid].astype(np.float64))
    host_layer = np.mean(host_squared, axis=(0, 2), dtype=np.float64)
    host_network_final = np.mean(host_squared[:, -1], axis=1, dtype=np.float64)
    candidate_by_local = {
        int(row["role_rank"]): float(row["final_mse"])
        for row in autonomous["per_network"]
    }
    wins = sum(
        candidate_by_local[int(local)] < float(host_value)
        for local, host_value in zip(valid, host_network_final, strict=True)
    )
    candidate_final = float(autonomous["final_mse"])
    host_final = float(host_layer[-1])
    maximum_network = max(float(row["final_mse"]) for row in autonomous["per_network"])
    finite = bool(autonomous["finite_full_rollout"])
    gate = {
        "finite_full_rollout": finite,
        "validation_final_mse_below_5e_8": candidate_final < 5.0e-8,
        "validation_ratio_to_host_below_0_6": candidate_final / host_final < 0.6,
        "networkwise_wins_at_least_28": wins >= 28,
        "maximum_network_final_mse_below_5e_7": maximum_network < 5.0e-7,
    }
    gate["conjunctive_pass"] = all(gate.values())
    max_rss = max(training_max_rss, resident_bytes())
    if max_rss >= MAX_RSS_BYTES:
        raise MemoryError(f"pilot RSS exceeded bound: {max_rss}")

    artifact = {
        "schema": "p2r0f2j-q24-native-transfer-pilot-model-v1",
        "state_dict": model.state_dict(),
        "model_seed": MODEL_SEED,
        "shuffle_stream_id": SHUFFLE_STREAM_ID,
        "pilot_preregistration_sha256": seal["preregistration_sha256"],
        "pilot_program_sha256": seal["program_sha256"],
        "native_row_aggregate_sha256": native_aggregate,
    }
    temporary_model = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary_model)
    temporary_model.replace(MODEL)
    result: dict[str, object] = {
        "schema": "p2r0f2j-q24-native-transfer-pilot-result-v1",
        "status": "complete",
        "node": "P2R0F2J_NATIVE",
        "observer_fit_networks": 200,
        "train_networks": 160,
        "validation_networks": 40,
        "train_role_ranks": [ROSTER[int(local)] for local in train],
        "validation_role_ranks": [ROSTER[int(local)] for local in valid],
        "observer_selection_targets_opened": 0,
        "name_or_board_join": False,
        "architecture": {
            "authority": "exact P2R0F2J source",
            "channels": 4,
            "hidden": 96,
            "input_features": 36,
            "parameters": 25600,
            "deployment_truth_inputs": 0,
        },
        "training": {
            "model_seed": MODEL_SEED,
            "shuffle_stream_id": SHUFFLE_STREAM_ID,
            "epochs": 6,
            "batch_size_networks": 3,
            "fixed_final_checkpoint": True,
            "trace": trace,
        },
        "validation": {
            "host_final_mse": host_final,
            "host_layer_mse": host_layer.tolist(),
            "autonomous_final_mse": candidate_final,
            "autonomous_layer_mse": autonomous["layer_mse"],
            "autonomous_ratio_to_host": candidate_final / host_final,
            "maximum_per_network_final_mse": maximum_network,
            "networkwise_wins": wins,
            "networks": 40,
        },
        "decision_gate": gate,
        "resource": {
            "maximum_process_rss_bytes": max_rss,
            "native_row_storage_bytes": native_storage,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "bindings": {
            "pilot_seal_sha256": sha256(SEAL),
            "preregistration_sha256": seal["preregistration_sha256"],
            "run_manifest_sha256": seal["run_manifest_sha256"],
            "program_sha256": seal["program_sha256"],
            "j_source_sha256": J_SOURCE_SHA256,
            "j_cv_result_sha256": J_CV_RESULT_SHA256,
            "instrumented_source_sha256": INSTRUMENTED_SHA256,
            "generation_seal_sha256": GENERATION_SEAL_SHA256,
            "target_bundle_sha256": TARGETS_SHA256,
            "native_row_aggregate_sha256": native_aggregate,
            "model_sha256": sha256(MODEL),
        },
    }
    write_json_once(RESULT, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "validation": result["validation"],
                "gate": gate,
                "model_sha256": result["bindings"]["model_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
