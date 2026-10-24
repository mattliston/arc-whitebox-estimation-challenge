#!/Users/matt/anaconda3/bin/python3
"""Frozen fixed-checkpoint exact-J final fit on all 688 Q24-native rows."""

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
PILOT = P2 / "public_q24_native_transport_energy_pilot_v1"
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"
J_SOURCE = J_DIR / "run_experiment.py"
J_CV_RESULT = J_DIR / "CV_RESULTS.json"
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
FEATURE_MANIFEST = FEATURE_STORE / "manifest.json"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
PILOT_ROWS = PILOT / "runs/native_rows_v1"
FINAL_ROWS = HERE / "runs/native_rows_remaining_v1"
PILOT_GENERATION_SEAL = PILOT / "GENERATION_SEAL.json"
FINAL_GENERATION_SEAL = HERE / "GENERATION_SEAL.json"
PILOT_RESULT = PILOT / "PILOT_RESULT.json"
PREREGISTRATION = HERE / "FINAL_FIT_PREREGISTRATION.json"
RUN_MANIFEST = HERE / "FINAL_FIT_RUN_MANIFEST.json"
SEAL = HERE / "FINAL_FIT_SEAL.json"
RESULT = HERE / "FINAL_FIT_RESULT.json"
MODEL = HERE / "FINAL_MODEL.pt"

J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
J_CV_RESULT_SHA256 = "70a0917258a2d656ab9593a67b3a04483af153d8ef59162be9570a6a62b9b5b4"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
TARGETS_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
PILOT_GENERATION_SEAL_SHA256 = "2310cfeacb3203a462492f0963568a8bfdff9fdaea3526658da9fd15ab226d4c"
FINAL_GENERATION_SEAL_SHA256 = "8959fd274df3120003b71229a5453b5ee2bc1c875c4e9a139d2e72d104135a6b"
PILOT_RESULT_SHA256 = "7ecdc3438ddd4934853134ef0002b517e8111a2ecbf3a0e9abdafd0efb7e89a7"
INSTRUMENTED_SHA256 = "f539889324fb82cc1ded35a41c8b5a359014d9ddac52eab62949257e8e58a0ef"
ALL_RANKS = tuple(range(688))
MODEL_SEED = 1_753_358_478
SHUFFLE_STREAM_ID = 40
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_NATIVE_STORAGE_BYTES = 2 * (1 << 30)


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
    if seal.get("status") != "frozen-after-transfer-pass-before-all688-target-open":
        raise AssertionError("final-fit seal status changed")
    expected = {
        "preregistration_sha256": sha256(PREREGISTRATION),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "program_sha256": sha256(Path(__file__)),
        "j_source_sha256": sha256(J_SOURCE),
        "j_cv_result_sha256": sha256(J_CV_RESULT),
        "feature_manifest_sha256": sha256(FEATURE_MANIFEST),
        "target_bundle_sha256": sha256(TARGETS),
        "pilot_generation_seal_sha256": sha256(PILOT_GENERATION_SEAL),
        "final_generation_seal_sha256": sha256(FINAL_GENERATION_SEAL),
        "pilot_result_sha256": sha256(PILOT_RESULT),
    }
    for key, observed in expected.items():
        if seal.get(key) != observed:
            raise AssertionError(f"final-fit seal changed: {key}: {observed}")
    literals = {
        "j_source_sha256": J_SOURCE_SHA256,
        "j_cv_result_sha256": J_CV_RESULT_SHA256,
        "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
        "target_bundle_sha256": TARGETS_SHA256,
        "pilot_generation_seal_sha256": PILOT_GENERATION_SEAL_SHA256,
        "final_generation_seal_sha256": FINAL_GENERATION_SEAL_SHA256,
        "pilot_result_sha256": PILOT_RESULT_SHA256,
    }
    for key, expected_value in literals.items():
        if expected[key] != expected_value:
            raise AssertionError(f"final-fit literal binding changed: {key}")
    pilot = json.loads(PILOT_RESULT.read_text(encoding="utf-8"))
    if not pilot.get("decision_gate", {}).get("conjunctive_pass"):
        raise RuntimeError("protected transfer did not pass")
    return seal


def row_dir(rank: int) -> Path:
    if 35 <= rank < 235:
        return PILOT_ROWS / f"row_{rank:04d}"
    return FINAL_ROWS / f"row_{rank:04d}"


def verify_native_rows() -> tuple[list[dict[str, object]], str, int]:
    records: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    storage = 0
    for rank in ALL_RANKS:
        directory = row_dir(rank)
        result_path = directory / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"all-688 native generation incomplete at rank {rank}")
        receipt = json.loads(result_path.read_text(encoding="utf-8"))
        expected_generation_seal = (
            PILOT_GENERATION_SEAL_SHA256
            if 35 <= rank < 235
            else FINAL_GENERATION_SEAL_SHA256
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
            raise AssertionError(f"native row receipt contract failed at rank {rank}")
        for name in ("host.npy", "raw20.npy", "sigma.npy"):
            path = directory / name
            record = receipt["files"][name]
            if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise AssertionError(f"native payload binding failed: rank {rank} {name}")
            storage += path.stat().st_size
        result_hash = sha256(result_path)
        aggregate.update(rank.to_bytes(4, "big"))
        aggregate.update(result_hash.encode("ascii"))
        records.append(receipt)
    if len(records) != 688:
        raise AssertionError("all-688 native coverage changed")
    mlp_ids = [int(record["mlp_id"]) for record in records]
    if len(set(mlp_ids)) != 688:
        raise AssertionError("all-688 native mlp_id roster is not unique")
    if storage >= MAX_NATIVE_STORAGE_BYTES:
        raise OSError(f"all-688 native store exceeds bound: {storage}")
    return records, aggregate.hexdigest(), storage


def load_targets(records: list[dict[str, object]]) -> np.ndarray:
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
    if not np.array_equal(ids, expected_ids):
        raise AssertionError("target/native all-688 identity alignment changed")
    return np.ascontiguousarray(targets, dtype=np.float32)


@dataclass(frozen=True)
class NativeJoinedRow:
    raw_locals: np.ndarray
    sigma: np.ndarray
    weights_path: Path


class All688NativeDataset:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.mlp_ids = np.asarray(
            [int(record["mlp_id"]) for record in records], dtype=np.int64
        )

    def load(self, rank: int) -> NativeJoinedRow:
        directory = row_dir(int(rank))
        feature_dir = FEATURE_STORE / f"rows/observer_fit/{int(rank):04d}"
        return NativeJoinedRow(
            raw_locals=np.load(directory / "raw20.npy", mmap_mode="r", allow_pickle=False),
            sigma=np.load(directory / "sigma.npy", mmap_mode="r", allow_pickle=False),
            weights_path=feature_dir / "weights.npy",
        )


def main() -> None:
    seal = verify_seal()
    if RESULT.exists() or MODEL.exists():
        raise FileExistsError("final fit output already exists")
    records, native_aggregate, native_storage = verify_native_rows()
    # First and only target open occurs after all 688 target-free row receipts
    # and payload hashes have passed above.
    targets = load_targets(records)
    hosts = np.ascontiguousarray(
        np.stack(
            [np.load(row_dir(rank) / "host.npy", allow_pickle=False) for rank in ALL_RANKS]
        ),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    dataset = All688NativeDataset(records)
    sums = np.zeros((688, 16, 19), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    for rank in ALL_RANKS:
        raw = np.load(row_dir(rank) / "raw20.npy", mmap_mode="r", allow_pickle=False)
        tail = np.asarray(raw[:, :, 1:], dtype=np.float64)
        sums[rank] = tail.sum(axis=1, dtype=np.float64)
        sums2[rank] = np.square(tail).sum(axis=1, dtype=np.float64)

    j = load_module("p2r0f2j_q24_native_final_exact_j", J_SOURCE)
    j.configure()
    train = np.arange(688, dtype=np.int64)
    centers, scales, layer_rms = j.moments_for(train, error, sums, sums2)
    torch.manual_seed(MODEL_SEED)
    model = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    if j.parameter_count(model) != 25_600:
        raise AssertionError("exact J parameter count changed")
    expected_state_keys = {
        "source_weight", "source_bias", "gate_weight", "gate_bias",
        "transport_delta", "mix_delta", "centers", "scales", "layer_rms",
        "trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias"
    }
    if set(model.state_dict()) != expected_state_keys:
        raise AssertionError("exact exporter-bound J state schema changed")
    started = time.perf_counter()
    trace, training_max_rss = j.fit_fold(
        SHUFFLE_STREAM_ID, model, dataset, train, error
    )
    # Fixed epoch-6 artifact is already determined.  This in-sample diagnostic
    # is recorded after training and performs no model/checkpoint selection.
    autonomous = j.score(model, dataset, train, error)
    host_layer = np.mean(
        np.square(error.astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    max_rss = max(training_max_rss, resident_bytes())
    if max_rss >= MAX_RSS_BYTES:
        raise MemoryError(f"final-fit RSS exceeded bound: {max_rss}")
    artifact = {
        "schema": "p2r0f2j-q24-native-all688-final-model-v1",
        "state_dict": model.state_dict(),
        "model_seed": MODEL_SEED,
        "shuffle_stream_id": SHUFFLE_STREAM_ID,
        "training_rows": 688,
        "fixed_epoch": 6,
        "final_fit_preregistration_sha256": seal["preregistration_sha256"],
        "final_fit_program_sha256": seal["program_sha256"],
        "native_row_aggregate_sha256": native_aggregate,
    }
    temporary_model = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary_model)
    temporary_model.replace(MODEL)
    result = {
        "schema": "p2r0f2j-q24-native-all688-final-fit-result-v1",
        "status": "complete-fixed-epoch6",
        "node": "P2R0H7B",
        "observer_fit_networks": 688,
        "training_networks": 688,
        "validation_networks": 0,
        "observer_selection_targets_opened": 0,
        "name_or_board_join": False,
        "checkpoint_selection": False,
        "architecture": {
            "authority": "exact P2R0F2J source",
            "channels": 4,
            "hidden": 96,
            "input_features": 36,
            "parameters": 25600,
            "deployment_truth_inputs": 0
        },
        "training": {
            "model_seed": MODEL_SEED,
            "shuffle_stream_id": SHUFFLE_STREAM_ID,
            "epochs": 6,
            "batch_size_networks": 3,
            "fixed_final_checkpoint": True,
            "trace": trace
        },
        "postfit_in_sample_diagnostic_not_used_for_selection": {
            "host_final_mse": float(host_layer[-1]),
            "autonomous_final_mse": float(autonomous["final_mse"]),
            "autonomous_ratio_to_host": float(autonomous["final_mse"] / host_layer[-1]),
            "finite_full_rollout": bool(autonomous["finite_full_rollout"]),
            "maximum_absolute_correction": float(autonomous["maximum_absolute_correction"])
        },
        "resource": {
            "maximum_process_rss_bytes": max_rss,
            "native_row_storage_bytes": native_storage,
            "elapsed_seconds": time.perf_counter() - started
        },
        "bindings": {
            "final_fit_seal_sha256": sha256(SEAL),
            "preregistration_sha256": seal["preregistration_sha256"],
            "run_manifest_sha256": seal["run_manifest_sha256"],
            "program_sha256": seal["program_sha256"],
            "j_source_sha256": J_SOURCE_SHA256,
            "pilot_result_sha256": PILOT_RESULT_SHA256,
            "pilot_generation_seal_sha256": PILOT_GENERATION_SEAL_SHA256,
            "final_generation_seal_sha256": FINAL_GENERATION_SEAL_SHA256,
            "target_bundle_sha256": TARGETS_SHA256,
            "native_row_aggregate_sha256": native_aggregate,
            "model_sha256": sha256(MODEL)
        }
    }
    write_json_once(RESULT, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "diagnostic": result["postfit_in_sample_diagnostic_not_used_for_selection"],
                "model_sha256": result["bindings"]["model_sha256"]
            },
            sort_keys=True
        ),
        flush=True
    )


if __name__ == "__main__":
    main()
