#!/Users/matt/anaconda3/bin/python3
"""Frozen one-shot protected 160/40 D3 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUN_MANIFEST = HERE / "RUN_MANIFEST.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
HOST_HELPER = HERE / "generate_host_row.py"
INSTRUMENTED = HERE / "instrumented_estimator.py"
SOURCE_HOST = P2 / "cp3_cap16384_mean_recurrence_ablation/estimator.py"
DYNAMIC4 = P2 / "public_full688_dynamic_multistate_residual_cv_v1/run_experiment.py"
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
FEATURE_MANIFEST = FEATURE_STORE / "manifest.json"
TARGET_BUNDLE = P2 / "data_gate/sealed_role_target_bundle"
TARGET_MANIFEST = TARGET_BUNDLE / "manifest.json"
TARGET_PARQUET = TARGET_BUNDLE / "observer_fit.parquet"
TARGET_AUTHORIZATION = P2 / "public_observer_fit_truth_scan_v1/TARGET_ACCESS_AUTHORIZATION.json"
METER_RESULT = HERE / "runs/integrated_meter_rank0000_v1/RESULT.json"
PREFLIGHT_ROW = HERE / "runs/integrated_meter_rank0000_v1"
HOST_PYTHON = P2 / "data_gate/.venv/bin/python"

RUN_ROOT = HERE / "runs/protected160_40_v1"
HOST_ROOT = RUN_ROOT / "hosts"
HOSTS_COMPLETE = RUN_ROOT / "HOSTS_COMPLETE.json"
TARGET_CLAIM = RUN_ROOT / "TARGET_OPEN_CLAIM.json"
MODEL_PATH = RUN_ROOT / "MODEL.pt"
RESULT_PATH = RUN_ROOT / "RESULT.json"

PILOT_ROWS = 200
TRAIN = np.arange(0, 160, dtype=np.int64)
EVAL = np.arange(160, 200, dtype=np.int64)
DEPTH = 16
WIDTH = 1024
LOCAL_FEATURES = 20
INTEGRATED_HOST_TAPE_FLOPS = 620_447_502_338
POST_TAPE_CORRECTION_UPPER = 652_201_984
TOTAL_UPPER = 621_099_704_322
PHASE2_BUDGET = 2**41
SCORE_MULTIPLIER = TOTAL_UPPER / PHASE2_BUDGET
BREAK_EVEN_RAW = 3.5405318023013399466359567564424759803548750915614e-8

INSTRUMENTED_SHA256 = (
    "76a87a1d161db9583dd318971b2688079d1919579d7a900fa9ec1a0230318105"
)
SOURCE_HOST_SHA256 = (
    "361c87cba008987c117871bf9892beecce75911634798ed7ad3fa26fd8ee5814"
)
DYNAMIC4_SHA256 = (
    "8871ac10f3f122d9231652e3618c888bdc69b71cc0de44ffa00a79630258225d"
)
FEATURE_MANIFEST_SHA256 = (
    "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
)
TARGET_MANIFEST_SHA256 = (
    "2c0ea0bb09d0c86d46ca07fdde3563c4cda046a1fff152cdff26fde965e1f7e6"
)
TARGET_PARQUET_SHA256 = (
    "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
)
TARGET_AUTHORIZATION_SHA256 = (
    "6a8eb53a567419a2d829822ae1747243b878f889735f1830d421ca98d4c15c7a"
)
METER_RESULT_SHA256 = (
    "6d7afc4af1e8a9a77d7e3611f56830e19955ed917775b9c93a225a419917826c"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_seal() -> dict[str, object]:
    seal = read_json(SEAL)
    if seal.get("status") != "frozen-before-host-generation-and-target-metrics":
        raise AssertionError("pilot seal status changed")
    paths = {
        "preregistration_sha256": PREREGISTRATION,
        "run_manifest_sha256": RUN_MANIFEST,
        "program_sha256": Path(__file__),
        "host_helper_sha256": HOST_HELPER,
        "instrumented_source_sha256": INSTRUMENTED,
        "exact_host_source_sha256": SOURCE_HOST,
        "dynamic4_source_sha256": DYNAMIC4,
        "feature_manifest_sha256": FEATURE_MANIFEST,
        "target_manifest_sha256": TARGET_MANIFEST,
        "target_parquet_sha256": TARGET_PARQUET,
        "target_authorization_sha256": TARGET_AUTHORIZATION,
        "meter_result_sha256": METER_RESULT,
    }
    for key, path in paths.items():
        observed = sha256(path)
        if seal.get(key) != observed:
            raise AssertionError(f"frozen pilot binding changed: {key}: {observed}")
    prereg = read_json(PREREGISTRATION)
    if prereg.get("status") != "frozen-before-host-generation-and-target-metrics":
        raise AssertionError("pilot preregistration changed")
    if prereg["split"]["train_role_ranks"] != [0, 159]:
        raise AssertionError("train split changed")
    if prereg["split"]["eval_role_ranks"] != [160, 199]:
        raise AssertionError("eval split changed")
    if float(prereg["decision_gate"]["raw_mse_strictly_below"]) != BREAK_EVEN_RAW:
        raise AssertionError("break-even raw gate changed")
    return seal


def feature_entry(rank: int) -> tuple[dict[str, object], Path, dict[str, object]]:
    manifest = read_json(FEATURE_MANIFEST)
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    entry = entries[rank]
    if int(entry["role_rank"]) != rank:
        raise AssertionError(f"rank {rank}: feature roster changed")
    path = FEATURE_STORE / str(entry["path"])
    if sha256(path) != str(entry["sha256"]):
        raise AssertionError(f"rank {rank}: row manifest changed")
    row = read_json(path)
    if row.get("role") != "observer_fit" or int(row.get("role_rank", -1)) != rank:
        raise AssertionError(f"rank {rank}: feature role changed")
    if int(row.get("mlp_id", -1)) != int(entry["mlp_id"]):
        raise AssertionError(f"rank {rank}: feature identity changed")
    return entry, path, row


def row_paths(rank: int) -> tuple[Path, Path, Path, Path]:
    if rank == 0:
        return (
            PREFLIGHT_ROW / "host.npy",
            PREFLIGHT_ROW / "raw20.npy",
            PREFLIGHT_ROW / "sigma.npy",
            PREFLIGHT_ROW / "RESULT.json",
        )
    directory = HOST_ROOT / f"row_{rank:04d}"
    return (
        directory / "host.npy",
        directory / "raw20.npy",
        directory / "sigma.npy",
        directory / "result.json",
    )


def validate_host_row(rank: int) -> dict[str, object]:
    host_path, raw_path, sigma_path, result_path = row_paths(rank)
    if not all(path.is_file() for path in (host_path, raw_path, sigma_path, result_path)):
        raise FileNotFoundError(f"rank {rank}: incomplete integrated host row")
    result = read_json(result_path)
    if rank == 0:
        if result.get("status") != "pass-admitted-before-target-fit":
            raise AssertionError("rank 0 admission result changed")
        meter_flops = int(result["literal_integrated_meter"]["flops_used"])
    else:
        if result.get("status") != "complete":
            raise AssertionError(f"rank {rank}: host status changed")
        if int(result.get("role_rank", -1)) != rank:
            raise AssertionError(f"rank {rank}: host role rank changed")
        if result.get("instrumented_source_sha256") != INSTRUMENTED_SHA256:
            raise AssertionError(f"rank {rank}: instrumented source changed")
        meter_flops = int(result["literal_meter"]["flops_used"])
    if meter_flops != INTEGRATED_HOST_TAPE_FLOPS:
        raise AssertionError(f"rank {rank}: integrated FLOPs changed")
    host = np.load(host_path, mmap_mode="r", allow_pickle=False)
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    sigma = np.load(sigma_path, mmap_mode="r", allow_pickle=False)
    if host.dtype != np.float32 or host.shape != (DEPTH, WIDTH):
        raise AssertionError(f"rank {rank}: host schema changed")
    if raw.dtype != np.float32 or raw.shape != (DEPTH, WIDTH, LOCAL_FEATURES):
        raise AssertionError(f"rank {rank}: raw20 schema changed")
    if sigma.dtype != np.float32 or sigma.shape != (DEPTH, WIDTH):
        raise AssertionError(f"rank {rank}: sigma schema changed")
    return result


def prepare_hosts(seal: dict[str, object]) -> dict[str, object]:
    if TARGET_CLAIM.exists() or MODEL_PATH.exists() or RESULT_PATH.exists():
        raise FileExistsError("target/model artifact exists before host preparation")
    HOST_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for rank in range(1, PILOT_ROWS):
        directory = HOST_ROOT / f"row_{rank:04d}"
        if directory.exists():
            validate_host_row(rank)
        else:
            subprocess.run(
                [
                    str(HOST_PYTHON),
                    str(HOST_HELPER),
                    "--rank",
                    str(rank),
                    "--output-dir",
                    str(directory),
                ],
                check=True,
                cwd=str(HERE.parents[4]),
            )
        if (rank + 1) % 25 == 0 or rank + 1 == PILOT_ROWS:
            print(f"integrated hosts {rank + 1}/{PILOT_ROWS}", flush=True)
    records: list[dict[str, object]] = []
    for rank in range(PILOT_ROWS):
        result = validate_host_row(rank)
        _, _, _, result_path = row_paths(rank)
        entry, _, _ = feature_entry(rank)
        records.append(
            {
                "role_rank": rank,
                "mlp_id": int(entry["mlp_id"]),
                "result_sha256": sha256(result_path),
                "host_source": "admission-row-reuse" if rank == 0 else "integrated-row-helper",
                "flops_used": (
                    int(result["literal_integrated_meter"]["flops_used"])
                    if rank == 0
                    else int(result["literal_meter"]["flops_used"])
                ),
            }
        )
    expected: dict[str, object] = {
        "schema": "p2r0c1d3-protected200-host-completion-v1",
        "status": "complete-before-target-open",
        "role": "observer_fit",
        "ordered_role_ranks": [0, PILOT_ROWS - 1],
        "rows": PILOT_ROWS,
        "ordered_result_digest": canonical_sha256(records),
        "records": records,
        "integrated_host_tape_flops_per_row": INTEGRATED_HOST_TAPE_FLOPS,
        "meter_result_sha256": METER_RESULT_SHA256,
        "instrumented_source_sha256": INSTRUMENTED_SHA256,
        "pilot_seal_sha256": sha256(SEAL),
        "program_sha256": seal["program_sha256"],
        "target_roles_opened": [],
        "target_columns_opened": [],
        "name_or_board_join": False,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    if HOSTS_COMPLETE.exists():
        observed = read_json(HOSTS_COMPLETE)
        stable_keys = set(expected) - {"elapsed_seconds_this_invocation"}
        if any(observed.get(key) != expected.get(key) for key in stable_keys):
            raise AssertionError("host completion receipt changed")
        return observed
    write_json_once(HOSTS_COMPLETE, expected)
    return expected


def load_targets_once(expected_ids: np.ndarray) -> np.ndarray:
    authorization = read_json(TARGET_AUTHORIZATION)
    if not (
        authorization.get("authorized") is True
        and authorization.get("allowed_roles") == ["observer_fit"]
        and authorization.get("allowed_columns") == ["mlp_id", "all_layer_means"]
        and authorization.get("raw_full_dataset_access") == "forbidden"
    ):
        raise PermissionError("observer_fit target authorization changed")
    parquet = pq.ParquetFile(TARGET_PARQUET)
    iterator = parquet.iter_batches(
        batch_size=PILOT_ROWS,
        columns=["mlp_id", "all_layer_means"],
        use_threads=False,
    )
    batch = next(iterator)
    if batch.num_rows != PILOT_ROWS:
        raise AssertionError("first protected target batch is not exactly 200 rows")
    table = pa.Table.from_batches([batch])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    if ids.dtype != np.int32 or not np.array_equal(ids, expected_ids):
        raise AssertionError("protected target/feature identity join failed")
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(PILOT_ROWS, DEPTH, WIDTH)
    if targets.dtype != np.float32 or not np.isfinite(targets).all():
        raise AssertionError("protected target schema/finiteness changed")
    return np.ascontiguousarray(targets)


@dataclass(frozen=True)
class PilotRow:
    raw_locals: np.ndarray
    sigma: np.ndarray
    weights_path: Path


class PilotDataset:
    def __init__(self, mlp_ids: np.ndarray):
        self.mlp_ids = mlp_ids

    def load(self, rank: int) -> PilotRow:
        _, raw_path, sigma_path, _ = row_paths(rank)
        _, manifest_path, _ = feature_entry(rank)
        return PilotRow(
            raw_locals=np.load(raw_path, mmap_mode="r", allow_pickle=False),
            sigma=np.load(sigma_path, mmap_mode="r", allow_pickle=False),
            weights_path=manifest_path.parent / "weights.npy",
        )


def load_dynamic4():
    spec = importlib.util.spec_from_file_location("p2r0c1d3_frozen_dynamic4", DYNAMIC4)
    if spec is None or spec.loader is None:
        raise ImportError(DYNAMIC4)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_once(seal: dict[str, object], hosts: dict[str, object]) -> dict[str, object]:
    if TARGET_CLAIM.exists() or MODEL_PATH.exists() or RESULT_PATH.exists():
        raise FileExistsError("one-shot target claim/model/result collision")
    mlp_ids = np.asarray([record["mlp_id"] for record in hosts["records"]], dtype=np.int32)
    if mlp_ids.shape != (PILOT_ROWS,) or len(np.unique(mlp_ids)) != PILOT_ROWS:
        raise AssertionError("protected host roster identities changed")
    claim: dict[str, object] = {
        "schema": "p2r0c1d3-protected160-40-target-open-claim-v1",
        "status": "irreversibly-claimed-no-retry",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "observer_fit",
        "ordered_role_ranks": [0, PILOT_ROWS - 1],
        "train_role_ranks": [0, 159],
        "eval_role_ranks": [160, 199],
        "target_projection": ["mlp_id", "all_layer_means"],
        "observer_selection_targets_opened": 0,
        "name_or_board_join": False,
        "hosts_complete_sha256": sha256(HOSTS_COMPLETE),
        "pilot_seal_sha256": sha256(SEAL),
        "program_sha256": seal["program_sha256"],
        "fits_authorized": 1,
        "autonomous_evaluations_authorized": 1,
        "retries_authorized": 0,
    }
    write_json_once(TARGET_CLAIM, claim)
    targets = load_targets_once(mlp_ids)
    hosts_array = np.empty((PILOT_ROWS, DEPTH, WIDTH), dtype=np.float32)
    for rank in range(PILOT_ROWS):
        host_path, _, _, _ = row_paths(rank)
        hosts_array[rank] = np.load(host_path, mmap_mode="r", allow_pickle=False)
    error = np.ascontiguousarray(targets - hosts_array, dtype=np.float32)

    sums = np.zeros((PILOT_ROWS, DEPTH, LOCAL_FEATURES - 1), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    dataset = PilotDataset(mlp_ids)
    for rank in TRAIN:
        raw = np.asarray(dataset.load(int(rank)).raw_locals[..., 1:], dtype=np.float64)
        sums[rank] = raw.sum(axis=1, dtype=np.float64)
        sums2[rank] = np.square(raw).sum(axis=1, dtype=np.float64)
    dynamic = load_dynamic4()
    dynamic.configure()
    centers, scales, layer_rms = dynamic.moments_for(TRAIN, error, sums, sums2)
    torch.manual_seed(dynamic.SEED)
    model = dynamic.DynamicMultiStateResidual(centers, scales, layer_rms)
    if dynamic.parameter_count(model) != 24_832:
        raise AssertionError("dynamic4 parameter count changed")
    dynamic_cost = dynamic.cost_record(model)
    if int(dynamic_cost["correction_source_structure_upper_estimate"]) != 920_637_440:
        raise AssertionError("dynamic4 correction upper changed")
    if (
        int(dynamic_cost["correction_source_structure_upper_estimate"])
        - int(dynamic_cost["raw20_materialization_reserve"])
        != POST_TAPE_CORRECTION_UPPER
    ):
        raise AssertionError("post-tape correction upper changed")

    started = time.perf_counter()
    trace, max_rss = dynamic.fit_fold(0, model, dataset, TRAIN, error)
    autonomous = dynamic.score(model, dataset, EVAL, error, exposure=1.0)
    host_layer = np.mean(
        np.square(error[EVAL].astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    raw_mse = float(autonomous["final_mse"])
    finite = bool(autonomous["finite_full_rollout"]) and np.isfinite(raw_mse)
    adjusted = raw_mse * SCORE_MULTIPLIER
    passed = finite and raw_mse < BREAK_EVEN_RAW and adjusted < 1.0e-8

    artifact = {
        "schema": "p2r0c1d3-protected160-40-model-v1",
        "state_dict": model.state_dict(),
        "train_role_ranks": [0, 159],
        "eval_role_ranks": [160, 199],
        "program_sha256": seal["program_sha256"],
        "pilot_seal_sha256": sha256(SEAL),
    }
    temporary_model = MODEL_PATH.with_name(f".{MODEL_PATH.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary_model)
    temporary_model.replace(MODEL_PATH)
    result: dict[str, object] = {
        "schema": "p2r0c1d3-protected160-40-result-v1",
        "status": "pass-promote-full-fourfold" if passed else "fail-close-pilot-spelling-only",
        "decision": {
            "finite_autonomous_rollout": finite,
            "raw_final_mse": raw_mse,
            "raw_gate_strictly_below": BREAK_EVEN_RAW,
            "raw_gate_pass": finite and raw_mse < BREAK_EVEN_RAW,
            "projected_adjusted_score": adjusted,
            "adjusted_below_1e_8": finite and adjusted < 1.0e-8,
            "promote_full_fourfold": passed,
            "on_fail_scope": "close only this protected D3 pilot spelling",
        },
        "scope": {
            "role": "observer_fit",
            "networks": PILOT_ROWS,
            "train_networks": len(TRAIN),
            "eval_networks": len(EVAL),
            "train_role_ranks": [0, 159],
            "eval_role_ranks": [160, 199],
            "fits": 1,
            "autonomous_evaluations": 1,
            "retries": 0,
            "observer_selection_targets_opened": 0,
            "name_or_board_join": False,
            "target_projection": ["mlp_id", "all_layer_means"],
        },
        "architecture": {
            "channels": dynamic.CHANNELS,
            "hidden": dynamic.HIDDEN,
            "parameters": dynamic.parameter_count(model),
            "first_corrected_layer": dynamic.FIRST_CORRECTED_LAYER,
            "training_seed": dynamic.SEED,
            "epochs": dynamic.EPOCHS,
            "exposure_schedule": list(dynamic.EXPOSURE_SCHEDULE),
            "batch_size_networks": dynamic.BATCH_SIZE,
        },
        "training_trace": trace,
        "observed_max_rss_bytes": max_rss,
        "host": {
            "final_mse": float(host_layer[-1]),
            "layer_mse": host_layer.tolist(),
        },
        "autonomous": autonomous,
        "cost": {
            "literal_integrated_host_tape_flops": INTEGRATED_HOST_TAPE_FLOPS,
            "post_tape_dynamic4_correction_upper": POST_TAPE_CORRECTION_UPPER,
            "host_tape_plus_correction_upper": TOTAL_UPPER,
            "phase2_budget": PHASE2_BUDGET,
            "score_multiplier": SCORE_MULTIPLIER,
            "headroom": PHASE2_BUDGET - TOTAL_UPPER,
        },
        "bindings": {
            "model_sha256": sha256(MODEL_PATH),
            "target_claim_sha256": sha256(TARGET_CLAIM),
            "hosts_complete_sha256": sha256(HOSTS_COMPLETE),
            "meter_result_sha256": METER_RESULT_SHA256,
            "pilot_seal_sha256": sha256(SEAL),
            "program_sha256": seal["program_sha256"],
            "instrumented_source_sha256": INSTRUMENTED_SHA256,
            "dynamic4_source_sha256": DYNAMIC4_SHA256,
        },
        "elapsed_seconds_fit_and_eval": time.perf_counter() - started,
    }
    write_json_once(RESULT_PATH, result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return result


def main() -> None:
    seal = verify_seal()
    if RESULT_PATH.is_file():
        existing = read_json(RESULT_PATH)
        if existing.get("bindings", {}).get("program_sha256") != seal["program_sha256"]:
            raise AssertionError("existing result belongs to another pilot")
        print(json.dumps(existing["decision"], sort_keys=True), flush=True)
        return
    if TARGET_CLAIM.exists() or MODEL_PATH.exists():
        raise RuntimeError("one-shot target claim was consumed without a terminal result; no retry")
    hosts = prepare_hosts(seal)
    run_once(seal, hosts)


if __name__ == "__main__":
    main()
