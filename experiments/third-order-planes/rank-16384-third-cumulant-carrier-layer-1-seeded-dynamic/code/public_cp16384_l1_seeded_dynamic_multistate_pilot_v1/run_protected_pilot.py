#!/Users/matt/anaconda3/bin/python3
"""Frozen one-shot P2R0C1D4 protected 160/40 pilot.

The only scientific change from D3 is FIRST_CORRECTED_LAYER=1.  Local array
indices 0..199 map exactly to observer_fit role ranks 200..399.
"""

from __future__ import annotations

import argparse
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
D3 = P2 / "public_cp16384_dynamic_multistate_pilot_v1"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUN_MANIFEST = HERE / "RUN_MANIFEST.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
DECOMPOSITION = HERE / "D3_FAILURE_DECOMPOSITION.json"
RELEASE = HERE / "HEAVY_LANE_RELEASE.json"
SMOKE_RESULT = HERE / "SMOKE_RESULT_V2.json"

HOST_HELPER = HERE / "generate_host_row.py"
INSTRUMENTED = D3 / "instrumented_estimator.py"
SOURCE_HOST = P2 / "cp3_cap16384_mean_recurrence_ablation/estimator.py"
DYNAMIC4 = P2 / "public_full688_dynamic_multistate_residual_cv_v1/run_experiment.py"
D3_TERMINAL = D3 / "runs/protected160_40_v1/TERMINAL_RECOVERY.json"
D3_PARTIAL = D3 / "runs/protected160_40_v1/.RESULT.json.28259.tmp"
D3_METER_RESULT = D3 / "runs/integrated_meter_rank0000_v1/RESULT.json"
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
HOST_PYTHON = P2 / "data_gate/.venv/bin/python"

RUN_ROOT = HERE / "runs/protected200_399_train160_eval40_v1"
HOST_ROOT = RUN_ROOT / "hosts"
HOSTS_COMPLETE = RUN_ROOT / "HOSTS_COMPLETE.json"
TARGET_CLAIM = RUN_ROOT / "TARGET_OPEN_CLAIM.json"
MODEL_PATH = RUN_ROOT / "MODEL.pt"
RESULT_PATH = RUN_ROOT / "RESULT.json"

ROLE_START = 200
PILOT_ROWS = 200
TRAIN = np.arange(0, 160, dtype=np.int64)
EVAL = np.arange(160, 200, dtype=np.int64)
TRAIN_ROLE_RANGE = [200, 359]
EVAL_ROLE_RANGE = [360, 399]
DEPTH = 16
WIDTH = 1024
LOCAL_FEATURES = 20
FIRST_CORRECTED_LAYER = 1

INTEGRATED_HOST_TAPE_FLOPS = 620_447_502_338
D3_POST_TAPE_CORRECTION_UPPER = 652_201_984
EXTRA_L1_CELL_UPPER = 38_752_256
EXTRA_L1_TO_L2_TRANSPORT_UPPER = 8_384_512
POST_TAPE_CORRECTION_UPPER = 699_338_752
TOTAL_UPPER = 621_146_841_090
PHASE2_BUDGET = 2**41
SCORE_MULTIPLIER = TOTAL_UPPER / PHASE2_BUDGET
BREAK_EVEN_RAW = 3.5402631231177368555906471767710946105625633e-8

INSTRUMENTED_SHA256 = "76a87a1d161db9583dd318971b2688079d1919579d7a900fa9ec1a0230318105"
SOURCE_HOST_SHA256 = "361c87cba008987c117871bf9892beecce75911634798ed7ad3fa26fd8ee5814"
DYNAMIC4_SHA256 = "8871ac10f3f122d9231652e3618c888bdc69b71cc0de44ffa00a79630258225d"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
TARGET_MANIFEST_SHA256 = "2c0ea0bb09d0c86d46ca07fdde3563c4cda046a1fff152cdff26fde965e1f7e6"
TARGET_PARQUET_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
TARGET_AUTHORIZATION_SHA256 = "6a8eb53a567419a2d829822ae1747243b878f889735f1830d421ca98d4c15c7a"
D3_METER_RESULT_SHA256 = "6d7afc4af1e8a9a77d7e3611f56830e19955ed917775b9c93a225a419917826c"
D3_TERMINAL_SHA256 = "404bb2a4f1bb3eebb7f3d961bcad0aa098d1e4c866fcd3e139cec9138c42d792"
D3_PARTIAL_SHA256 = "f487c7c360b066549263f5b8506aeb5a55ece637e944ebd17fa844d693282121"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
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
        raise AssertionError("D4 seal status changed")
    paths = {
        "preregistration_sha256": PREREGISTRATION,
        "run_manifest_sha256": RUN_MANIFEST,
        "program_sha256": Path(__file__),
        "decomposition_sha256": DECOMPOSITION,
        "host_helper_sha256": HOST_HELPER,
        "instrumented_source_sha256": INSTRUMENTED,
        "exact_host_source_sha256": SOURCE_HOST,
        "dynamic4_source_sha256": DYNAMIC4,
        "feature_manifest_sha256": FEATURE_MANIFEST,
        "target_manifest_sha256": TARGET_MANIFEST,
        "target_parquet_sha256": TARGET_PARQUET,
        "target_authorization_sha256": TARGET_AUTHORIZATION,
        "d3_meter_result_sha256": D3_METER_RESULT,
        "d3_terminal_recovery_sha256": D3_TERMINAL,
        "d3_partial_result_sha256": D3_PARTIAL,
    }
    for key, path in paths.items():
        observed = sha256(path)
        if seal.get(key) != observed:
            raise AssertionError(f"frozen D4 binding changed: {key}: {observed}")
    prereg = read_json(PREREGISTRATION)
    if prereg.get("status") != "frozen-before-host-generation-and-target-metrics":
        raise AssertionError("D4 preregistration status changed")
    if prereg["split"]["train_role_ranks"] != TRAIN_ROLE_RANGE:
        raise AssertionError("D4 train split changed")
    if prereg["split"]["eval_role_ranks"] != EVAL_ROLE_RANGE:
        raise AssertionError("D4 eval split changed")
    if prereg["architecture"]["first_corrected_layer"] != FIRST_CORRECTED_LAYER:
        raise AssertionError("D4 causal intervention changed")
    if float(prereg["decision_gate"]["raw_mse_strictly_below"]) != BREAK_EVEN_RAW:
        raise AssertionError("D4 break-even raw gate changed")
    return seal


def verify_release() -> dict[str, object]:
    release = read_json(RELEASE)
    if not (
        release.get("status") == "released-by-root-after-f2j-terminal"
        and release.get("node") == "P2R0C1D4"
        and release.get("authorized") is True
    ):
        raise PermissionError("D4 heavy lane is not root-released")
    terminal = P2 / str(release["f2j_terminal_relative_to_p2"])
    if sha256(terminal) != release.get("f2j_terminal_sha256"):
        raise AssertionError("F2J terminal binding changed")
    return release


def actual_rank(local_index: int) -> int:
    if not 0 <= local_index < PILOT_ROWS:
        raise IndexError(local_index)
    return ROLE_START + local_index


def feature_entry(local_index: int) -> tuple[dict[str, object], Path, dict[str, object]]:
    rank = actual_rank(local_index)
    manifest = read_json(FEATURE_MANIFEST)
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    if int(entry["role_rank"]) != rank:
        raise AssertionError(f"rank {rank}: feature roster changed")
    path = FEATURE_STORE / str(entry["path"])
    if sha256(path) != str(entry["sha256"]):
        raise AssertionError(f"rank {rank}: feature row changed")
    row = read_json(path)
    if row.get("role") != "observer_fit" or int(row.get("role_rank", -1)) != rank:
        raise AssertionError(f"rank {rank}: feature role changed")
    if int(row.get("mlp_id", -1)) != int(entry["mlp_id"]):
        raise AssertionError(f"rank {rank}: feature identity changed")
    return entry, path, row


def row_paths(local_index: int) -> tuple[Path, Path, Path, Path]:
    rank = actual_rank(local_index)
    directory = HOST_ROOT / f"role_rank_{rank:04d}"
    return (
        directory / "host.npy",
        directory / "raw20.npy",
        directory / "sigma.npy",
        directory / "result.json",
    )


def validate_host_row(local_index: int) -> dict[str, object]:
    rank = actual_rank(local_index)
    host_path, raw_path, sigma_path, result_path = row_paths(local_index)
    if not all(path.is_file() for path in (host_path, raw_path, sigma_path, result_path)):
        raise FileNotFoundError(f"rank {rank}: incomplete integrated host row")
    result = read_json(result_path)
    if result.get("status") != "complete" or int(result.get("role_rank", -1)) != rank:
        raise AssertionError(f"rank {rank}: host receipt changed")
    if result.get("instrumented_source_sha256") != INSTRUMENTED_SHA256:
        raise AssertionError(f"rank {rank}: instrumented source changed")
    if int(result["literal_meter"]["flops_used"]) != INTEGRATED_HOST_TAPE_FLOPS:
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
    for local_index in range(PILOT_ROWS):
        rank = actual_rank(local_index)
        directory = HOST_ROOT / f"role_rank_{rank:04d}"
        if directory.exists():
            validate_host_row(local_index)
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
        complete = local_index + 1
        if complete % 25 == 0 or complete == PILOT_ROWS:
            print(f"D4 integrated hosts {complete}/{PILOT_ROWS}", flush=True)
    records: list[dict[str, object]] = []
    for local_index in range(PILOT_ROWS):
        result = validate_host_row(local_index)
        _, _, _, result_path = row_paths(local_index)
        entry, _, _ = feature_entry(local_index)
        records.append(
            {
                "local_index": local_index,
                "role_rank": actual_rank(local_index),
                "mlp_id": int(entry["mlp_id"]),
                "result_sha256": sha256(result_path),
                "flops_used": int(result["literal_meter"]["flops_used"]),
            }
        )
    expected: dict[str, object] = {
        "schema": "p2r0c1d4-protected200-host-completion-v1",
        "status": "complete-before-target-open",
        "role": "observer_fit",
        "ordered_role_ranks": [ROLE_START, ROLE_START + PILOT_ROWS - 1],
        "rows": PILOT_ROWS,
        "ordered_result_digest": canonical_sha256(records),
        "records": records,
        "integrated_host_tape_flops_per_row": INTEGRATED_HOST_TAPE_FLOPS,
        "d3_meter_result_sha256": D3_METER_RESULT_SHA256,
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
            raise AssertionError("D4 host completion receipt changed")
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
    prior_d3_batch = next(iterator)
    if prior_d3_batch.num_rows != PILOT_ROWS:
        raise AssertionError("prior D3 stream prefix changed")
    batch = next(iterator)
    if batch.num_rows != PILOT_ROWS:
        raise AssertionError("D4 protected target batch is not exactly 200 rows")
    table = pa.Table.from_batches([batch])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    if ids.dtype != np.int32 or not np.array_equal(ids, expected_ids):
        raise AssertionError("D4 target/feature identity join failed")
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(PILOT_ROWS, DEPTH, WIDTH)
    if targets.dtype != np.float32 or not np.isfinite(targets).all():
        raise AssertionError("D4 target schema/finiteness changed")
    return np.ascontiguousarray(targets)


@dataclass(frozen=True)
class PilotRow:
    raw_locals: np.ndarray
    sigma: np.ndarray
    weights_path: Path


class PilotDataset:
    def __init__(self, mlp_ids: np.ndarray):
        self.mlp_ids = mlp_ids

    def load(self, local_index: int) -> PilotRow:
        _, raw_path, sigma_path, _ = row_paths(local_index)
        _, manifest_path, _ = feature_entry(local_index)
        return PilotRow(
            raw_locals=np.load(raw_path, mmap_mode="r", allow_pickle=False),
            sigma=np.load(sigma_path, mmap_mode="r", allow_pickle=False),
            weights_path=manifest_path.parent / "weights.npy",
        )


def load_dynamic4():
    spec = importlib.util.spec_from_file_location("p2r0c1d4_frozen_dynamic4", DYNAMIC4)
    if spec is None or spec.loader is None:
        raise ImportError(DYNAMIC4)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.FIRST_CORRECTED_LAYER != 2:
        raise AssertionError("D3 source no longer has frozen L2 start")
    module.FIRST_CORRECTED_LAYER = FIRST_CORRECTED_LAYER
    return module


def smoke(seal: dict[str, object]) -> dict[str, object]:
    dynamic = load_dynamic4()
    dynamic.configure()
    torch.manual_seed(dynamic.SEED)
    centers = np.zeros((DEPTH, LOCAL_FEATURES - 1), dtype=np.float32)
    scales = np.ones_like(centers)
    layer_rms = np.ones(DEPTH, dtype=np.float32)
    model = dynamic.DynamicMultiStateResidual(centers, scales, layer_rms)
    if dynamic.parameter_count(model) != 24_832:
        raise AssertionError("D4 parameter count changed")
    cost = dynamic.cost_record(model)
    if int(cost["correction_source_structure_upper_estimate"]) != 967_774_208:
        raise AssertionError("D4 correction upper changed")
    if int(cost["correction_source_structure_upper_estimate"]) - int(
        cost["raw20_materialization_reserve"]
    ) != POST_TAPE_CORRECTION_UPPER:
        raise AssertionError("D4 post-tape upper changed")
    generator = torch.Generator().manual_seed(20260824)
    raw = 0.05 * torch.randn(1, DEPTH, WIDTH, LOCAL_FEATURES, generator=generator)
    raw[..., 0] = torch.linspace(-1.0, 1.0, DEPTH).view(1, DEPTH, 1)
    raw[..., 5] = 0.5
    sigma = torch.ones(1, DEPTH, WIDTH)
    identity = torch.eye(WIDTH).view(1, 1, WIDTH, WIDTH)
    weights = identity.expand(1, DEPTH, WIDTH, WIDTH)
    with torch.no_grad():
        dynamic.FIRST_CORRECTED_LAYER = 2
        baseline = model(raw, sigma, weights, exposure=1.0)
        dynamic.FIRST_CORRECTED_LAYER = 1
        candidate, states = model(
            raw, sigma, weights, exposure=1.0, return_states=True
        )
    if not isinstance(candidate, torch.Tensor) or not isinstance(states, torch.Tensor):
        raise AssertionError("D4 smoke return contract changed")
    baseline_l1_max = float(baseline[:, 1].abs().max())
    candidate_l1_max = float(candidate[:, 1].abs().max())
    l2_incoming_message_max = float((0.5 * states[:, 1]).abs().max())
    passed = bool(
        torch.isfinite(candidate).all()
        and baseline_l1_max == 0.0
        and candidate_l1_max > 0.0
        and l2_incoming_message_max > 0.0
    )
    result: dict[str, object] = {
        "schema": "p2r0c1d4-target-free-l1-carry-smoke-v1",
        "status": "pass" if passed else "fail",
        "program_sha256": seal["program_sha256"],
        "pilot_seal_sha256": sha256(SEAL),
        "dynamic4_source_sha256": DYNAMIC4_SHA256,
        "original_first_corrected_layer": 2,
        "candidate_first_corrected_layer": 1,
        "sole_scientific_change": True,
        "parameters": dynamic.parameter_count(model),
        "baseline_l1_max_abs": baseline_l1_max,
        "candidate_l1_max_abs": candidate_l1_max,
        "candidate_l2_incoming_message_max_abs": l2_incoming_message_max,
        "candidate_all_finite": bool(torch.isfinite(candidate).all()),
        "post_tape_correction_upper": POST_TAPE_CORRECTION_UPPER,
        "total_upper": TOTAL_UPPER,
        "break_even_raw_strictly_below": format(BREAK_EVEN_RAW, ".17g"),
        "target_values_opened": 0,
        "host_generation_runs": 0,
        "observer_selection_targets_opened": 0,
        "name_or_board_join": False,
    }
    if not passed:
        raise AssertionError(result)
    if SMOKE_RESULT.exists():
        observed = read_json(SMOKE_RESULT)
        if observed != result:
            raise AssertionError("D4 smoke result changed")
        return observed
    write_json_once(SMOKE_RESULT, result)
    return result


def run_once(seal: dict[str, object], hosts: dict[str, object]) -> dict[str, object]:
    if TARGET_CLAIM.exists() or MODEL_PATH.exists() or RESULT_PATH.exists():
        raise FileExistsError("D4 one-shot target claim/model/result collision")
    mlp_ids = np.asarray([record["mlp_id"] for record in hosts["records"]], dtype=np.int32)
    if mlp_ids.shape != (PILOT_ROWS,) or len(np.unique(mlp_ids)) != PILOT_ROWS:
        raise AssertionError("D4 protected host identities changed")
    claim: dict[str, object] = {
        "schema": "p2r0c1d4-protected160-40-target-open-claim-v1",
        "status": "irreversibly-claimed-no-retry",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "observer_fit",
        "ordered_role_ranks": [200, 399],
        "train_role_ranks": TRAIN_ROLE_RANGE,
        "eval_role_ranks": EVAL_ROLE_RANGE,
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
    for local_index in range(PILOT_ROWS):
        host_path, _, _, _ = row_paths(local_index)
        hosts_array[local_index] = np.load(host_path, mmap_mode="r", allow_pickle=False)
    error = np.ascontiguousarray(targets - hosts_array, dtype=np.float32)

    sums = np.zeros((PILOT_ROWS, DEPTH, LOCAL_FEATURES - 1), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    dataset = PilotDataset(mlp_ids)
    for local_index in TRAIN:
        raw = np.asarray(dataset.load(int(local_index)).raw_locals[..., 1:], dtype=np.float64)
        sums[local_index] = raw.sum(axis=1, dtype=np.float64)
        sums2[local_index] = np.square(raw).sum(axis=1, dtype=np.float64)
    dynamic = load_dynamic4()
    dynamic.configure()
    centers, scales, layer_rms = dynamic.moments_for(TRAIN, error, sums, sums2)
    torch.manual_seed(dynamic.SEED)
    model = dynamic.DynamicMultiStateResidual(centers, scales, layer_rms)
    if dynamic.parameter_count(model) != 24_832:
        raise AssertionError("D4 parameter count changed")
    dynamic_cost = dynamic.cost_record(model)
    if int(dynamic_cost["correction_source_structure_upper_estimate"]) != 967_774_208:
        raise AssertionError("D4 correction upper changed")
    if int(dynamic_cost["correction_source_structure_upper_estimate"]) - int(
        dynamic_cost["raw20_materialization_reserve"]
    ) != POST_TAPE_CORRECTION_UPPER:
        raise AssertionError("D4 post-tape correction upper changed")

    started = time.perf_counter()
    trace, max_rss = dynamic.fit_fold(0, model, dataset, TRAIN, error)
    autonomous = dynamic.score(model, dataset, EVAL, error, exposure=1.0)
    host_layer = np.mean(
        np.square(error[EVAL].astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    raw_mse = float(autonomous["final_mse"])
    finite = bool(autonomous["finite_full_rollout"]) and bool(np.isfinite(raw_mse))
    adjusted = raw_mse * SCORE_MULTIPLIER
    passed = bool(finite and raw_mse < BREAK_EVEN_RAW and adjusted < 1.0e-8)

    artifact = {
        "schema": "p2r0c1d4-protected160-40-model-v1",
        "state_dict": model.state_dict(),
        "train_role_ranks": TRAIN_ROLE_RANGE,
        "eval_role_ranks": EVAL_ROLE_RANGE,
        "program_sha256": seal["program_sha256"],
        "pilot_seal_sha256": sha256(SEAL),
    }
    temporary_model = MODEL_PATH.with_name(f".{MODEL_PATH.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary_model)
    temporary_model.replace(MODEL_PATH)
    result: dict[str, object] = {
        "schema": "p2r0c1d4-protected160-40-result-v1",
        "status": "pass-promote-full-fourfold" if passed else "fail-close-pilot-spelling-only",
        "decision": {
            "finite_autonomous_rollout": finite,
            "raw_final_mse": raw_mse,
            "raw_gate_strictly_below": BREAK_EVEN_RAW,
            "raw_gate_pass": bool(finite and raw_mse < BREAK_EVEN_RAW),
            "projected_adjusted_score": adjusted,
            "adjusted_below_1e_8": bool(finite and adjusted < 1.0e-8),
            "promote_full_fourfold": passed,
            "on_fail_scope": "close only this protected D4 L1-seeded spelling",
        },
        "scope": {
            "role": "observer_fit",
            "networks": PILOT_ROWS,
            "train_networks": len(TRAIN),
            "eval_networks": len(EVAL),
            "train_role_ranks": TRAIN_ROLE_RANGE,
            "eval_role_ranks": EVAL_ROLE_RANGE,
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
        "host": {"final_mse": float(host_layer[-1]), "layer_mse": host_layer.tolist()},
        "autonomous": autonomous,
        "cost": {
            "literal_integrated_host_tape_flops": INTEGRATED_HOST_TAPE_FLOPS,
            "post_tape_correction_upper": POST_TAPE_CORRECTION_UPPER,
            "host_tape_plus_correction_upper": TOTAL_UPPER,
            "phase2_budget": PHASE2_BUDGET,
            "score_multiplier": SCORE_MULTIPLIER,
            "headroom": PHASE2_BUDGET - TOTAL_UPPER,
        },
        "bindings": {
            "model_sha256": sha256(MODEL_PATH),
            "target_claim_sha256": sha256(TARGET_CLAIM),
            "hosts_complete_sha256": sha256(HOSTS_COMPLETE),
            "d3_meter_result_sha256": D3_METER_RESULT_SHA256,
            "d3_terminal_recovery_sha256": D3_TERMINAL_SHA256,
            "decomposition_sha256": sha256(DECOMPOSITION),
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


def launch() -> None:
    seal = verify_seal()
    verify_release()
    if not SMOKE_RESULT.exists() or read_json(SMOKE_RESULT).get("status") != "pass":
        raise RuntimeError("D4 target-free smoke must pass before launch")
    if RESULT_PATH.is_file():
        existing = read_json(RESULT_PATH)
        if existing.get("bindings", {}).get("program_sha256") != seal["program_sha256"]:
            raise AssertionError("existing result belongs to another pilot")
        print(json.dumps(existing["decision"], sort_keys=True), flush=True)
        return
    if TARGET_CLAIM.exists() or MODEL_PATH.exists():
        raise RuntimeError("D4 one-shot target claim consumed without terminal result; no retry")
    hosts = prepare_hosts(seal)
    run_once(seal, hosts)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        result = smoke(verify_seal())
        print(json.dumps(result, sort_keys=True), flush=True)
    else:
        launch()


if __name__ == "__main__":
    main()
