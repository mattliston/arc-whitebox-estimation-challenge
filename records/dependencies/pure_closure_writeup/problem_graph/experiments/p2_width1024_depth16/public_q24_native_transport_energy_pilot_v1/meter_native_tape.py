#!/usr/bin/env python3
"""Target-free literal meter and exact-source prediction parity for Q24 native tape."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
INSTRUMENTED = HERE / "instrumented_estimator.py"
EXACT_SOURCE = (
    P2
    / "fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1"
    / "estimator.py"
)
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
FEATURE_MANIFEST = FEATURE_STORE / "manifest.json"
PREREGISTRATION = HERE / "METER_PREREGISTRATION.json"
RUN_MANIFEST = HERE / "METER_RUN_MANIFEST.json"
SEAL = HERE / "METER_SEAL.json"
OUTPUT = HERE / "runs/native_meter_rank0035_v1"

INSTRUMENTED_SHA256 = "f539889324fb82cc1ded35a41c8b5a359014d9ddac52eab62949257e8e58a0ef"
EXACT_SOURCE_SHA256 = "26aca6ed1fb52ec431132f87739fe729ec6855cc2fa2c80642fadd1c4c786e5d"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
ROLE_RANK = 35
PHASE2_BUDGET = 2**41
POST_TAPE_J_UPPER = 783_220_736
MAX_RSS_BYTES = 8 * (1 << 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_seal() -> dict[str, object]:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "frozen-before-target-free-literal-meter":
        raise AssertionError("meter seal status changed")
    expected = {
        "preregistration_sha256": sha256(PREREGISTRATION),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "program_sha256": sha256(Path(__file__)),
        "instrumented_source_sha256": sha256(INSTRUMENTED),
        "exact_source_sha256": sha256(EXACT_SOURCE),
        "feature_manifest_sha256": sha256(FEATURE_MANIFEST),
    }
    for key, observed in expected.items():
        if seal.get(key) != observed:
            raise AssertionError(f"sealed binding changed: {key}: {observed}")
    if expected["instrumented_source_sha256"] != INSTRUMENTED_SHA256:
        raise AssertionError("instrumented source literal binding changed")
    if expected["exact_source_sha256"] != EXACT_SOURCE_SHA256:
        raise AssertionError("exact Q24 source literal binding changed")
    if expected["feature_manifest_sha256"] != FEATURE_MANIFEST_SHA256:
        raise AssertionError("feature manifest literal binding changed")
    return seal


def main() -> None:
    seal = verify_seal()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    entry = manifest["roles"]["observer_fit"]["row_manifests"][ROLE_RANK]
    if int(entry["role_rank"]) != ROLE_RANK:
        raise AssertionError("meter role rank changed")
    row_manifest_path = FEATURE_STORE / str(entry["path"])
    if sha256(row_manifest_path) != str(entry["sha256"]):
        raise AssertionError("row manifest hash changed")
    row = json.loads(row_manifest_path.read_text(encoding="utf-8"))
    if row.get("role") != "observer_fit" or int(row.get("role_rank", -1)) != ROLE_RANK:
        raise AssertionError("meter row role changed")
    weights_path = row_manifest_path.parent / "weights.npy"
    if sha256(weights_path) != row["files"]["weights.npy"]["sha256"]:
        raise AssertionError("meter weight hash changed")
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise AssertionError("meter weight schema changed")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
        name="target-blind-observer-fit-rank-0035-q24-native-meter",
    )
    mlp.validate()

    instrumented = load_module("p2r0f2j_q24_native_meter", INSTRUMENTED)
    integrated_started = time.perf_counter()
    with threadpool_limits(limits=4):
        integrated_context = flops.BudgetContext(
            flop_budget=PHASE2_BUDGET,
            wall_time_limit_s=120.0,
            quiet=True,
            namespace="p2r0f2j_q24_native_tape_rank0035",
        )
        with integrated_context:
            prediction_value, raw_value, sigma_value = (
                instrumented.predict_with_native_tape(mlp)
            )
    integrated_wall = time.perf_counter() - integrated_started
    integrated_summary = integrated_context.summary_dict(by_namespace=False)
    prediction = np.ascontiguousarray(np.asarray(prediction_value), dtype=np.dtype("<f4"))
    raw20 = np.ascontiguousarray(np.asarray(raw_value), dtype=np.dtype("<f4"))
    sigma = np.ascontiguousarray(np.asarray(sigma_value), dtype=np.dtype("<f4"))
    del prediction_value, raw_value, sigma_value, instrumented
    gc.collect()

    exact = load_module("p2r0f2j_q24_exact_meter", EXACT_SOURCE)
    exact_started = time.perf_counter()
    with threadpool_limits(limits=4):
        exact_context = flops.BudgetContext(
            flop_budget=PHASE2_BUDGET,
            wall_time_limit_s=120.0,
            quiet=True,
            namespace="p2r0f2j_exact_q24_parity_rank0035",
        )
        with exact_context:
            exact_value = exact._predict(mlp)
    exact_wall = time.perf_counter() - exact_started
    exact_summary = exact_context.summary_dict(by_namespace=False)
    exact_prediction = np.ascontiguousarray(
        np.asarray(exact_value), dtype=np.dtype("<f4")
    )
    del exact_value, exact
    gc.collect()

    if prediction.shape != (16, 1024) or not np.isfinite(prediction).all():
        raise AssertionError("instrumented prediction contract failed")
    if raw20.shape != (16, 1024, 20) or not np.isfinite(raw20).all():
        raise AssertionError("native raw20 contract failed")
    if sigma.shape != (16, 1024) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise AssertionError("native sigma contract failed")
    if not np.array_equal(prediction, exact_prediction):
        maximum = float(np.max(np.abs(prediction.astype(np.float64) - exact_prediction)))
        raise AssertionError(f"Q24 prediction byte parity failed: max abs {maximum}")
    expected_depth = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    if not np.array_equal(
        raw20[:, :, 0], np.broadcast_to(expected_depth[:, None], (16, 1024))
    ):
        raise AssertionError("native depth feature changed")
    expected_rank = np.arange(16, dtype=np.float32) / np.float32(16.0)
    if not np.array_equal(
        raw20[:, :, 1], np.broadcast_to(expected_rank[:, None], (16, 1024))
    ):
        raise AssertionError("native CP3 rank feature changed")
    if not np.array_equal(raw20[:, :, 2], np.ones((16, 1024), dtype=np.float32)):
        raise AssertionError("native retained feature changed")

    integrated_flops = int(integrated_summary["flops_used"])
    exact_flops = int(exact_summary["flops_used"])
    combined_upper = integrated_flops + POST_TAPE_J_UPPER
    max_rss = resident_bytes()
    if combined_upper > PHASE2_BUDGET:
        raise RuntimeError(f"native host+tape+J upper exceeds Phase2: {combined_upper}")
    if max_rss >= MAX_RSS_BYTES:
        raise MemoryError(f"native meter RSS exceeds bound: {max_rss}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    np.save(OUTPUT / "host.npy", prediction, allow_pickle=False)
    np.save(OUTPUT / "raw20.npy", raw20, allow_pickle=False)
    np.save(OUTPUT / "sigma.npy", sigma, allow_pickle=False)
    result: dict[str, object] = {
        "schema": "p2r0f2j-q24-native-tape-literal-meter-v1",
        "status": "PASS_TARGET_FREE_NATIVE_TAPE_ADMISSION",
        "role": "observer_fit",
        "role_rank": ROLE_RANK,
        "mlp_id": int(entry["mlp_id"]),
        "target_columns_opened": [],
        "observer_selection_opened": False,
        "name_or_board_join": False,
        "prediction": {
            "byte_equal_to_exact_source": True,
            "instrumented_array_sha256": array_sha256(prediction),
            "exact_source_array_sha256": array_sha256(exact_prediction),
            "shape": list(prediction.shape),
            "dtype": prediction.dtype.str
        },
        "native_tape": {
            "raw20_array_sha256": array_sha256(raw20),
            "sigma_array_sha256": array_sha256(sigma),
            "raw20_shape": list(raw20.shape),
            "sigma_shape": list(sigma.shape),
            "finite": True,
            "sigma_strictly_positive": True,
            "cp1792_value_parity_claimed": False
        },
        "literal_meter": {
            "integrated_host_tape": integrated_summary,
            "exact_source": exact_summary,
            "integrated_wall_seconds": integrated_wall,
            "exact_wall_seconds": exact_wall,
            "maximum_process_rss_bytes": max_rss,
            "post_tape_j_upper": POST_TAPE_J_UPPER,
            "integrated_host_tape_plus_j_upper": combined_upper,
            "phase2_budget": PHASE2_BUDGET,
            "admitted": True
        },
        "bindings": {
            "seal_sha256": sha256(SEAL),
            "preregistration_sha256": seal["preregistration_sha256"],
            "run_manifest_sha256": seal["run_manifest_sha256"],
            "program_sha256": seal["program_sha256"],
            "instrumented_source_sha256": INSTRUMENTED_SHA256,
            "exact_source_sha256": EXACT_SOURCE_SHA256,
            "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
            "row_manifest_sha256": str(entry["sha256"]),
            "weights_file_sha256": row["files"]["weights.npy"]["sha256"]
        }
    }
    write_json_once(OUTPUT / "RESULT.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "prediction_byte_parity": True,
                "exact_flops": exact_flops,
                "integrated_flops": integrated_flops,
                "combined_upper": combined_upper,
                "max_rss_bytes": max_rss
            },
            sort_keys=True
        ),
        flush=True
    )


if __name__ == "__main__":
    main()
