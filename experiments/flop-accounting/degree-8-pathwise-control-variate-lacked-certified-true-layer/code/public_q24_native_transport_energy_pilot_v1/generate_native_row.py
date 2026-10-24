#!/usr/bin/env python3
"""Generate one sealed target-free Q24-native host/raw20/sigma row."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
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
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
FEATURE_MANIFEST = FEATURE_STORE / "manifest.json"
PREREGISTRATION = HERE / "GENERATION_PREREGISTRATION.json"
RUN_MANIFEST = HERE / "GENERATION_RUN_MANIFEST.json"
SEAL = HERE / "GENERATION_SEAL.json"
METER_RESULT = HERE / "runs/native_meter_rank0035_v1/RESULT.json"
METER_DIR = METER_RESULT.parent
DEFAULT_OUTPUT_ROOT = HERE / "runs/native_rows_v1"

INSTRUMENTED_SHA256 = "f539889324fb82cc1ded35a41c8b5a359014d9ddac52eab62949257e8e58a0ef"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
METER_RESULT_SHA256 = "27edd79f91d02c265bfc051c146b1b9162647c4c42cea37e7ff6de602307a5c3"
METER_FILE_SHA256 = {
    "host.npy": "2df0935183dad7ee2442ceb2456304d18f85115f42fe950fdab7a9201475b206",
    "raw20.npy": "0768502bbd8beb26b85ffdd6b347f0312e1f59e7d4416c849efac8c2dec88365",
    "sigma.npy": "c22932b89b08a4ad6be821245a275b1126ef8410cebdc7d75335b62359e55ff5",
}
PHASE2_BUDGET = 2**41
THREADS = 2
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


def load_module(path: Path):
    tag = f"p2r0f2j_q24_native_row_{os.getpid()}"
    specification = importlib.util.spec_from_file_location(tag, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[tag] = module
    specification.loader.exec_module(module)
    return module


def verify_seal() -> dict[str, object]:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "frozen-after-meter-pass-before-row-generation":
        raise AssertionError("generation seal status changed")
    expected = {
        "preregistration_sha256": sha256(PREREGISTRATION),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "program_sha256": sha256(Path(__file__)),
        "instrumented_source_sha256": sha256(INSTRUMENTED),
        "feature_manifest_sha256": sha256(FEATURE_MANIFEST),
        "meter_result_sha256": sha256(METER_RESULT),
    }
    for key, observed in expected.items():
        if seal.get(key) != observed:
            raise AssertionError(f"sealed binding changed: {key}: {observed}")
    if expected["instrumented_source_sha256"] != INSTRUMENTED_SHA256:
        raise AssertionError("instrumented source changed")
    if expected["feature_manifest_sha256"] != FEATURE_MANIFEST_SHA256:
        raise AssertionError("feature manifest changed")
    if expected["meter_result_sha256"] != METER_RESULT_SHA256:
        raise AssertionError("meter result changed")
    return seal


def validate_arrays(host: np.ndarray, raw20: np.ndarray, sigma: np.ndarray) -> None:
    if host.dtype != np.dtype("<f4") or host.shape != (16, 1024):
        raise AssertionError("host schema changed")
    if raw20.dtype != np.dtype("<f4") or raw20.shape != (16, 1024, 20):
        raise AssertionError("raw20 schema changed")
    if sigma.dtype != np.dtype("<f4") or sigma.shape != (16, 1024):
        raise AssertionError("sigma schema changed")
    if not np.isfinite(host).all() or np.any(host < 0):
        raise FloatingPointError("host sign/finiteness contract failed")
    if not np.isfinite(raw20).all():
        raise FloatingPointError("raw20 finiteness contract failed")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise FloatingPointError("sigma sign/finiteness contract failed")


def write_row(
    output: Path,
    host: np.ndarray,
    raw20: np.ndarray,
    sigma: np.ndarray,
    receipt: dict[str, object],
) -> None:
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    np.save(temporary / "host.npy", host, allow_pickle=False)
    np.save(temporary / "raw20.npy", raw20, allow_pickle=False)
    np.save(temporary / "sigma.npy", sigma, allow_pickle=False)
    files = {
        name: {
            "sha256": sha256(temporary / name),
            "bytes": (temporary / name).stat().st_size,
            "array_sha256": array_sha256(value),
        }
        for name, value in (
            ("host.npy", host),
            ("raw20.npy", raw20),
            ("sigma.npy", sigma),
        )
    }
    receipt["files"] = files
    result_path = temporary / "result.json"
    result_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    rank = int(args.rank)
    if rank < 35 or rank >= 235:
        raise ValueError("rank is outside frozen native-pilot roster 35:235")
    seal = verify_seal()
    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    if int(entry["role_rank"]) != rank:
        raise AssertionError("feature manifest rank changed")
    row_manifest_path = FEATURE_STORE / str(entry["path"])
    if sha256(row_manifest_path) != str(entry["sha256"]):
        raise AssertionError("row manifest hash changed")
    row = json.loads(row_manifest_path.read_text(encoding="utf-8"))
    if row.get("role") != "observer_fit" or int(row.get("role_rank", -1)) != rank:
        raise AssertionError("row role changed")
    weights_path = row_manifest_path.parent / "weights.npy"
    if sha256(weights_path) != row["files"]["weights.npy"]["sha256"]:
        raise AssertionError("weight file hash changed")
    output = args.output_root / f"row_{rank:04d}"
    started = time.perf_counter()

    if rank == 35:
        meter = json.loads(METER_RESULT.read_text(encoding="utf-8"))
        if meter.get("status") != "PASS_TARGET_FREE_NATIVE_TAPE_ADMISSION":
            raise AssertionError("rank35 meter is not reusable")
        for name, expected in METER_FILE_SHA256.items():
            if sha256(METER_DIR / name) != expected:
                raise AssertionError(f"rank35 meter file changed: {name}")
        host = np.ascontiguousarray(
            np.load(METER_DIR / "host.npy", allow_pickle=False), dtype=np.dtype("<f4")
        )
        raw20 = np.ascontiguousarray(
            np.load(METER_DIR / "raw20.npy", allow_pickle=False), dtype=np.dtype("<f4")
        )
        sigma = np.ascontiguousarray(
            np.load(METER_DIR / "sigma.npy", allow_pickle=False), dtype=np.dtype("<f4")
        )
        meter_record: dict[str, object] = {
            "mode": "reused_exact_target_free_literal_meter_tape",
            "meter_result_sha256": METER_RESULT_SHA256,
            "flops_used": meter["literal_meter"]["integrated_host_tape"]["flops_used"],
            "wall_time_s": meter["literal_meter"]["integrated_wall_seconds"],
        }
    else:
        weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
        if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
            raise AssertionError("weight schema changed")
        mlp = MLP(
            width=1024,
            depth=16,
            weights=[weights[layer] for layer in range(16)],
            seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
            name=f"target-blind-observer-fit-rank-{rank:04d}-q24-native",
        )
        mlp.validate()
        module = load_module(INSTRUMENTED)
        with threadpool_limits(limits=THREADS):
            context = flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=120.0,
                quiet=True,
                namespace=f"p2r0f2j_q24_native_rank_{rank:04d}",
            )
            with context:
                host_value, raw_value, sigma_value = module.predict_with_native_tape(mlp)
        summary = context.summary_dict(by_namespace=False)
        host = np.ascontiguousarray(np.asarray(host_value), dtype=np.dtype("<f4"))
        raw20 = np.ascontiguousarray(np.asarray(raw_value), dtype=np.dtype("<f4"))
        sigma = np.ascontiguousarray(np.asarray(sigma_value), dtype=np.dtype("<f4"))
        if int(summary["flops_used"]) > PHASE2_BUDGET:
            raise RuntimeError("native tape exceeded Phase2 FLOP budget")
        if float(summary["wall_time_s"]) >= 120.0:
            raise RuntimeError("native tape exceeded wall bound")
        meter_record = {
            "mode": "fresh_literal_integrated_host_tape",
            **summary,
        }

    validate_arrays(host, raw20, sigma)
    max_rss = resident_bytes()
    if max_rss >= MAX_RSS_BYTES:
        raise MemoryError(f"worker RSS exceeded bound: {max_rss}")
    receipt: dict[str, object] = {
        "schema": "p2r0f2j-q24-native-row-v1",
        "status": "complete-target-free",
        "role": "observer_fit",
        "role_rank": rank,
        "mlp_id": int(entry["mlp_id"]),
        "mlp_seed": int(row["mlp_seed"]),
        "target_columns_opened": [],
        "observer_selection_opened": False,
        "name_or_board_join": False,
        "instrumented_source_sha256": INSTRUMENTED_SHA256,
        "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
        "row_manifest_sha256": str(entry["sha256"]),
        "weights_file_sha256": row["files"]["weights.npy"]["sha256"],
        "meter": meter_record,
        "maximum_process_rss_bytes": max_rss,
        "elapsed_seconds": time.perf_counter() - started,
        "generation_seal_sha256": sha256(SEAL),
        "generation_program_sha256": seal["program_sha256"],
    }
    write_row(output, host, raw20, sigma, receipt)
    print(
        json.dumps(
            {
                "rank": rank,
                "status": receipt["status"],
                "flops_used": meter_record["flops_used"],
                "elapsed_seconds": receipt["elapsed_seconds"],
                "maximum_process_rss_bytes": max_rss,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
