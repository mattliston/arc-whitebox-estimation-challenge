#!/usr/bin/env python3
"""Generate one target-blind integrated CP16384 host/raw20/sigma row."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
FEATURE_MANIFEST = FEATURE_STORE / "manifest.json"
INSTRUMENTED = HERE / "instrumented_estimator.py"
SOURCE_HOST = P2 / "cp3_cap16384_mean_recurrence_ablation/estimator.py"
METER_RESULT = HERE / "runs/integrated_meter_rank0000_v1/RESULT.json"

FEATURE_MANIFEST_SHA256 = (
    "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
)
INSTRUMENTED_SHA256 = (
    "76a87a1d161db9583dd318971b2688079d1919579d7a900fa9ec1a0230318105"
)
SOURCE_HOST_SHA256 = (
    "361c87cba008987c117871bf9892beecce75911634798ed7ad3fa26fd8ee5814"
)
METER_RESULT_SHA256 = (
    "6d7afc4af1e8a9a77d7e3611f56830e19955ed917775b9c93a225a419917826c"
)
EXPECTED_INTEGRATED_FLOPS = 620_447_502_338
PHASE2_BUDGET = 2**41
PILOT_ROWS = 200


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


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_instrumented():
    spec = importlib.util.spec_from_file_location(
        "p2r0c1d3_frozen_integrated_host", INSTRUMENTED
    )
    if spec is None or spec.loader is None:
        raise ImportError(INSTRUMENTED)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(args.rank)
    # Rank zero is the already-metered and parity-checked admission row.
    if rank <= 0 or rank >= PILOT_ROWS:
        raise ValueError(f"rank {rank} is outside frozen generated roster 1:{PILOT_ROWS}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    bindings = {
        FEATURE_MANIFEST: FEATURE_MANIFEST_SHA256,
        INSTRUMENTED: INSTRUMENTED_SHA256,
        SOURCE_HOST: SOURCE_HOST_SHA256,
        METER_RESULT: METER_RESULT_SHA256,
    }
    for path, expected in bindings.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(f"frozen binding changed: {path}: {observed}")
    meter = json.loads(METER_RESULT.read_text(encoding="utf-8"))
    if meter.get("status") != "pass-admitted-before-target-fit":
        raise AssertionError("integrated meter did not admit D3")
    if int(meter["literal_integrated_meter"]["flops_used"]) != EXPECTED_INTEGRATED_FLOPS:
        raise AssertionError("integrated meter FLOPs changed")

    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    if len(entries) != 688:
        raise AssertionError("feature-store observer_fit roster changed")
    entry = entries[rank]
    if int(entry["role_rank"]) != rank:
        raise AssertionError("feature manifest is not ordered by role rank")
    row_manifest_path = FEATURE_STORE / str(entry["path"])
    if sha256(row_manifest_path) != str(entry["sha256"]):
        raise AssertionError("sealed row manifest hash changed")
    row_manifest = json.loads(row_manifest_path.read_text(encoding="utf-8"))
    if row_manifest.get("role") != "observer_fit":
        raise AssertionError("row role changed")
    if int(row_manifest.get("role_rank", -1)) != rank:
        raise AssertionError("row role rank changed")
    if int(row_manifest.get("mlp_id", -1)) != int(entry["mlp_id"]):
        raise AssertionError("row MLP identity changed")
    weights_path = row_manifest_path.parent / "weights.npy"
    weights_record = row_manifest["files"]["weights.npy"]
    if sha256(weights_path) != str(weights_record["sha256"]):
        raise AssertionError("sealed weight file hash changed")
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise AssertionError("weight schema changed")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=int(derive_estimator_seed(int(row_manifest["mlp_seed"]))),
        name=f"target-blind-observer-fit-rank-{rank:04d}-d3-integrated",
    )
    mlp.validate()

    module = load_instrumented()
    flops.configure(symmetry_warnings=False)
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        context = flops.BudgetContext(
            flop_budget=PHASE2_BUDGET,
            wall_time_limit_s=120.0,
            quiet=True,
            namespace=f"p2r0c1d3_integrated_rank_{rank:04d}",
        )
        with context:
            host_value, raw_value, sigma_value = module.predict_with_tape(mlp)
    wall = time.perf_counter() - started
    summary = context.summary_dict(by_namespace=False)
    host = np.ascontiguousarray(np.asarray(host_value), dtype=np.dtype("<f4"))
    raw20 = np.ascontiguousarray(np.asarray(raw_value), dtype=np.dtype("<f4"))
    sigma = np.ascontiguousarray(np.asarray(sigma_value), dtype=np.dtype("<f4"))
    del host_value, raw_value, sigma_value, module
    gc.collect()
    if int(summary["flops_used"]) != EXPECTED_INTEGRATED_FLOPS:
        raise AssertionError(
            f"integrated FLOP bill changed: {summary['flops_used']} != {EXPECTED_INTEGRATED_FLOPS}"
        )
    if host.shape != (16, 1024) or not np.isfinite(host).all():
        raise AssertionError("host contract failed")
    if raw20.shape != (16, 1024, 20) or not np.isfinite(raw20).all():
        raise AssertionError("raw20 contract failed")
    if sigma.shape != (16, 1024) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise AssertionError("sigma contract failed")
    if not np.array_equal(raw20[:, :, 2], np.ones((16, 1024), dtype=np.float32)):
        raise AssertionError("full-birth retained feature changed")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays = {"host.npy": host, "raw20.npy": raw20, "sigma.npy": sigma}
    files: dict[str, object] = {}
    for name, value in arrays.items():
        path = args.output_dir / name
        np.save(path, value, allow_pickle=False)
        files[name] = {
            "file_sha256": sha256(path),
            "array_sha256": array_sha256(value),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    receipt: dict[str, object] = {
        "schema": "p2r0c1d3-integrated-host-row-v1",
        "status": "complete",
        "role": "observer_fit",
        "role_rank": rank,
        "mlp_id": int(entry["mlp_id"]),
        "mlp_seed": int(row_manifest["mlp_seed"]),
        "name_or_board_join": False,
        "target_roles_opened": [],
        "target_columns_opened": [],
        "instrumented_source_sha256": INSTRUMENTED_SHA256,
        "exact_host_source_sha256": SOURCE_HOST_SHA256,
        "meter_result_sha256": METER_RESULT_SHA256,
        "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
        "row_manifest_sha256": str(entry["sha256"]),
        "weights_file_sha256": str(weights_record["sha256"]),
        "files": files,
        "literal_meter": {
            "flops_used": int(summary["flops_used"]),
            "flop_budget": int(summary["flop_budget"]),
            "flops_remaining": int(summary["flops_remaining"]),
            "wall_time_s": float(summary["wall_time_s"]),
            "residual_wall_time_s": float(summary["residual_wall_time_s"]),
            "integrated_wall_time_s": wall,
            "operations": summary["operations"],
        },
    }
    write_json_once(args.output_dir / "result.json", receipt)
    print(
        json.dumps(
            {
                "rank": rank,
                "mlp_id": int(entry["mlp_id"]),
                "flops_used": int(summary["flops_used"]),
                "wall_time_s": wall,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
