#!/usr/bin/env python3
"""One-row CP1536/CP1792 target-blind host-feature worker.

The address-space guard is installed before NumPy or Flopscope is imported.
This worker accepts only a parent-written target-blind weight file and emits
one unsealed row-local directory.  The parent performs the atomic row seal.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any


PROCESS_LIMIT_BYTES = 8 * (1 << 30)
_old_soft, _old_hard = resource.getrlimit(resource.RLIMIT_AS)
_infinity = resource.RLIM_INFINITY
_new_soft = PROCESS_LIMIT_BYTES
if _old_hard != _infinity:
    _new_soft = min(_new_soft, _old_hard)
resource.setrlimit(resource.RLIMIT_AS, (_new_soft, _old_hard))

import numpy as np  # noqa: E402  (intentionally after RLIMIT_AS)
import flopscope as flops  # noqa: E402

from contract import (  # noqa: E402
    CAPS,
    DEPTH,
    EXPECTED_ENVIRONMENT,
    FROZEN_BINDINGS,
    LOCAL_FEATURES,
    PER_PROCESS_LIMIT_BYTES,
    ROLE_ORDER,
    ROW_SCHEMA,
    SCRATCH_PATH,
    WIDTH,
    array_payload_sha256,
    canonical_json_bytes,
    fsync_directory,
    raw_array_sha256,
    sha256_file,
    verify_frozen_bindings,
    write_exclusive_fsynced,
)


FLOP_BUDGET = 2**41


def _strict_child(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    base = parent.resolve()
    if resolved == base or base not in resolved.parents:
        raise PermissionError(f"worker path escapes scratch root: {path}")
    return resolved


def _load_frozen_host_worker() -> Any:
    source = Path(FROZEN_BINDINGS["v4_host_worker"]["path"])
    if sha256_file(source) != FROZEN_BINDINGS["v4_host_worker"]["sha256"]:
        raise AssertionError("frozen V4 host worker changed")
    spec = importlib.util.spec_from_file_location("v4_feature_store_frozen_host", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen V4 host worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _write_npy_exclusive(path: Path, value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=np.dtype("<f4"))
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "sha256": sha256_file(path),
        "array_sha256": array_payload_sha256(array),
        "bytes": path.stat().st_size,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _max_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _validate_task(task: dict[str, Any], task_path: Path) -> tuple[Path, dict[str, Any]]:
    if task.get("schema") != "phase2-observer-v4-feature-row-task-v1":
        raise PermissionError("wrong row task schema")
    if task.get("host_caps") != list(CAPS):
        raise PermissionError("row task caps changed")
    if task.get("target_values_decoded") != 0:
        raise PermissionError("row task claims target values")
    if task.get("role") not in ROLE_ORDER:
        raise PermissionError("worker role is not fit/selection")
    row_dir = _strict_child(Path(task["row_partial_path"]), SCRATCH_PATH)
    resolved_task = _strict_child(task_path, SCRATCH_PATH)
    if resolved_task.parent != row_dir.parent or resolved_task.parent.name == row_dir.name:
        raise PermissionError("task and row scratch paths do not share one invocation root")
    if not row_dir.is_dir() or row_dir.is_symlink():
        raise PermissionError("row scratch directory is absent or a symlink")
    if set(path.name for path in row_dir.iterdir()) != {"weights.npy"}:
        raise PermissionError("row scratch directory has unexpected pre-worker members")
    return row_dir, task


def run(task_path: Path) -> dict[str, Any]:
    if PROCESS_LIMIT_BYTES != PER_PROCESS_LIMIT_BYTES or _new_soft > PER_PROCESS_LIMIT_BYTES:
        raise AssertionError("worker process guard changed")
    environment = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "flopscope": importlib.metadata.version("flopscope"),
        "whestbench": importlib.metadata.version("whestbench"),
    }
    if environment != EXPECTED_ENVIRONMENT:
        raise PermissionError("feature worker environment changed")
    verify_frozen_bindings()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    row_dir, task = _validate_task(task, task_path)

    weights_path = row_dir / "weights.npy"
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or tuple(weights.shape) != (DEPTH, WIDTH, WIDTH):
        raise AssertionError("worker weights schema changed")
    if not weights.flags.c_contiguous or not np.isfinite(weights).all():
        raise AssertionError("worker weights are invalid")
    if raw_array_sha256(weights) != task["weights_raw_sha256"]:
        raise AssertionError("worker weights differ from roster attestation")

    files: dict[str, dict[str, Any]] = {
        "weights.npy": {
            "sha256": sha256_file(weights_path),
            "array_sha256": array_payload_sha256(weights),
            "bytes": weights_path.stat().st_size,
            "dtype": weights.dtype.str,
            "shape": list(weights.shape),
        }
    }
    frozen = _load_frozen_host_worker()
    flops.configure(symmetry_warnings=False)
    hosts: dict[str, dict[str, Any]] = {}
    started_all = time.perf_counter()
    for cap in CAPS:
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as context:
            tape, report = frozen._host_tape(weights, cap)
        wall_seconds = time.perf_counter() - started
        raw = np.ascontiguousarray(tape["raw32"], dtype=np.dtype("<f4"))
        host = np.ascontiguousarray(tape["host"], dtype=np.dtype("<f4"))
        sigma = np.ascontiguousarray(tape["sigma"], dtype=np.dtype("<f4"))
        if raw.shape != (DEPTH, WIDTH, LOCAL_FEATURES) or not np.isfinite(raw).all():
            raise AssertionError("frozen host raw-local schema changed")
        if host.shape != (DEPTH, WIDTH) or not np.isfinite(host).all() or np.any(host < 0):
            raise AssertionError("frozen host-row schema changed")
        if sigma.shape != (DEPTH, WIDTH) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise AssertionError("frozen sigma schema changed")
        if report["raw32_sha256"] != array_payload_sha256(raw):
            raise AssertionError("frozen host raw-local receipt mismatch")
        if report["host_sha256"] != array_payload_sha256(host):
            raise AssertionError("frozen host-row receipt mismatch")
        if report["sigma_sha256"] != array_payload_sha256(sigma):
            raise AssertionError("frozen sigma receipt mismatch")

        files[f"cp{cap}_raw_locals.npy"] = _write_npy_exclusive(
            row_dir / f"cp{cap}_raw_locals.npy", raw
        )
        files[f"cp{cap}_host_rows.npy"] = _write_npy_exclusive(
            row_dir / f"cp{cap}_host_rows.npy", host
        )
        files[f"cp{cap}_sigma.npy"] = _write_npy_exclusive(
            row_dir / f"cp{cap}_sigma.npy", sigma
        )
        hashes = list(report["entering_cp_state_hashes"])
        if len(hashes) != DEPTH:
            raise AssertionError("frozen entering-CP chain changed")
        hosts[str(cap)] = {
            **report,
            "flops_used": int(context.flops_used),
            "wall_seconds": wall_seconds,
            "process_max_rss_bytes_after_cap": _max_rss_bytes(),
            "process_limit_bytes": PER_PROCESS_LIMIT_BYTES,
        }
        del tape, raw, host, sigma
        gc.collect()

    if _max_rss_bytes() >= PER_PROCESS_LIMIT_BYTES:
        raise MemoryError("worker reached the strict 8 GiB process boundary")
    manifest = {
        "schema": ROW_SCHEMA,
        "status": "SEALED_ROW_LOCAL_FEATURES",
        "role": task["role"],
        "role_rank": int(task["role_rank"]),
        "mlp_id": int(task["mlp_id"]),
        "mlp_seed": int(task["mlp_seed"]),
        "source_ordinal": int(task["source_ordinal"]),
        "source_shard_index": int(task["source_shard_index"]),
        "source_row_in_shard": int(task["source_row_in_shard"]),
        "decoded_columns": ["mlp_id", "mlp_name", "mlp_seed", "weights"],
        "weights_raw_sha256": task["weights_raw_sha256"],
        "host_caps": list(CAPS),
        "host_latch": [True] * DEPTH,
        "edge_materialization": "deterministic_on_demand_from_weights",
        "files": files,
        "hosts": hosts,
        "worker_environment": environment,
        "worker_source_sha256": sha256_file(Path(__file__)),
        "frozen_host_worker_sha256": FROZEN_BINDINGS["v4_host_worker"]["sha256"],
        "frozen_host_base_sha256": FROZEN_BINDINGS["v3_host_base"]["sha256"],
        "total_worker_wall_seconds": time.perf_counter() - started_all,
        "worker_max_rss_bytes": _max_rss_bytes(),
        "target_columns_projected": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
    }
    write_exclusive_fsynced(row_dir / "row_manifest.json", canonical_json_bytes(manifest))
    fsync_directory(row_dir)
    return {
        "status": "PASS_ROW_LOCAL_TARGET_BLIND_HOST_FEATURES__UNSEALED_BY_PARENT",
        "role": task["role"],
        "role_rank": int(task["role_rank"]),
        "mlp_id": int(task["mlp_id"]),
        "row_manifest_sha256": sha256_file(row_dir / "row_manifest.json"),
        "worker_max_rss_bytes": _max_rss_bytes(),
        "target_values_decoded": 0,
        "fits": 0,
        "scores": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.task)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
