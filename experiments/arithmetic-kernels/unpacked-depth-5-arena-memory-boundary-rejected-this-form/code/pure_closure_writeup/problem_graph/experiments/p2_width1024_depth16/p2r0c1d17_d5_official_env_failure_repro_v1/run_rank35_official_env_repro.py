#!/usr/bin/env python3
"""One-shot target-free rank-35 reproduction of failed submission #328627.

This runner must be invoked with ``data_gate/.venv/bin/python``.  It executes
the unchanged submitted estimator exactly once and records either its output or
the complete exception together with the partial literal meter, residual, RSS,
and environment versions.  It does not package, submit, or open target data.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D15 = P2 / "p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1"
D16 = P2 / "p2r0c1d16_cp3_arena_depth5_accounting_repair_v1"
SOURCE = D15 / "estimator.py"
SUBMIT_WATCH = D16 / "SUBMIT_WATCH.log"
FIXTURE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3/"
    "sealed_target_blind_feature_store_v3/rows/observer_fit/0035/weights.npy"
)
ATTEMPT = HERE / "ATTEMPT_MARKER.json"
RESULT = HERE / "OFFICIAL_ENV_RANK35_REPRO.json"

SOURCE_SHA256 = "74095215f842a0a993ad89c94cbaa27baf2cbc2f8cb2868e490972be8b9ab5e5"
SUBMIT_WATCH_SHA256 = "38c824fc579fdf726d5dc5136f6510a3d92e6bff7ae87817943956e6532a4d19"
FIXTURE_SHA256 = "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2"
PHASE2_BUDGET = 2**41
THREADS = 2


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


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_estimator_module():
    spec = importlib.util.spec_from_file_location("d5_official_env_repro_estimator", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if RESULT.exists() or ATTEMPT.exists():
        raise FileExistsError("official-environment reproduction already attempted")
    pins = {
        SOURCE: SOURCE_SHA256,
        SUBMIT_WATCH: SUBMIT_WATCH_SHA256,
        FIXTURE: FIXTURE_SHA256,
    }
    for path, expected in pins.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))

    watch = json.loads(SUBMIT_WATCH.read_text().strip())
    submission = watch.get("submission", {})
    if (
        watch.get("ok") is not True
        or submission.get("id") != 328627
        or submission.get("grading_status_cd") != "failed"
        or submission.get("score") is not None
    ):
        raise AssertionError("#328627 failed-board binding changed")

    versions = {
        "python": sys.version,
        "executable": sys.executable,
        "flopscope_runtime": getattr(flops, "__version__", None),
        "flopscope_distribution": importlib.metadata.version("flopscope"),
        "whestbench_distribution": importlib.metadata.version("whestbench"),
        "numpy": np.__version__,
    }
    expected_executable_tail = "data_gate/.venv/bin/python"
    if expected_executable_tail not in str(Path(sys.executable)):
        # The venv python is a symlink, so sys.executable may resolve to the
        # environment's bin path but must still identify data_gate explicitly.
        invoked = os.environ.get("D5_OFFICIAL_ENV_INVOKED_AS", "")
        if expected_executable_tail not in invoked:
            raise AssertionError(("wrong reproduction interpreter", versions))
    if (
        versions["flopscope_distribution"] != "0.12.0"
        or versions["whestbench_distribution"] != "0.16.0"
        or versions["numpy"] != "2.4.6"
    ):
        raise AssertionError(("official environment version drift", versions))

    write_json_once(
        ATTEMPT,
        {
            "schema": "p2r0c1-d5-official-env-rank35-attempt-v1",
            "submission_id": 328627,
            "pid": os.getpid(),
            "created_unix_seconds": time.time(),
            "source_sha256": SOURCE_SHA256,
            "submit_watch_sha256": SUBMIT_WATCH_SHA256,
            "fixture_sha256": FIXTURE_SHA256,
            "versions": versions,
            "target_free": True,
            "authorized_calls": 1,
        },
    )

    weights = np.load(FIXTURE, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise AssertionError("rank35 fixture schema changed")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=0,
        name="sealed-weight-fixture",
    )
    mlp.validate()
    estimator = load_estimator_module().Estimator()
    context = flops.BudgetContext(
        flop_budget=PHASE2_BUDGET,
        wall_time_limit_s=120.0,
        quiet=True,
        namespace="d5_official_env_rank35_repro",
    )
    started = time.perf_counter()
    status = "UNKNOWN"
    exception_payload = None
    output_payload = None
    try:
        with threadpool_limits(limits=THREADS):
            with context:
                value = estimator.predict(mlp, PHASE2_BUDGET)
        raw = np.asarray(value)
        output_payload = {
            "dtype": raw.dtype.str,
            "shape": list(raw.shape),
            "finite": bool(np.isfinite(raw).all()),
            "array_sha256": array_sha256(raw),
        }
        status = "RETURNED_OUTPUT"
    except BaseException as error:
        status = "RAISED_EXCEPTION"
        exception_payload = {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "repr": repr(error),
            "traceback": traceback.format_exc(),
        }
    outer_wall = time.perf_counter() - started
    try:
        meter = context.summary_dict(by_namespace=False)
    except BaseException as meter_error:
        meter = {
            "summary_error_type": f"{type(meter_error).__module__}.{type(meter_error).__qualname__}",
            "summary_error_message": str(meter_error),
        }
    payload = {
        "schema": "p2r0c1-d5-official-env-rank35-repro-result-v1",
        "status": status,
        "submission_binding": {
            "id": 328627,
            "grading_status_cd": "failed",
            "score": None,
            "submit_watch_sha256": SUBMIT_WATCH_SHA256,
        },
        "source_sha256": SOURCE_SHA256,
        "fixture_sha256": FIXTURE_SHA256,
        "role_rank": 35,
        "candidate_calls": 1,
        "reference_calls": 0,
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened_during_repro": False,
        "submission_during_repro": False,
        "versions": versions,
        "exact_outer_wall_seconds": outer_wall,
        "maximum_process_rss_bytes": resident_bytes(),
        "literal_meter": meter,
        "exception": exception_payload,
        "output": output_payload,
        "bindings": {
            "runner_sha256": sha256(Path(__file__)),
            "attempt_marker_sha256": sha256(ATTEMPT),
        },
    }
    write_json_once(RESULT, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
