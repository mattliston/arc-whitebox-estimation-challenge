#!/usr/bin/env python3
"""D4 rank-35 control through FlopScope's RemoteArray client."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import types
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1" / "estimator.py"
FIXTURE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0035/weights.npy"
)
D4_RESULT = (
    P2
    / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
    / "D4_CANDIDATE_RESULT.json"
)
D16_DIR = P2 / "p2r0c1d16_cp3_arena_depth5_accounting_repair_v1"
WATCH_328627 = D16_DIR / "SUBMIT_WATCH.log"
WATCH_328628 = D16_DIR / "SUBMIT_WATCH_OFFICIAL.log"
D21_SETUP_FAILURE = (
    P2
    / "p2r0c1d21_d5_remote_rank35_repro_v1"
    / "D21_SERVER_TELEMETRY_SANDBOX_BIND_FAILURE_aaad0363.json"
)
RESULT = HERE / "D23_D4_REMOTE_RANK35_CONTROL_RESULT.json"
RAW_OUTPUT = HERE / "D23_D4_REMOTE_RANK35_OUTPUT.f32"
D22_DIR = P2 / "p2r0c1d22_d5_remote_rank35_ipc_permission_v1"
D22_RESULT = D22_DIR / "D22_REMOTE_RANK35_CLIENT_RESULT.json"
D22_TELEMETRY = D22_DIR / "D22_SERVER_TELEMETRY.json"

SOURCE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
FIXTURE_SHA256 = "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2"
D4_RESULT_SHA256 = "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1"
WATCH_328627_SHA256 = "38c824fc579fdf726d5dc5136f6510a3d92e6bff7ae87817943956e6532a4d19"
WATCH_328628_SHA256 = "f12fafbf5ea8397f691f561c759a2f3a7f80f3fbff1e17db2e1840cf93bf64f3"
D21_SETUP_FAILURE_SHA256 = (
    "aaad0363c6344232646836de375646e212f0462877cb47cea91ee4d919dfa263"
)
D22_RESULT_SHA256 = "a20d4acf382dccdc96b6403521c098648fedb1db75e2207265a7a37896687b29"
D22_TELEMETRY_SHA256 = "513f6fbf8e486e8e6708080fbb6ab70ebb9767b4feef4dfc0c4206dc93a77fb8"
PHASE2_BUDGET = 2**41
EXPECTED_NATIVE_OUTPUT_SHA256 = (
    "2acfbe6f8acbe6ba525284f681ed90158751c018e1d838415a9ef8e00d5781af"
)
PROC_PIDTASKINFO = 4


class TaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


LIBSYSTEM = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def current_rss_bytes() -> int:
    if platform.system() != "Darwin":
        raise RuntimeError("D23 requires Darwin")
    info = TaskInfo()
    written = LIBSYSTEM.proc_pidinfo(
        ctypes.c_int(os.getpid()),
        ctypes.c_int(PROC_PIDTASKINFO),
        ctypes.c_uint64(0),
        ctypes.byref(info),
        ctypes.c_int(ctypes.sizeof(info)),
    )
    if written != ctypes.sizeof(info):
        raise RuntimeError(("proc_pidinfo", written, ctypes.sizeof(info)))
    return int(info.pti_resident_size)


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def atomic_write(payload: dict[str, object]) -> None:
    temporary = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULT)


def atomic_write_raw(raw: bytes) -> None:
    temporary = RAW_OUTPUT.with_name(f".{RAW_OUTPUT.name}.{os.getpid()}.tmp")
    temporary.write_bytes(raw)
    temporary.replace(RAW_OUTPUT)


def install_whestbench_stub() -> type:
    """Install only the two inert grader interfaces imported by the source."""

    class BaseEstimator:
        pass

    @dataclass(frozen=True)
    class MLP:
        width: int
        depth: int
        weights: list[Any]
        seed: int = 0
        name: str = ""

        def validate(self) -> None:
            if self.width <= 0 or self.depth <= 0:
                raise ValueError("invalid MLP geometry")
            if len(self.weights) != self.depth:
                raise ValueError("MLP depth mismatch")
            for weight in self.weights:
                if tuple(weight.shape) != (self.width, self.width):
                    raise ValueError(("weight shape", weight.shape))

    package = types.ModuleType("whestbench")
    domain = types.ModuleType("whestbench.domain")
    package.BaseEstimator = BaseEstimator
    package.MLP = MLP
    package.domain = domain
    domain.MLP = MLP
    sys.modules["whestbench"] = package
    sys.modules["whestbench.domain"] = domain
    return MLP


def load_estimator_module():
    spec = importlib.util.spec_from_file_location("d23_remote_d4_estimator", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def describe_remote_tree(value: Any, path: str = "root") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[int] = set()

    def visit(item: Any, item_path: str) -> None:
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        handle = getattr(item, "handle_id", None)
        if isinstance(handle, str):
            rows.append(
                {
                    "path": item_path,
                    "handle_id": handle,
                    "shape": list(getattr(item, "shape", ())),
                    "dtype": str(getattr(item, "dtype", "unknown")),
                    "nbytes": int(getattr(item, "nbytes", 0)),
                }
            )
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]")
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{item_path}[{key!r}]")

    visit(value, path)
    return rows


def watch_binding(path: Path, expected_id: int, expected_sha: str) -> dict[str, object]:
    if sha256(path) != expected_sha:
        raise AssertionError((path, sha256(path), expected_sha))
    payload = json.loads(path.read_text().strip())
    submission = payload.get("submission", {})
    if (
        payload.get("submission_id") != expected_id
        or submission.get("id") != expected_id
        or submission.get("grading_status_cd") != "failed"
        or submission.get("score") is not None
    ):
        raise AssertionError((expected_id, payload))
    return {
        "id": expected_id,
        "status": "failed",
        "score": None,
        "watch_sha256": expected_sha,
    }


def array_sha256(raw: bytes, shape: tuple[int, ...], dtype: str) -> str:
    dtype_spelling = {"float32": "<f4"}.get(dtype)
    if dtype_spelling is None:
        raise ValueError(("unexpected output dtype", dtype))
    digest = hashlib.sha256()
    digest.update(dtype_spelling.encode("ascii"))
    digest.update(json.dumps(list(shape), separators=(",", ":")).encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def main() -> None:
    for output_path in (RESULT, RAW_OUTPUT):
        if output_path.exists():
            raise FileExistsError(output_path)
    if importlib.util.find_spec("numpy") is not None:
        raise AssertionError("client environment unexpectedly contains NumPy")
    pins = {
        SOURCE: SOURCE_SHA256,
        FIXTURE: FIXTURE_SHA256,
        D4_RESULT: D4_RESULT_SHA256,
        WATCH_328627: WATCH_328627_SHA256,
        WATCH_328628: WATCH_328628_SHA256,
        D21_SETUP_FAILURE: D21_SETUP_FAILURE_SHA256,
        D22_RESULT: D22_RESULT_SHA256,
        D22_TELEMETRY: D22_TELEMETRY_SHA256,
    }
    for path, expected in pins.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))
    versions = {
        "python": sys.version,
        "executable": sys.executable,
        "flopscope_client_distribution": importlib.metadata.version(
            "flopscope-client"
        ),
        "flopscope_client_runtime": getattr(flops, "__version__", None),
        "numpy_available": False,
        "server_url": os.environ.get("FLOPSCOPE_SERVER_URL"),
    }
    if versions["flopscope_client_distribution"] != "0.12.0":
        raise AssertionError(versions)

    mlp_type = install_whestbench_stub()
    module = load_estimator_module()
    estimator = module.Estimator()
    watches = [
        watch_binding(WATCH_328627, 328627, WATCH_328627_SHA256),
        watch_binding(WATCH_328628, 328628, WATCH_328628_SHA256),
    ]
    before_rss = current_rss_bytes()
    before_peak = peak_rss_bytes()
    context = flops.BudgetContext(
        flop_budget=PHASE2_BUDGET,
        quiet=True,
        namespace="d23_d4_remote_rank35",
    )
    output = None
    output_record = None
    fixture_record = None
    exception = None
    context_exception = None
    started = time.perf_counter()
    try:
        with context:
            weights_blob = fnp.load(str(FIXTURE))
            weights = [weights_blob[layer] for layer in range(16)]
            mlp = mlp_type(
                width=1024,
                depth=16,
                weights=weights,
                seed=0,
                name="sealed-weight-fixture-rank35",
            )
            mlp.validate()
            fixture_record = {
                "root_handle_id": weights_blob.handle_id,
                "root_shape": list(weights_blob.shape),
                "root_dtype": str(weights_blob.dtype),
                "layer_handle_ids": [weight.handle_id for weight in weights],
            }
            try:
                output = estimator.predict(mlp, PHASE2_BUDGET)
                raw, shape, dtype = output._fetch_data()
                atomic_write_raw(raw)
                digest = array_sha256(raw, tuple(shape), dtype)
                output_record = {
                    "handle_id": output.handle_id,
                    "shape": list(shape),
                    "dtype": dtype,
                    "array_sha256": digest,
                    "byte_identical_to_native_d4": digest
                    == EXPECTED_NATIVE_OUTPUT_SHA256,
                    "raw_output_file": RAW_OUTPUT.name,
                    "raw_output_file_sha256": sha256(RAW_OUTPUT),
                }
            except BaseException as error:
                exception = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "live_client_flops_used_after_error": context.flops_used,
                    "live_client_flops_remaining_after_error": context.flops_remaining,
                }
    except BaseException as error:
        context_exception = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    outer_wall = time.perf_counter() - started
    try:
        summary = context.summary_dict(by_namespace=True)
    except BaseException as error:
        summary = None
        if context_exception is None:
            context_exception = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
    after_rss = current_rss_bytes()
    after_peak = peak_rss_bytes()
    scratch = describe_remote_tree(
        getattr(module, "_PRODUCTION_SCRATCH_CONTEXT", None),
        "_PRODUCTION_SCRATCH_CONTEXT",
    )
    payload = {
        "schema": "p2r0c1-d4-flopscope-client-server-rank35-control-v1",
        "status": (
            "REMOTE_RETURNED_OUTPUT"
            if output_record is not None and exception is None and context_exception is None
            else "REMOTE_EXCEPTION_REPRODUCED"
        ),
        "source_changed": False,
        "role_rank": 35,
        "reference_estimator_calls": 0,
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "training": False,
        "network": False,
        "official_client_environment": versions,
        "failed_submission_bindings": watches,
        "fixture": fixture_record,
        "exception": exception,
        "context_exception": context_exception,
        "output": output_record,
        "literal_meter": summary,
        "exact_outer_wall_seconds": outer_wall,
        "client_memory": {
            "current_rss_before_bytes": before_rss,
            "current_rss_after_bytes": after_rss,
            "current_rss_growth_bytes": after_rss - before_rss,
            "peak_rss_before_bytes": before_peak,
            "peak_rss_after_bytes": after_peak,
            "peak_rss_growth_bytes": after_peak - before_peak,
        },
        "persistent_scratch_remote_handles": scratch,
        "persistent_scratch_remote_handle_count": len(scratch),
        "bindings": {
            "runner_sha256": sha256(Path(__file__)),
            "candidate_source_sha256": sha256(SOURCE),
            "fixture_sha256": sha256(FIXTURE),
            "d4_native_full_gate_result_sha256": sha256(D4_RESULT),
            "d21_sandbox_bind_failure_sha256": sha256(D21_SETUP_FAILURE),
            "d22_remote_d5_result_sha256": sha256(D22_RESULT),
            "d22_remote_d5_telemetry_sha256": sha256(D22_TELEMETRY),
        },
    }
    atomic_write(payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
