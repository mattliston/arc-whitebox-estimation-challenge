#!/usr/bin/env python3
"""Audited one-session generator and finalizer for V4 target-blind features.

The default command is metadata-only preflight.  Generation is impossible
without a future independent source-audit receipt and exact release instance.
No target reader, optimizer, fit, or scorer is imported anywhere in this file.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from contract import (
    CAPS,
    CANDIDATE_SCHEMA,
    CLAIM_PATH,
    ENVIRONMENT_PYTHON,
    EXPECTED_ENVIRONMENT,
    FINALIZATION_RELEASE_SCHEMA,
    FINAL_SCHEMA,
    FROZEN_BINDINGS,
    GENERATION_RELEASE_SCHEMA,
    HERE,
    MAX_INVOCATIONS,
    MAX_WORKERS,
    MIN_FREE_BYTES,
    OUTPUT_PATH,
    PER_PROCESS_LIMIT_BYTES,
    RELEASE_SCHEMA_PATH,
    ROLE_BINDINGS,
    ROLE_ORDER,
    RSS_SAMPLE_INTERVAL_SECONDS,
    SCRATCH_PATH,
    SOURCE_MANIFEST_PATH,
    STAGING_PATH,
    SYSTEM_MEMORY_FRACTION_DENOMINATOR,
    SYSTEM_MEMORY_FRACTION_NUMERATOR,
    TERMINATION_GRACE_SECONDS,
    canonical_json_bytes,
    fsync_directory,
    load_exact_roster,
    raw_array_sha256,
    row_relative_path,
    sha256_file,
    validate_candidate_metadata,
    validate_row_directory,
    verify_frozen_bindings,
    write_exclusive_fsynced,
)


WORKER_SOURCE = HERE / "host_feature_worker.py"
PROVIDER_SOURCE = HERE / "sealed_feature_store_provider.py"
GENERATION_AUDIT_STATUS = "PASS_PHASE2_OBSERVER_V4_FEATURE_STORE_SOURCE_AUDIT"
ARTIFACT_AUDIT_STATUS = "PASS_PHASE2_OBSERVER_V4_FEATURE_STORE_ARTIFACT_AUDIT"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path))


def _physical_memory_bytes() -> int:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    total = page_size * pages
    if total <= 0:
        raise RuntimeError("cannot establish physical memory")
    return total


def _free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return int(stats.f_bavail) * int(stats.f_frsize)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "flopscope": importlib.metadata.version("flopscope"),
        "whestbench": importlib.metadata.version("whestbench"),
    }


def _load_after_claim(name: str, source: Path) -> Any:
    if not CLAIM_PATH.is_file() or CLAIM_PATH.is_symlink():
        raise PermissionError("generation claim is absent")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load post-claim source: {source.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def preflight() -> dict[str, Any]:
    frozen = verify_frozen_bindings()
    roster = load_exact_roster()
    resource_probe = _resource_capability_probe()
    physical = _physical_memory_bytes()
    runtime = {
        "claim": CLAIM_PATH.exists() or CLAIM_PATH.is_symlink(),
        "staging": STAGING_PATH.exists() or STAGING_PATH.is_symlink(),
        "output": OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink(),
        "scratch": SCRATCH_PATH.exists() or SCRATCH_PATH.is_symlink(),
    }
    release_instances = sorted(
        path.name
        for path in HERE.glob("*.json")
        if path.name not in {
            "STATIC_CONTRACT.json", "RELEASE_SCHEMA.json", "SOURCE_MANIFEST.json",
            "STATIC_PREFLIGHT_RECEIPT.json",
        }
        and "AUDIT" not in path.name
    )
    return {
        "status": "PASS_TARGET_FREE_FEATURE_STORE_SOURCE_PREFLIGHT__HOLD_GENERATION",
        "frozen_bindings": frozen,
        "roles": {role: len(rows) for role, rows in roster.items()},
        "caps": list(CAPS),
        "hard_max_workers": MAX_WORKERS,
        "system_memory_strictly_below_fraction": "1/4",
        "per_process_limit_bytes": PER_PROCESS_LIMIT_BYTES,
        "rss_capability_probe": resource_probe,
        "physical_memory_bytes": physical,
        "strict_process_tree_limit_bytes": physical // 4,
        "workspace_free_bytes": _free_bytes(HERE),
        "minimum_generation_free_bytes": MIN_FREE_BYTES,
        "runtime_paths": runtime,
        "release_schema_sha256": sha256_file(RELEASE_SCHEMA_PATH),
        "release_instances_present": release_instances,
        "feature_worker_imported": False,
        "full_reader_imported": False,
        "full_rows_iterated": 0,
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
    }


def _validate_common_release_paths(release: dict[str, Any]) -> None:
    exact = {
        "claim_path": str(CLAIM_PATH),
        "staging_path": str(STAGING_PATH),
        "output_path": str(OUTPUT_PATH),
        "scratch_path": str(SCRATCH_PATH),
        "environment_python": str(ENVIRONMENT_PYTHON),
    }
    for key, value in exact.items():
        if release.get(key) != value:
            raise PermissionError(f"release path changed: {key}")


def _validate_generation_release(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("generation release must be a regular file")
    release_sha = sha256_file(path)
    release = json.loads(path.read_text(encoding="utf-8"))
    if release.get("schema") != GENERATION_RELEASE_SCHEMA:
        raise PermissionError("wrong generation release schema")
    if release.get("status") != "AUTHORIZED_PHASE2_OBSERVER_V4_TARGET_BLIND_FEATURE_GENERATION":
        raise PermissionError("generation is not authorized")
    if release.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise PermissionError("generation release binds another source freeze")
    _validate_common_release_paths(release)
    exact = {
        "roles": {"observer_fit": 688, "observer_selection": 88},
        "caps": list(CAPS),
        "max_invocations": MAX_INVOCATIONS,
        "system_memory_fraction": "1/4",
        "per_process_limit_bytes": PER_PROCESS_LIMIT_BYTES,
        "v4_implementation_manifest_sha256": FROZEN_BINDINGS["v4_implementation_manifest"]["sha256"],
        "v4_launch_audit_sha256": FROZEN_BINDINGS["v4_independent_launch_audit"]["sha256"],
        "custody_post_seal_audit_sha256": FROZEN_BINDINGS["custody_post_seal_audit"]["sha256"],
        "roster_sha256": FROZEN_BINDINGS["full_roster"]["sha256"],
        "retry_policy": "same_consumed_session_claim_only",
        "target_access_authorized": False,
        "fit_or_score_authorized": False,
    }
    for key, value in exact.items():
        if release.get(key) != value:
            raise PermissionError(f"generation release binding changed: {key}")
    if not isinstance(release.get("max_workers"), int) or not 1 <= release["max_workers"] <= MAX_WORKERS:
        raise PermissionError("generation release worker bound changed")
    for key in ("authorized_by", "authorized_at_utc", "nonce"):
        if not isinstance(release.get(key), str) or not release[key]:
            raise PermissionError(f"generation release lacks {key}")
    audit_path = Path(release["independent_source_audit_path"])
    if audit_path.is_symlink() or not audit_path.is_file():
        raise PermissionError("independent source audit is absent")
    if sha256_file(audit_path) != release.get("independent_source_audit_sha256"):
        raise PermissionError("independent source audit hash changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != GENERATION_AUDIT_STATUS:
        raise PermissionError("independent source audit is not PASS")
    if audit.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise PermissionError("independent source audit binds another freeze")
    if release.get("independent_source_audit_status") != GENERATION_AUDIT_STATUS:
        raise PermissionError("release source-audit status changed")
    return release, release_sha


def _consume_claim(release: dict[str, Any], release_sha: str) -> dict[str, Any]:
    payload = {
        "schema": "phase2-observer-v4-feature-store-generation-session-claim-v1",
        "status": "CONSUMED_BEFORE_FULL_READER_IMPORT_OR_ANY_FULL_ROW",
        "created_at_utc": _utc_now(),
        "release_sha256": release_sha,
        "release_nonce": release["nonce"],
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "roles": {role: ROLE_BINDINGS[role]["rows"] for role in ROLE_ORDER},
        "caps": list(CAPS),
        "same_claim_resume_allowed": True,
        "maximum_invocations": MAX_INVOCATIONS,
        "another_release_or_session_allowed": False,
        "full_reader_imported_before_claim": False,
        "full_rows_iterated_before_claim": 0,
        "role_target_artifacts_opened_or_hashed_before_claim": 0,
        "target_values_decoded_before_claim": 0,
        "optimizers_constructed_before_claim": 0,
        "fits_before_claim": 0,
        "scores_before_claim": 0,
    }
    encoded = canonical_json_bytes(payload)
    write_exclusive_fsynced(CLAIM_PATH, encoded)
    return {**payload, "claim_sha256": hashlib.sha256(encoded).hexdigest()}


def _load_claim(release: dict[str, Any], release_sha: str) -> dict[str, Any]:
    if CLAIM_PATH.is_symlink() or not CLAIM_PATH.is_file():
        raise PermissionError("resume requires the consumed regular claim")
    claim = json.loads(CLAIM_PATH.read_text(encoding="utf-8"))
    if claim.get("status") != "CONSUMED_BEFORE_FULL_READER_IMPORT_OR_ANY_FULL_ROW":
        raise PermissionError("generation claim status changed")
    if claim.get("release_sha256") != release_sha or claim.get("release_nonce") != release["nonce"]:
        raise PermissionError("resume release differs from consumed claim")
    if claim.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise PermissionError("generation claim binds another source freeze")
    return {**claim, "claim_sha256": sha256_file(CLAIM_PATH)}


def _initialize_session(claim: dict[str, Any], release: dict[str, Any]) -> None:
    STAGING_PATH.mkdir(mode=0o700)
    (STAGING_PATH / "rows").mkdir(mode=0o700)
    for role in ROLE_ORDER:
        (STAGING_PATH / "rows" / role).mkdir(mode=0o700)
    (STAGING_PATH / "invocations").mkdir(mode=0o700)
    SCRATCH_PATH.mkdir(mode=0o700)
    session = {
        "schema": "phase2-observer-v4-feature-store-generation-session-v1",
        "status": "ACTIVE_CLAIMED_TARGET_BLIND_GENERATION_SESSION",
        "claim_sha256": claim["claim_sha256"],
        "release_nonce": release["nonce"],
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "created_at_utc": _utc_now(),
        "roles": {role: ROLE_BINDINGS[role]["rows"] for role in ROLE_ORDER},
        "caps": list(CAPS),
        "target_values_decoded": 0,
    }
    write_exclusive_fsynced(STAGING_PATH / "session.json", canonical_json_bytes(session))
    for path in (STAGING_PATH / "rows" / role for role in ROLE_ORDER):
        fsync_directory(path)
    fsync_directory(STAGING_PATH / "rows")
    fsync_directory(STAGING_PATH)
    fsync_directory(SCRATCH_PATH)


def _validate_session(claim: dict[str, Any], release: dict[str, Any]) -> None:
    if STAGING_PATH.is_symlink() or not STAGING_PATH.is_dir():
        raise PermissionError("resume staging directory is absent")
    if SCRATCH_PATH.is_symlink() or not SCRATCH_PATH.is_dir():
        raise PermissionError("resume scratch directory is absent")
    session = json.loads((STAGING_PATH / "session.json").read_text(encoding="utf-8"))
    if session.get("claim_sha256") != claim["claim_sha256"]:
        raise PermissionError("staging session binds another claim")
    if session.get("release_nonce") != release["nonce"]:
        raise PermissionError("staging session binds another release")


def _consume_invocation(claim: dict[str, Any], *, resume: bool) -> tuple[int, Path]:
    starts = sorted((STAGING_PATH / "invocations").glob("invocation-*.START.json"))
    if len(starts) >= MAX_INVOCATIONS:
        raise PermissionError("generation invocation budget exhausted")
    index = len(starts)
    path = STAGING_PATH / "invocations" / f"invocation-{index:02d}.START.json"
    payload = {
        "schema": "phase2-observer-v4-feature-store-invocation-start-v1",
        "status": "STARTED_UNDER_EXISTING_SESSION_CLAIM",
        "invocation_index": index,
        "resume": resume,
        "started_at_utc": _utc_now(),
        "claim_sha256": claim["claim_sha256"],
        "target_values_decoded": 0,
    }
    write_exclusive_fsynced(path, canonical_json_bytes(payload))
    return index, path


def _completed_rows() -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for role in ROLE_ORDER:
        parent = STAGING_PATH / row_relative_path(role, 0).parent
        observed = sorted(path for path in parent.iterdir() if path.is_dir())
        for path in observed:
            if not path.name.isdigit() or len(path.name) != 4:
                raise AssertionError("unexpected completed-row directory")
            rank = int(path.name)
            validate_row_directory(path, role=role, role_rank=rank)
            completed.add((role, rank))
    return completed


class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32),
        ("syscalls_unix", ctypes.c_int32),
        ("csw", ctypes.c_int32),
        ("threadnum", ctypes.c_int32),
        ("numrunning", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


def _darwin_libproc() -> Any:
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pidinfo.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
    ]
    library.proc_pidinfo.restype = ctypes.c_int
    library.proc_listpids.argtypes = [
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int,
    ]
    library.proc_listpids.restype = ctypes.c_int
    return library


def _pid_rss_bytes(pid: int, library: Any | None = None) -> int:
    if sys.platform == "darwin":
        libproc = library or _darwin_libproc()
        info = _ProcTaskInfo()
        returned = libproc.proc_pidinfo(
            pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info)
        )
        if returned != ctypes.sizeof(info):
            return 0
        return int(info.resident_size)
    statm = Path(f"/proc/{pid}/statm")
    if not statm.is_file():
        return 0
    resident_pages = int(statm.read_text(encoding="ascii").split()[1])
    return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))


def _process_group_pids(pgid: int, library: Any | None = None) -> list[int]:
    if sys.platform != "darwin":
        result = []
        for candidate in Path("/proc").iterdir():
            if not candidate.name.isdigit():
                continue
            pid = int(candidate.name)
            try:
                if os.getpgid(pid) == pgid:
                    result.append(pid)
            except (ProcessLookupError, PermissionError):
                continue
        return sorted(result)
    libproc = library or _darwin_libproc()
    capacity_bytes = max(4096, int(libproc.proc_listpids(2, pgid, None, 0)))
    buffer = (ctypes.c_int32 * (capacity_bytes // ctypes.sizeof(ctypes.c_int32)))()
    returned = int(libproc.proc_listpids(2, pgid, buffer, ctypes.sizeof(buffer)))
    if returned < 0:
        raise OSError(ctypes.get_errno(), "proc_listpids(PROC_PGRP_ONLY) failed")
    return sorted(pid for pid in buffer[: returned // 4] if pid > 0)


def _resource_capability_probe() -> dict[str, Any]:
    library = _darwin_libproc() if sys.platform == "darwin" else None
    rss = _pid_rss_bytes(os.getpid(), library)
    if rss <= 0:
        raise RuntimeError("live RSS capability probe failed before claim")
    return {
        "backend": "darwin_libproc_process_groups" if sys.platform == "darwin" else "linux_procfs_process_groups",
        "probe_pid": os.getpid(),
        "probe_rss_bytes": rss,
    }


def _sample_resources(
    running: dict[int, dict[str, Any]], physical: int, sample_handle: Any
) -> dict[str, Any]:
    library = _darwin_libproc() if sys.platform == "darwin" else None
    parent_rss = _pid_rss_bytes(os.getpid(), library)
    worker_rss = {}
    worker_group_pids = {}
    for pid in sorted(running):
        group_pids = _process_group_pids(pid, library)
        if pid not in group_pids and running[pid]["process"].poll() is None:
            raise RuntimeError("live worker is absent from its isolated process group")
        worker_group_pids[str(pid)] = group_pids
        worker_rss[str(pid)] = sum(_pid_rss_bytes(member, library) for member in group_pids)
    tree_rss = parent_rss + sum(worker_rss.values())
    sample = {
        "monotonic_seconds": time.monotonic(),
        "process_tree_rss_bytes": tree_rss,
        "worker_subtree_rss_bytes": worker_rss,
        "worker_process_group_pids": worker_group_pids,
        "parent_rss_bytes": parent_rss,
        "physical_memory_bytes": physical,
        "strict_system_limit_bytes": (
            physical * SYSTEM_MEMORY_FRACTION_NUMERATOR
        ) // SYSTEM_MEMORY_FRACTION_DENOMINATOR,
        "per_process_limit_bytes": PER_PROCESS_LIMIT_BYTES,
        "running_workers": len(running),
    }
    sample_handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
    sample_handle.flush()
    if tree_rss * SYSTEM_MEMORY_FRACTION_DENOMINATOR >= physical * SYSTEM_MEMORY_FRACTION_NUMERATOR:
        raise MemoryError("process tree reached the strict one-quarter system-memory boundary")
    if any(value >= PER_PROCESS_LIMIT_BYTES for value in worker_rss.values()):
        raise MemoryError("worker subtree reached the strict 8 GiB boundary")
    return sample


def _read_resource_samples(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            result.append(json.loads(line))
    return result


def _terminate_running(running: dict[int, dict[str, Any]]) -> dict[str, Any]:
    terminated: list[int] = []
    killed: list[int] = []
    for pid, item in sorted(running.items()):
        process = item["process"]
        if process.poll() is None:
            process.terminate()
            terminated.append(pid)
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline and any(
        item["process"].poll() is None for item in running.values()
    ):
        time.sleep(0.05)
    for pid, item in sorted(running.items()):
        process = item["process"]
        if process.poll() is None:
            process.kill()
            killed.append(pid)
        process.wait(timeout=10)
        item["stdout"].close()
        item["stderr"].close()
    return {
        "sigterm_pids": terminated,
        "sigkill_pids": killed,
        "workers_reaped": len(running),
    }


def _write_array_exclusive(path: Path, value: Any) -> None:
    import numpy as np  # post-claim only

    array = np.ascontiguousarray(value, dtype=np.dtype("<f4"))
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _launch_row(
    row: Any,
    spec: dict[str, Any],
    role: str,
    role_rank: int,
    invocation_scratch: Path,
) -> tuple[int, dict[str, Any]]:
    partial = invocation_scratch / f"{int(row.ordinal):04d}-{role}-{role_rank:04d}.partial"
    partial.mkdir(mode=0o700)
    weights_path = partial / "weights.npy"
    _write_array_exclusive(weights_path, row.weights)
    if raw_array_sha256(row.weights) != spec["weights_sha256"]:
        raise AssertionError("full-row weights differ from frozen roster attestation")
    task = {
        "schema": "phase2-observer-v4-feature-row-task-v1",
        "role": role,
        "role_rank": role_rank,
        "mlp_id": int(spec["mlp_id"]),
        "mlp_seed": int(spec["mlp_seed"]),
        "source_ordinal": int(spec["ordinal"]),
        "source_shard_index": int(spec["shard_index"]),
        "source_row_in_shard": int(spec["row_in_shard"]),
        "weights_raw_sha256": spec["weights_sha256"],
        "row_partial_path": str(partial),
        "host_caps": list(CAPS),
        "target_values_decoded": 0,
    }
    task_path = invocation_scratch / f"{partial.name}.task.json"
    write_exclusive_fsynced(task_path, canonical_json_bytes(task))
    fsync_directory(partial)
    stdout_path = invocation_scratch / f"{partial.name}.stdout.log"
    stderr_path = invocation_scratch / f"{partial.name}.stderr.log"
    stdout = stdout_path.open("xb")
    stderr = stderr_path.open("xb")
    process = subprocess.Popen(
        [str(ENVIRONMENT_PYTHON), str(WORKER_SOURCE), "--task", str(task_path)],
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        close_fds=True,
        start_new_session=True,
    )
    return process.pid, {
        "process": process,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "partial": partial,
        "role": role,
        "role_rank": role_rank,
        "mlp_id": int(spec["mlp_id"]),
        "launched_monotonic": time.monotonic(),
    }


def _seal_completed_process(pid: int, item: dict[str, Any]) -> dict[str, Any]:
    process = item["process"]
    returncode = process.wait()
    item["stdout"].close()
    item["stderr"].close()
    if returncode != 0:
        stderr_tail = item["stderr_path"].read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"feature worker {pid} failed ({returncode}): {stderr_tail}")
    role = item["role"]
    rank = item["role_rank"]
    validate_row_directory(item["partial"], role=role, role_rank=rank)
    final_dir = STAGING_PATH / row_relative_path(role, rank)
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError("atomic row destination already exists")
    os.replace(item["partial"], final_dir)
    fsync_directory(final_dir.parent)
    return {
        "role": role,
        "role_rank": rank,
        "mlp_id": item["mlp_id"],
        "worker_pid": pid,
        "wall_seconds": time.monotonic() - item["launched_monotonic"],
        "row_manifest_sha256": sha256_file(final_dir / "row_manifest.json"),
    }


def _poll_until_slot(
    running: dict[int, dict[str, Any]],
    max_workers: int,
    physical: int,
    sample_handle: Any,
    samples: list[dict[str, Any]],
    completed_receipts: list[dict[str, Any]],
) -> None:
    while len(running) >= max_workers:
        samples.append(_sample_resources(running, physical, sample_handle))
        ready = [pid for pid, item in running.items() if item["process"].poll() is not None]
        if not ready:
            time.sleep(RSS_SAMPLE_INTERVAL_SECONDS)
            continue
        for pid in sorted(ready):
            completed_receipts.append(_seal_completed_process(pid, running.pop(pid)))


def _write_abort_receipt(
    invocation_index: int,
    reason: BaseException,
    actions: dict[str, Any],
    launched: int,
    completed: int,
    completed_before: int,
) -> None:
    receipt = {
        "schema": "phase2-observer-v4-feature-store-abort-receipt-v1",
        "status": "ABORTED__CLAIM_REMAINS_CONSUMED__RESUME_SAME_SESSION_ONLY",
        "invocation_index": invocation_index,
        "aborted_at_utc": _utc_now(),
        "reason_type": type(reason).__name__,
        "reason": str(reason),
        "launched_rows": launched,
        "completed_rows_this_invocation": completed,
        "unlaunched_rows_cancelled": 776 - completed_before - completed,
        "termination": actions,
        "target_values_decoded": 0,
        "fits": 0,
        "scores": 0,
    }
    path = STAGING_PATH / "invocations" / f"invocation-{invocation_index:02d}.ABORT.json"
    write_exclusive_fsynced(path, canonical_json_bytes(receipt))


def _candidate_manifest(claim: dict[str, Any], execution_receipt_sha: str) -> dict[str, Any]:
    roster = load_exact_roster()
    roles: dict[str, Any] = {}
    for role in ROLE_ORDER:
        entries = []
        for rank in range(ROLE_BINDINGS[role]["rows"]):
            manifest_path = STAGING_PATH / row_relative_path(role, rank) / "row_manifest.json"
            entries.append(
                {
                    "role_rank": rank,
                    "mlp_id": int(roster[role][rank]["mlp_id"]),
                    "path": str(row_relative_path(role, rank) / "row_manifest.json"),
                    "sha256": sha256_file(manifest_path),
                }
            )
        roles[role] = {
            "rows": ROLE_BINDINGS[role]["rows"],
            "ordered_mlp_ids_sha256": ROLE_BINDINGS[role]["ordered_mlp_ids_sha256"],
            "row_manifests": entries,
        }
    return {
        "schema": CANDIDATE_SCHEMA,
        "status": "READY_FOR_INDEPENDENT_ARTIFACT_AUDIT__NOT_FINAL_SEALED",
        "created_at_utc": _utc_now(),
        "claim_sha256": claim["claim_sha256"],
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "provider_source_sha256": sha256_file(PROVIDER_SOURCE),
        "host_caps": list(CAPS),
        "roles": roles,
        "fields": [
            "mlp_id", "raw_locals", "edge_matrices", "host_rows", "sigma",
            "entering_cp_state_hashes", "host_latch",
        ],
        "edge_materialization": "deterministic_on_demand_from_sealed_weights",
        "execution_receipt_sha256": execution_receipt_sha,
        "resource_samples_sha256": sha256_file(STAGING_PATH / "resource_samples.jsonl"),
        "target_blind_full_source_sha256": FROZEN_BINDINGS["target_blind_full_reader"]["sha256"],
        "full_cache_audit_sha256": FROZEN_BINDINGS["full_cache_audit"]["sha256"],
        "roster_sha256": FROZEN_BINDINGS["full_roster"]["sha256"],
        "v4_implementation_manifest_sha256": FROZEN_BINDINGS["v4_implementation_manifest"]["sha256"],
        "v4_launch_audit_sha256": FROZEN_BINDINGS["v4_independent_launch_audit"]["sha256"],
        "custody_post_seal_audit_sha256": FROZEN_BINDINGS["custody_post_seal_audit"]["sha256"],
        "raw_full_dataset_target_projection": "forbidden",
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
        "independent_feature_store_audit_status": "HOLD",
    }


def execute_generation(release_path: Path, *, resume: bool) -> dict[str, Any]:
    release, release_sha = _validate_generation_release(release_path)
    if _real(Path(sys.executable)) != _real(ENVIRONMENT_PYTHON):
        raise PermissionError("generation must use the exact bound feature environment")
    if _environment() != EXPECTED_ENVIRONMENT:
        raise PermissionError("feature generation package environment changed")
    resource_capability = _resource_capability_probe()
    verify_frozen_bindings()
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError("final feature store already exists")
    physical = _physical_memory_bytes()
    strict_limit = physical // 4
    if strict_limit <= PER_PROCESS_LIMIT_BYTES:
        raise MemoryError("system cannot support even one guarded worker below one-quarter RAM")
    if _free_bytes(HERE) < MIN_FREE_BYTES:
        raise OSError("feature generation requires at least 60 GiB free")

    if resume:
        claim = _load_claim(release, release_sha)
        _validate_session(claim, release)
    else:
        for path in (CLAIM_PATH, STAGING_PATH, SCRATCH_PATH):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"fresh generation path is not absent: {path.name}")
        claim = _consume_claim(release, release_sha)
        _initialize_session(claim, release)
    invocation_index, _ = _consume_invocation(claim, resume=resume)
    invocation_scratch = SCRATCH_PATH / f"invocation-{invocation_index:02d}"
    invocation_scratch.mkdir(mode=0o700)
    fsync_directory(SCRATCH_PATH)

    completed_before = _completed_rows()
    if (STAGING_PATH / "candidate_manifest.json").exists():
        if len(completed_before) != 776:
            raise AssertionError("candidate exists without all rows")
        validate_candidate_metadata(STAGING_PATH / "candidate_manifest.json")
        return {
            "status": "READY_FOR_INDEPENDENT_ARTIFACT_AUDIT__NOT_FINAL_SEALED",
            "candidate_manifest_sha256": sha256_file(STAGING_PATH / "candidate_manifest.json"),
            "rows": 776,
            "target_values_decoded": 0,
        }

    # Every data-capable import is below the durable O_EXCL claim.
    import numpy as np  # noqa: F401  (post-claim environment assertion)
    full_reader = _load_after_claim(
        "v4_feature_store_target_blind_full_reader",
        Path(FROZEN_BINDINGS["target_blind_full_reader"]["path"]),
    )
    roster = load_exact_roster()
    lookup: dict[int, tuple[str, int, dict[str, Any]]] = {}
    for role in ROLE_ORDER:
        for rank, spec in enumerate(roster[role]):
            lookup[int(spec["mlp_id"])] = (role, rank, spec)

    running: dict[int, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    completed_receipts: list[dict[str, Any]] = []
    launched = 0
    seen_selected: set[int] = set()
    started = time.perf_counter()
    samples_path = STAGING_PATH / "resource_samples.jsonl"
    sample_mode = "a" if samples_path.exists() else "x"
    sample_handle = samples_path.open(sample_mode, encoding="utf-8", buffering=1)
    try:
        for row in full_reader.iter_target_blind_full():
            selected = lookup.get(int(row.mlp_id))
            if selected is None:
                continue
            role, rank, spec = selected
            seen_selected.add(int(row.mlp_id))
            if (
                int(row.ordinal) != int(spec["ordinal"])
                or int(row.shard_index) != int(spec["shard_index"])
                or int(row.row_in_shard) != int(spec["row_in_shard"])
                or int(row.mlp_seed) != int(spec["mlp_seed"])
                or row.mlp_name != spec["mlp_name"]
                or tuple(row.decoded_columns) != ("mlp_id", "mlp_name", "mlp_seed", "weights")
            ):
                raise AssertionError("target-blind full row differs from frozen roster")
            if (role, rank) in completed_before:
                continue
            _poll_until_slot(
                running, int(release["max_workers"]), physical, sample_handle,
                samples, completed_receipts,
            )
            pid, item = _launch_row(row, spec, role, rank, invocation_scratch)
            if pid in running:
                raise AssertionError("worker PID collision")
            running[pid] = item
            launched += 1
            samples.append(_sample_resources(running, physical, sample_handle))
        if seen_selected != set(lookup):
            raise AssertionError("full reader did not cover the exact 776-row roster")
        while running:
            _poll_until_slot(
                running, 1, physical, sample_handle, samples, completed_receipts
            )
    except BaseException as error:
        actions = _terminate_running(running)
        running.clear()
        sample_handle.flush()
        os.fsync(sample_handle.fileno())
        sample_handle.close()
        _write_abort_receipt(
            invocation_index, error, actions, launched, len(completed_receipts),
            len(completed_before),
        )
        raise
    sample_handle.flush()
    os.fsync(sample_handle.fileno())
    sample_handle.close()

    completed_after = _completed_rows()
    if len(completed_after) != 776:
        raise AssertionError("generation ended without exactly 776 validated rows")
    all_samples = _read_resource_samples(samples_path)
    if not all_samples:
        raise AssertionError("generation completed without a resource sample")
    peak_tree = max(sample["process_tree_rss_bytes"] for sample in all_samples)
    peak_worker = max(
        (
            max(sample["worker_subtree_rss_bytes"].values(), default=0)
            for sample in all_samples
        ),
        default=0,
    )
    if peak_tree * 4 >= physical or peak_worker >= PER_PROCESS_LIMIT_BYTES:
        raise MemoryError("completed generation receipt violates strict memory bounds")
    execution_receipt = {
        "schema": "phase2-observer-v4-feature-store-generation-receipt-v1",
        "status": "COMPLETE_TARGET_BLIND_FEATURE_ROWS__HOLD_FINAL_SEAL_FOR_INDEPENDENT_AUDIT",
        "completed_at_utc": _utc_now(),
        "claim_sha256": claim["claim_sha256"],
        "invocation_index": invocation_index,
        "completed_before_invocation": len(completed_before),
        "launched_this_invocation": launched,
        "completed_this_invocation": len(completed_receipts),
        "total_validated_rows": len(completed_after),
        "roles": {role: ROLE_BINDINGS[role]["rows"] for role in ROLE_ORDER},
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "resource_samples": len(all_samples),
        "peak_process_tree_rss_bytes": peak_tree,
        "physical_memory_bytes": physical,
        "strict_system_limit_bytes": physical // 4,
        "peak_worker_subtree_rss_bytes": peak_worker,
        "strict_per_process_limit_bytes": PER_PROCESS_LIMIT_BYTES,
        "hard_max_workers": int(release["max_workers"]),
        "resource_samples_sha256": sha256_file(samples_path),
        "deterministic_schedule": "ascending_full_source_ordinal",
        "rss_backend": resource_capability["backend"],
        "preclaim_rss_capability_probe_bytes": resource_capability["probe_rss_bytes"],
        "atomic_row_seals": len(completed_receipts),
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
    }
    execution_path = STAGING_PATH / "GENERATION_RECEIPT.json"
    if execution_path.exists():
        prior = json.loads(execution_path.read_text(encoding="utf-8"))
        if (
            prior.get("status")
            != "COMPLETE_TARGET_BLIND_FEATURE_ROWS__HOLD_FINAL_SEAL_FOR_INDEPENDENT_AUDIT"
            or prior.get("claim_sha256") != claim["claim_sha256"]
            or prior.get("total_validated_rows") != 776
            or prior.get("resource_samples_sha256") != sha256_file(samples_path)
        ):
            raise AssertionError("pre-existing generation receipt is not resumable")
    else:
        write_exclusive_fsynced(execution_path, canonical_json_bytes(execution_receipt))
    candidate = _candidate_manifest(claim, sha256_file(execution_path))
    candidate_path = STAGING_PATH / "candidate_manifest.json"
    write_exclusive_fsynced(candidate_path, canonical_json_bytes(candidate))
    fsync_directory(STAGING_PATH)
    return {
        "status": candidate["status"],
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "execution_receipt_sha256": sha256_file(execution_path),
        "rows": len(completed_after),
        "target_values_decoded": 0,
        "fits": 0,
        "scores": 0,
    }


def _validate_finalization_release(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("finalization release must be a regular file")
    release = json.loads(path.read_text(encoding="utf-8"))
    if release.get("schema") != FINALIZATION_RELEASE_SCHEMA:
        raise PermissionError("wrong finalization release schema")
    if release.get("status") != "AUTHORIZED_PHASE2_OBSERVER_V4_FEATURE_STORE_FINALIZATION":
        raise PermissionError("feature finalization is not authorized")
    exact = {
        "staging_path": str(STAGING_PATH),
        "output_path": str(OUTPUT_PATH),
        "target_access_authorized": False,
        "fit_or_score_authorized": False,
    }
    for key, value in exact.items():
        if release.get(key) != value:
            raise PermissionError(f"finalization release changed: {key}")
    candidate_path = STAGING_PATH / "candidate_manifest.json"
    if release.get("candidate_manifest_sha256") != sha256_file(candidate_path):
        raise PermissionError("finalization release binds another candidate")
    if release.get("generation_claim_sha256") != sha256_file(CLAIM_PATH):
        raise PermissionError("finalization release binds another claim")
    audit_path = Path(release["independent_artifact_audit_path"])
    if audit_path.is_symlink() or not audit_path.is_file():
        raise PermissionError("independent artifact audit is absent")
    if release.get("independent_artifact_audit_sha256") != sha256_file(audit_path):
        raise PermissionError("independent artifact audit hash changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != ARTIFACT_AUDIT_STATUS:
        raise PermissionError("independent artifact audit is not PASS")
    if audit.get("candidate_manifest_sha256") != sha256_file(candidate_path):
        raise PermissionError("independent artifact audit binds another candidate")
    if audit.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise PermissionError("independent artifact audit binds another source freeze")
    if release.get("independent_artifact_audit_status") != ARTIFACT_AUDIT_STATUS:
        raise PermissionError("finalization audit status changed")
    for key in ("authorized_by", "authorized_at_utc", "nonce"):
        if not isinstance(release.get(key), str) or not release[key]:
            raise PermissionError(f"finalization release lacks {key}")
    return release, audit


def _fsync_tree(path: Path) -> None:
    for directory, subdirectories, filenames in os.walk(path, topdown=False):
        current = Path(directory)
        for name in filenames:
            file_path = current / name
            descriptor = os.open(file_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        fsync_directory(current)


def finalize_store(release_path: Path) -> dict[str, Any]:
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError("final feature store already exists")
    if STAGING_PATH.is_symlink() or not STAGING_PATH.is_dir():
        raise PermissionError("candidate staging store is absent")
    release, audit = _validate_finalization_release(release_path)
    candidate_path = STAGING_PATH / "candidate_manifest.json"
    candidate = validate_candidate_metadata(candidate_path)
    for role in ROLE_ORDER:
        for rank, entry in enumerate(candidate["roles"][role]["row_manifests"]):
            row_dir = STAGING_PATH / row_relative_path(role, rank)
            if sha256_file(row_dir / "row_manifest.json") != entry["sha256"]:
                raise AssertionError("candidate row-manifest digest changed")
            validate_row_directory(row_dir, role=role, role_rank=rank)

    audit_source = Path(release["independent_artifact_audit_path"])
    audit_copy = STAGING_PATH / "INDEPENDENT_FEATURE_STORE_ARTIFACT_AUDIT.json"
    write_exclusive_fsynced(audit_copy, audit_source.read_bytes())
    if sha256_file(audit_copy) != release["independent_artifact_audit_sha256"]:
        raise AssertionError("copied artifact audit digest changed")
    final_manifest = {
        "schema": FINAL_SCHEMA,
        "status": "SEALED_TARGET_BLIND_FEATURE_STORE",
        "sealed_at_utc": _utc_now(),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "provider_source_sha256": sha256_file(PROVIDER_SOURCE),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "generation_claim_sha256": sha256_file(CLAIM_PATH),
        "host_caps": list(CAPS),
        "roles": candidate["roles"],
        "fields": candidate["fields"],
        "edge_materialization": candidate["edge_materialization"],
        "target_blind_full_source_sha256": FROZEN_BINDINGS["target_blind_full_reader"]["sha256"],
        "full_cache_audit_sha256": FROZEN_BINDINGS["full_cache_audit"]["sha256"],
        "roster_sha256": FROZEN_BINDINGS["full_roster"]["sha256"],
        "v4_implementation_manifest_sha256": FROZEN_BINDINGS["v4_implementation_manifest"]["sha256"],
        "v4_launch_audit_sha256": FROZEN_BINDINGS["v4_independent_launch_audit"]["sha256"],
        "custody_post_seal_audit_sha256": FROZEN_BINDINGS["custody_post_seal_audit"]["sha256"],
        "independent_feature_store_audit_status": "PASS",
        "independent_feature_store_audit_sha256": release["independent_artifact_audit_sha256"],
        "raw_full_dataset_target_projection": "forbidden",
        "role_target_artifacts_opened_or_hashed": 0,
        "target_columns_decoded": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
    }
    manifest_path = STAGING_PATH / "manifest.json"
    write_exclusive_fsynced(manifest_path, canonical_json_bytes(final_manifest))
    finalization_receipt = {
        "schema": "phase2-observer-v4-feature-store-finalization-receipt-v1",
        "status": "ATOMIC_FINAL_SEAL_COMPLETE",
        "finalized_at_utc": _utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "independent_artifact_audit_sha256": release["independent_artifact_audit_sha256"],
        "rows": 776,
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "fits": 0,
        "scores": 0,
    }
    write_exclusive_fsynced(
        STAGING_PATH / "FINALIZATION_RECEIPT.json", canonical_json_bytes(finalization_receipt)
    )
    _fsync_tree(STAGING_PATH)
    for directory, subdirectories, filenames in os.walk(STAGING_PATH, topdown=False):
        current = Path(directory)
        for name in filenames:
            (current / name).chmod(0o400)
        for name in subdirectories:
            (current / name).chmod(0o500)
    STAGING_PATH.chmod(0o500)
    fsync_directory(STAGING_PATH)
    os.replace(STAGING_PATH, OUTPUT_PATH)
    fsync_directory(HERE)
    return {
        "status": "SEALED_TARGET_BLIND_FEATURE_STORE",
        "manifest_sha256": sha256_file(OUTPUT_PATH / "manifest.json"),
        "rows": 776,
        "target_values_decoded": 0,
        "fits": 0,
        "scores": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--resume", action="store_true")
    actions.add_argument("--finalize", action="store_true")
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if not (args.execute or args.resume or args.finalize):
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return
    if args.release is None:
        raise SystemExit("an independently issued exact release is required")
    if args.finalize:
        report = finalize_store(args.release)
    else:
        report = execute_generation(args.release, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
