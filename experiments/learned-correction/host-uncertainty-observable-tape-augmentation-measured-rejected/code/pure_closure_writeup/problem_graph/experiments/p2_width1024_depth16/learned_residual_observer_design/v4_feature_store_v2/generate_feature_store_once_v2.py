#!/usr/bin/env python3
"""V2 guarded-cursor feature generator; default invocation is preflight only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from contract_v2 import (
    CAPS, CLAIM_PATH, ENVIRONMENT_PYTHON, EXPECTED_ENVIRONMENT, FINAL_SCHEMA,
    FROZEN_BINDINGS, HERE, MAX_HOST_WORKERS, MAX_INVOCATIONS, MIN_FREE_BYTES,
    OUTPUT_PATH, PER_PROCESS_LIMIT_BYTES, RELEASE_SCHEMA_PATH, ROLE_BINDINGS,
    ROLE_ORDER, SCRATCH_PATH, SOURCE_MANIFEST_PATH, STAGING_PATH, V1_BINDINGS,
    WATCHDOG_INTERVAL_SECONDS, MAX_WATCHDOG_GAP_SECONDS, canonical_json_bytes,
    fsync_directory, load_exact_roster, row_relative_path, sha256_file,
    validate_row_directory, verify_v1_bindings, write_exclusive_fsynced,
)
from resource_safety_v2 import (
    GroupRecord, GroupRegistry, ResourceViolation, ResourceWatchdog,
    pid_rss_bytes, retire_completed_group,
)


CURSOR_SOURCE = HERE / "target_blind_cursor_v2.py"
HOST_GATE_SOURCE = HERE / "host_worker_exec_gate_v2.py"
PROVIDER_SOURCE = HERE / "sealed_feature_store_provider_v2.py"
SOURCE_AUDIT_STATUS = "PASS_PHASE2_OBSERVER_V4_FEATURE_STORE_V2_SOURCE_AUDIT"
ARTIFACT_AUDIT_STATUS = "PASS_PHASE2_OBSERVER_V4_FEATURE_STORE_V2_ARTIFACT_AUDIT"
GENERATION_RELEASE_KEYS = {
    "schema", "status", "authorized_by", "authorized_at_utc", "nonce",
    "source_manifest_sha256", "independent_v2_source_audit_path",
    "independent_v2_source_audit_sha256", "independent_v2_source_audit_status",
    "v1_hold_audit_sha256", "claim_path", "staging_path", "output_path",
    "scratch_path", "environment_python", "roles", "caps", "max_host_workers",
    "watchdog_interval_seconds", "maximum_watchdog_gap_seconds",
    "system_memory_fraction", "per_group_limit_bytes", "target_access_authorized",
    "fit_or_score_authorized",
}
FINALIZATION_RELEASE_KEYS = {
    "schema", "status", "authorized_by", "authorized_at_utc", "nonce",
    "source_manifest_sha256", "v1_hold_audit_sha256", "generation_claim_sha256",
    "candidate_manifest_sha256", "independent_artifact_audit_path",
    "independent_artifact_audit_sha256", "independent_artifact_audit_status",
    "staging_path", "output_path", "target_access_authorized",
    "fit_or_score_authorized",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _physical_memory_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


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


def preflight() -> dict[str, Any]:
    v1 = verify_v1_bindings()
    audit = json.loads(V1_BINDINGS["independent_source_audit_hold"]["path"].read_text(encoding="utf-8"))
    if audit.get("status") != "HOLD_PHASE2_OBSERVER_V4_FEATURE_STORE_SOURCE_AUDIT__RESOURCE_SUPERVISOR_INCOMPLETE":
        raise AssertionError("V1 independent HOLD status changed")
    roster = load_exact_roster()
    runtime = {
        path.name: path.exists() or path.is_symlink()
        for path in (CLAIM_PATH, STAGING_PATH, OUTPUT_PATH, SCRATCH_PATH)
    }
    return {
        "status": "PASS_TARGET_FREE_V2_PREFLIGHT__HOLD_REAL_GENERATION_FOR_INDEPENDENT_AUDIT",
        "v1_bindings": v1,
        "v1_hold_status": audit["status"],
        "roles": {role: len(roster[role]) for role in ROLE_ORDER},
        "caps": list(CAPS),
        "runtime_paths": runtime,
        "rss_capability_probe_bytes": pid_rss_bytes(os.getpid()),
        "watchdog_interval_seconds": WATCHDOG_INTERVAL_SECONDS,
        "maximum_watchdog_gap_seconds": MAX_WATCHDOG_GAP_SECONDS,
        "group_abort": "killpg_TERM_grace_killpg_KILL_leader_wait_three_empty_group_polls",
        "cursor_protocol": "BOOT_0__START_iterator_0__guard_each_NEXT__guard_each_WRITE",
        "release_instances": 0,
        "claims": 0,
        "real_full_rows_decoded": 0,
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "fits": 0,
        "scores": 0,
    }


def _validate_release(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("V2 generation release must be a regular file")
    release = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if release.get("schema") != "phase2-observer-v4-feature-store-v2-generation-release-v1":
        raise PermissionError("wrong V2 generation release schema")
    if release.get("status") != "AUTHORIZED_PHASE2_OBSERVER_V4_TARGET_BLIND_FEATURE_GENERATION_V2":
        raise PermissionError("V2 generation is not authorized")
    if set(release) != GENERATION_RELEASE_KEYS:
        raise PermissionError("V2 generation release field membership changed")
    exact = {
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "v1_hold_audit_sha256": V1_BINDINGS["independent_source_audit_hold"]["sha256"],
        "claim_path": str(CLAIM_PATH), "staging_path": str(STAGING_PATH),
        "output_path": str(OUTPUT_PATH), "scratch_path": str(SCRATCH_PATH),
        "environment_python": str(ENVIRONMENT_PYTHON),
        "roles": {"observer_fit": 688, "observer_selection": 88},
        "caps": list(CAPS), "watchdog_interval_seconds": WATCHDOG_INTERVAL_SECONDS,
        "maximum_watchdog_gap_seconds": MAX_WATCHDOG_GAP_SECONDS,
        "system_memory_fraction": "1/4", "per_group_limit_bytes": PER_PROCESS_LIMIT_BYTES,
        "target_access_authorized": False, "fit_or_score_authorized": False,
    }
    for key, value in exact.items():
        if release.get(key) != value:
            raise PermissionError(f"V2 release binding changed: {key}")
    if not isinstance(release.get("max_host_workers"), int) or not 1 <= release["max_host_workers"] <= 2:
        raise PermissionError("V2 host-worker bound changed")
    for key in ("authorized_by", "authorized_at_utc", "nonce"):
        if not isinstance(release.get(key), str) or not release[key]:
            raise PermissionError(f"V2 release lacks {key}")
    audit_path = Path(release["independent_v2_source_audit_path"])
    if audit_path.is_symlink() or not audit_path.is_file():
        raise PermissionError("V2 independent source audit is absent")
    if sha256_file(audit_path) != release.get("independent_v2_source_audit_sha256"):
        raise PermissionError("V2 independent source audit hash changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != SOURCE_AUDIT_STATUS:
        raise PermissionError("V2 independent source audit is not PASS")
    if audit.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise PermissionError("V2 source audit binds another freeze")
    if release.get("independent_v2_source_audit_status") != SOURCE_AUDIT_STATUS:
        raise PermissionError("V2 release audit status changed")
    return release, digest


def _claim(release: dict[str, Any], release_sha: str) -> dict[str, Any]:
    payload = {
        "schema": "phase2-observer-v4-feature-store-v2-session-claim-v1",
        "status": "CONSUMED_BEFORE_CURSOR_PROCESS_READER_IMPORT_ITERATOR_OR_ROW_DECODE",
        "created_at_utc": _utc_now(), "release_sha256": release_sha,
        "release_nonce": release["nonce"], "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "cursor_processes_started_before_claim": 0, "readers_imported_before_claim": 0,
        "iterators_constructed_before_claim": 0, "rows_decoded_before_claim": 0,
        "role_target_artifacts_opened_or_hashed": 0, "target_values_decoded": 0,
        "fits": 0, "scores": 0, "same_claim_resume_only": True,
    }
    encoded = canonical_json_bytes(payload)
    write_exclusive_fsynced(CLAIM_PATH, encoded)
    return {**payload, "claim_sha256": hashlib.sha256(encoded).hexdigest()}


def _load_claim(release: dict[str, Any], release_sha: str) -> dict[str, Any]:
    if CLAIM_PATH.is_symlink() or not CLAIM_PATH.is_file():
        raise PermissionError("V2 resume requires the consumed claim")
    claim = json.loads(CLAIM_PATH.read_text(encoding="utf-8"))
    if claim.get("release_sha256") != release_sha or claim.get("release_nonce") != release["nonce"]:
        raise PermissionError("V2 resume release differs from claim")
    return {**claim, "claim_sha256": sha256_file(CLAIM_PATH)}


def _initialize(claim: dict[str, Any], release: dict[str, Any]) -> None:
    STAGING_PATH.mkdir(mode=0o700)
    (STAGING_PATH / "rows").mkdir(mode=0o700)
    for role in ROLE_ORDER:
        (STAGING_PATH / "rows" / role).mkdir(mode=0o700)
    (STAGING_PATH / "invocations").mkdir(mode=0o700)
    SCRATCH_PATH.mkdir(mode=0o700)
    session = {
        "schema": "phase2-observer-v4-feature-store-v2-session-v1",
        "status": "ACTIVE_GUARDED_CURSOR_SESSION", "claim_sha256": claim["claim_sha256"],
        "release_nonce": release["nonce"], "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "target_values_decoded": 0,
    }
    write_exclusive_fsynced(STAGING_PATH / "session.json", canonical_json_bytes(session))


def _validate_session(claim: dict[str, Any], release: dict[str, Any]) -> None:
    session = json.loads((STAGING_PATH / "session.json").read_text(encoding="utf-8"))
    if session.get("claim_sha256") != claim["claim_sha256"] or session.get("release_nonce") != release["nonce"]:
        raise PermissionError("V2 staging session differs from claim/release")


def _invocation(claim: dict[str, Any], resume: bool) -> tuple[int, Path]:
    starts = sorted((STAGING_PATH / "invocations").glob("invocation-*.START.json"))
    if len(starts) >= MAX_INVOCATIONS:
        raise PermissionError("V2 invocation budget exhausted")
    index = len(starts)
    path = STAGING_PATH / "invocations" / f"invocation-{index:02d}.START.json"
    write_exclusive_fsynced(path, canonical_json_bytes({
        "schema": "phase2-observer-v4-feature-store-v2-invocation-start-v1",
        "status": "STARTED_UNDER_CONSUMED_SESSION_CLAIM", "index": index,
        "resume": resume, "claim_sha256": claim["claim_sha256"], "started_at_utc": _utc_now(),
    }))
    return index, path


def _completed() -> set[tuple[str, int]]:
    result = set()
    for role in ROLE_ORDER:
        parent = STAGING_PATH / "rows" / role
        for path in sorted(item for item in parent.iterdir() if item.is_dir()):
            if not path.name.isdigit():
                raise AssertionError("V2 row directory name changed")
            rank = int(path.name)
            validate_row_directory(path, role=role, role_rank=rank)
            result.add((role, rank))
    return result


def _send_cursor(process: subprocess.Popen[str], payload: dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("cursor stdin is absent")
    process.stdin.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    process.stdin.flush()


def _read_cursor(process: subprocess.Popen[str], watchdog: ResourceWatchdog) -> dict[str, Any]:
    if process.stdout is None:
        raise RuntimeError("cursor stdout is absent")
    line = process.stdout.readline()  # watchdog thread remains continuous while blocked
    watchdog.check()
    if not line:
        raise RuntimeError("cursor exited before protocol reply")
    return json.loads(line)


def _start_cursor(registry: GroupRegistry, watchdog: ResourceWatchdog, scratch: Path) -> tuple[GroupRecord, Any]:
    watchdog.guard("pre_cursor_process_start")
    stderr = (scratch / "cursor.stderr.log").open("xb")
    process = subprocess.Popen(
        [str(ENVIRONMENT_PYTHON), str(CURSOR_SOURCE)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1, close_fds=True,
        start_new_session=True,
    )
    record = registry.register(process, "target-blind-one-row-cursor")
    boot = _read_cursor(process, watchdog)
    if (
        boot.get("event") != "BOOTED_ZERO_DECODE_CURSOR"
        or boot.get("decoded_rows") != 0
        or boot.get("process_limit_bytes") != PER_PROCESS_LIMIT_BYTES
        or boot.get("strict_group_rss_watchdog_required") is not True
        or not isinstance(boot.get("rlimit_as_installed"), bool)
    ):
        raise AssertionError("cursor boot crossed the decode boundary")
    watchdog.guard("pre_iterator_construction")
    _send_cursor(process, {"op": "START"})
    ready = _read_cursor(process, watchdog)
    if ready.get("event") != "ITERATOR_READY_ZERO_DECODE" or ready.get("decoded_rows") != 0:
        raise AssertionError("cursor iterator construction decoded a row")
    return record, stderr


def _launch_host(
    registry: GroupRegistry, watchdog: ResourceWatchdog, partial: Path,
    spec: dict[str, Any], role: str, rank: int, scratch: Path,
) -> tuple[GroupRecord, dict[str, Any]]:
    task = {
        "schema": "phase2-observer-v4-feature-row-task-v1", "role": role,
        "role_rank": rank, "mlp_id": int(spec["mlp_id"]), "mlp_seed": int(spec["mlp_seed"]),
        "source_ordinal": int(spec["ordinal"]), "source_shard_index": int(spec["shard_index"]),
        "source_row_in_shard": int(spec["row_in_shard"]),
        "weights_raw_sha256": spec["weights_sha256"], "row_partial_path": str(partial),
        "host_caps": list(CAPS), "target_values_decoded": 0,
    }
    task_path = scratch / f"{partial.name}.task.json"
    write_exclusive_fsynced(task_path, canonical_json_bytes(task))
    stdout = (scratch / f"{partial.name}.stdout.log").open("xb")
    stderr = (scratch / f"{partial.name}.stderr.log").open("xb")
    gate_path = scratch / f"{partial.name}.HEAVY_EXEC.GATE"
    boot_path = scratch / f"{partial.name}.LIGHTWEIGHT_BOOT.json"
    watchdog.guard("pre_host_process_start")
    process = subprocess.Popen(
        [
            str(ENVIRONMENT_PYTHON), str(HOST_GATE_SOURCE), "--gate", str(gate_path),
            "--boot", str(boot_path), "--task", str(task_path),
        ],
        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True,
        start_new_session=True,
    )
    record = registry.register(process, f"host-{role}-{rank:04d}")
    while not boot_path.is_file():
        watchdog.check()
        if process.poll() is not None:
            raise RuntimeError("lightweight host exec gate exited before BOOT")
        time.sleep(0.01)
    boot = json.loads(boot_path.read_text(encoding="utf-8"))
    if boot.get("status") != "BOOTED_LIGHTWEIGHT_ZERO_HEAVY_IMPORTS_WAITING_FOR_MONITORED_GATE":
        raise AssertionError("host exec-gate BOOT status changed")
    watchdog.guard("pre_monitored_host_heavy_import_and_compute_exec")
    gate_partial = gate_path.with_name(gate_path.name + ".partial")
    write_exclusive_fsynced(gate_partial, b"MONITORED_HEAVY_EXEC\n")
    os.replace(gate_partial, gate_path)
    fsync_directory(gate_path.parent)
    return record, {
        "record": record, "partial": partial, "role": role, "rank": rank,
        "stdout": stdout, "stderr": stderr, "stderr_path": Path(stderr.name),
        "started": time.monotonic(),
    }


def _finish_host(registry: GroupRegistry, item: dict[str, Any]) -> dict[str, Any]:
    record = item["record"]
    returncode = record.process.wait()
    item["stdout"].close(); item["stderr"].close()
    quiescence = retire_completed_group(registry, record)
    if returncode != 0:
        tail = item["stderr_path"].read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"V2 host worker failed ({returncode}): {tail}")
    role, rank = item["role"], item["rank"]
    validate_row_directory(item["partial"], role=role, role_rank=rank)
    destination = STAGING_PATH / row_relative_path(role, rank)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("V2 row destination already exists")
    os.replace(item["partial"], destination)
    fsync_directory(destination.parent)
    return {
        "role": role, "role_rank": rank,
        "row_manifest_sha256": sha256_file(destination / "row_manifest.json"),
        "wall_seconds": time.monotonic() - item["started"], "group_quiescence": quiescence,
    }


def _wait_host_slot(
    running: dict[int, dict[str, Any]], maximum: int, registry: GroupRegistry,
    watchdog: ResourceWatchdog, completions: list[dict[str, Any]],
) -> None:
    while len(running) >= maximum:
        watchdog.check()
        ready = [pgid for pgid, item in running.items() if item["record"].process.poll() is not None]
        if not ready:
            time.sleep(0.05)
            continue
        for pgid in sorted(ready):
            completions.append(_finish_host(registry, running.pop(pgid)))


def _roles_manifest() -> dict[str, Any]:
    roster = load_exact_roster(); roles = {}
    for role in ROLE_ORDER:
        entries = []
        for rank, spec in enumerate(roster[role]):
            path = STAGING_PATH / row_relative_path(role, rank) / "row_manifest.json"
            entries.append({
                "role_rank": rank, "mlp_id": int(spec["mlp_id"]),
                "path": str(row_relative_path(role, rank) / "row_manifest.json"),
                "sha256": sha256_file(path),
            })
        roles[role] = {
            "rows": ROLE_BINDINGS[role]["rows"],
            "ordered_mlp_ids_sha256": ROLE_BINDINGS[role]["ordered_mlp_ids_sha256"],
            "row_manifests": entries,
        }
    return roles


def _resource_log_manifest() -> list[dict[str, Any]]:
    records = []
    parent = STAGING_PATH / "invocations"
    for path in sorted(parent.glob("invocation-*.RESOURCE.jsonl")):
        if path.is_symlink() or not path.is_file():
            raise AssertionError("V2 invocation resource log is not a regular file")
        records.append({
            "path": str(path.relative_to(STAGING_PATH)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    if not records:
        raise AssertionError("V2 candidate lacks invocation resource evidence")
    return records


def _validate_candidate_for_resume(candidate_path: Path, claim_sha256: str) -> dict[str, Any]:
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise PermissionError("V2 candidate must be a regular file")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    exact = {
        "schema": "phase2-observer-v4-target-blind-feature-store-v2-candidate-v1",
        "status": "READY_FOR_V2_INDEPENDENT_ARTIFACT_AUDIT__NOT_FINAL_SEALED",
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "provider_source_sha256": sha256_file(PROVIDER_SOURCE),
        "claim_sha256": claim_sha256,
        "host_caps": list(CAPS),
        "target_blind_full_source_sha256": FROZEN_BINDINGS["target_blind_full_reader"]["sha256"],
        "full_cache_audit_sha256": FROZEN_BINDINGS["full_cache_audit"]["sha256"],
        "roster_sha256": FROZEN_BINDINGS["full_roster"]["sha256"],
        "v1_hold_audit_sha256": V1_BINDINGS["independent_source_audit_hold"]["sha256"],
        "raw_full_dataset_target_projection": "forbidden",
        "target_values_decoded": 0,
        "role_target_artifacts_opened_or_hashed": 0,
        "fits": 0,
        "scores": 0,
    }
    for key, value in exact.items():
        if candidate.get(key) != value:
            raise AssertionError(f"V2 resume candidate binding changed: {key}")
    if set(candidate.get("roles", {})) != set(ROLE_ORDER):
        raise AssertionError("V2 resume candidate role membership changed")
    rows = 0
    for role in ROLE_ORDER:
        record = candidate["roles"][role]
        if record.get("rows") != ROLE_BINDINGS[role]["rows"]:
            raise AssertionError("V2 resume candidate role count changed")
        if record.get("ordered_mlp_ids_sha256") != ROLE_BINDINGS[role]["ordered_mlp_ids_sha256"]:
            raise AssertionError("V2 resume candidate role ordering changed")
        entries = record.get("row_manifests")
        if not isinstance(entries, list) or len(entries) != ROLE_BINDINGS[role]["rows"]:
            raise AssertionError("V2 resume candidate row membership changed")
        for rank, entry in enumerate(entries):
            relative = row_relative_path(role, rank) / "row_manifest.json"
            if entry.get("role_rank") != rank or entry.get("path") != str(relative):
                raise AssertionError("V2 resume candidate row address changed")
            row_dir = STAGING_PATH / relative.parent
            if sha256_file(row_dir / "row_manifest.json") != entry.get("sha256"):
                raise AssertionError("V2 resume candidate row hash changed")
            validate_row_directory(row_dir, role=role, role_rank=rank)
            rows += 1
    if rows != 776:
        raise AssertionError("V2 resume candidate did not validate exactly 776 rows")
    receipt_path = STAGING_PATH / "GENERATION_RECEIPT_V2.json"
    if sha256_file(receipt_path) != candidate.get("generation_receipt_sha256"):
        raise AssertionError("V2 resume candidate generation receipt changed")
    observed_logs = _resource_log_manifest()
    if candidate.get("resource_sample_logs") != observed_logs:
        raise AssertionError("V2 resume candidate resource evidence changed")
    return candidate


def execute_generation(release_path: Path, resume: bool) -> dict[str, Any]:
    release, release_sha = _validate_release(release_path)
    if Path(os.path.realpath(sys.executable)) != Path(os.path.realpath(ENVIRONMENT_PYTHON)):
        raise PermissionError("V2 generation must use the exact bound environment")
    if _environment() != EXPECTED_ENVIRONMENT:
        raise PermissionError("V2 generation environment changed")
    verify_v1_bindings()
    physical = _physical_memory_bytes()
    if pid_rss_bytes(os.getpid()) * 4 >= physical:
        raise MemoryError("preclaim parent RSS violates strict one-quarter boundary")
    if _free_bytes(HERE) < MIN_FREE_BYTES:
        raise OSError("V2 generation requires at least 60 GiB free")
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError("V2 final store already exists")
    if resume:
        claim = _load_claim(release, release_sha); _validate_session(claim, release)
    else:
        for path in (CLAIM_PATH, STAGING_PATH, SCRATCH_PATH):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"fresh V2 runtime path exists: {path.name}")
        claim = _claim(release, release_sha); _initialize(claim, release)
    candidate_path = STAGING_PATH / "candidate_manifest.json"
    if candidate_path.exists() or candidate_path.is_symlink():
        candidate = _validate_candidate_for_resume(candidate_path, claim["claim_sha256"])
        return {
            "status": candidate["status"],
            "candidate_manifest_sha256": sha256_file(candidate_path),
            "rows": 776,
        }
    invocation, _ = _invocation(claim, resume)
    scratch = SCRATCH_PATH / f"invocation-{invocation:02d}"
    scratch.mkdir(mode=0o700); fsync_directory(SCRATCH_PATH)
    completed_before = _completed()

    registry = GroupRegistry()
    resource_log_path = (
        STAGING_PATH / "invocations" / f"invocation-{invocation:02d}.RESOURCE.jsonl"
    )
    if resource_log_path.exists() or resource_log_path.is_symlink():
        raise FileExistsError("V2 invocation resource log already exists")
    watchdog = ResourceWatchdog(registry, physical, resource_log_path)
    watchdog.start()
    running: dict[int, dict[str, Any]] = {}; completions = []
    cursor_record = None; cursor_stderr = None; decoded_rows = 0; seen_selected = set()
    started = time.perf_counter()
    try:
        cursor_record, cursor_stderr = _start_cursor(registry, watchdog, scratch)
        cursor = cursor_record.process
        roster = load_exact_roster(); lookup = {}
        for role in ROLE_ORDER:
            for rank, spec in enumerate(roster[role]):
                lookup[int(spec["mlp_id"])] = (role, rank, spec)
        while True:
            watchdog.guard("pre_decode_next_64mib_full_row")
            _send_cursor(cursor, {"op": "NEXT"})
            reply = _read_cursor(cursor, watchdog)
            if reply.get("event") == "END":
                decoded_rows = int(reply["decoded_rows"]); break
            if reply.get("event") != "ROW_HELD":
                raise AssertionError("cursor NEXT reply changed")
            decoded_rows = int(reply["decoded_rows"])
            selected = lookup.get(int(reply["mlp_id"]))
            if selected is None:
                _send_cursor(cursor, {"op": "DISCARD"})
                if _read_cursor(cursor, watchdog).get("event") != "DISCARDED":
                    raise AssertionError("cursor discard reply changed")
                continue
            role, rank, spec = selected; seen_selected.add(int(spec["mlp_id"]))
            for key in ("ordinal", "shard_index", "row_in_shard", "mlp_id", "mlp_seed"):
                if int(reply[key]) != int(spec[key]):
                    raise AssertionError(f"cursor/roster metadata mismatch: {key}")
            if reply["mlp_name"] != spec["mlp_name"] or reply["decoded_columns"] != ["mlp_id", "mlp_name", "mlp_seed", "weights"]:
                raise AssertionError("cursor target-blind projection changed")
            if (role, rank) in completed_before:
                _send_cursor(cursor, {"op": "DISCARD"}); _read_cursor(cursor, watchdog)
                continue
            _wait_host_slot(running, int(release["max_host_workers"]), registry, watchdog, completions)
            watchdog.guard("pre_partial_row_directory_creation")
            partial = scratch / f"{int(spec['ordinal']):04d}-{role}-{rank:04d}.partial"
            partial.mkdir(mode=0o700); fsync_directory(scratch)
            watchdog.guard("pre_64mib_weight_write_allocation")
            _send_cursor(cursor, {
                "op": "WRITE", "weights_path": str(partial / "weights.npy"),
                "weights_raw_sha256": spec["weights_sha256"], "mlp_id": int(spec["mlp_id"]),
                "mlp_seed": int(spec["mlp_seed"]), "ordinal": int(spec["ordinal"]),
            })
            written = _read_cursor(cursor, watchdog)
            if written.get("event") != "WRITTEN" or written.get("weights_raw_sha256") != spec["weights_sha256"]:
                raise AssertionError("guarded cursor weight-write receipt changed")
            record, item = _launch_host(registry, watchdog, partial, spec, role, rank, scratch)
            running[record.pgid] = item
        if decoded_rows != 1000 or seen_selected != set(lookup):
            raise AssertionError("V2 cursor did not cover exact full/selected roster")
        while running:
            _wait_host_slot(running, 1, registry, watchdog, completions)
        watchdog.guard("pre_cursor_stop")
        _send_cursor(cursor, {"op": "STOP"})
        stopped = _read_cursor(cursor, watchdog)
        if stopped.get("event") != "STOPPED" or stopped.get("decoded_rows") != 1000:
            raise AssertionError("V2 cursor stop receipt changed")
        cursor.wait(); cursor_stderr.close()
        cursor_quiescence = retire_completed_group(registry, cursor_record)
        if len(_completed()) != 776:
            raise AssertionError("V2 generation ended without 776 validated rows")
        watchdog_receipt = watchdog.stop()
    except BaseException as error:
        termination = watchdog.abort_all(f"{type(error).__name__}: {error}")
        abort_watchdog = watchdog.join_after_abort()
        for item in running.values():
            item["stdout"].close(); item["stderr"].close()
        if cursor_stderr is not None and not cursor_stderr.closed:
            cursor_stderr.close()
        abort = {
            "schema": "phase2-observer-v4-feature-store-v2-abort-v1",
            "status": "ABORTED_ALL_PROCESS_GROUPS_QUIESCENT__CLAIM_CONSUMED",
            "reason": f"{type(error).__name__}: {error}", "decoded_rows_observed": decoded_rows,
            "completed_before": len(completed_before), "completed_this_invocation": len(completions),
            "termination": termination, "watchdog": abort_watchdog,
            "target_values_decoded": 0, "fits": 0, "scores": 0,
        }
        write_exclusive_fsynced(
            STAGING_PATH / "invocations" / f"invocation-{invocation:02d}.ABORT.json",
            canonical_json_bytes(abort),
        )
        raise

    receipt = {
        "schema": "phase2-observer-v4-feature-store-v2-generation-receipt-v1",
        "status": "COMPLETE_TARGET_BLIND_ROWS__HOLD_FINAL_SEAL_FOR_V2_ARTIFACT_AUDIT",
        "claim_sha256": claim["claim_sha256"], "decoded_full_rows": decoded_rows,
        "selected_rows": 776, "completed_before": len(completed_before),
        "completed_this_invocation": len(completions), "wall_seconds": time.perf_counter() - started,
        "watchdog": watchdog_receipt, "cursor_group_quiescence": cursor_quiescence,
        "resource_log": {
            "path": str(resource_log_path.relative_to(STAGING_PATH)),
            "sha256": sha256_file(resource_log_path),
            "bytes": resource_log_path.stat().st_size,
        },
        "all_host_groups_quiescent": all(x["group_quiescence"]["three_empty_polls"] for x in completions),
        "target_values_decoded": 0, "fits": 0, "scores": 0,
    }
    receipt_path = STAGING_PATH / "GENERATION_RECEIPT_V2.json"
    write_exclusive_fsynced(receipt_path, canonical_json_bytes(receipt))
    candidate = {
        "schema": "phase2-observer-v4-target-blind-feature-store-v2-candidate-v1",
        "status": "READY_FOR_V2_INDEPENDENT_ARTIFACT_AUDIT__NOT_FINAL_SEALED",
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "provider_source_sha256": sha256_file(PROVIDER_SOURCE), "claim_sha256": claim["claim_sha256"],
        "host_caps": list(CAPS), "roles": _roles_manifest(),
        "fields": ["mlp_id", "raw_locals", "edge_matrices", "host_rows", "sigma", "entering_cp_state_hashes", "host_latch"],
        "edge_materialization": "deterministic_on_demand_from_sealed_weights",
        "generation_receipt_sha256": sha256_file(receipt_path),
        "resource_sample_logs": _resource_log_manifest(),
        "target_blind_full_source_sha256": FROZEN_BINDINGS["target_blind_full_reader"]["sha256"],
        "full_cache_audit_sha256": FROZEN_BINDINGS["full_cache_audit"]["sha256"],
        "roster_sha256": FROZEN_BINDINGS["full_roster"]["sha256"],
        "v1_hold_audit_sha256": V1_BINDINGS["independent_source_audit_hold"]["sha256"],
        "raw_full_dataset_target_projection": "forbidden", "target_values_decoded": 0,
        "role_target_artifacts_opened_or_hashed": 0, "fits": 0, "scores": 0,
    }
    write_exclusive_fsynced(candidate_path, canonical_json_bytes(candidate)); fsync_directory(STAGING_PATH)
    return {"status": candidate["status"], "candidate_manifest_sha256": sha256_file(candidate_path), "rows": 776}


def finalize_store(release_path: Path) -> dict[str, Any]:
    if release_path.is_symlink() or not release_path.is_file():
        raise PermissionError("V2 finalization release must be a regular file")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema") != "phase2-observer-v4-feature-store-v2-finalization-release-v1" or release.get("status") != "AUTHORIZED_PHASE2_OBSERVER_V4_FEATURE_STORE_V2_FINALIZATION":
        raise PermissionError("wrong V2 finalization release")
    if set(release) != FINALIZATION_RELEASE_KEYS:
        raise PermissionError("V2 finalization release field membership changed")
    for key in ("authorized_by", "authorized_at_utc", "nonce"):
        if not isinstance(release.get(key), str) or not release[key]:
            raise PermissionError(f"V2 finalization release lacks {key}")
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError("V2 final store already exists")
    if STAGING_PATH.is_symlink() or not STAGING_PATH.is_dir():
        raise PermissionError("V2 staging directory is absent or a symlink")
    if CLAIM_PATH.is_symlink() or not CLAIM_PATH.is_file():
        raise PermissionError("V2 generation claim is absent or a symlink")
    candidate_path = STAGING_PATH / "candidate_manifest.json"
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise PermissionError("V2 candidate is absent or a symlink")
    exact = {
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "v1_hold_audit_sha256": V1_BINDINGS["independent_source_audit_hold"]["sha256"],
        "generation_claim_sha256": sha256_file(CLAIM_PATH),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "staging_path": str(STAGING_PATH), "output_path": str(OUTPUT_PATH),
        "target_access_authorized": False, "fit_or_score_authorized": False,
    }
    for key, value in exact.items():
        if release.get(key) != value:
            raise PermissionError(f"V2 finalization binding changed: {key}")
    audit_path = Path(release["independent_artifact_audit_path"])
    if audit_path.is_symlink() or not audit_path.is_file():
        raise PermissionError("V2 artifact audit is absent or a symlink")
    if sha256_file(audit_path) != release.get("independent_artifact_audit_sha256"):
        raise PermissionError("V2 artifact audit hash changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        release.get("independent_artifact_audit_status") != ARTIFACT_AUDIT_STATUS
        or audit.get("status") != ARTIFACT_AUDIT_STATUS
        or audit.get("candidate_manifest_sha256") != sha256_file(candidate_path)
        or audit.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH)
    ):
        raise PermissionError("V2 independent artifact audit is not exact PASS")
    candidate = _validate_candidate_for_resume(candidate_path, sha256_file(CLAIM_PATH))
    for role in ROLE_ORDER:
        for rank, entry in enumerate(candidate["roles"][role]["row_manifests"]):
            row = STAGING_PATH / row_relative_path(role, rank)
            if sha256_file(row / "row_manifest.json") != entry["sha256"]:
                raise AssertionError("V2 candidate row hash changed")
            validate_row_directory(row, role=role, role_rank=rank)
    final = {
        "schema": FINAL_SCHEMA, "status": "SEALED_TARGET_BLIND_FEATURE_STORE",
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "provider_source_sha256": sha256_file(PROVIDER_SOURCE),
        "candidate_manifest_sha256": sha256_file(candidate_path), "host_caps": list(CAPS),
        "roles": candidate["roles"], "fields": candidate["fields"],
        "edge_materialization": candidate["edge_materialization"],
        "target_blind_full_source_sha256": FROZEN_BINDINGS["target_blind_full_reader"]["sha256"],
        "full_cache_audit_sha256": FROZEN_BINDINGS["full_cache_audit"]["sha256"],
        "independent_feature_store_audit_status": "PASS",
        "independent_feature_store_audit_sha256": release["independent_artifact_audit_sha256"],
        "raw_full_dataset_target_projection": "forbidden", "target_columns_decoded": 0,
        "target_values_decoded": 0, "fits": 0, "scores": 0,
    }
    write_exclusive_fsynced(STAGING_PATH / "manifest.json", canonical_json_bytes(final))
    write_exclusive_fsynced(STAGING_PATH / "FINALIZATION_RECEIPT_V2.json", canonical_json_bytes({
        "schema": "phase2-observer-v4-feature-store-v2-finalization-receipt-v1",
        "status": "ATOMIC_FINAL_SEAL_COMPLETE", "manifest_sha256": sha256_file(STAGING_PATH / "manifest.json"),
        "rows": 776, "target_values_decoded": 0, "fits": 0, "scores": 0,
    }))
    for directory, subdirectories, filenames in os.walk(STAGING_PATH, topdown=False):
        current = Path(directory); fsync_directory(current)
        for name in filenames: (current / name).chmod(0o400)
        for name in subdirectories: (current / name).chmod(0o500)
    STAGING_PATH.chmod(0o500); os.replace(STAGING_PATH, OUTPUT_PATH); fsync_directory(HERE)
    return {"status": "SEALED_TARGET_BLIND_FEATURE_STORE", "manifest_sha256": sha256_file(OUTPUT_PATH / "manifest.json"), "rows": 776}


def main() -> None:
    parser = argparse.ArgumentParser(); actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true"); actions.add_argument("--resume", action="store_true"); actions.add_argument("--finalize", action="store_true")
    parser.add_argument("--release", type=Path); args = parser.parse_args()
    if not (args.execute or args.resume or args.finalize):
        print(json.dumps(preflight(), indent=2, sort_keys=True)); return
    if args.release is None: raise SystemExit("independent exact V2 release required")
    report = finalize_store(args.release) if args.finalize else execute_generation(args.release, args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
