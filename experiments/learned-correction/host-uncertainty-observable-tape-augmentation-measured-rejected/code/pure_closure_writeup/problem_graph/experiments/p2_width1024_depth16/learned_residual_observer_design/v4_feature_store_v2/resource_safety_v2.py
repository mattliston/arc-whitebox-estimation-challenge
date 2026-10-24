#!/usr/bin/env python3
"""Continuous RSS watchdog and process-group-complete termination for V2."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from contract_v2 import (
    MAX_WATCHDOG_GAP_SECONDS,
    PER_PROCESS_LIMIT_BYTES,
    QUIESCENCE_TIMEOUT_SECONDS,
    TERMINATION_GRACE_SECONDS,
    WATCHDOG_INTERVAL_SECONDS,
)


class ResourceViolation(RuntimeError):
    """A strict resource or watchdog-continuity boundary was reached."""


class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64), ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64), ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32), ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32), ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32), ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32), ("syscalls_unix", ctypes.c_int32),
        ("csw", ctypes.c_int32), ("threadnum", ctypes.c_int32),
        ("numrunning", ctypes.c_int32), ("priority", ctypes.c_int32),
    ]


def _libproc() -> Any:
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


def pid_rss_bytes(pid: int, library: Any | None = None) -> int:
    if sys.platform == "darwin":
        libproc = library or _libproc()
        info = _ProcTaskInfo()
        returned = libproc.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        return int(info.resident_size) if returned == ctypes.sizeof(info) else 0
    statm = Path(f"/proc/{pid}/statm")
    if not statm.is_file():
        return 0
    return int(statm.read_text(encoding="ascii").split()[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def process_group_pids(pgid: int, library: Any | None = None) -> list[int]:
    if sys.platform == "darwin":
        libproc = library or _libproc()
        capacity = max(4096, int(libproc.proc_listpids(2, pgid, None, 0)))
        buffer = (ctypes.c_int32 * (capacity // 4))()
        returned = int(libproc.proc_listpids(2, pgid, buffer, ctypes.sizeof(buffer)))
        if returned < 0:
            raise OSError(ctypes.get_errno(), "proc_listpids process-group query failed")
        return sorted(pid for pid in buffer[: returned // 4] if pid > 0)
    result = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        pid = int(candidate.name)
        try:
            if os.getpgid(pid) == pgid:
                result.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    return sorted(result)


@dataclass(frozen=True)
class GroupRecord:
    pgid: int
    label: str
    process: subprocess.Popen[Any]


class GroupRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._groups: dict[int, GroupRecord] = {}

    def register(self, process: subprocess.Popen[Any], label: str) -> GroupRecord:
        pgid = process.pid
        if os.getpgid(process.pid) != pgid:
            raise RuntimeError("child is not leader of its isolated process group")
        record = GroupRecord(pgid=pgid, label=label, process=process)
        with self._lock:
            if pgid in self._groups:
                raise RuntimeError("duplicate process-group registration")
            self._groups[pgid] = record
        return record

    def snapshot(self) -> list[GroupRecord]:
        with self._lock:
            return list(self._groups.values())

    def remove(self, pgid: int) -> None:
        with self._lock:
            self._groups.pop(pgid, None)


def _signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def _quiescent_three_polls(pgid: int, timeout: float) -> tuple[bool, list[int]]:
    deadline = time.monotonic() + timeout
    empty = 0
    last: list[int] = []
    while time.monotonic() < deadline:
        last = process_group_pids(pgid)
        if not last:
            empty += 1
            if empty >= 3:
                return True, []
        else:
            empty = 0
        time.sleep(0.05)
    return False, last


def terminate_process_groups(records: list[GroupRecord], reason: str) -> dict[str, Any]:
    """TERM whole groups, grace, KILL survivors, reap leaders, prove quiescence."""

    unique = {record.pgid: record for record in records}
    before = {str(pgid): process_group_pids(pgid) for pgid in sorted(unique)}
    term_groups = [pgid for pgid in sorted(unique) if _signal_group(pgid, signal.SIGTERM)]
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        for record in unique.values():
            record.process.poll()  # reap a terminated leader as soon as possible
        if all(not process_group_pids(pgid) for pgid in unique):
            break
        time.sleep(0.05)
    survivors = {
        pgid: process_group_pids(pgid) for pgid in sorted(unique) if process_group_pids(pgid)
    }
    kill_groups = [pgid for pgid in sorted(survivors) if _signal_group(pgid, signal.SIGKILL)]
    leader_returncodes = {}
    for pgid, record in sorted(unique.items()):
        try:
            leader_returncodes[str(pgid)] = record.process.wait(timeout=QUIESCENCE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_group(pgid, signal.SIGKILL)
            leader_returncodes[str(pgid)] = record.process.wait(timeout=QUIESCENCE_TIMEOUT_SECONDS)
    quiescence = {}
    for pgid in sorted(unique):
        passed, remaining = _quiescent_three_polls(pgid, QUIESCENCE_TIMEOUT_SECONDS)
        quiescence[str(pgid)] = {"three_empty_polls": passed, "remaining_pids": remaining}
        if not passed:
            raise ResourceViolation(f"process group {pgid} failed verified quiescence")
    return {
        "reason": reason,
        "members_before": before,
        "sigterm_groups": term_groups,
        "survivors_after_grace": {str(key): value for key, value in survivors.items()},
        "sigkill_groups": kill_groups,
        "leader_returncodes": leader_returncodes,
        "quiescence": quiescence,
        "all_groups_quiescent": True,
    }


def retire_completed_group(registry: GroupRegistry, record: GroupRecord) -> dict[str, Any]:
    returncode = record.process.wait()
    passed, remaining = _quiescent_three_polls(record.pgid, QUIESCENCE_TIMEOUT_SECONDS)
    if not passed:
        report = terminate_process_groups([record], "completed leader leaked descendants")
        registry.remove(record.pgid)
        raise ResourceViolation(json.dumps(report, sort_keys=True))
    registry.remove(record.pgid)
    return {
        "pgid": record.pgid,
        "leader_returncode": returncode,
        "remaining_pids": remaining,
        "three_empty_polls": True,
    }


class ResourceWatchdog:
    def __init__(
        self,
        registry: GroupRegistry,
        physical_memory_bytes: int,
        log_path: Path,
        *,
        synthetic_mode: bool = False,
        injected_guards: dict[str, str] | None = None,
    ) -> None:
        self.registry = registry
        self.physical = int(physical_memory_bytes)
        self.log_path = log_path
        self.synthetic_mode = synthetic_mode
        self.injected = dict(injected_guards or {})
        if self.injected and not synthetic_mode:
            raise PermissionError("resource injections are synthetic-test only")
        self._stop = threading.Event()
        self._violation = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.RLock()
        self._sample_lock = threading.Lock()
        self._termination_lock = threading.Lock()
        self._last_sample: float | None = None
        self._log_created = False
        self.max_gap_seconds = 0.0
        self.samples = 0
        self.violation_reason: str | None = None
        self.termination_report: dict[str, Any] | None = None

    def _measure(self, phase: str) -> dict[str, Any]:
        library = _libproc() if sys.platform == "darwin" else None
        groups = self.registry.snapshot()
        group_pids = {str(record.pgid): process_group_pids(record.pgid, library) for record in groups}
        group_rss = {
            key: sum(pid_rss_bytes(pid, library) for pid in pids) for key, pids in group_pids.items()
        }
        parent_rss = pid_rss_bytes(os.getpid(), library)
        sample = {
            "monotonic_seconds": time.monotonic(),
            "phase": phase,
            "parent_rss_bytes": parent_rss,
            "group_rss_bytes": group_rss,
            "group_pids": group_pids,
            "aggregate_rss_bytes": parent_rss + sum(group_rss.values()),
            "physical_memory_bytes": self.physical,
            "strict_system_limit_bytes": self.physical // 4,
            "strict_group_limit_bytes": PER_PROCESS_LIMIT_BYTES,
        }
        injection = self.injected.pop(phase, None)
        if injection == "system_25_percent":
            sample["aggregate_rss_bytes"] = self.physical // 4
            sample["synthetic_injection"] = injection
        elif injection == "group_8gib":
            key = next(iter(group_rss), "synthetic-group")
            sample["group_rss_bytes"][key] = PER_PROCESS_LIMIT_BYTES
            sample["synthetic_injection"] = injection
        elif injection == "exception":
            raise RuntimeError("synthetic injected exception before guarded operation")
        return sample

    def _record(self, sample: dict[str, Any]) -> None:
        now = float(sample["monotonic_seconds"])
        with self._state_lock:
            if self._last_sample is not None:
                self.max_gap_seconds = max(self.max_gap_seconds, now - self._last_sample)
            self._last_sample = now
            self.samples += 1
            mode = "a" if self._log_created else "x"
            with self.log_path.open(mode, encoding="utf-8") as handle:
                handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._log_created = True

    @staticmethod
    def _boundary_reason(sample: dict[str, Any]) -> str | None:
        if int(sample["aggregate_rss_bytes"]) * 4 >= int(sample["physical_memory_bytes"]):
            return "aggregate RSS reached strict one-quarter system-memory boundary"
        if any(int(value) >= PER_PROCESS_LIMIT_BYTES for value in sample["group_rss_bytes"].values()):
            return "process-group RSS reached strict 8 GiB boundary"
        return None

    def _trip(self, reason: str) -> None:
        with self._state_lock:
            if self.violation_reason is None:
                self.violation_reason = reason
                self._violation.set()
        with self._termination_lock:
            if self.termination_report is None:
                self.termination_report = terminate_process_groups(
                    self.registry.snapshot(), reason
                )

    def _sample_and_enforce(self, phase: str) -> dict[str, Any]:
        with self._sample_lock:
            if self._violation.is_set():
                raise ResourceViolation(self.violation_reason or "resource watchdog tripped")
            try:
                sample = self._measure(phase)
                self._record(sample)
                reason = self._boundary_reason(sample)
                if reason is not None:
                    self._trip(reason)
                    raise ResourceViolation(reason)
                with self._state_lock:
                    if self.max_gap_seconds > MAX_WATCHDOG_GAP_SECONDS:
                        reason = f"watchdog sample gap exceeded {MAX_WATCHDOG_GAP_SECONDS}s"
                        self._trip(reason)
                        raise ResourceViolation(reason)
                return sample
            except BaseException as error:
                if not isinstance(error, ResourceViolation):
                    self._trip(f"resource monitor exception: {type(error).__name__}: {error}")
                raise

    def _run(self) -> None:
        while not self._stop.wait(WATCHDOG_INTERVAL_SECONDS):
            try:
                self._sample_and_enforce("continuous_watchdog")
            except BaseException:
                return

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("watchdog already started")
        self._sample_and_enforce("watchdog_start")
        self._thread = threading.Thread(target=self._run, name="v4-feature-store-v2-rss", daemon=True)
        self._thread.start()

    def guard(self, phase: str) -> dict[str, Any]:
        self.check()
        return self._sample_and_enforce(phase)

    def check(self) -> None:
        if self._violation.is_set():
            raise ResourceViolation(self.violation_reason or "resource watchdog tripped")

    def abort_all(self, reason: str) -> dict[str, Any]:
        self._stop.set()
        self._trip(reason)
        return self.termination_report or {
            "reason": reason, "all_groups_quiescent": True, "members_before": {}
        }

    def stop(self) -> dict[str, Any]:
        self.guard("watchdog_stop")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                self._trip("watchdog thread failed to join")
                raise ResourceViolation("watchdog thread failed to join")
        self.check()
        if self.max_gap_seconds > MAX_WATCHDOG_GAP_SECONDS:
            raise ResourceViolation("watchdog continuity bound violated")
        return {
            "samples": self.samples,
            "maximum_sample_gap_seconds": self.max_gap_seconds,
            "required_maximum_gap_seconds": MAX_WATCHDOG_GAP_SECONDS,
            "violation": False,
        }

    def join_after_abort(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise ResourceViolation("aborted watchdog thread failed to join")
        if self.termination_report is None or not self.termination_report.get("all_groups_quiescent"):
            raise ResourceViolation("abort lacks verified process-group quiescence")
        return {
            "samples": self.samples,
            "maximum_sample_gap_seconds": self.max_gap_seconds,
            "required_maximum_gap_seconds": MAX_WATCHDOG_GAP_SECONDS,
            "violation": True,
            "violation_reason": self.violation_reason,
            "termination": self.termination_report,
        }
