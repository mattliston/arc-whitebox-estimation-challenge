#!/usr/bin/env python3
"""Exact FlopScope 0.12 server with read-only RSS/handle telemetry."""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time

from flopscope_server._request_handler import MAX_ARRAY_BYTES
from flopscope_server._server import FlopscopeServer


HERE = Path(__file__).resolve().parent
TELEMETRY = HERE / "D23_SERVER_TELEMETRY.json"
URL = os.environ.get("D23_SERVER_URL", "ipc:///private/tmp/d23-flopscope.sock")
SAMPLE_SECONDS = 0.02
PROC_PIDTASKINFO = 4
OFFICIAL_MAX_ARRAY_BYTES = 4_294_967_296


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


def current_rss_bytes() -> int:
    if platform.system() != "Darwin":
        raise RuntimeError("D23 telemetry requires Darwin")
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
    temporary = TELEMETRY.with_name(f".{TELEMETRY.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(TELEMETRY)


def main() -> None:
    if TELEMETRY.exists():
        raise FileExistsError(TELEMETRY)
    if MAX_ARRAY_BYTES != OFFICIAL_MAX_ARRAY_BYTES:
        raise AssertionError(
            ("official grader single-array cap changed", MAX_ARRAY_BYTES,
             OFFICIAL_MAX_ARRAY_BYTES)
        )
    server = FlopscopeServer(url=URL, session_timeout_s=600.0)
    started = time.time()
    stop = threading.Event()
    samples: list[dict[str, object]] = []
    aggregate = {
        "maximum_current_rss_bytes": 0,
        "maximum_peak_rss_bytes": 0,
        "maximum_live_array_handles": 0,
    }

    def monitor() -> None:
        ordinal = 0
        while not stop.is_set():
            current = current_rss_bytes()
            peak = peak_rss_bytes()
            handles = int(server._conn_store.arrays.count)
            sample = {
                "ordinal": ordinal,
                "elapsed_seconds": time.time() - started,
                "current_rss_bytes": current,
                "peak_rss_bytes": peak,
                "live_array_handles": handles,
                "session_present": server._session is not None,
                "session_open": bool(
                    server._session is not None and server._session.is_open
                ),
            }
            samples.append(sample)
            aggregate["maximum_current_rss_bytes"] = max(
                int(aggregate["maximum_current_rss_bytes"]), current
            )
            aggregate["maximum_peak_rss_bytes"] = max(
                int(aggregate["maximum_peak_rss_bytes"]), peak
            )
            aggregate["maximum_live_array_handles"] = max(
                int(aggregate["maximum_live_array_handles"]), handles
            )
            # Keep a bounded trace while preserving every handle/RSS change.
            if len(samples) > 5000:
                samples.pop(0)
            if ordinal % 5 == 0:
                atomic_write(
                    {
                        "schema": "d23-flopscope-server-telemetry-v1",
                        "status": "RUNNING",
                        "pid": os.getpid(),
                        "url": URL,
                        "flopscope_server_distribution": importlib.metadata.version(
                            "flopscope-server"
                        ),
                        "flopscope_distribution": importlib.metadata.version(
                            "flopscope"
                        ),
                        "numpy": importlib.metadata.version("numpy"),
                        "max_array_bytes": MAX_ARRAY_BYTES,
                        "max_array_count": int(
                            os.environ.get("FLOPSCOPE_MAX_ARRAY_COUNT", "100000")
                        ),
                        "aggregate": aggregate,
                        "samples": samples,
                    }
                )
            ordinal += 1
            stop.wait(SAMPLE_SECONDS)

    thread = threading.Thread(target=monitor, daemon=True, name="d23-telemetry")
    thread.start()
    print(
        json.dumps(
            {
                "status": "SERVER_STARTING",
                "pid": os.getpid(),
                "url": URL,
                "max_array_bytes": MAX_ARRAY_BYTES,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    terminal = "STOPPED"
    try:
        server.run()
    except KeyboardInterrupt:
        terminal = "STOPPED_BY_OWNER"
    except BaseException:
        terminal = "SERVER_EXCEPTION"
        raise
    finally:
        stop.set()
        thread.join(timeout=1.0)
        atomic_write(
            {
                "schema": "d23-flopscope-server-telemetry-v1",
                "status": terminal,
                "pid": os.getpid(),
                "url": URL,
                "flopscope_server_distribution": importlib.metadata.version(
                    "flopscope-server"
                ),
                "flopscope_distribution": importlib.metadata.version("flopscope"),
                "numpy": importlib.metadata.version("numpy"),
                "max_array_bytes": MAX_ARRAY_BYTES,
                "max_array_count": int(
                    os.environ.get("FLOPSCOPE_MAX_ARRAY_COUNT", "100000")
                ),
                "aggregate": aggregate,
                "samples": samples,
            }
        )


if __name__ == "__main__":
    main()
