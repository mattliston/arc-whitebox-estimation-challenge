#!/usr/bin/env python3
"""Lightweight monitored exec gate before the heavy V2 host-worker import."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from contract_v2 import ENVIRONMENT_PYTHON, HERE, SCRATCH_PATH, fsync_directory, write_exclusive_fsynced


def _child(path: Path) -> Path:
    resolved = path.resolve()
    if SCRATCH_PATH.resolve() not in resolved.parents:
        raise PermissionError("host exec-gate path escapes V2 scratch")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--boot", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    gate = _child(args.gate); boot = _child(args.boot); task = _child(args.task)
    boot_partial = boot.with_name(boot.name + ".partial")
    if gate.exists() or boot.exists() or boot_partial.exists() or not task.is_file():
        raise PermissionError("host exec-gate initial state changed")
    write_exclusive_fsynced(boot_partial, (json.dumps({
        "status": "BOOTED_LIGHTWEIGHT_ZERO_HEAVY_IMPORTS_WAITING_FOR_MONITORED_GATE",
        "pid": os.getpid(),
    }, sort_keys=True) + "\n").encode("utf-8"))
    os.replace(boot_partial, boot)
    fsync_directory(boot.parent)
    while not gate.is_file():
        time.sleep(0.01)
    if gate.is_symlink() or gate.read_text(encoding="utf-8") != "MONITORED_HEAVY_EXEC\n":
        raise PermissionError("host exec gate payload changed")
    os.execv(
        str(ENVIRONMENT_PYTHON),
        [str(ENVIRONMENT_PYTHON), str(HERE / "host_feature_worker_v2.py"), "--task", str(task)],
    )


if __name__ == "__main__":
    main()
