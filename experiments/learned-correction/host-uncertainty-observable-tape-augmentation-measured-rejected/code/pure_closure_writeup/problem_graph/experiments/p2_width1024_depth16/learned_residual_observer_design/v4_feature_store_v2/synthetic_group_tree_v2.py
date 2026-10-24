#!/usr/bin/env python3
"""Target-free synthetic group leader with one descendant for V2 kill tests."""

from __future__ import annotations

import json
import subprocess
import sys
import time


child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('READY', flush=True); time.sleep(3600)",
    ],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    close_fds=True,
)
assert child.stdout is not None
if child.stdout.readline().strip() != "READY":
    raise RuntimeError("synthetic stubborn descendant failed readiness handshake")
print(json.dumps({"leader_pid": __import__("os").getpid(), "descendant_pid": child.pid}), flush=True)
while True:
    time.sleep(1.0)
