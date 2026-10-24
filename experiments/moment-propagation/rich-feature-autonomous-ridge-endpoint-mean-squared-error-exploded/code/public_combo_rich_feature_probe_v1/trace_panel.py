#!/usr/bin/env python3
"""Sequential low-memory launcher for rich combo traces."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent / "data_gate/.venv/bin/python"
WORKER = HERE / "trace_rich_row.py"


for rank in range(35):
    output = HERE / "rich_rows" / f"row_{rank:04d}" / "rich_features.npy"
    if output.is_file():
        continue
    completed = subprocess.run(
        [str(PYTHON), str(WORKER), "--role-rank", str(rank)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout.strip(), flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)
print("RICH_PANEL_COMPLETE", flush=True)
