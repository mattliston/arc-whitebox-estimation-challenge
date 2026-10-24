#!/usr/bin/env python3
"""Stage-1 orchestrator: combined arm only, 200 rows, 2 rank-parallel workers.

TAG=combined-pilot-gen  watch: tail -F gen.log
Ranks are processed in pairs so two forwards run concurrently; a stop leaves
completed rows that the generator skips on resume.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parent / "data_gate/.venv/bin/python"
RANKS = list(range(35, 235))
ARM = "combined"


def spawn(rank):
    return subprocess.Popen(
        ["nice", "-n", "10", str(PY), str(HERE / "generate_row.py"),
         "--arm", ARM, "--rank", str(rank)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def main():
    started = time.time()
    done = 0
    for index in range(0, len(RANKS), 2):
        pair = RANKS[index:index + 2]
        jobs = [(rank, spawn(rank)) for rank in pair]
        for rank, job in jobs:
            output, _ = job.communicate()
            if job.returncode:
                print(f"FAIL {ARM} {rank}\n{output}", flush=True)
                sys.exit(1)
            line = output.strip().splitlines()[-1] if output.strip() else "{}"
            print(line, flush=True)
            done += 1
        if pair[0] % 10 <= 1:
            elapsed = time.time() - started
            print(json.dumps({"progress": f"{done}/200",
                              "elapsed_minutes": round(elapsed / 60, 1)}), flush=True)
    print(json.dumps({"status": "STAGE1_COMPLETE", "rows": done}), flush=True)


if __name__ == "__main__":
    main()
