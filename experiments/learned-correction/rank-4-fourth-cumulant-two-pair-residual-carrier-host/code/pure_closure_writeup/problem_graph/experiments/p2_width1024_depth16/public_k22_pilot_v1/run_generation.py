#!/usr/bin/env python3
"""Stage-1 orchestrator: 200 rows x 2 arms, 2 workers, paired row order.

TAG=k22pilot-gen  watch: tail -F gen.log
Row-major over ranks; within each rank runs base then k22r4 across the two
worker slots so the pair lands together and a stop leaves aligned coverage.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parent / "data_gate/.venv/bin/python"
RANKS = list(range(35, 235))
ARMS = ("base", "k22r4")


def spawn(arm, rank):
    return subprocess.Popen(
        ["nice", "-n", "10", str(PY), str(HERE / "generate_pilot_row.py"),
         "--arm", arm, "--rank", str(rank)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def main():
    started = time.time()
    done = 0
    for rank in RANKS:
        jobs = [(arm, spawn(arm, rank)) for arm in ARMS]
        for arm, job in jobs:
            output, _ = job.communicate()
            if job.returncode:
                print(f"FAIL {arm} {rank}\n{output}", flush=True)
                sys.exit(1)
            line = output.strip().splitlines()[-1] if output.strip() else "{}"
            print(line, flush=True)
            done += 1
        if rank % 10 == 0:
            elapsed = time.time() - started
            print(json.dumps({"progress": f"{done}/400",
                              "elapsed_minutes": round(elapsed / 60, 1)}), flush=True)
    print(json.dumps({"status": "STAGE1_COMPLETE", "rows": done}), flush=True)


if __name__ == "__main__":
    main()
