#!/usr/bin/env python3
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = HERE.parent / "data_gate/.venv/bin/python"
RANKS = list(range(35, 235))
start = time.time()
for i in range(0, len(RANKS), 2):
    pair = RANKS[i:i+2]
    jobs = [(r, subprocess.Popen(["nice", "-n", "10", str(PY), str(HERE / "generate_joint_rows.py"), str(r)],
             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)) for r in pair]
    for r, j in jobs:
        out, _ = j.communicate()
        if j.returncode:
            print(f"FAIL {r}\n{out}", flush=True); sys.exit(1)
        print(out.strip().splitlines()[-1] if out.strip() else "{}", flush=True)
    if pair[0] % 20 <= 1:
        print(json.dumps({"progress": i + len(pair), "min": round((time.time()-start)/60, 1)}), flush=True)
print(json.dumps({"status": "COMPLETE"}), flush=True)
