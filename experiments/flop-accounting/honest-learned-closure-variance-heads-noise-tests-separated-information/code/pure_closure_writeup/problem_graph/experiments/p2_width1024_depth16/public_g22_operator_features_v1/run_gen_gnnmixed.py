#!/usr/bin/env python3
"""Drive gen_rows_k4panel.py over ranks 35..234, TWO at a time (memory:
two card runs ~16 GB; never more), resume-safe, nice 10.
Log: gen_k4panel.log. Usage: run_gen_k4panel.py [first_rank] [last_rank]"""
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = "/Users/matt/arc-white-box-estimation/.venv/bin/python"
lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
hi = int(sys.argv[2]) if len(sys.argv) > 2 else 234
RANKS = [r for r in range(lo, hi + 1)
         if not (Path(__import__("os").environ["G22_OUT"]) / f"row_{r:04d}" / "result.json").exists()]
start = time.time()
print(json.dumps({"todo": len(RANKS)}), flush=True)
for i in range(0, len(RANKS), 2):
    pair = RANKS[i:i + 2]
    jobs = [(r, subprocess.Popen(["nice", "-n", "10", PY, str(HERE / "gen_rows_gnnmixed.py"), str(r)],
             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)) for r in pair]
    for r, j in jobs:
        out, _ = j.communicate()
        if j.returncode:
            print(f"FAIL {r}\n{out[-3000:]}", flush=True); sys.exit(1)
        print(out.strip().splitlines()[-1] if out.strip() else "{}", flush=True)
    if (i // 2) % 10 == 0:
        print(json.dumps({"progress": i + len(pair), "of": len(RANKS),
                          "min": round((time.time() - start) / 60, 1)}), flush=True)
print(json.dumps({"status": "COMPLETE", "min": round((time.time() - start) / 60, 1)}), flush=True)
