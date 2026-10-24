#!/usr/bin/env python3
"""Run submitted-Q24 truth rows, reporting only hits and 50-row checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROW_RUNNER = HERE / "run_truth_row_v1.py"
RUNS = HERE / "runs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-hits", type=int, default=3)
    parser.add_argument("--max-role-rank", type=int, default=687)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()
    env = dict(os.environ)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONWARNINGS": "ignore",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "VECLIB_MAXIMUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
    })
    known = {}
    if RUNS.is_dir():
        for path in RUNS.glob("row_*/result.json"):
            row = json.loads(path.read_text())
            known[int(row["role_rank"])] = row
    hits = sum(bool(row["hit"]) for row in known.values())
    for role_rank in range(args.max_role_rank + 1):
        if role_rank not in known:
            completed = subprocess.run(
                [sys.executable, "-B", str(ROW_RUNNER), "--role-rank", str(role_rank)],
                cwd=HERE,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode != 0:
                print(completed.stdout, end="")
                print(completed.stderr, end="", file=sys.stderr)
                print(json.dumps({"status": "TERMINAL_ROW_FAILURE", "role_rank": role_rank}), flush=True)
                return completed.returncode
            row = json.loads((RUNS / f"row_{role_rank:04d}" / "result.json").read_text())
            known[role_rank] = row
            if row["hit"]:
                hits += 1
                metrics = row["metrics"]
                print(json.dumps({
                    "status": "HIT",
                    "hit_number": hits,
                    "role_rank": role_rank,
                    "mlp_id": row["mlp_id"],
                    "outcome": row["outcome"],
                    "final_mse": None if metrics is None else metrics["final_layer_mse"],
                    "onset": None if metrics is None else metrics["first_layer_above_5e_7"],
                }, allow_nan=False), flush=True)
        if hits >= args.stop_hits:
            print(json.dumps({"status": "COMPLETE", "rows": len(known), "hits": hits}), flush=True)
            return 0
        if len(known) % args.checkpoint_every == 0:
            print(json.dumps({"status": "CHECKPOINT", "rows": len(known), "hits": hits}), flush=True)
    print(json.dumps({"status": "EXHAUSTED", "rows": len(known), "hits": hits}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
