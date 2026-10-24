#!/usr/bin/env python3
"""Run one candidate over a contiguous public-fit panel, quietly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROW_RUNNER = HERE / "run_candidate_truth_row_v1.py"
PARENT_RUNS = HERE.parent / "public_observer_fit_truth_scan_v1/runs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--max-role-rank", type=int, required=True)
    args = parser.parse_args()
    candidate_runs = HERE / "runs" / args.candidate
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
    for rank in range(args.max_role_rank + 1):
        result_path = candidate_runs / f"row_{rank:04d}/result.json"
        if result_path.is_file():
            continue
        completed = subprocess.run(
            [sys.executable, "-B", str(ROW_RUNNER), "--candidate", args.candidate, "--role-rank", str(rank)],
            cwd=HERE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode

    comparisons = []
    for rank in range(args.max_role_rank + 1):
        parent = json.loads((PARENT_RUNS / f"row_{rank:04d}/result.json").read_text())
        child = json.loads((candidate_runs / f"row_{rank:04d}/result.json").read_text())
        parent_mse = float(parent["metrics"]["final_layer_mse"])
        child_mse = float(child["metrics"]["final_layer_mse"])
        comparisons.append((rank, parent_mse, child_mse))
    improved = sum(child < parent for _, parent, child in comparisons)
    worsened = sum(child > parent for _, parent, child in comparisons)
    parent_sum = sum(parent for _, parent, _ in comparisons)
    child_sum = sum(child for _, _, child in comparisons)
    tail_before = sum(parent > 5e-7 for _, parent, _ in comparisons)
    tail_after = sum(child > 5e-7 for _, _, child in comparisons)
    worst_after = sorted(comparisons, key=lambda row: row[2], reverse=True)[:5]
    print(json.dumps({
        "status": "COMPLETE",
        "candidate": args.candidate,
        "rows": len(comparisons),
        "improved": improved,
        "worsened": worsened,
        "parent_mean_final_mse": parent_sum / len(comparisons),
        "candidate_mean_final_mse": child_sum / len(comparisons),
        "mean_ratio": child_sum / parent_sum,
        "tail_count_before": tail_before,
        "tail_count_after": tail_after,
        "worst_after": [
            {"role_rank": rank, "parent": parent, "candidate": child}
            for rank, parent, child in worst_after
        ],
    }, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
