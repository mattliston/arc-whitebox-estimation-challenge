#!/usr/bin/env python3
"""TAG g19g-loo-grade: leave-one-out grading of a supply's correction
chain across the panel. For each held-out net: fit joint (k4,k3,v)
betas by pooled least squares on the OTHER nets' end-frame vectors,
apply frozen to the held-out net. Reports per-net capture and mse plus
mean +- sd. Usage: run_loo_grade.py <supply> <rank> [rank ...]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import os
CH = tuple(os.environ.get("LOO_CH", "k4,k3,v").split(","))


def load(rank, supply):
    rv = np.load(HERE / f"rvec_{supply}_rank{rank:04d}.npz")
    tr = np.load(HERE / f"mps_truth_rank{rank:04d}_P8M.npz")
    dep = np.load(HERE / f"deployed_rank{rank:04d}.npy")
    resid = tr["mc_post"][15] - dep[15]
    noise = float(np.sum(tr["mc_se"][15] ** 2))
    return np.stack([rv[k] for k in CH]), resid, noise


def main():
    supply = sys.argv[1]
    ranks = [int(a) for a in sys.argv[2:]]
    data = {r: load(r, supply) for r in ranks}
    out = {}
    caps = []
    for h in ranks:
        G = np.zeros((len(CH), len(CH)))
        bv = np.zeros(len(CH))
        for r in ranks:
            if r == h:
                continue
            R, resid, _ = data[r]
            G += R @ R.T
            bv += R @ resid
        beta = np.linalg.solve(G + 1e-12 * np.eye(len(CH)), bv)
        R, resid, noise = data[h]
        c = beta @ R
        before = float(resid @ resid) - noise
        after = float((resid - c) @ (resid - c)) - noise
        cap = 100 * (1 - after / before)
        caps.append(cap)
        out[str(h)] = {"betas": [round(float(x), 3) for x in beta],
                       "capture_pct": round(cap, 1),
                       "mse_nc": after / 1024}
    out["mean_capture"] = round(float(np.mean(caps)), 1)
    out["sd_capture"] = round(float(np.std(caps)), 1)
    print(json.dumps(out, indent=1))
    (HERE / f"LOO_GRADE_{supply}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
