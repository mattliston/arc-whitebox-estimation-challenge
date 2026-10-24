#!/usr/bin/env python3
"""TAG g19g-beta-transfer: estimate-quality betas, cross-net frozen.
Fit joint (k4,k3,v) betas by least squares on net A's end-frame vectors
(rvec_{supply}_rankA.npz from DUMP_R=1 consumption), apply FROZEN to net
B; report capture vs the fixed universal betas both directions.
Usage: run_beta_transfer.py <supply> <rankA> <rankB>"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH = ("k4", "k3", "v")
FIXED = {"k4": 0.16, "k3": 0.33, "v": 0.82}


def load(rank, supply):
    rv = np.load(HERE / f"rvec_{supply}_rank{rank:04d}.npz")
    tr = np.load(HERE / f"mps_truth_rank{rank:04d}_P8M.npz")
    dep = np.load(HERE / f"deployed_rank{rank:04d}.npy")
    resid = tr["mc_post"][15] - dep[15]
    noise = float(np.sum(tr["mc_se"][15] ** 2))
    R = np.stack([rv[k] for k in CH])
    return R, resid, noise


def capture(R, resid, noise, betas):
    c = sum(b * r for b, r in zip(betas, R))
    before = float(resid @ resid) - noise
    after = float((resid - c) @ (resid - c)) - noise
    return 100 * (1 - after / before), after / 1024


def fit(R, resid):
    G = R @ R.T
    b = R @ resid
    return np.linalg.solve(G, b)


def main():
    supply = sys.argv[1]
    ra, rb = int(sys.argv[2]), int(sys.argv[3])
    out = {}
    data = {r: load(r, supply) for r in (ra, rb)}
    for src, dst in ((ra, rb), (rb, ra)):
        Rs, rs, _ = data[src]
        beta = fit(Rs, rs)
        Rd, rd, nd = data[dst]
        cap_t, mse_t = capture(Rd, rd, nd, beta)
        cap_f, mse_f = capture(Rd, rd, nd, [FIXED[k] for k in CH])
        out[f"fit{src}_apply{dst}"] = {
            "betas": {k: round(float(b), 3) for k, b in zip(CH, beta)},
            "capture_pct": round(cap_t, 1), "mse_nc": mse_t,
            "fixedbeta_capture_pct": round(cap_f, 1), "fixedbeta_mse_nc": mse_f}
    print(json.dumps(out, indent=1))
    (HERE / f"BETA_TRANSFER_{supply}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
