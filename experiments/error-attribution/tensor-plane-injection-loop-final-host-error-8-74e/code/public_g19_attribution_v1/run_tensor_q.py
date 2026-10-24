#!/usr/bin/env python3
"""TAG g19f-tensor-q-v1: quality of the tensor recursion's plane deltas.
q_l = corr(K_tensor_pre_l - K_closure_l, K_true_l - K_closure_l), the
number the q2 arithmetic prices (target 0.925-0.983 stacked).
Usage: run_tensor_q.py [rank]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    suf = sys.argv[2] if len(sys.argv) > 2 else ""
    tp = np.load(HERE / f"tensor_planes_rank{rank:04d}{suf}.npz")
    Pl = np.load(HERE / f"mps_planes_rank{rank:04d}.npz")
    P1 = np.load(HERE / f"mps_planes_rank{rank:04d}_pre1.npz")
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    out = {}
    for l in range(2, 16):
        key = f"pre_{l}"
        tkey = f"K21_{l}"
        if key not in tp.files or l - 1 >= z["k21"].shape[0]:
            continue
        truth = (P1[tkey] if l == 1 and tkey in P1.files
                 else Pl[tkey] if tkey in Pl.files else None)
        if truth is None:
            continue
        truth = truth.astype(np.float64)
        K_t = tp[key].astype(np.float64)
        C = z["k21"][l - 1].astype(np.float64)
        dt = K_t - C
        dT = truth - C
        q = float(np.sum(dt * dT) / np.sqrt(np.sum(dt ** 2) * np.sum(dT ** 2)))
        beta = float(np.sum(dt * dT) / np.sum(dt ** 2))
        out[f"L{l}"] = {"q": round(q, 3), "beta": round(beta, 3),
                        "dnorm_ratio": round(float(np.sqrt(np.sum(dt ** 2)
                                             / np.sum(dT ** 2))), 3)}
    print(json.dumps(out, indent=1))
    (HERE / f"TENSOR_Q_rank{rank:04d}{suf}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
