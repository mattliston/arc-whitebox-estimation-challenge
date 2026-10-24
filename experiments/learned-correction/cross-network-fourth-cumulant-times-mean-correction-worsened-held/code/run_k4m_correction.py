#!/usr/bin/env python3
"""TAG g19g-k4m-correction: the universal deep-plane rank-1 deficit.
Fit alpha_l (coef of k4 (x) m in est-minus-truth) on the FIT net only,
FREEZE the profile, apply to the APPLY net with its own k4/m:
  pre_corr = pre - alpha_l * outer(k4_l, m_l);  k3u_corr follows the diag.
Writes tensor_planes_rank{apply}_k4m.npz (all other keys copied) and
prints held-out plane rel before/after per layer.
Usage: run_k4m_correction.py <fit_rank> <apply_rank>"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(rank):
    tp = np.load(HERE / f"tensor_planes_rank{rank:04d}.npz")
    Pl = np.load(HERE / f"mps_planes_rank{rank:04d}.npz")
    T = np.load(HERE / f"mps_cov2_rank{rank:04d}.npz")
    return tp, Pl, T


def main():
    rf, ra = int(sys.argv[1]), int(sys.argv[2])
    tpf, Plf, Tf = load(rf)
    alpha = {}
    for l in range(2, 15):
        if f"pre_{l}" not in tpf.files or f"K21_{l}" not in Plf.files:
            continue
        err = (tpf[f"pre_{l}"].astype(np.float64)
               - Plf[f"K21_{l}"].astype(np.float64))
        A = np.outer(Tf["mc_pre_k4"][l].astype(np.float64),
                     Tf["mc_pre_m"][l].astype(np.float64))
        alpha[l] = float(np.sum(err * A) / np.sum(A * A))
    print("frozen alpha profile (fit rank %d):" % rf,
          {l: round(a, 5) for l, a in alpha.items()})
    tpa, Pla, Ta = load(ra)
    out = {k: tpa[k] for k in tpa.files}
    rep = {}
    for l, a in alpha.items():
        if f"pre_{l}" not in tpa.files:
            continue
        A = np.outer(Ta["mc_pre_k4"][l].astype(np.float64),
                     Ta["mc_pre_m"][l].astype(np.float64))
        pre = tpa[f"pre_{l}"].astype(np.float64)
        cor = pre - a * A
        out[f"pre_{l}"] = cor.astype(np.float32)
        if f"k3u_{l}" in tpa.files:
            out[f"k3u_{l}"] = (tpa[f"k3u_{l}"].astype(np.float64)
                               - a * np.diag(A)).astype(np.float32)
        if f"K21_{l}" in Pla.files:
            tru = Pla[f"K21_{l}"].astype(np.float64)
            s = np.sqrt(np.mean(tru ** 2))
            rep[l] = {"rel_before": round(float(np.sqrt(np.mean(
                (pre - tru) ** 2))) / s, 4),
                "rel_after": round(float(np.sqrt(np.mean(
                    (cor - tru) ** 2))) / s, 4)}
    np.savez(HERE / f"tensor_planes_rank{ra:04d}_k4m.npz", **out)
    print(json.dumps({"apply_rank": ra, "layers": rep}, indent=1))


if __name__ == "__main__":
    main()
