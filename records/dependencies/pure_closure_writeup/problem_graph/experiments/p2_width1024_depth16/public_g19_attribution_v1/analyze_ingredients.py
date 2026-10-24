#!/usr/bin/env python3
"""TAG g19a-ingredient-split-v1: which ingredient drives the late-layer mean
error - the transported mean recursion (m), variance (v), skew (k3), or
kurtosis (k4)? One-at-a-time swaps of host vs MC-true state through the
order-2 emission formula, from saved states. Also reports the per-layer
state-fidelity ladder for this host and the error growth profile."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_attribution import edg2_relu_mean


def analyze(rank):
    d = np.load(HERE / f"states_rank{rank:04d}.npz")
    mc_m, mc_v = d["mc_pre_m"], d["mc_pre_v"]
    mc_k3, mc_k4 = d["mc_pre_k3"], d["mc_pre_k4"]
    hm, hv, hk3, hk4 = d["host_m"], d["host_v"], d["host_k3"], d["host_k4"]
    mc_post, mc_se, host_post = d["mc_post"], d["mc_se"], d["host_post"]

    out = {"rank": int(rank), "layers": {}}
    for l in range(1, 16):
        i = l - 1
        if i >= hm.shape[0]:
            continue
        base = edg2_relu_mean(hm[i], hv[i], hk3[i], hk4[i])
        full = edg2_relu_mean(mc_m[l], mc_v[l], mc_k3[l], mc_k4[l])
        dm = edg2_relu_mean(mc_m[l], hv[i], hk3[i], hk4[i]) - base
        dv = edg2_relu_mean(hm[i], mc_v[l], hk3[i], hk4[i]) - base
        dk3 = edg2_relu_mean(hm[i], hv[i], mc_k3[l], hk4[i]) - base
        dk4 = edg2_relu_mean(hm[i], hv[i], hk3[i], mc_k4[l]) - base
        tb = full - base
        tb_e = float(np.sum(tb ** 2))
        rec = {
            "term_B_energy": tb_e,
            "share_m": float(np.sum(dm ** 2)) / tb_e if tb_e else None,
            "share_v": float(np.sum(dv ** 2)) / tb_e if tb_e else None,
            "share_k3": float(np.sum(dk3 ** 2)) / tb_e if tb_e else None,
            "share_k4": float(np.sum(dk4 ** 2)) / tb_e if tb_e else None,
            "additivity": float(np.sum((dm + dv + dk3 + dk4 - tb) ** 2)) / tb_e
                          if tb_e else None,
            "state_rel_err": {
                "m": float(np.sqrt(np.mean((hm[i] - mc_m[l]) ** 2)
                                   / np.mean(mc_m[l] ** 2))),
                "v": float(np.sqrt(np.mean((hv[i] - mc_v[l]) ** 2)
                                   / np.mean(mc_v[l] ** 2))),
                "k3": float(np.sqrt(np.mean((hk3[i] - mc_k3[l]) ** 2)
                                    / np.mean(mc_k3[l] ** 2))),
                "k4": float(np.sqrt(np.mean((hk4[i] - mc_k4[l]) ** 2)
                                    / np.mean(mc_k4[l] ** 2))),
            },
            "post_err_energy": float(np.sum((mc_post[l] - host_post[l]) ** 2)),
        }
        out["layers"][str(l)] = rec

    late = [out["layers"][str(l)] for l in range(10, 16)
            if str(l) in out["layers"]]
    tot = sum(r["term_B_energy"] for r in late)
    out["late_summary"] = {
        "share_m": sum(r["share_m"] * r["term_B_energy"] for r in late) / tot,
        "share_v": sum(r["share_v"] * r["term_B_energy"] for r in late) / tot,
        "share_k3": sum(r["share_k3"] * r["term_B_energy"] for r in late) / tot,
        "share_k4": sum(r["share_k4"] * r["term_B_energy"] for r in late) / tot,
        "additivity": sum(r["additivity"] * r["term_B_energy"]
                          for r in late) / tot,
    }
    (HERE / f"INGREDIENTS_rank{rank:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps({"rank": rank, **out["late_summary"]}, indent=1))
    print("state ladder (relative RMS error by layer):")
    for l in (2, 5, 8, 11, 14, 15):
        r = out["layers"].get(str(l))
        if r:
            s = r["state_rel_err"]
            print(f"  L{l:2d}: m={s['m']:.4f} v={s['v']:.4f} "
                  f"k3={s['k3']:.4f} k4={s['k4']:.4f}")


if __name__ == "__main__":
    for rank in (int(a) for a in sys.argv[1:] or ("35", "100")):
        analyze(rank)
