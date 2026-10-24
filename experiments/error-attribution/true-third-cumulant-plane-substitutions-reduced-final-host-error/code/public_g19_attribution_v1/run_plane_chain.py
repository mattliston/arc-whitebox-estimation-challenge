#!/usr/bin/env python3
"""TAG g19d-plane-chain-v1: plane-error tangent recursion oracle.
Seed dK21pre at layer 2 with TRUTH, propagate deltas with the analytic law:
  emission echo: dM21 = (d/2) o S21(c2,c1) + (d^T/2) o S12(c2,c1)
                 dM11 = (d/2) o S21(c1,c1) + (d^T/2) o S12(c1,c1)
                 dK21post = dM21 - 2 r1 (row) o dM11
  push (plane-supported congruence): d_next = (W o W)^T dK21post W
Score corr vs true dK21pre at check layers. If corr holds deep, the seed
(exactly computable at deploy) drives a deterministic dK21 supply.
Usage: run_plane_chain.py <rank> [seed_layer]"""
import json
import sys
from math import erf, factorial

import numpy as np

_erf = np.vectorize(erf)
sys.path.insert(0, ".")
from run_attribution import edg2_relu_mean


def columns(m, v, qmax=16):
    v = np.maximum(v, 1e-30)
    sd = np.sqrt(v)
    a = m / sd
    pdf = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(a / np.sqrt(2)))
    ma = -a
    He = [np.ones_like(a), ma]
    for k in range(2, qmax + 2):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    c1 = [sd * pdf + m * cdf, sd * cdf]
    for q in range(2, qmax):
        c1.append(sd * He[q - 2] * pdf)
    raw1 = c1[0]
    raw2 = (m * m + v) * cdf + m * sd * pdf
    c2 = [raw2, 2 * sd * raw1, 2 * v * cdf]
    for q in range(3, qmax):
        c2.append(2 * v * He[q - 3] * pdf)
    return np.stack(c1), np.stack(c2), sd


def main():
    rank = int(sys.argv[1])
    seed_l = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    d = np.load(f"states_rank{rank:04d}.npz")
    z = np.load(f"host_cov_rank{rank:04d}_base.npz")
    Pl = np.load(f"mps_planes_rank{rank:04d}.npz")
    import json as J
    from pathlib import Path
    P2 = Path(".").resolve().parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    manifest = J.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")

    dpre = (Pl[f"K21_{seed_l}"].astype(np.float64)
            - z["k21"][seed_l - 1].astype(np.float64))
    out = {"rank": rank, "seed_layer": seed_l, "layers": {}}
    for l in range(seed_l, 15):
        i = l - 1
        m_, v_ = d["host_m"][i], d["host_v"][i]
        k3, k4 = d["host_k3"][i], d["host_k4"][i]
        C = z["cov"][i].astype(np.float64)
        c1, c2, sd = columns(m_, v_)
        rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)

        def S(colsA, a, b, tmax=10):
            acc = np.zeros_like(rho)
            rp = np.ones_like(rho)
            for t in range(tmax + 1):
                if t > 0:
                    rp = rp * rho
                acc += rp / factorial(t) * np.outer(
                    colsA[a + t] / sd ** a, c1[b + t] / sd ** b)
            return acc

        S21_2, S12_2 = S(c2, 2, 1), S(c2, 1, 2)
        S21_1, S12_1 = S(c1, 2, 1), S(c1, 1, 2)
        dM21 = (dpre / 2) * S21_2 + (dpre.T / 2) * S12_2
        dM11 = (dpre / 2) * S21_1 + (dpre.T / 2) * S12_1
        r1 = edg2_relu_mean(m_, v_, k3, k4)
        dpost = dM21 - 2 * r1[:, None] * dM11
        Wd = np.asarray(W[l + 1], dtype=np.float64)
        dpre = (Wd * Wd).T @ dpost @ Wd
        nl = l + 1
        key = f"K21_{nl}"
        if key in Pl.files and nl - 1 < z["k21"].shape[0]:
            T = (Pl[key].astype(np.float64)
                 - z["k21"][nl - 1].astype(np.float64))
            corr = float(np.sum(dpre * T)
                         / np.sqrt(np.sum(dpre ** 2) * np.sum(T ** 2)))
            beta = float(np.sum(dpre * T) / np.sum(dpre ** 2))
            out["layers"][str(nl)] = {"corr": round(corr, 3),
                                      "beta": round(beta, 3),
                                      "pred_rms": float(np.sqrt(np.mean(dpre ** 2))),
                                      "true_rms": float(np.sqrt(np.mean(T ** 2)))}
    print(json.dumps(out, indent=1))
    open(f"PLANE_CHAIN_rank{rank:04d}_seed{seed_l}.json", "w").write(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
