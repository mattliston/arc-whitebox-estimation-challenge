#!/usr/bin/env python3
"""TAG g19g-collective-patch: add the validated collective-skewness
correction to the plane supply. Tracker (held-out validated +-1.3%):
  K3S_{l+1} = (G_l/sigma_l)^3 K3S_l + C_UNIV * fresh_ridge_l,  K3S_1 = 0
with G/sigma/fresh from the exact 1D ridge at each layer's state and
C_UNIV = 2.9 (frozen, fit r35, held out r100). Correction per layer:
  pre_l += (K3S_l - k3impl_l) * outer(w*w, w),  w = m-hat_l
Writes tensor_planes_rank{r}_coll.npz. Usage: run_collective_patch.py <rank>"""
import json
import sys
from math import erf
from pathlib import Path

import numpy as np

_erf = np.vectorize(erf)
HERE = Path(__file__).resolve().parent
STORE = (HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
C_UNIV = 2.9


def main():
    rank = int(sys.argv[1])
    T = np.load(HERE / f"mps_cov2_rank{rank:04d}.npz")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy",
                mmap_mode="r")
    tp = np.load(HERE / f"tensor_planes_rank{rank:04d}.npz")

    def step(l):
        m_l = T["mc_pre_m"][l].astype(np.float64)
        C_l = T["mc_pre_cov"][l].astype(np.float64)
        w_l = m_l / max(np.linalg.norm(m_l), 1e-30)
        m_n = T["mc_pre_m"][l + 1].astype(np.float64)
        w_n = m_n / max(np.linalg.norm(m_n), 1e-30)
        u = np.asarray(W[l + 1], dtype=np.float64) @ w_n
        sig = np.sqrt(max(w_l @ C_l @ w_l, 1e-30))
        b = (C_l @ w_l) / sig
        vp = np.maximum(np.diag(C_l) - b * b, 1e-30)
        sd = np.sqrt(vp)

        def F(t):
            mu = m_l + t * b
            al = mu / sd
            pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
            cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
            return float(u @ (sd * pdf + mu * cdf))

        G = (F(0.05) - F(-0.05)) / 0.1
        x, wq = np.polynomial.hermite_e.hermegauss(60)
        wq = wq / np.sqrt(2 * np.pi)
        S = np.array([F(t) for t in x])
        E1, E2, E3 = wq @ S, wq @ S ** 2, wq @ S ** 3
        return (G / sig) ** 3, E3 - 3 * E2 * E1 + 2 * E1 ** 3

    K3S = {1: 0.0}
    k3 = 0.0
    for l in range(1, 15):
        g3, fr = step(l)
        k3 = g3 * k3 + C_UNIV * fr
        K3S[l + 1] = k3
    out = {k: tp[k] for k in tp.files}
    rep = {}
    for l in range(2, 15):
        if f"pre_{l}" not in tp.files or f"k3impl_{l}" not in tp.files:
            continue
        m = T["mc_pre_m"][l].astype(np.float64)
        w = m / max(np.linalg.norm(m), 1e-30)
        delta = float(K3S[l]) - float(tp[f"k3impl_{l}"])
        pre = tp[f"pre_{l}"].astype(np.float64)
        cor = pre + delta * np.outer(w * w, w)
        out[f"pre_{l}"] = cor.astype(np.float32)
        out[f"k3u_{l}"] = np.diag(cor).astype(np.float32)
        rep[l] = {"K3S": round(K3S[l], 3),
                  "implied": round(float(tp[f"k3impl_{l}"]), 3),
                  "delta": round(delta, 3)}
    np.savez(HERE / f"tensor_planes_rank{rank:04d}_coll.npz", **out)
    print(json.dumps({"rank": rank, "layers": rep}, indent=1))


if __name__ == "__main__":
    main()
