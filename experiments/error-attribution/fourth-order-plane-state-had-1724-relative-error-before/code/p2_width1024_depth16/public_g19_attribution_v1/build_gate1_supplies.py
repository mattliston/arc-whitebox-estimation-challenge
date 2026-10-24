#!/usr/bin/env python3
"""TAG g19g-gate1-supplies: build the two BILLABLE fire-candidate
supplies per net, deploy-native calibrations only:
  _gc (collective-only, zero-bill): closure + (K3S_l - implied_closure_l)
       * outer(w^2, w), tracker with universal c=2.9.
  _gj (joint): _gc + gamma(l) * (honest-hub delta), gamma(l) = the
       frozen r35 lsq profile (cross-net verified +-10%).
Writes tensor_planes_rank{r}_gc.npz / _gj.npz (pre_l + k3u_l keys).
Usage: build_gate1_supplies.py <rank> [gamma_fit_rank=35]"""
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


def weights_for(rank):
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    return np.load((STORE / entry["path"]).parent / "weights.npy",
                   mmap_mode="r")


def tracker(rank):
    T = np.load(HERE / f"mps_cov2_rank{rank:04d}.npz")
    W = weights_for(rank)

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
    return K3S, T


def gamma_profile(fit_rank):
    Pl = np.load(HERE / f"mps_planes_rank{fit_rank:04d}.npz")
    z = np.load(HERE / f"host_cov_rank{fit_rank:04d}_base.npz")
    hub = np.load(HERE / f"hub_planes_rank{fit_rank:04d}_honest.npz")
    gam = {}
    for l in range(2, 15):
        if f"pre_{l}" not in hub.files or f"K21_{l}" not in Pl.files:
            continue
        dK = (hub[f"pre_{l}"].astype(np.float64)
              - z["k21"][l - 1].astype(np.float64))
        dT = (Pl[f"K21_{l}"].astype(np.float64)
              - z["k21"][l - 1].astype(np.float64))
        gam[l] = float(np.sum(dK * dT) / np.sum(dK ** 2))
    return gam


def main():
    rank = int(sys.argv[1])
    fitr = int(sys.argv[2]) if len(sys.argv) > 2 else 35
    K3S, T = tracker(rank)
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    hub = np.load(HERE / f"hub_planes_rank{rank:04d}_honest.npz")
    gam = gamma_profile(fitr)
    gc, gj = {}, {}
    for l in range(2, 15):
        if l - 1 >= z["k21"].shape[0]:
            continue
        zc = z["k21"][l - 1].astype(np.float64)
        m = T["mc_pre_m"][l].astype(np.float64)
        w = m / max(np.linalg.norm(m), 1e-30)
        impl = float((w * w) @ zc @ w)
        delta = float(K3S[l]) - impl
        coll = delta * np.outer(w * w, w)
        pre_c = zc + coll
        gc[f"pre_{l}"] = pre_c.astype(np.float32)
        gc[f"k3u_{l}"] = np.diag(pre_c).astype(np.float32)
        hd = (hub[f"pre_{l}"].astype(np.float64) - zc) \
            if f"pre_{l}" in hub.files else 0.0
        pre_j = zc + coll + gam.get(l, 0.0) * hd
        gj[f"pre_{l}"] = pre_j.astype(np.float32)
        gj[f"k3u_{l}"] = np.diag(pre_j).astype(np.float32)
    np.savez(HERE / f"tensor_planes_rank{rank:04d}_gc.npz", **gc)
    np.savez(HERE / f"tensor_planes_rank{rank:04d}_gj.npz", **gj)
    print(json.dumps({"rank": rank, "layers": len(gc) // 2}))


if __name__ == "__main__":
    main()
