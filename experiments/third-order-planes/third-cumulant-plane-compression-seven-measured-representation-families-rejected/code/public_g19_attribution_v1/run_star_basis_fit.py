#!/usr/bin/env python3
"""TAG g19carrier-star-basis: is the per-step fresh K21 error explained
by the ONE-STEP STAR basis  fresh_l ~= sum_c t_c (q_c*q_c) x q_c,
q_c = e1-scaled column c of W_l (the one-step propagator image of unit
c's fresh third-cumulant emission)?  Exact linear fit of the 1024
amplitudes via the separable Gram:
  Gram[c,c'] = <q_c^2, q_c'^2> * <q_c, q_c'>   (hadamard of two Grams)
  rhs[c]     = q_c^2 . fresh . q_c             (diag of batched forms)
Targets: (a) fresh_l = dK21_{l+1} - beta*P(dK21_l)  [recursion residual]
         (b) dK21_{l+1} raw                        [total error]
Reports fitted fraction per layer + corr of fitted t_c with the analytic
per-unit emission proxy (host fresh kappa3 of the ReLU marginal).
Usage: run_star_basis_fit.py <rank> [...]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_tangent_v2 import c1_std_factors


def analytic_emission(m, v, k3):
    """Per-unit fresh kappa3 emitted by the ReLU marginal: kappa3 of
    phi(z_c) under the GC marginal minus the linear-response carry of
    the input kappa3 (e1^3 k3): the closure's per-unit k3u recursion
    carries exactly the diagonal; the STAR spread is sourced by the
    full emitted kappa3."""
    v = np.maximum(v, 1e-30)
    s = np.sqrt(v)
    a = m / s
    pdf = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    from math import erf
    cdf = 0.5 * (1 + np.vectorize(erf)(a / np.sqrt(2)))
    # ReLU Gaussian moments
    m1 = s * pdf + m * cdf
    m2 = (m * m + v) * cdf + m * s * pdf
    m3 = (m ** 3 + 3 * m * v) * cdf + (m * m + 2 * v) * s * pdf
    mu2 = m2 - m1 ** 2
    k3out = m3 - 3 * m2 * m1 + 2 * m1 ** 3
    e1 = cdf  # d m1 / d m  (leading linear response)
    return k3out - e1 ** 3 * k3, k3out, mu2


def fit_layer(fresh, Q):
    """Q: (1024 units, 1024 dims) rows q_c. Solve min ||fresh - sum t_c
    q_c^2 x q_c|| via separable Gram."""
    Q2 = Q * Q
    G = (Q2 @ Q2.T) * (Q @ Q.T)
    rhs = np.einsum("ci,ij,cj->c", Q2, fresh, Q, optimize=True)
    lam = 1e-12 * float(np.trace(G)) / 1024
    t = np.linalg.solve(G + lam * np.eye(1024), rhs)
    expl = float(t @ rhs)                      # ||proj||^2
    tot = float(np.sum(fresh ** 2))
    return t, expl / max(tot, 1e-300)


def probe(rank):
    d = (np.load(HERE / f"states_rank{rank:04d}_lite.npz")
         if (HERE / f"states_rank{rank:04d}_lite.npz").exists()
         and not (HERE / f"states_rank{rank:04d}.npz").exists()
         else np.load(HERE / f"states_rank{rank:04d}.npz"))
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    tp = np.load(HERE / f"tensor_planes_rank{rank:04d}.npz")
    P2 = HERE.parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy",
                mmap_mode="r")
    out = {}
    for l in range(4, 14):
        i = l - 1
        if f"pre_{l}" not in tp.files or f"pre_{l+1}" not in tp.files:
            continue
        dK = (tp[f"pre_{l}"].astype(np.float64)
              - z["k21"][i].astype(np.float64))
        dKn = (tp[f"pre_{l+1}"].astype(np.float64)
               - z["k21"][i + 1].astype(np.float64))
        m_, v_ = d["host_m"][i], d["host_v"][i]
        hk3 = d["host_k3"][i]
        c1, e, sd = c1_std_factors(m_, v_)
        e1 = c1[1] / sd
        e2 = c1[2] / sd ** 2
        Wl = np.asarray(W[l + 1], dtype=np.float64)
        P = (Wl * Wl).T @ (dK * np.outer(e2, e1)) @ Wl
        beta = float(np.sum(P * dKn) / max(np.sum(P * P), 1e-300))
        fresh = dKn - beta * P
        # star basis: q_c = row c of Wl (unit c's fan-out image)
        Q = Wl.copy()
        t_fr, fr_frac = fit_layer(fresh, Q)
        t_tot, tot_frac = fit_layer(dKn, Q)
        em_delta, em_raw, mu2 = analytic_emission(m_, v_, hk3)
        def cc(a, b):
            return float(np.sum(a * b)
                         / np.sqrt(max(np.sum(a * a) * np.sum(b * b),
                                       1e-300)))
        out[l + 1] = {
            "star_frac_of_fresh": round(fr_frac, 3),
            "star_frac_of_total": round(tot_frac, 3),
            "t_corr_emission_delta": round(cc(t_fr, em_delta), 3),
            "t_corr_emission_raw": round(cc(t_fr, em_raw), 3),
            "t_tot_corr_emission_delta": round(cc(t_tot, em_delta), 3)}
        print(rank, f"L{l+1}", json.dumps(out[l + 1]))
    (HERE / f"STAR_BASIS_FIT_rank{rank:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    for a in sys.argv[1:] or ["35"]:
        probe(int(a))
