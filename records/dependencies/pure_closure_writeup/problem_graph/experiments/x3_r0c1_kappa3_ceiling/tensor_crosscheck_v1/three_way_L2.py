#!/usr/bin/env python3
"""TAG r0c1-crosscheck-step2-v1: the step-two fork. Inputs at pre-1 are
exact-grade (true mean/cov from 64M MC; kappa3 tensor = my step-1 output,
validated at the MC floor). Apply ONE ReLU step two ways - (A) my tensor
map, (B) the A2f emitter - push to pre-2, compare K21 vs truth. Both maps
are FIRST order in kappa3pre: if both miss by ~16 pct, the blocker is the
missing second-order kappa3 terms in both engines; if B does much better,
my transport/feed truncation is the gap. Usage: three_way_L2.py"""
import json
import sys
import time
from math import erf, factorial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tensor_recursion_w256 import generate_weights, relu_columns, pair_map
from three_way_L1 import my_post_tensor

_erf = np.vectorize(erf)
WIDTH = 256
DTOT = 8


def my_relu_step(m, C, T_pre, K21pre, k3u, k4u):
    """One tensor ReLU step (numpy f64): F3 transport + connected feed +
    Gaussian completion on all entries, pair-map plane slices, exact diag."""
    v = np.maximum(np.diag(C), 1e-30)
    sd = np.sqrt(v)
    qmax = 2 * DTOT + 2
    c1, c2, bcols, dcols, sd_, pdf, cdf, He = relu_columns(m, v, qmax)
    rho = C / np.outer(sd, sd)
    np.fill_diagonal(rho, 0.0)
    rpow = {t: rho ** t for t in range(1, DTOT + 1)}
    terms = []
    for tij in range(0, DTOT + 1):
        for tik in range(0, DTOT + 1 - tij):
            for tjk in range(0, DTOT + 1 - tij - tik):
                di, dj, dk = tij + tik, tij + tjk, tik + tjk
                if di == 0 or dj == 0 or dk == 0:
                    continue
                terms.append((tij, tik, tjk, di, dj, dk,
                              1.0 / (factorial(tij) * factorial(tik)
                                     * factorial(tjk))))
    allterms = [(0, 0, 0, 0, 0, 0, 1.0)] + terms

    def bshape(cols, t, pos):
        vv = cols[t]
        return (vv[:, None, None] if pos == 0 else
                vv[None, :, None] if pos == 1 else vv[None, None, :])

    def rshape(t, p1, p2):
        if {p1, p2} == {0, 1}:
            return rpow[t][:, :, None]
        if {p1, p2} == {0, 2}:
            return rpow[t][:, None, :]
        return rpow[t][None, :, :]

    F3 = np.zeros((WIDTH, WIDTH, WIDTH))
    for (tij, tik, tjk, di, dj, dk, coef) in allterms:
        t_ = (bshape(bcols, di, 0) * bshape(bcols, dj, 1)
              * bshape(bcols, dk, 2)) * coef
        if tij:
            t_ = t_ * rshape(tij, 0, 1)
        if tik:
            t_ = t_ * rshape(tik, 0, 2)
        if tjk:
            t_ = t_ * rshape(tjk, 1, 2)
        F3 += t_
    T = T_pre * F3
    del F3
    pats = [(K21pre[:, :, None], 0, 1, 2), (K21pre[:, None, :], 0, 2, 1),
            (K21pre.T[:, :, None], 1, 0, 2), (K21pre[None, :, :], 1, 2, 0),
            (K21pre.T[:, None, :], 2, 0, 1), (K21pre.T[None, :, :], 2, 1, 0)]
    for (Kb, up, wp, sp) in pats:
        acc = None
        for tuw in range(0, 5):
            for tus in range(0, 5 - tuw):
                for tws in range(0, 5 - tuw - tus):
                    if tus + tws < 1:
                        continue
                    du, dw, ds = tuw + tus, tuw + tws, tus + tws
                    cf = 0.5 / (factorial(tuw) * factorial(tus)
                                * factorial(tws))
                    t_ = (bshape(dcols, du, up) * bshape(bcols, dw, wp)
                          * bshape(c1, ds, sp)) * cf
                    if tuw:
                        t_ = t_ * rshape(tuw, up, wp)
                    if tus:
                        t_ = t_ * rshape(tus, up, sp)
                    if tws:
                        t_ = t_ * rshape(tws, wp, sp)
                    acc = t_ if acc is None else acc + t_
        T += Kb * acc
        del acc
    for (tij, tik, tjk, di, dj, dk, coef) in terms:
        t_ = (bshape(c1, di, 0) * bshape(c1, dj, 1)
              * bshape(c1, dk, 2)) * coef
        if tij:
            t_ = t_ * rshape(tij, 0, 1)
        if tik:
            t_ = t_ * rshape(tik, 0, 2)
        if tjk:
            t_ = t_ * rshape(tjk, 1, 2)
        T += t_
    # kappa4-feed, RAW response + numerical product-rule assembly (the
    # rig-validated spelling; no connected-filter guesswork). kappa4 fields
    # from the two-vertex law, calibrated (tree measured 1.9x hot here).
    vv = v
    K4_CAL = 0.52
    Pv = K21pre / vv[None, :]
    k4d = K4_CAL * np.einsum("ix,ix->i", K21pre, Pv)
    K31f = K4_CAL * 3.0 * np.einsum("im,ijm->ij", Pv, T_pre)
    K22f = K4_CAL * (Pv @ K21pre.T + 2.0 * np.einsum(
        "ijm,ijm->ij", T_pre, T_pre / vv[None, None, :]))
    K4T = K4_CAL * (np.einsum("im,jkm->ijk", Pv, T_pre)
                    + 2.0 * np.einsum("ijm,ikm->ijk", T_pre,
                                      T_pre / vv[None, None, :]))
    import os as _os
    if _os.environ.get("K4_DIAG_ONLY") == "1":
        K31f = K31f * 0.0
        K22f = K22f * 0.0
        K4T = K4T * 0.0
    # column families (vectors over units): fam[n][t]
    al_ = m / sd
    pdf_ = np.exp(-0.5 * al_ * al_) / np.sqrt(2 * np.pi)
    cdf_ = 0.5 * (1 + _erf(al_ / np.sqrt(2)))
    ma_ = -al_
    HeV = [np.ones(WIDTH), ma_]
    for kk_ in range(2, 16):
        HeV.append(ma_ * HeV[-1] - (kk_ - 1) * HeV[-2])
    fam = {0: c1, 1: bcols, 2: dcols,
           3: np.stack([HeV[1 + t] * pdf_ / sd ** 2 for t in range(9)]),
           4: np.stack([HeV[2 + t] * pdf_ / sd ** 3 for t in range(9)])}
    from second_order_triple import K4_PATTERNS
    k4fields = {"k4_0": ("v0", k4d), "k4_1": ("v1", k4d), "k4_2": ("v2", k4d),
                "K31_01": ("m01", K31f), "K31_10": ("m01", K31f.T),
                "K31_02": ("m02", K31f), "K31_20": ("m02", K31f.T),
                "K31_12": ("m12", K31f), "K31_21": ("m12", K31f.T),
                "K22_01": ("m01", K22f), "K22_02": ("m02", K22f),
                "K22_12": ("m12", K22f),
                "K4T_0": ("t", K4T), "K4T_1": ("t", K4T.transpose(1, 0, 2)),
                "K4T_2": ("t", K4T.transpose(2, 0, 1))}

    def bcast(kind, val):
        if kind == "v0":
            return val[:, None, None]
        if kind == "v1":
            return val[None, :, None]
        if kind == "v2":
            return val[None, None, :]
        if kind == "m01":
            return val[:, :, None]
        if kind == "m02":
            return val[:, None, :]
        if kind == "m12":
            return val[None, :, :]
        return val

    # raw dE3 over all kappa4 patterns x bonds (cap 2), UNFILTERED
    dE3 = np.zeros((WIDTH, WIDTH, WIDTH))
    for (name, counts, key, base) in K4_PATTERNS:
        kind, val = k4fields[key]
        fld = bcast(kind, val)
        for t01 in range(0, 3):
            for t02 in range(0, 3 - t01):
                for t12 in range(0, 3 - t01 - t02):
                    bc_ = base / (factorial(t01) * factorial(t02)
                                  * factorial(t12))
                    deg = (t01 + t02, t01 + t12, t02 + t12)
                    t_ = (fam[counts[0]][deg[0]][:, None, None]
                          * fam[counts[1]][deg[1]][None, :, None]
                          * fam[counts[2]][deg[2]][None, None, :]) * bc_
                    if t01:
                        t_ = t_ * rpow[t01][:, :, None]
                    if t02:
                        t_ = t_ * rpow[t02][:, None, :]
                    if t12:
                        t_ = t_ * rpow[t12][None, :, :]
                    dE3 += fld * t_
    # assembly subtractions
    r1g_ = sd * pdf_ + m * cdf_                     # E1^0 (Gaussian base)
    dE1 = (k4d / 24.0) * fam[4][0]                  # kappa4 univariate resp
    # pair pieces (matrices over the pair, Gaussian base + kappa4 response)
    rho_m = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
    np.fill_diagonal(rho_m, 0.0)
    E2_0 = np.zeros((WIDTH, WIDTH))
    dE2 = np.zeros((WIDTH, WIDTH))
    rp_ = np.ones_like(rho_m)
    for t in range(0, 9):
        if t > 0:
            rp_ = rp_ * rho_m
        w_ = rp_ / factorial(t)
        E2_0 += w_ * np.outer(c1[t], c1[t])
        dE2 += w_ * ((k4d / 24.0 * fam[4][t])[:, None] * c1[t][None, :]
                     + c1[t][:, None] * (k4d / 24.0 * fam[4][t])[None, :]
                     + (K31f / 6.0) * np.outer(fam[3][t], bcols[t])
                     + (K31f.T / 6.0) * np.outer(bcols[t], fam[3][t])
                     + (K22f / 4.0) * np.outer(dcols[t], dcols[t]))
    dcum = dE3.copy()
    dcum -= dE1[:, None, None] * E2_0[None, :, :]
    dcum -= dE1[None, :, None] * E2_0[:, None, :]
    dcum -= dE1[None, None, :] * E2_0[:, :, None]
    dcum -= r1g_[:, None, None] * dE2[None, :, :]
    dcum -= r1g_[None, :, None] * dE2[:, None, :]
    dcum -= r1g_[None, None, :] * dE2[:, :, None]
    dcum += 2.0 * (dE1[:, None, None] * r1g_[None, :, None]
                   * r1g_[None, None, :]
                   + r1g_[:, None, None] * dE1[None, :, None]
                   * r1g_[None, None, :]
                   + r1g_[:, None, None] * r1g_[None, :, None]
                   * dE1[None, None, :])
    T += dcum
    del dE3, dcum
    r1, post_C, K21post, _ = pair_map(m, v, k3u, k4u, C, K21pre)
    idx = np.arange(WIDTH)
    for i in range(WIDTH):
        T[i, i, :] = K21post[i]
        T[i, :, i] = K21post[i]
        T[:, i, i] = K21post[:, i]
    al = m / sd
    pdf1 = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf1 = 0.5 * (1 + _erf(al / np.sqrt(2)))
    r1g = sd * pdf1 + m * cdf1
    r2g = (m * m + v) * cdf1 + m * sd * pdf1
    r3g = (m ** 3 + 3 * m * v) * cdf1 + (m * m + 2 * v) * sd * pdf1
    T[idx, idx, idx] = r3g - 3 * r2g * r1g + 2 * r1g ** 3
    return T


def main():
    W = generate_weights(1000000)
    W0, W1, W2 = (W[i].astype(np.float64) for i in range(3))
    mc = np.load(HERE / "mc_w256_seed1000000.npz")
    m0 = np.zeros(WIDTH)
    C0 = W0.T @ W0
    T0, _ = my_post_tensor(m0, C0)          # exact-grade post-0 tensor

    def push(T, Wl):
        X = np.einsum("ia,ijk->ajk", Wl, T, optimize=True)
        X = np.einsum("jb,ajk->abk", Wl, X, optimize=True)
        return np.einsum("kc,abk->abc", Wl, X, optimize=True)

    T1 = push(T0, W1)                        # pre-1 kappa3 tensor
    del T0
    T1 = (T1 + T1.transpose(0, 2, 1) + T1.transpose(1, 0, 2)
          + T1.transpose(1, 2, 0) + T1.transpose(2, 0, 1)
          + T1.transpose(2, 1, 0)) / 6.0
    idx = np.arange(WIDTH)
    K21pre1 = T1[idx, idx, :].copy()
    k3u1 = T1[idx, idx, idx].copy()
    m1 = mc["pre_m"][1].astype(np.float64)
    C1 = mc["pre_cov"][1].astype(np.float64)
    C1 = 0.5 * (C1 + C1.T)
    v1 = np.maximum(np.diag(C1), 1e-30)
    k4u1 = np.einsum("ix,ix->i", K21pre1, K21pre1 / v1[None, :])
    truth2 = mc["pre_K21"][2].astype(np.float64)
    scale = float(np.sqrt(np.mean(truth2 ** 2)))
    out = {}

    t0 = time.time()
    Tmine = my_relu_step(m1, C1, T1, K21pre1, k3u1, k4u1)
    import os as _os
    if _os.environ.get("TRUE_POST_SLICES") == "1":
        mcv2 = np.load(HERE / "mc_w256_seed1000000_v2.npz")
        Kp_true = mcv2["post_K21"][1].astype(np.float64)
        for i_ in range(WIDTH):
            Tmine[i_, i_, :] = Kp_true[i_]
            Tmine[i_, :, i_] = Kp_true[i_]
            Tmine[:, i_, i_] = Kp_true[:, i_]
    K2_mine = push(Tmine, W2)[idx, idx, :]
    del Tmine
    out["mine_step2"] = {
        "rel_vs_truth": round(float(np.sqrt(np.mean(
            (K2_mine - truth2) ** 2))) / scale, 4),
        "wall_s": round(time.time() - t0, 1)}
    print(json.dumps({"mine_step2": out["mine_step2"]}), flush=True)

    t0 = time.time()
    sys.path.insert(0, str(HERE.parent / "full_kappa3_emitter_v2_candidate"))
    import emitter as em
    res = em.emit_post_relu_kappa3(m1, C1, T1, gaussian_order=48,
                                   enforce_thread1=False)
    Ta = np.asarray(res.cumulant30, dtype=np.float64) + np.asarray(
        res.cumulant31, dtype=np.float64)
    clip = res.diagnostics.get("orthant3_frechet_clip_max")
    K2_a2f = push(Ta, W2)[idx, idx, :]
    del Ta
    out["a2f_step2"] = {
        "rel_vs_truth": round(float(np.sqrt(np.mean(
            (K2_a2f - truth2) ** 2))) / scale, 4),
        "frechet_clip_max": clip,
        "wall_s": round(time.time() - t0, 1)}
    out["mine_vs_a2f"] = round(float(np.sqrt(np.mean(
        (K2_mine - K2_a2f) ** 2))) / scale, 4)
    print(json.dumps(out, indent=1))
    (HERE / "THREE_WAY_L2.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
