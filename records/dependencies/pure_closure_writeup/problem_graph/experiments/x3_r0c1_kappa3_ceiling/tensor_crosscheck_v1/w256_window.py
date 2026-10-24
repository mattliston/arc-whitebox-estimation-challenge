#!/usr/bin/env python3
"""TAG r0c1-crosscheck-recursion-v1: width-256 depth-32 port of the validated
full-cum3 tensor recursion (g19 v2b physics: F3 = E[f'f'f'] rho-series
transport, CONNECTED plane-feed E[f''f'f], Gaussian Kibble completion,
Mehler emission on plane slices, exact diag; exact congruence push).
Self-contained state recursion: mean and covariance are ALSO propagated by
the closed Gaussian+Edgeworth pair map (no MC inputs), so the final-layer
post-mean is a genuine sampling-free prediction. Per-layer diagnostics vs
the MC truth file. Independent implementation vs the A2f engine (different
derivation: Kibble series vs orthant quadrature) - agreement is evidence
for both. Usage: tensor_recursion_w256.py <screen_seed> [arm]
arm in {full, gauss, oracle}  (oracle = true mean/cov/univs fed per layer;
isolates the tensor map from state-recursion drift)"""
import json
import sys
import time
from math import erf, factorial
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtri

_erf = np.vectorize(erf)
HERE = Path(__file__).resolve().parent
WIDTH, DEPTH = 256, 32
DTOT = int(__import__('os').environ.get('DTOT', 6))
TMAX_PAIR = 24


def generate_weights(seed: int) -> np.ndarray:
    uniforms = np.random.Generator(np.random.Philox(key=int(seed))).random(
        (DEPTH, WIDTH, WIDTH), dtype=np.float64
    )
    return (
        ndtri(np.maximum(uniforms, 2.0 ** -54)) * np.sqrt(2.0 / WIDTH)
    ).astype(np.float32)


def relu_columns(m, v, qmax):
    v = np.maximum(v, 1e-30)
    sd = np.sqrt(v)
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
    ma = -al
    He = [np.ones_like(al), ma]
    for k in range(2, qmax + 2):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    c1 = [sd * pdf + m * cdf, sd * cdf]
    for q in range(2, qmax + 1):
        c1.append(sd * He[q - 2] * pdf)
    raw1 = c1[0]
    raw2 = (m * m + v) * cdf + m * sd * pdf
    c2 = [raw2, 2 * sd * raw1, 2 * v * cdf]
    for q in range(3, qmax + 1):
        c2.append(2 * v * He[q - 3] * pdf)
    b = [cdf] + [He[t - 1] * pdf for t in range(1, qmax + 1)]
    d = [He[t] * pdf / sd for t in range(0, qmax + 1)]
    raw3 = (m ** 3 + 3 * m * v) * cdf + (m * m + 2 * v) * sd * pdf
    c3 = [raw3, 3 * sd * raw2, 6 * v * raw1, 6 * v * sd * cdf]
    for q in range(4, qmax + 1):
        c3.append(6 * v * sd * He[q - 4] * pdf)
    return (np.stack(c1), np.stack(c2), np.stack(b), np.stack(d),
            sd, pdf, cdf, He, np.stack(c3))


def edg2_relu_mean(m, v, k3, k4):
    sd = np.sqrt(np.maximum(v, 1e-30))
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
    ma = -al
    he1 = ma
    he2 = ma * ma - 1
    he4 = ma ** 4 - 6 * ma * ma + 3
    g3 = k3 / sd ** 3
    g4 = k4 / sd ** 4
    return (sd * pdf + m * cdf + (g3 / 6) * (sd * he1 * pdf)
            + (g4 / 24) * (sd * he2 * pdf)
            + (g3 * g3 / 72) * (sd * he4 * pdf))


def fourth_emission(m, v, k3, k4, C, K21, K22c, K31c):
    """Post-ReLU fourth-order planes (k4diag, K22, K31), connected, from
    the pre state via M22 (c2xc2) and M31 (c3xc1) Mehler maps."""
    qmax = TMAX_PAIR + 8
    c1, c2, b, d, sd, pdf, cdf, He, c3 = relu_columns(m, v, qmax)
    rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)

    def series(colsA, colsB, a, bb, tmax=TMAX_PAIR):
        acc = np.zeros_like(rho); rp = np.ones_like(rho)
        for t in range(tmax + 1):
            if t > 0:
                rp = rp * rho
            acc += rp / float(factorial(t)) * np.outer(
                colsA[a + t] / sd ** a, colsB[bb + t] / sd ** bb)
        return acc

    def dM(colsA, colsB):
        return ((k3[:, None] / 6.0) * series(colsA, colsB, 3, 0)
                + (k3[None, :] / 6.0) * series(colsA, colsB, 0, 3)
                + (K21 / 2.0) * series(colsA, colsB, 2, 1)
                + (K21.T / 2.0) * series(colsA, colsB, 1, 2)
                + (k4[:, None] / 24.0) * series(colsA, colsB, 4, 0)
                + (k4[None, :] / 24.0) * series(colsA, colsB, 0, 4)
                + (K22c / 4.0) * series(colsA, colsB, 2, 2)
                + (K31c / 6.0) * series(colsA, colsB, 3, 1)
                + (K31c.T / 6.0) * series(colsA, colsB, 1, 3))

    M11 = series(c1, c1, 0, 0) + dM(c1, c1)
    M21 = series(c2, c1, 0, 0) + dM(c2, c1)
    M22 = series(c2, c2, 0, 0) + dM(c2, c2)
    M31 = series(c3, c1, 0, 0) + dM(c3, c1)
    r1 = edg2_relu_mean(m, v, k3, k4)
    al = m / sd; ma = -al
    he1, he3 = ma, ma ** 3 - 3 * ma
    g3v, g4v = k3 / sd ** 3, k4 / sd ** 4
    r2 = ((m * m + v) * cdf + m * sd * pdf + (g3v / 6) * (2 * v * pdf)
          + (g4v / 24) * (2 * v * he1 * pdf)
          + (g3v * g3v / 72) * (2 * v * he3 * pdf))
    r3 = c3[0]
    # centrals from raws (pairwise)
    mu11 = M11 - np.outer(r1, r1)
    mu22 = (M22 - 2 * r1[None, :] * M21 - 2 * r1[:, None] * M21.T
            + np.outer(r1 ** 2, r2) + np.outer(r2, r1 ** 2)
            + 4 * np.outer(r1, r1) * M11
            - 3 * np.outer(r1 ** 2, r1 ** 2))
    mu31 = (M31 - r1[None, :] * r3[:, None]
            - 3 * r1[:, None] * M21
            + 3 * np.outer(r1, r1) * r2[:, None]
            + 3 * (r1 ** 2)[:, None] * M11
            - 3 * np.outer(r1 ** 3, r1))
    vpost = np.maximum(r2 - r1 * r1, 1e-30)
    K22post = mu22 - np.outer(vpost, vpost) - 2 * mu11 ** 2
    K31post = mu31 - 3 * vpost[:, None] * mu11
    k4post = np.diag(K22post).copy()
    np.fill_diagonal(K22post, 0.0)
    np.fill_diagonal(K31post, 0.0)
    return k4post, K22post, K31post


def fourth_push(k4post, K22post, K31post, Wl):
    """Plane-supported congruence of the fourth-order planes.
    K22next[a,b] = kappa4next[aabb], K31next[a,b] = kappa4next[aaab],
    from the post plane content {k4 diag, K22, K31}; all-distinct
    fourth-order content dropped (plane-supported compression)."""
    W2 = Wl * Wl
    W3 = W2 * Wl
    k4n = np.einsum("ia,ia,i->a", W2, W2, k4post, optimize=True)
    P_ab = np.einsum("ia,ib->iab", Wl, Wl)
    K22n = (np.einsum("ia,ib,i->ab", W2, W2, k4post, optimize=True)
            + W2.T @ K22post @ W2
            + 4.0 * np.einsum("iab,jab,ij->ab", P_ab, P_ab, K22post,
                              optimize=True)
            + 2.0 * np.einsum("ia,ib,jb,ij->ab", W2, Wl, Wl, K31post,
                              optimize=True)
            + 2.0 * np.einsum("ia,ja,ib,ij->ab", Wl, Wl, W2, K31post,
                              optimize=True))
    del P_ab
    K31n = (np.einsum("ia,ib,i->ab", W3, Wl, k4post, optimize=True)
            + W3.T @ K31post @ Wl
            + 3.0 * np.einsum("ia,ja,ib,ij->ab", W2, Wl, Wl, K31post,
                              optimize=True)
            + 3.0 * np.einsum("ia,ja,jb,ij->ab", W2, Wl, Wl, K22post,
                              optimize=True))
    # k4 diagonal also gains the K22/K31 plane pushes (diag of K22n form)
    k4n = (k4n
           + 3.0 * np.einsum("ia,ja,ij->a", W2, W2, K22post, optimize=True)
           + 4.0 * np.einsum("ia,ja,ij->a", W3, Wl, K31post, optimize=True))
    np.fill_diagonal(K22n, 0.0)
    np.fill_diagonal(K31n, 0.0)
    return k4n, K22n, K31n


def pair_map(m, v, k3, k4, C, K21, K22c=None, K31c=None):
    """Mehler pair map: post mean and post covariance (+post K21 planes)
    from pre state. Channels: k3 (3,0)/(0,3), K21 (2,1)/(1,2), k4 (4,0)/(0,4).
    """
    qmax = TMAX_PAIR + 8
    c1, c2, b, d, sd, pdf, cdf, He, c3 = relu_columns(m, v, qmax)
    rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)

    def series(colsA, colsB, a, bb, tmax=TMAX_PAIR):
        acc = np.zeros_like(rho)
        rp = np.ones_like(rho)
        for t in range(tmax + 1):
            if t > 0:
                rp = rp * rho
            acc += rp / float(factorial(t)) * np.outer(colsA[a + t] / sd ** a,
                                                colsB[bb + t] / sd ** bb)
        return acc

    def dM(colsA, colsB):
        out_ = ((k3[:, None] / 6.0) * series(colsA, colsB, 3, 0)
                + (k3[None, :] / 6.0) * series(colsA, colsB, 0, 3)
                + (K21 / 2.0) * series(colsA, colsB, 2, 1)
                + (K21.T / 2.0) * series(colsA, colsB, 1, 2)
                + (k4[:, None] / 24.0) * series(colsA, colsB, 4, 0)
                + (k4[None, :] / 24.0) * series(colsA, colsB, 0, 4))
        # second order in kappa3 (canonical Edgeworth A^2/2 channels):
        k3o = np.outer(k3, k3)
        out_ = out_ + ((k3 * k3)[:, None] / 72.0) * series(colsA, colsB, 6, 0)
        out_ = out_ + ((k3 * k3)[None, :] / 72.0) * series(colsA, colsB, 0, 6)
        out_ = out_ + (k3o / 36.0) * series(colsA, colsB, 3, 3)
        # kappa3 x K21 cross products (patterns (3,0)x(2,1) etc.)
        out_ = out_ + ((k3[:, None] * K21) / 12.0) * series(colsA, colsB, 5, 1)
        out_ = out_ + ((k3[None, :] * K21.T) / 12.0) * series(colsA, colsB, 1, 5)
        out_ = out_ + ((k3[:, None] * K21.T) / 12.0) * series(colsA, colsB, 4, 2)
        out_ = out_ + ((k3[None, :] * K21) / 12.0) * series(colsA, colsB, 2, 4)
        out_ = out_ + ((K21 * K21.T) / 4.0) * series(colsA, colsB, 3, 3)
        out_ = out_ + ((K21 * K21) / 8.0) * series(colsA, colsB, 4, 2)
        out_ = out_ + ((K21.T * K21.T) / 8.0) * series(colsA, colsB, 2, 4)
        if K22c is not None:
            out_ = out_ + (K22c / 4.0) * series(colsA, colsB, 2, 2)
        if K31c is not None:
            out_ = out_ + (K31c / 6.0) * series(colsA, colsB, 3, 1)
            out_ = out_ + (K31c.T / 6.0) * series(colsA, colsB, 1, 3)
        return out_

    M11 = series(c1, c1, 0, 0) + dM(c1, c1)
    M21 = series(c2, c1, 0, 0) + dM(c2, c1)
    r1 = edg2_relu_mean(m, v, k3, k4)
    al = m / sd
    ma = -al
    he1, he3 = ma, ma ** 3 - 3 * ma
    g3v, g4v = k3 / sd ** 3, k4 / sd ** 4
    r2 = ((m * m + v) * cdf + m * sd * pdf + (g3v / 6) * (2 * v * pdf)
          + (g4v / 24) * (2 * v * he1 * pdf)
          + (g3v * g3v / 72) * (2 * v * he3 * pdf))
    post_cov = M11 - np.outer(r1, r1)
    np.fill_diagonal(post_cov, np.maximum(r2 - r1 * r1, 1e-30))
    K21post = (M21 - 2 * r1[:, None] * M11 - r2[:, None] * r1[None, :]
               + 2 * (r1 ** 2)[:, None] * r1[None, :])
    return r1, post_cov, K21post, rho


def main():
    seed = int(sys.argv[1])
    arm = sys.argv[2] if len(sys.argv) > 2 else "full"
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Wnp = generate_weights(seed)
    mc = None
    mcp = HERE / f"mc_w256_seed{seed}_v4.npz"
    if mcp.exists():
        mc = np.load(mcp)

    mean = np.zeros(WIDTH)
    C = np.eye(WIDTH)
    k3u = np.zeros(WIDTH)
    k4u = np.zeros(WIDTH)
    k4d_post = np.zeros(WIDTH)
    K22_post = np.zeros((WIDTH, WIDTH))
    K31_post = np.zeros((WIDTH, WIDTH))
    import os as _os
    WIN = int(_os.environ.get("WINDOW", "4"))
    TC = []            # age cohorts, newest last; window-W exact carrier
    route_pending = None
    preK21 = np.zeros((WIDTH, WIDTH))
    out = {"seed": seed, "arm": arm, "layers": {}}
    t00 = time.time()
    idx = torch.arange(WIDTH, device=dev)

    for l in range(DEPTH):
        t0 = time.time()
        Wl = Wnp[l].astype(np.float64)
        pre_m = mean @ Wl
        pre_C = Wl.T @ C @ Wl
        if arm == "oracle" and mc is not None:
            pre_m = mc["pre_m"][l].astype(np.float64)
            pre_C = mc["pre_cov"][l].astype(np.float64)
        # push the carried post-tensor into pre-frame
        Wt = torch.from_numpy(Wl.astype(np.float32)).to(dev)
        if arm in ("full", "oracle"):
            for ci in range(len(TC)):
                Tc = torch.einsum("ia,ijk->ajk", Wt, TC[ci])
                Tc = torch.einsum("jb,ajk->abk", Wt, Tc)
                TC[ci] = torch.einsum("kc,abk->abc", Wt, Tc)
        if arm != "gauss" and TC:
            Tsum_pl = sum(Tc[idx, idx, :] for Tc in TC)
            preK21 = Tsum_pl.cpu().numpy().astype(np.float64)
            if route_pending is not None:
                # dropped-cohort plane image, pushed one layer by the
                # exact plane-class congruence; entering preK21 it then
                # rides the pair-map -> emission -> assembly loop forever
                RP = route_pending
                W2r = Wl * Wl
                Roff = RP.copy()
                np.fill_diagonal(Roff, 0.0)
                preK21 += (np.einsum("ia,ib,i->ab", W2r, Wl,
                                     np.diag(RP).copy(), optimize=True)
                           + W2r.T @ Roff @ Wl
                           + 2.0 * np.einsum("ia,ib,ij,ja->ab", Wl, Wl,
                                             Roff, Wl, optimize=True))
                route_pending = None
            pre_k3u = sum(Tc[idx, idx, idx]
                          for Tc in TC).cpu().numpy().astype(np.float64)
        else:
            preK21 = np.zeros((WIDTH, WIDTH))
            pre_k3u = np.zeros(WIDTH)
        # carried fourth-order planes, pushed to the pre frame
        v_pre = np.maximum(np.diag(pre_C), 1e-30)
        if arm != "gauss":
            k4u_pre, K22_pre, K31_pre = fourth_push(
                k4d_post, K22_post, K31_post, Wl)
            from k211_completion import push_k211
            if l > 0:
                k4c, K22c_, K31c_ = push_k211(
                    prev_pre_m, np.maximum(np.diag(prev_pre_C), 1e-30),
                    prev_pre_C, Wl)
            else:
                k4c = np.zeros(WIDTH)
                K22c_ = np.zeros((WIDTH, WIDTH))
                K31c_ = np.zeros((WIDTH, WIDTH))
            k4u_pre = k4u_pre + k4c
            K22_pre = K22_pre + K22c_
            K31_pre = K31_pre + K31c_
        else:
            k4u_pre = np.zeros(WIDTH)
            K22_pre = np.zeros((WIDTH, WIDTH))
            K31_pre = np.zeros((WIDTH, WIDTH))

        # diagnostics vs MC truth at the pre frame
        rec = {}
        if mc is not None:
            tv = mc["pre_v"][l]
            rec["dv_rel"] = round(float(np.sqrt(np.mean(
                (np.diag(pre_C) - tv) ** 2) / np.mean(tv ** 2))), 5)
            tK = mc["pre_K21"][l].astype(np.float64)
            sc = float(np.sqrt(np.mean(tK ** 2)))
            if sc > 0 and arm != "gauss":
                rec["K21_rel"] = round(float(np.sqrt(np.mean(
                    (preK21 - tK) ** 2))) / sc, 4)

        if mc is not None and arm != "gauss":
            tk3 = mc["pre_k3"][l].astype(np.float64)
            tk4 = mc["pre_k4"][l].astype(np.float64)
            s3 = float(np.sqrt(np.mean(tk3 ** 2)))
            s4 = float(np.sqrt(np.mean(tk4 ** 2)))
            if s3 > 0:
                rec["k3u_rel"] = round(float(np.sqrt(np.mean(
                    (pre_k3u - tk3) ** 2))) / s3, 3)
            if s4 > 0:
                rec["k4u_rel"] = round(float(np.sqrt(np.mean(
                    (k4u_pre - tk4) ** 2))) / s4, 3)
        # diagnostics for the carried 4th planes vs truth (layers 1-2)
        if mc is not None and arm != "gauss" and "pre_K22" in getattr(
                mc, "files", []) and l <= 2:
            tK22 = mc["pre_K22"][l].astype(np.float64)
            s22 = float(np.sqrt(np.mean(tK22 ** 2)))
            if s22 > 0:
                rec["K22_rel"] = round(float(np.sqrt(np.mean(
                    (K22_pre - tK22) ** 2))) / s22, 4)
        # ReLU map: mean/cov/planes via the pair map (full channels)
        r1, post_C, K21post, rho = pair_map(
            pre_m, np.diag(pre_C), pre_k3u, k4u_pre, pre_C, preK21,
            K22c=K22_pre if arm != "gauss" else None,
            K31c=K31_pre if arm != "gauss" else None)
        if mc is not None:
            pm = mc["post_mean"][l]
            se = mc["post_se"][l]
            err = float(np.sum((r1 - pm) ** 2))
            noise = float(np.sum(se ** 2))
            rec["post_mean_mse_nc"] = (err - noise) / WIDTH
        rho_off = rho - np.diag(np.diag(rho))
        rec["rho_absmax"] = round(float(np.abs(rho_off).max()), 3)

        if arm in ("full", "oracle"):
            # tensor ReLU map (g19 v2b physics) on the distinct entries
            m_, v_ = pre_m, np.maximum(np.diag(pre_C), 1e-30)
            qmax = 2 * DTOT + 2
            c1, c2, bcols, dcols, sd, pdf, cdf, He, c3cols = relu_columns(m_, v_, qmax)
            at = torch.from_numpy(c1[:2 * DTOT + 1].astype(np.float32)).to(dev)
            bt = torch.from_numpy(bcols[:DTOT + 3].astype(np.float32)).to(dev)
            dt_ = torch.from_numpy(dcols[:DTOT + 3].astype(np.float32)).to(dev)
            rt = torch.from_numpy(np.clip(
                pre_C / np.outer(sd, sd), -0.9999, 0.9999
            ).astype(np.float32)).to(dev)
            rt.fill_diagonal_(0.0)
            rpow = {t: rt ** t for t in range(1, DTOT + 1)}
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
            preKt = torch.from_numpy(preK21.astype(np.float32)).to(dev)
            F3 = torch.zeros((WIDTH, WIDTH, WIDTH), dtype=torch.float32,
                             device=dev)
            for (tij, tik, tjk, di, dj, dk, coef) in allterms:
                t_ = (bt[di][:, None, None] * bt[dj][None, :, None]
                      * bt[dk][None, None, :]) * coef
                if tij:
                    t_ = t_ * rpow[tij][:, :, None]
                if tik:
                    t_ = t_ * rpow[tik][:, None, :]
                if tjk:
                    t_ = t_ * rpow[tjk][None, :, :]
                F3 += t_
            for ci in range(len(TC)):
                TC[ci] = TC[ci] * F3          # exact transport per cohort
            del F3
            blk = torch.zeros((WIDTH, WIDTH, WIDTH), dtype=torch.float32,
                              device=dev)     # the FRESH cohort
            pats = [
                (preKt[:, :, None] * 1.0, 0, 1, 2),
                (preKt[:, None, :] * 1.0, 0, 2, 1),
                (preKt.T[:, :, None] * 1.0, 1, 0, 2),
                (preKt[None, :, :] * 1.0, 1, 2, 0),
                (preKt.T[:, None, :] * 1.0, 2, 0, 1),
                (preKt.T[None, :, :] * 1.0, 2, 1, 0),
            ]

            def cshape(cols, t, pos):
                v2 = cols[t]
                if pos == 0:
                    return v2[:, None, None]
                if pos == 1:
                    return v2[None, :, None]
                return v2[None, None, :]

            def rshape(t, p1, p2):
                if {p1, p2} == {0, 1}:
                    return rpow[t][:, :, None]
                if {p1, p2} == {0, 2}:
                    return rpow[t][:, None, :]
                return rpow[t][None, :, :]

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
                            t_ = (cshape(dt_, du, up) * cshape(bt, dw, wp)
                                  * cshape(at, ds, sp)) * cf
                            if tuw:
                                t_ = t_ * rshape(tuw, up, wp)
                            if tus:
                                t_ = t_ * rshape(tus, up, sp)
                            if tws:
                                t_ = t_ * rshape(tws, wp, sp)
                            acc = t_ if acc is None else acc + t_
                blk = blk + Kb * acc
                del acc
            for (tij, tik, tjk, di, dj, dk, coef) in terms:
                t_ = (at[di][:, None, None] * at[dj][None, :, None]
                      * at[dk][None, None, :]) * coef
                if tij:
                    t_ = t_ * rpow[tij][:, :, None]
                if tik:
                    t_ = t_ * rpow[tik][:, None, :]
                if tjk:
                    t_ = t_ * rpow[tjk][None, :, :]
                blk = blk + t_
            import os as _os
            if _os.environ.get("W256_SO2") == "1":
                # distinct-tensor SECOND-ORDER pattern pairs (kappa3^2
                # class; scalar framework second_order_triple.py, this
                # broadcast form crosschecked exactly against it)
                from second_order_triple import second_order_terms
                ma_ = -(m_ / sd)
                Hl = [np.ones_like(ma_), ma_]
                for kk_ in range(2, 16):
                    Hl.append(ma_ * Hl[-1] - (kk_ - 1) * Hl[-2])
                fams = {}

                def famn(n):
                    if n not in fams:
                        if n == 0:
                            base = ([sd * pdf + m_ * cdf, sd * cdf]
                                    + [sd * Hl[q - 2] * pdf
                                       for q in range(2, 12)])
                            fams[n] = np.stack(base)
                        elif n == 1:
                            fams[n] = np.stack(
                                [cdf] + [Hl[t - 1] * pdf
                                         for t in range(1, 12)])
                        else:
                            fams[n] = np.stack(
                                [Hl[(n - 2) + t] * pdf / sd ** (n - 1)
                                 for t in range(0, 12)])
                    return fams[n]

                k3t_v = torch.from_numpy(
                    pre_k3u.astype(np.float32)).to(dev)
                KMt = preKt.T.contiguous()

                def wfac(key):
                    return {
                        "k3_0": k3t_v[:, None, None],
                        "k3_1": k3t_v[None, :, None],
                        "k3_2": k3t_v[None, None, :],
                        "K21_01": preKt[:, :, None],
                        "K21_10": KMt[:, :, None],
                        "K21_02": preKt[:, None, :],
                        "K21_20": KMt[:, None, :],
                        "K21_12": preKt[None, :, :],
                        "K21_21": KMt[None, :, :]}.get(key)

                def colt(n, deg, pos):
                    arr = torch.from_numpy(
                        famn(n)[deg].astype(np.float32)).to(dev)
                    if pos == 0:
                        return arr[:, None, None]
                    if pos == 1:
                        return arr[None, :, None]
                    return arr[None, None, :]

                for (k1_, k2_, (t01, t02, t12), (n0, n1, n2), cf2) in \
                        second_order_terms(bond_cap=3):
                    W1, W2 = wfac(k1_), wfac(k2_)
                    if W1 is None or W2 is None:
                        continue        # T-pattern pairs out of scope
                    t_ = (cf2 * W1 * W2 * colt(n0, t01 + t02, 0)
                          * colt(n1, t01 + t12, 1)
                          * colt(n2, t02 + t12, 2))
                    if t01:
                        t_ = t_ * rpow[t01][:, :, None]
                    if t02:
                        t_ = t_ * rpow[t02][:, None, :]
                    if t12:
                        t_ = t_ * rpow[t12][None, :, :]
                    blk = blk + t_
                    del t_
            TC.append(blk)
            if len(TC) > WIN:
                dropped = TC.pop(0)
                if _os.environ.get("ROUTE_DROPPED") == "1":
                    route_pending = dropped[idx, idx, :].cpu().numpy() \
                        .astype(np.float64)
                del dropped
            # plane and diagonal slices: zero in all cohorts, set on the
            # NEWEST so the cohort SUM carries exactly the pair-map plane
            Kp = torch.from_numpy(K21post.astype(np.float32)).to(dev)
            for Tc in TC[:-1]:
                Tc[idx, idx, :] = 0.0
                Tc[idx, :, idx] = 0.0
                Tc[:, idx, idx] = 0.0
            TC[-1][idx, idx, :] = Kp
            TC[-1][idx, :, idx] = Kp
            TC[-1][:, idx, idx] = Kp.T
            al = pre_m / sd
            pdf_ = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
            cdf_ = 0.5 * (1 + _erf(al / np.sqrt(2)))
            r1g = sd * pdf_ + pre_m * cdf_
            r2g = (pre_m ** 2 + v_) * cdf_ + pre_m * sd * pdf_
            r3g = ((pre_m ** 3 + 3 * pre_m * v_) * cdf_
                   + (pre_m ** 2 + 2 * v_) * sd * pdf_)
            k3post = r3g - 3 * r2g * r1g + 2 * r1g ** 3
            TC[-1][idx, idx, idx] = torch.from_numpy(
                k3post.astype(np.float32)).to(dev)

        prev_pre_m, prev_pre_C = pre_m, pre_C
        if arm != "gauss":
            k4d_post, K22_post, K31_post = fourth_emission(
                pre_m, np.maximum(np.diag(pre_C), 1e-30), pre_k3u,
                k4u_pre, pre_C, preK21, K22_pre, K31_pre)
        mean, C = r1, post_C
        rec["wall_s"] = round(time.time() - t0, 2)
        out["layers"][str(l)] = rec
        print(json.dumps({f"l{l}": rec}), flush=True)

    if mc is not None:
        pm = mc["post_mean"][DEPTH - 1]
        se = mc["post_se"][DEPTH - 1]
        err = float(np.sum((mean - pm) ** 2))
        noise = float(np.sum(se ** 2))
        out["final_post_mean_mse_nc"] = (err - noise) / WIDTH
        out["final_noise_mse"] = noise / WIDTH
        print(json.dumps({"FINAL": {"mse_nc": out["final_post_mean_mse_nc"],
                                    "noise": out["final_noise_mse"]}}))
    out["total_wall_s"] = round(time.time() - t00, 1)
    (HERE / f"RECURSION_W256WIN_seed{seed}_{arm}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
