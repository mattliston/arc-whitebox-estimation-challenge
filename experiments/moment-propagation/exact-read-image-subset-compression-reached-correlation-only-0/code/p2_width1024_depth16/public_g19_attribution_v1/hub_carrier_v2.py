#!/usr/bin/env python3
"""TAG g19g-hub-carrier-v2 (PHASE 2, 1024x16): the numbers-closed design
from the 08-30 campaign (graph da2dc03a):
  - t<=1 Mehler transport with KRHO~16 rho-spectral modes (6-13 modes =
    50-70% of deep ||rho||_F; the KRHO=4 null was ~30% coverage)
  - NO hard source cap: population controlled by PLANE-READ-AWARE
    compression to RANK columns - objective = the consumed functionals
    (the K21 plane read now and after the next FUT actual W-pushes),
    via sketched orthogonal matching pursuit over column read-images +
    joint coefficient refit (Frobenius-ALS refuted: keeps wrong columns)
  - birth content, collision subtraction, plane channels = v0
Target: q 0.62 -> 0.75+ (the measured chain conversion threshold), at
e10-class deploy billing. Usage: hub_carrier_v2.py <rank> [Lmax]"""
import json
import os
import sys
import time
from math import factorial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hub_carrier_v0 import (STORE, W_, state_cols, collision_tensor_cols,
                            _centers)

RANK = int(os.environ.get("HUB_RANK", "96"))
KRHO = int(os.environ.get("HUB_KRHO", "16"))
FUT = int(os.environ.get("HUB_FUT", "2"))
SKD = int(os.environ.get("HUB_SKD", "4096"))
PQ11 = [(1, 1)]


def fresh_columns(m, v, C):
    a, g, sd = state_cols(m, v)
    rho = C / np.outer(sd, sd)
    np.fill_diagonal(rho, 0.0)
    Us, Xs, Ys = [], [], []
    for (p, q) in PQ11:
        coef = 1.0 / (factorial(p) * factorial(q))
        rp = rho ** p
        cen = coef * _centers(a, sd, p, q)
        U = np.zeros((W_, W_))
        np.fill_diagonal(U, cen)
        X = (rp * a[p][None, :]).T
        Y = (rp * a[q][None, :]).T
        for (uu, xx, yy) in ((U, X, Y), (X, U, Y), (X, Y, U)):
            Us.append(uu)
            Xs.append(xx)
            Ys.append(yy)
    return (np.concatenate(Us, 1), np.concatenate(Xs, 1),
            np.concatenate(Ys, 1))


def transport_columns(U, X, Y, g, pdf, rho, krho):
    w, S = np.linalg.eigh(rho)
    ordr = np.argsort(-np.abs(w))[:krho]
    lam, sig = w[ordr], S[:, ordr]
    gU, gX, gY = g[:, None] * U, g[:, None] * X, g[:, None] * Y
    Us, Xs, Ys = [gU], [gX], [gY]
    for r in range(len(lam)):
        sr = pdf * sig[:, r]
        srl = lam[r] * sr
        Us += [srl[:, None] * U, srl[:, None] * U, gU]
        Xs += [sr[:, None] * X, gX, srl[:, None] * X]
        Ys += [gY, sr[:, None] * Y, sr[:, None] * Y]
    return (np.concatenate(Us, 1), np.concatenate(Xs, 1),
            np.concatenate(Ys, 1))


def read_sketch(U, X, Y, Wfut, rng_proj):
    """Sketch of each column's consumed read-images: for k=0..FUT the
    plane read after k pushes; JL-projected to SKD dims per k.
    Returns PHI (H x FUT+1 x SKD) as one (H, D) matrix."""
    H = U.shape[1]
    outs = []
    Uk, Xk, Yk = U, X, Y
    for k in range(FUT + 1):
        A = Uk * Xk                                   # N x H
        # column read-image = outer(A[:,h], Yk[:,h]); sketch:
        # phi = (P1^T A[:,h]) kron-ish (P2^T Yk[:,h]) with small P1,P2
        P1, P2 = rng_proj[k]
        S1 = P1.T @ A                                 # d1 x H
        S2 = P2.T @ Yk                                # d2 x H
        outs.append((S1[:, None, :] * S2[None, :, :]).reshape(-1, H))
        if k < FUT:
            Wk = Wfut[k]
            Uk, Xk, Yk = Wk.T @ Uk, Wk.T @ Xk, Wk.T @ Yk
    return np.concatenate(outs, 0).T                  # H x D


def read_compress(U, X, Y, R, Wfut, seed=0):
    """EXACT-read subset compression: read-image inner products in
    closed form via factor Grams (no sketching): for pushed stage k,
    <R_h, R_g>_k = ((Uk*Xk)^T (Uk*Xk))[h,g] * (Yk^T Yk)[h,g]."""
    H = U.shape[1]
    if H <= R:
        return U, X, Y
    G = np.zeros((H, H))
    b = np.zeros(H)
    Uk, Xk, Yk = U, X, Y
    for k in range(FUT + 1):
        A = Uk * Xk
        GA = A.T @ A
        GY = Yk.T @ Yk
        Gk = GA * GY
        G += Gk
        b += Gk.sum(1)                 # <R_h, sum_g R_g>
        if k < FUT:
            Wk = Wfut[k]
            Uk, Xk, Yk = Wk.T @ Uk, Wk.T @ Xk, Wk.T @ Yk
    # greedy selection maximizing exact read-energy capture
    sel = []
    nrm = np.sqrt(np.diag(G)) + 1e-300
    resid_b = b.copy()
    for _ in range(R):
        scores = np.abs(resid_b) / nrm
        if sel:
            scores[sel] = -1
        j = int(np.argmax(scores))
        sel.append(j)
        Gs = G[np.ix_(sel, sel)]
        c = np.linalg.lstsq(Gs + 1e-10 * np.eye(len(sel)),
                            b[sel], rcond=None)[0]
        resid_b = b - G[:, sel] @ c
    c = np.linalg.lstsq(G[np.ix_(sel, sel)] + 1e-10 * np.eye(R),
                        b[sel], rcond=None)[0]
    Us = U[:, sel] * c[None, :]
    return Us, X[:, sel].copy(), Y[:, sel].copy()


def main():
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    T = np.load(HERE / f"mps_cov2_rank{rank:04d}.npz")
    Pl = np.load(HERE / f"mps_planes_rank{rank:04d}.npz")
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    weights = np.load((STORE / entry["path"]).parent / "weights.npy",
                      mmap_mode="r")
    from run_tensor_recursion_v2b import emission_K21
    from run_tensor_recursion_v4 import fourth_emission_1024, fourth_push_1024
    import torch as _th
    dev = _th.device("mps" if _th.backends.mps.is_available() else "cpu")

    Uc = Xc = Yc = None
    pending_colZ = None
    PLSAVE = {}
    planeK21 = np.zeros((W_, W_))
    k4d_post = np.zeros(W_)
    K22_post = np.zeros((W_, W_))
    K31_post = np.zeros((W_, W_))
    out = {"rank": rank, "layers": {}}
    for l in range(0, lmax):
        t0 = time.time()
        if l == 0:
            W0 = np.asarray(weights[0], dtype=np.float64)
            C = W0.T @ W0
            m = np.zeros(W_)
            v = np.diag(C).copy()
            k3u = np.zeros(W_)
            k4u = np.zeros(W_)
            K22c = np.zeros((W_, W_))
            K31c = np.zeros((W_, W_))
        else:
            m = T["mc_pre_m"][l].astype(np.float64)
            v = T["mc_pre_v"][l].astype(np.float64)
            C = T["mc_pre_cov"][l].astype(np.float64)
        Wl = np.asarray(weights[l], dtype=np.float64)
        if Uc is not None:
            Uc, Xc, Yc = Wl.T @ Uc, Wl.T @ Xc, Wl.T @ Yc
        W2 = Wl * Wl
        Koff = planeK21.copy()
        np.fill_diagonal(Koff, 0.0)
        k3post_prev = np.diag(planeK21).copy()
        preK21 = (np.einsum("ia,ib,i->ab", W2, Wl, k3post_prev,
                            optimize=True)
                  + W2.T @ Koff @ Wl
                  + 2.0 * np.einsum("ia,ib,ij,ja->ab", Wl, Wl, Koff, Wl,
                                    optimize=True))
        if Uc is not None:
            preK21 += (Uc * Xc) @ Yc.T
        if pending_colZ is not None:
            A = np.einsum("ij,ia,ja,jb->ab", pending_colZ, Wl, Wl, Wl,
                          optimize=True)
            B = np.einsum("ij,ja,ib->ab", pending_colZ, Wl * Wl, Wl,
                          optimize=True)
            preK21 -= (2.0 * A + B)
            pending_colZ = None
        if l > 0:
            k3u = np.diag(preK21).copy()
            k4u, K22c, K31c = fourth_push_1024(
                dev, k4d_post, K22_post, K31_post, Wl)
            PLSAVE[f"pre_{l}"] = preK21.astype(np.float32)
            PLSAVE[f"k3u_{l}"] = k3u.astype(np.float32)
            PLSAVE[f"k4u_{l}"] = k4u.astype(np.float32)
        rec = {}
        key = f"K21_{l}"
        if l >= 2 and key in Pl.files and l - 1 < z["k21"].shape[0]:
            truth = Pl[key].astype(np.float64)
            scale = float(np.sqrt(np.mean(truth ** 2)))
            rec["rel"] = round(float(np.sqrt(np.mean(
                (preK21 - truth) ** 2))) / scale, 4)
            carried = z["k21"][l - 1].astype(np.float64)
            rec["closure_rel"] = round(float(np.sqrt(np.mean(
                (carried - truth) ** 2))) / scale, 4)
            dK = preK21 - carried
            dT = truth - carried
            rec["q"] = round(float(np.sum(dK * dT) / np.sqrt(
                np.sum(dK ** 2) * np.sum(dT ** 2))), 3)
        a, g, sd = state_cols(m, v)
        al = m / sd
        pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
        rho = C / np.outer(sd, sd)
        np.fill_diagonal(rho, 0.0)
        if Uc is not None:
            Uc, Xc, Yc = transport_columns(Uc, Xc, Yc, g, pdf, rho, KRHO)
        Uf, Xf, Yf = fresh_columns(m, v, C)
        pending_colZ = collision_tensor_cols(m, v, C)
        if Uc is None:
            Uc, Xc, Yc = Uf, Xf, Yf
        else:
            Uc = np.concatenate([Uc, Uf], 1)
            Xc = np.concatenate([Xc, Xf], 1)
            Yc = np.concatenate([Yc, Yf], 1)
        Wfut = [np.asarray(weights[min(l + 1 + k, 15)], dtype=np.float64)
                for k in range(FUT)]
        Uc, Xc, Yc = read_compress(Uc, Xc, Yc, RANK, Wfut, seed=l)
        planeK21 = emission_K21(m, v, k3u, k4u, C, preK21, K22c, K31c)
        k4d_post, K22_post, K31_post = fourth_emission_1024(
            m, v, k3u, k4u, C, preK21, K22c, K31c)
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["cols"] = int(Uc.shape[1])
        out["layers"][str(l)] = rec
        print(json.dumps({f"l{l}": rec}), flush=True)
    (HERE / f"HUB_CARRIER_V2_rank{rank:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")
    np.savez(HERE / f"hub_planes_rank{rank:04d}_v2.npz", **PLSAVE)


if __name__ == "__main__":
    main()
