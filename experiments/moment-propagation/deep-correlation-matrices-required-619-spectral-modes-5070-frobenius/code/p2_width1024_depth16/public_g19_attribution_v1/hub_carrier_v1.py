#!/usr/bin/env python3
"""TAG g19g-hub-carrier-v1 (PHASE 2, 1024x16): billable distinct-content
carrier with MEASUREMENT-SPECIFIED upgrades over v0:
  (1) t<=1 Mehler TRANSPORT at each ReLU (v0's g-Hadamard = t=0 only;
      cap-8 wedge showed old content rots under it): the one-rho-bond
      terms are added by splitting rho into its top-KRHO spectral modes,
      each carried triad spawning 1 + 3*KRHO triads:
        t0:      (g.u) x (g.x) x (g.y)
        bond ij: sum_r lam_r (pdf.s_r.u) x (pdf.s_r.x) x (g.y)
        bond ik: sum_r lam_r (pdf.s_r.u) x (g.x) x (pdf.s_r.y)
        bond jk: sum_r (g.u) x (lam_r pdf.s_r.x) x (pdf.s_r.y)
      (b[1] = He_0(-alpha) pdf = pdf; lam folded into one factor)
  (2) per-layer CP RECOMPRESSION to RANK columns by warm-started ALS on
      the implicit tensor (Gram-matrix algebra only, never 1024^3);
      truth distinct content is rank 1-6, so low-loss.
  Birth content, collision subtraction, plane channels = v0 (CAP=1
  wedge showed birth quality matches v4 fresh-only).
Usage: hub_carrier_v1.py <rank> [Lmax]   (RANK, KRHO, SWEEPS via env)"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hub_carrier_v0 import (STORE, W_, PQ_KEEP, state_cols,
                            collision_tensor_cols, _centers)
from math import factorial

RANK = int(os.environ.get("HUB_RANK", "32"))
KRHO = int(os.environ.get("HUB_KRHO", "4"))
SWEEPS = int(os.environ.get("HUB_SWEEPS", "3"))


def fresh_columns(m, v, C, k3u=None, k4u=None):
    """v0 fresh_source's triads as stacked column matrices (N x H)."""
    a, g, sd = state_cols(m, v)
    rho = C / np.outer(sd, sd)
    np.fill_diagonal(rho, 0.0)
    Us, Xs, Ys = [], [], []
    for (p, q) in PQ_KEEP:
        coef = 1.0 / (factorial(p) * factorial(q))
        rp = rho ** p
        rq = rho ** q if q != p else rp
        cen = coef * _centers(a, sd, p, q, k3u, k4u)
        U = np.zeros((W_, W_))
        np.fill_diagonal(U, cen)
        X = (rp * a[p][None, :]).T
        Y = (rq * a[q][None, :]).T
        for (uu, xx, yy) in ((U, X, Y), (X, U, Y), (X, Y, U)):
            Us.append(uu)
            Xs.append(xx)
            Ys.append(yy)
    return (np.concatenate(Us, 1), np.concatenate(Xs, 1),
            np.concatenate(Ys, 1))


def cp_compress(U, X, Y, R, sweeps=SWEEPS):
    """ALS rank-R CP of sum_h u_h (x) x_h (x) y_h. Init: top-R triads by
    norm. All algebra via H x R Grams; never materializes the tensor."""
    H = U.shape[1]
    if H <= R:
        return U, X, Y
    nrm = (np.linalg.norm(U, axis=0) * np.linalg.norm(X, axis=0)
           * np.linalg.norm(Y, axis=0))
    top = np.argsort(-nrm)[:R]
    Ut, Xt, Yt = U[:, top].copy(), X[:, top].copy(), Y[:, top].copy()
    for _ in range(sweeps):
        for mode in range(3):
            A, B = ((Xt, Yt), (Ut, Yt), (Ut, Xt))[mode]
            Af, Bf = ((X, Y), (U, Y), (U, X))[mode]
            T = (U, X, Y)[mode]
            G = (A.T @ A) * (B.T @ B)                      # R x R
            M = T @ ((Af.T @ A) * (Bf.T @ B))              # N x R
            sol = np.linalg.lstsq(G + 1e-12 * np.eye(R), M.T,
                                  rcond=None)[0].T
            if mode == 0:
                Ut = sol
            elif mode == 1:
                Xt = sol
            else:
                Yt = sol
    return Ut, Xt, Yt


def transport_columns(U, X, Y, g, pdf, rho, krho):
    """t<=1 Mehler transport of the carrier at this ReLU."""
    w, S = np.linalg.eigh(rho)
    ordr = np.argsort(-np.abs(w))[:krho]
    lam, sig = w[ordr], S[:, ordr]
    gU, gX, gY = g[:, None] * U, g[:, None] * X, g[:, None] * Y
    Us, Xs, Ys = [gU], [gX], [gY]
    for r in range(len(lam)):
        sr = pdf * sig[:, r]
        srl = lam[r] * sr
        # bond ij (slots u,x), ik (u,y), jk (x,y)
        Us += [srl[:, None] * U, srl[:, None] * U, gU]
        Xs += [sr[:, None] * X, gX, srl[:, None] * X]
        Ys += [gY, sr[:, None] * Y, sr[:, None] * Y]
    return (np.concatenate(Us, 1), np.concatenate(Xs, 1),
            np.concatenate(Ys, 1))


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

    Uc = Xc = Yc = None                 # the carrier (N x <=RANK)
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
        # push carrier to this layer's pre frame
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
                __import__("torch").device(
                    "mps" if __import__("torch").backends.mps.is_available()
                    else "cpu"),
                k4d_post, K22_post, K31_post, Wl)
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
        # ReLU: transport carrier (t<=1 Mehler), add fresh, recompress
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
        Uc, Xc, Yc = cp_compress(Uc, Xc, Yc, RANK)
        # plane emission with carried fourth-order channels (as v0)
        planeK21 = emission_K21(m, v, k3u, k4u, C, preK21, K22c, K31c)
        k4d_post, K22_post, K31_post = fourth_emission_1024(
            m, v, k3u, k4u, C, preK21, K22c, K31c)
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["cols"] = int(Uc.shape[1])
        out["layers"][str(l)] = rec
        print(json.dumps({f"l{l}": rec}), flush=True)
    (HERE / f"HUB_CARRIER_V1_rank{rank:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")
    np.savez(HERE / f"hub_planes_rank{rank:04d}_v1.npz", **PLSAVE)


if __name__ == "__main__":
    main()
