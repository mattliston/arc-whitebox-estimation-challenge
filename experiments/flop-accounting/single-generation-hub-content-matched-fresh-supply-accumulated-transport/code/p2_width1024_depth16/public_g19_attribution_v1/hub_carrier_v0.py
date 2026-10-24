#!/usr/bin/env python3
"""TAG g19g-hub-carrier-v0 (PHASE 2, 1024x16): the budget-compatible
carrier for the distinct third-order content.

Construction: each layer's fresh V-emission (leading (p,q)=(1,1) class,
three orientations) is a HUB tensor sum_h c_h e_h (x) x_h (x) y_h, stored
as factor matrices (U, X, Y) with hub columns. This class is CLOSED under:
  - the layer push: U' = W^T U, X' = W^T X, Y' = W^T Y  (3 matmuls)
  - the ReLU transport (leading order): columns Hadamard-scaled by the
    gate vector g = E[f'] per factor
  - plane extraction: K21[a,b] += (U o X) Y^T + orientation permutations
Old sources decay by prod(g^3) and are DROPPED beyond SOURCE_CAP layers.
Fresh appends: center-i / center-j / center-k orientations of
  coef * a[p+q]_i * rho^p_ij a[p]_j * rho^q_ik a[q]_k   (p=q=1)
Per-layer cost is matmul-grade (~1e11 flops at cap 4) - deployable class.

v0 measures: carried pre-K21 vs truth per layer, and q vs the closure,
against the v4 full-tensor reference. Validation frame: true mean/cov per
layer (as v4). Usage: hub_carrier_v0.py <rank> [Lmax]"""
import json
import sys
from math import erf, factorial
from pathlib import Path

import numpy as np

_erf = np.vectorize(erf)
HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
import os
W_ = 1024
SOURCE_CAP = int(os.environ.get("HUB_CAP", "4"))
PQ_KEEP = ([(1, 1)] if os.environ.get("HUB_MIN") == "1"
           else [(1, 1), (1, 2), (2, 1)])


def state_cols(m, v, qmax=8):
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
    return np.stack(c1), cdf, sd


import torch as _th
_DEV = _th.device("mps" if _th.backends.mps.is_available() else "cpu")


def _t(x):
    return _th.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(_DEV)


class HubSource:
    """Hub factors on MPS (f32). H = sum_h U[:,h] (x) X[:,h] (x) Y[:,h].
    With HUB_T1=1, each source records its MOST RECENT transport's
    t=1 Mehler bond pieces exactly (pre-transport factors scaled by
    pdf, the full rho at that layer, and an accumulated push matrix P)
    and adds the exact lazy bond-read at每 plane read."""

    __slots__ = ("tris", "t1")

    def __init__(self, tris):
        self.tris = [(u if _th.is_tensor(u) else _t(u),
                      x if _th.is_tensor(x) else _t(x),
                      y if _th.is_tensor(y) else _t(y), o)
                     for (u, x, y, o) in tris]
        self.t1 = None

    def push(self, WlT):
        self.tris = [(WlT @ U, WlT @ X, WlT @ Y, o)
                     for (U, X, Y, o) in self.tris]
        if self.t1 is not None:
            self.t1["P"] = WlT @ self.t1["P"]
            self.t1["Y"] = [WlT @ Yv for Yv in self.t1["Y"]]

    def attenuate(self, g):
        gc = g[:, None]
        if self.t1 is not None:
            self.t1["Y"] = [Yv * gc for Yv in self.t1["Y"]]
        self.tris = [(U * gc, X * gc, Y * gc, o)
                     for (U, X, Y, o) in self.tris]

    def record_t1(self, g, pdf, rho_t):
        """Call INSTEAD of attenuate when HUB_T1=1: t=0 Hadamard as
        usual, plus record the (1,2)-slot bond pieces at this layer."""
        pc = _t(pdf)[:, None]
        self.t1 = {
            "rho": rho_t,
            "U": [(U * pc) for (U, X, Y, o) in self.tris],
            "X": [(X * pc) for (U, X, Y, o) in self.tris],
            "Y": [(Y * _t(g)[:, None]) for (U, X, Y, o) in self.tris],
            "P": _th.eye(self.tris[0][0].shape[0], device=_DEV),
        }
        self.attenuate(_t(g))

    def k21_plane(self, acc):
        for (U, X, Y, o) in self.tris:
            acc += (U * X) @ Y.T
        if self.t1 is not None:
            # exact lazy t1 bond read (slots u,x bonded at layer s):
            # read[a,b] += sum_h [sum_ij rho_s[i,j] Ut[i,h] Xt[j,h]
            #                     P[i,a] P[j,a]] * Yt[b,h]
            # staged: R2[j,(h,a)] = Xt[j,h]P[j,a]; M = rho @ R2;
            #         A[h,a] = sum_i Ut[i,h]P[i,a]M[i,(h,a)]
            P = self.t1["P"]
            rho = self.t1["rho"]
            N = P.shape[0]
            CH = 128                       # a-chunk to bound memory
            for (Ut, Xt, Yt) in zip(self.t1["U"], self.t1["X"],
                                    self.t1["Y"]):
                H = Ut.shape[1]
                Aha = _th.zeros(H, N, device=_DEV)
                for a0 in range(0, N, CH):
                    Pa = P[:, a0:a0 + CH]                    # [j, c]
                    R2 = Xt[:, :, None] * Pa[:, None, :]     # [j,H,c]
                    M = (rho @ R2.reshape(N, -1)).reshape(N, H, -1)
                    Aha[:, a0:a0 + CH] = _th.einsum(
                        "ih,ic,ihc->hc", Ut, Pa, M)
                acc += Aha.T @ Yt.T
                del Aha
        return acc


def feed_source(m, v, C, K21post):
    """Connected plane-feed content as hub sources (leading bond order).
    Pattern (u=i, w=j, spectator k via one rho bond):
      T[i,j,k] = 0.5 * d0_i * (K21[i,j] b0_j) * (rho_ik a1_k)
    plus the (u=i, w=k) mirror and the j/k-hub orientations from the
    symmetric patterns. v0.1 keeps the two leading i-hub patterns."""
    a, g, sd = state_cols(m, v)
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    d0 = pdf / sd
    b0 = g
    rho = C / np.outer(sd, sd)
    np.fill_diagonal(rho, 0.0)
    K = K21post.copy()
    np.fill_diagonal(K, 0.0)
    tris = []
    U = np.zeros((W_, W_))
    np.fill_diagonal(U, 0.5 * d0)
    Xj = (K * b0[None, :]).T                # X[:,h] = K[h,:]*b0
    Yk = (rho * a[1][None, :]).T            # Y[:,h] = rho[h,:]*a1
    tris.append((U, Xj, Yk, 0))             # hub i, j via K, k via rho
    tris.append((U, Yk, Xj, 0))             # mirror (j via rho, k via K)
    tris.append((Xj, U, Yk, 1))             # hub j orientation
    tris.append((Xj, Yk, U, 2))             # hub k orientation
    return HubSource(tris)


def _centers(a, sd, p, q, k3u=None, k4u=None):
    """Hub center weights for one (p,q) leg pair: the Gaussian V-emission
    a[p+q], plus (Kibble hub expansion) the carried-univariate stars:
    kappa3u -> (k3u/6) a[p+q+3]/sd^3, kappa4u -> (k4u/24) a[p+q+4]/sd^4.
    Same legs/collisions as the V-class - only the center shifts order."""
    cen = a[p + q].copy()
    if k3u is not None:
        cen += (k3u / 6.0) * a[p + q + 3] / sd ** 3
    if k4u is not None:
        cen += (k4u / 24.0) * a[p + q + 4] / sd ** 4
    return cen


def collision_tensor_cols(m, v, C, k3u=None, k4u=None):
    """The j=k collision content of the fresh V-emissions, as the pair
    (cen_pq, Z_pq) with T_col[i,j,j] = cen_i * Z[i,j]. Returned summed
    over PQ_KEEP as one (cen=1 folded) matrix pair: (czen, Z)."""
    a, g, sd = state_cols(m, v)
    rho = C / np.outer(sd, sd)
    np.fill_diagonal(rho, 0.0)
    Zsum = np.zeros((W_, W_))
    for (p, q) in PQ_KEEP:
        coef = 1.0 / (factorial(p) * factorial(q))
        Z = (rho ** p * a[p][None, :]) * (rho ** q * a[q][None, :])
        Zsum += coef * _centers(a, sd, p, q, k3u, k4u)[:, None] * Z
    return Zsum        # T_col[i,j,j] = Zsum[i,j] (center-i class)


def fresh_source(m, v, C, k3u=None, k4u=None):
    """Fresh V-emission hubs at the current pre-state (Gaussian part,
    plus the carried-univariate kappa3/kappa4 star centers when given)."""
    a, g, sd = state_cols(m, v)
    rho = C / np.outer(sd, sd)
    np.fill_diagonal(rho, 0.0)
    tris = []
    for (p, q) in PQ_KEEP:
        coef = 1.0 / (factorial(p) * factorial(q))
        rp = rho ** p
        rq = rho ** q if q != p else rp
        cen = coef * _centers(a, sd, p, q, k3u, k4u)
        # center tensor: T[i,j,k] = cen_i (rp_ij a[p]_j)(rq_ik a[q]_k)
        # hub-CP with hub h == i: U = diag(cen) -> store U as cen on the
        # identity; X columns = rp[:, h]-row? Express: T = sum_h cen_h
        # e_h (x) (rp[h,:] * a[p]) (x) (rq[h,:] * a[q])
        U = np.zeros((W_, W_))
        np.fill_diagonal(U, cen)
        X = (rp * a[p][None, :]).T          # X[:,h] = rp[h,:]*a[p]
        Y = (rq * a[q][None, :]).T
        # orientation 0: hub on index i (slots i,j,k) = (hub, x, y)
        tris.append((U, X, Y, 0))
        # orientation 1: center on j -> T[i,j,k] with hub=j:
        #   sum_h (rp[h,:]*a[p]) (x) cen_h e_h (x) (rq[h,:]*a[q])
        tris.append((X, U, Y, 1))
        # orientation 2: center on k
        tris.append((X, Y, U, 2))
    return HubSource(tris)


def main():
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    T = np.load(HERE / f"mps_cov2_rank{rank:04d}.npz")
    _plp = HERE / f"mps_planes_rank{rank:04d}.npz"
    if _plp.exists():
        Pl = np.load(_plp)
    else:
        class _NoPl:
            files = []
        Pl = _NoPl()
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    weights = np.load((STORE / entry["path"]).parent / "weights.npy",
                      mmap_mode="r")
    sys.path.insert(0, str(HERE))
    from run_attribution import edg2_relu_mean

    sources = []
    pending_colZ = None
    PLSAVE = {}
    planeK21 = np.zeros((W_, W_))
    k4d_post = np.zeros(W_)
    K22_post = np.zeros((W_, W_))
    K31_post = np.zeros((W_, W_))       # plane-supported carried K21 (post)
    out = {"rank": rank, "layers": {}}
    import time
    for l in range(0, lmax):
        t0 = time.time()
        if l == 0:
            W0 = np.asarray(weights[0], dtype=np.float64)
            C = W0.T @ W0
            m = np.zeros(W_)
            v = np.diag(C).copy()
            k3u = np.zeros(W_)
            k4u = np.zeros(W_)
        else:
            m = T["mc_pre_m"][l].astype(np.float64)
            v = T["mc_pre_v"][l].astype(np.float64)
            C = T["mc_pre_cov"][l].astype(np.float64)
            k3u = None  # set below from carried planes
        Wl = np.asarray(weights[l], dtype=np.float64)
        WlT_t = _t(Wl.T)
        # push carried content to the pre frame of layer l
        for s in sources:
            s.push(WlT_t)
        W2 = Wl * Wl
        Koff = planeK21.copy()
        np.fill_diagonal(Koff, 0.0)
        k3post_prev = np.diag(planeK21).copy()
        preK21 = (np.einsum("ia,ib,i->ab", W2, Wl, k3post_prev,
                            optimize=True)
                  + W2.T @ Koff @ Wl
                  + 2.0 * np.einsum("ia,ib,ij,ja->ab", Wl, Wl, Koff, Wl,
                                    optimize=True))
        acc = _th.zeros(W_, W_, device=_DEV)
        for s in sources:
            s.k21_plane(acc)
        preK21 += acc.cpu().numpy().astype(np.float64)
        if pending_colZ is not None:
            # exact j=k collision subtraction for the newest source's
            # first plane read (2A + B enumeration; older tails decay)
            A = np.einsum("ij,ia,ja,jb->ab", pending_colZ, Wl, Wl, Wl,
                          optimize=True)
            B = np.einsum("ij,ja,ib->ab", pending_colZ * 1.0,
                          Wl * Wl, Wl, optimize=True)
            preK21 -= (2.0 * A + B)
            pending_colZ = None
        from run_tensor_recursion_v4 import (fourth_emission_1024,
                                             fourth_push_1024)
        if l == 0:
            K22c = np.zeros((W_, W_))
            K31c = np.zeros((W_, W_))
        else:
            k3u = np.diag(preK21).copy()
            k4u, K22c, K31c = fourth_push_1024(
                _DEV, k4d_post, K22_post, K31_post, Wl)
            PLSAVE[f"pre_{l}"] = preK21.astype(np.float32)
            PLSAVE[f"k3u_{l}"] = k3u.astype(np.float32)
            PLSAVE[f"k4u_{l}"] = k4u.astype(np.float32)
        # diagnostics
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
        # ReLU step: attenuate carried sources by the gate, append fresh,
        # emit the plane part via the pair map channels
        a, g, sd = state_cols(m, v)
        g_t = _t(g)
        if os.environ.get("HUB_T1") == "1" and len(sources) >= 2:
            al_ = m / sd
            pdf_ = np.exp(-0.5 * al_ * al_) / np.sqrt(2 * np.pi)
            rho_np = C / np.outer(sd, sd)
            np.fill_diagonal(rho_np, 0.0)
            rho_t1 = _t(rho_np)
            newest_fresh = sources[-2]
            for s in sources:
                if s is newest_fresh:
                    s.record_t1(g, pdf_, rho_t1)
                else:
                    s.attenuate(g_t)
        else:
            for s in sources:
                s.attenuate(g_t)
        star3 = None if os.environ.get("HUB_NOSTARS") == "1" else k3u
        star4 = None if os.environ.get("HUB_NOSTARS") == "1" else k4u
        sources.append(fresh_source(m, v, C, star3, star4))
        if os.environ.get("HUB_COLCARRY") == "1":
            # carry the fresh source's collision content as a NEGATIVE
            # hub source (T_col = sum_j Z[:,j] x e_j x e_j and its two
            # orientations) - transported and read exactly like the
            # source it corrects, replacing the one-time subtraction
            Zc = collision_tensor_cols(m, v, C, star3, star4)
            Ieye = np.eye(W_)
            sources.append(HubSource([(-Zc, Ieye, Ieye, 0),
                                      (Ieye, -Zc, Ieye, 1),
                                      (Ieye, Ieye, -Zc, 2)]))
            pending_colZ = None
        else:
            pending_colZ = collision_tensor_cols(m, v, C, star3, star4)
        sources.append(feed_source(m, v, C, preK21))
        _lim = (3 if os.environ.get("HUB_COLCARRY") == "1" else 2)
        while len(sources) > _lim * SOURCE_CAP:
            sources.pop(0)
        # plane-part emission with the carried fourth-order channels
        from run_tensor_recursion_v2b import emission_K21
        planeK21 = emission_K21(m, v, k3u, k4u, C, preK21, K22c, K31c)
        k4d_post, K22_post, K31_post = fourth_emission_1024(
            m, v, k3u, k4u, C, preK21, K22c, K31c)
        rec["wall_s"] = round(time.time() - t0, 1)
        out["layers"][str(l)] = rec
        print(json.dumps({f"l{l}": rec}), flush=True)
    (HERE / f"HUB_CARRIER_rank{rank:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")
    np.savez(HERE / f"hub_planes_rank{rank:04d}.npz", **PLSAVE)


if __name__ == "__main__":
    main()
