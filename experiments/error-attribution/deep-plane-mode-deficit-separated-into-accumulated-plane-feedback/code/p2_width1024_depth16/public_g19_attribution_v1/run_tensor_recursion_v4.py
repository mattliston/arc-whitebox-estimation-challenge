#!/usr/bin/env python3
"""TAG g19g-tensor-recursion-v4 (PHASE 2, 1024x16): full third-cumulant tensor recursion (MPS).
State: full cum3 tensor T of the PRE state at layer l (f16 storage on MPS,
f32 compute). Seed T_pre0 = 0 exactly (input Gaussian). Per layer:
  ReLU map:
    distinct entries: T_post = g_i g_j g_k T_pre + Gaussian completion
      (degree-organized Hermite, total bond order <= DTOT)
    plane slices (i,i,j): K21post from the VALIDATED emission machinery
      (M21/M11 Mehler stack + Edgeworth channels) fed with the tensor's own
      preK21 planes, closure-carried K22/K31, true univariates + cov
    diagonal: analytic univariate k3post
  Push: exact 3-mode congruence with W[l+1] (chunked f32 GEMMs).
TRUE univariates + cov are fed per layer (mc) to isolate tensor physics.
Diagnostic per layer: K21 plane of pushed tensor vs MC truth (floors known).
Usage: run_tensor_recursion.py <rank> [Lmax]"""
import json
import os
import sys
import time
from math import erf, factorial
from pathlib import Path

import numpy as np
import torch

_erf = np.vectorize(erf)
HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
STOREDT = None
DTOT = int(os.environ.get("DTOT", "6"))
W_ = 1024
SLAB = 64
CHUNK = 131072


def columns(m, v, qmax=30):
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
    for q in range(2, qmax):
        c1.append(sd * He[q - 2] * pdf)
    raw1 = c1[0]
    raw2 = (m * m + v) * cdf + m * sd * pdf
    c2 = [raw2, 2 * sd * raw1, 2 * v * cdf]
    for q in range(3, qmax):
        c2.append(2 * v * He[q - 3] * pdf)
    return np.stack(c1), np.stack(c2), sd, pdf, cdf


def emission_K21(m, v, k3, k4, C, K21c, K22c, K31c):
    """Validated analytic post-K21 (connected), from the decomp harness."""
    c1, c2, sd, pdf, cdf = columns(m, v)
    rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)

    def series(colsA, colsB, a, b, tmax=20):
        acc = np.zeros_like(rho)
        rp = np.ones_like(rho)
        for t in range(tmax + 1):
            if t > 0:
                rp = rp * rho
            acc += rp / factorial(t) * np.outer(colsA[a + t] / sd ** a,
                                                colsB[b + t] / sd ** b)
        return acc

    def dM(colsA, colsB):
        return ((k3[:, None] / 6.0) * series(colsA, colsB, 3, 0)
                + (k3[None, :] / 6.0) * series(colsA, colsB, 0, 3)
                + (K21c / 2.0) * series(colsA, colsB, 2, 1)
                + (K21c.T / 2.0) * series(colsA, colsB, 1, 2)
                + (k4[:, None] / 24.0) * series(colsA, colsB, 4, 0)
                + (k4[None, :] / 24.0) * series(colsA, colsB, 0, 4)
                + (K22c / 4.0) * series(colsA, colsB, 2, 2)
                + (K31c / 6.0) * series(colsA, colsB, 3, 1)
                + (K31c.T / 6.0) * series(colsA, colsB, 1, 3))

    def dM2(colsA, colsB):
        """Second-order pattern-pair response on the pair (kappa3-squared
        class): patterns D_a(3,0) k3_a/6, D_b(0,3) k3_b/6, P_ab(2,1)
        K21[a,b]/2, P_ba(1,2) K21[b,a]/2; unordered pairs, same-pattern
        pairs carry 1/2. Scalar framework validated in
        second_order_triple.py (w256, transferable)."""
        k3r = k3[:, None]
        k3c = k3[None, :]
        Kt = K21c.T
        return (
            (k3r * k3r / 72.0) * series(colsA, colsB, 6, 0)
            + (k3c * k3c / 72.0) * series(colsA, colsB, 0, 6)
            + (k3r * k3c / 36.0) * series(colsA, colsB, 3, 3)
            + (k3r * K21c / 12.0) * series(colsA, colsB, 5, 1)
            + (k3r * Kt / 12.0) * series(colsA, colsB, 4, 2)
            + (k3c * Kt / 12.0) * series(colsA, colsB, 1, 5)
            + (k3c * K21c / 12.0) * series(colsA, colsB, 2, 4)
            + (K21c * K21c / 8.0) * series(colsA, colsB, 4, 2)
            + (Kt * Kt / 8.0) * series(colsA, colsB, 2, 4)
            + (K21c * Kt / 4.0) * series(colsA, colsB, 3, 3))

    so2 = os.environ.get("SECOND_ORDER") == "1"
    M21 = series(c2, c1, 0, 0) + dM(c2, c1) + (dM2(c2, c1) if so2 else 0.0)
    M11 = series(c1, c1, 0, 0) + dM(c1, c1) + (dM2(c1, c1) if so2 else 0.0)
    al = m / sd
    ma = -al
    he1, he3 = ma, ma ** 3 - 3 * ma
    g3v, g4v = k3 / sd ** 3, k4 / sd ** 4
    sys.path.insert(0, str(HERE))
    from run_attribution import edg2_relu_mean
    r1 = edg2_relu_mean(m, v, k3, k4)
    r2 = ((m * m + v) * cdf + m * sd * pdf + (g3v / 6) * (2 * v * pdf)
          + (g4v / 24) * (2 * v * he1 * pdf)
          + (g3v * g3v / 72) * (2 * v * he3 * pdf))
    return (M21 - 2 * r1[:, None] * M11 - r2[:, None] * r1[None, :]
            + 2 * (r1 ** 2)[:, None] * r1[None, :])



def fourth_emission_1024(m, v, k3, k4, C, K21c, K22c, K31c):
    """Post-ReLU fourth-order planes (k4diag, K22, K31), connected, via
    M22 (c2xc2) and M31 (c3xc1) Mehler maps. Validated at w256 vs 64M MC
    (per-element 0.98-1.05 at Gaussian pairs)."""
    c1, c2, sd, pdf0, cdf0 = columns(m, v)
    v_ = np.maximum(v, 1e-30)
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
    qn = c1.shape[0]
    raw1 = c1[0]
    raw2 = (m * m + v_) * cdf + m * sd * pdf
    raw3 = (m ** 3 + 3 * m * v_) * cdf + (m * m + 2 * v_) * sd * pdf
    ma = -al
    He = [np.ones_like(al), ma]
    for k in range(2, qn + 2):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    c3 = [raw3, 3 * sd * raw2, 6 * v_ * raw1, 6 * v_ * sd * cdf]
    for q in range(4, qn):
        c3.append(6 * v_ * sd * He[q - 4] * pdf)
    c3 = np.stack(c3)
    rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)

    def series(colsA, colsB, a, b, tmax=16):
        acc = np.zeros_like(rho)
        rp = np.ones_like(rho)
        for t in range(tmax + 1):
            if t > 0:
                rp = rp * rho
            acc += rp / float(factorial(t)) * np.outer(
                colsA[a + t] / sd ** a, colsB[b + t] / sd ** b)
        return acc

    def dM(colsA, colsB):
        return ((k3[:, None] / 6.0) * series(colsA, colsB, 3, 0)
                + (k3[None, :] / 6.0) * series(colsA, colsB, 0, 3)
                + (K21c / 2.0) * series(colsA, colsB, 2, 1)
                + (K21c.T / 2.0) * series(colsA, colsB, 1, 2)
                + (k4[:, None] / 24.0) * series(colsA, colsB, 4, 0)
                + (k4[None, :] / 24.0) * series(colsA, colsB, 0, 4)
                + (K22c / 4.0) * series(colsA, colsB, 2, 2)
                + (K31c / 6.0) * series(colsA, colsB, 3, 1)
                + (K31c.T / 6.0) * series(colsA, colsB, 1, 3))

    M11 = series(c1, c1, 0, 0) + dM(c1, c1)
    M21 = series(c2, c1, 0, 0) + dM(c2, c1)
    M22 = series(c2, c2, 0, 0) + dM(c2, c2)
    M31 = series(c3, c1, 0, 0) + dM(c3, c1)
    sys.path.insert(0, str(HERE))
    from run_attribution import edg2_relu_mean
    r1 = edg2_relu_mean(m, v_, k3, k4)
    g3v, g4v = k3 / sd ** 3, k4 / sd ** 4
    he1, he3 = ma, ma ** 3 - 3 * ma
    r2 = (raw2 + (g3v / 6) * (2 * v_ * pdf) + (g4v / 24) * (2 * v_ * he1 * pdf)
          + (g3v * g3v / 72) * (2 * v_ * he3 * pdf))
    r3v = raw3
    if os.environ.get("FOURTH_R3FIX") == "1":
        # G29 (T-76): E[phi^3] with the same Edgeworth corrections as M31/r1/r2 (Stein: c3[q] = sd^q E[d^q phi^3]);
        # without them mu31 is wrong at first order in kappa3 and the connected K31 (a small remainder) is garbage
        r3v = (c3[0] + (k3 / 6.0) / sd ** 3 * c3[3] + (k4 / 24.0) / sd ** 4 * c3[4]
               + (k3 * k3 / 72.0) / sd ** 6 * c3[6])
    mu11 = M11 - np.outer(r1, r1)
    mu22 = (M22 - 2 * r1[None, :] * M21 - 2 * r1[:, None] * M21.T
            + np.outer(r1 ** 2, r2) + np.outer(r2, r1 ** 2)
            + 4 * np.outer(r1, r1) * M11 - 3 * np.outer(r1 ** 2, r1 ** 2))
    mu31 = (M31 - r1[None, :] * r3v[:, None] - 3 * r1[:, None] * M21
            + 3 * np.outer(r1, r1) * r2[:, None]
            + 3 * (r1 ** 2)[:, None] * M11 - 3 * np.outer(r1 ** 3, r1))
    vpost = np.maximum(r2 - r1 * r1, 1e-30)
    K22post = mu22 - np.outer(vpost, vpost) - 2 * mu11 ** 2
    K31post = mu31 - 3 * vpost[:, None] * mu11
    k4post = np.diag(K22post).copy()
    np.fill_diagonal(K22post, 0.0)
    np.fill_diagonal(K31post, 0.0)
    return k4post, K22post, K31post


def _big_pair_gemm(dev, M, Wa, Wb, weight_vec=None, chunk=131072):
    """out[a,b] = sum_ij Wa[i,a]Wb[i,b] M[i,j] Wa[j,a]Wb[j,b]
       (weight_vec optional on i) via Z[j,ab]-chunked GEMMs on MPS."""
    import torch as th
    W_ = Wa.shape[0]
    n = Wa.shape[1]
    Mt = th.from_numpy(M.astype(np.float32)).to(dev)
    Wat = th.from_numpy(Wa.astype(np.float32)).to(dev)
    Wbt = th.from_numpy(Wb.astype(np.float32)).to(dev)
    out = th.zeros(n * n, device=dev)
    for c0 in range(0, n * n, chunk):
        ce = min(c0 + chunk, n * n)
        aa = th.div(th.arange(c0, ce, device=dev), n, rounding_mode="floor")
        bb = th.arange(c0, ce, device=dev) % n
        Z = Wat[:, aa] * Wbt[:, bb]              # [i, chunk]
        Y = Mt @ Z                               # [i, chunk]
        out[c0:ce] = (Z * Y).sum(dim=0)
    return out.reshape(n, n).cpu().numpy().astype(np.float64)


def fourth_push_1024(dev, k4post, K22post, K31post, Wl):
    """Plane-supported congruence + the w256-validated pattern set,
    with the two large mixed terms as chunked MPS GEMMs."""
    W2 = Wl * Wl
    W3 = W2 * Wl
    k4n = (np.einsum("ia,ia,i->a", W2, W2, k4post, optimize=True)
           + 3.0 * np.einsum("ia,ja,ij->a", W2, W2, K22post, optimize=True)
           + 4.0 * np.einsum("ia,ja,ij->a", W3, Wl, K31post, optimize=True))
    K22n = (np.einsum("ia,ib,i->ab", W2, W2, k4post, optimize=True)
            + W2.T @ K22post @ W2
            + (0.0 if os.environ.get("NOBIGPAIR") == "1" else 4.0) * _big_pair_gemm(dev, K22post, Wl, Wl)   # G29: the 1024^4 pair term (deployability test)
            + 2.0 * np.einsum("ia,ib,jb,ij->ab", W2, Wl, Wl, K31post,
                              optimize=True)
            + 2.0 * np.einsum("ia,ja,ib,ij->ab", Wl, Wl, W2, K31post,
                              optimize=True))
    K31n = (np.einsum("ia,ib,i->ab", W3, Wl, k4post, optimize=True)
            + W3.T @ K31post @ Wl
            + 3.0 * np.einsum("ia,ja,ib,ij->ab", W2, Wl, Wl, K31post,
                              optimize=True)
            + 3.0 * np.einsum("ia,ja,jb,ij->ab", W2, Wl, Wl, K22post,
                              optimize=True))
    np.fill_diagonal(K22n, 0.0)
    np.fill_diagonal(K31n, 0.0)
    return k4n, K22n, K31n

def push_mode(Tt, Wn32, dev):
    """One mode-1 congruence: out[a,j,k] = sum_i W[i,a] T[i,j,k]; then the
    caller permutes. Chunked f32 GEMM over flattened (j,k)."""
    out = torch.empty_like(Tt)
    X = Tt.reshape(W_, W_ * W_)
    Y = out.reshape(W_, W_ * W_)
    for c0 in range(0, W_ * W_, CHUNK):
        ce = min(c0 + CHUNK, W_ * W_)
        Y[:, c0:ce] = (Wn32.T @ X[:, c0:ce].to(torch.float32)).to(Tt.dtype)
    return out


def main():
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    global STOREDT
    STOREDT = torch.float32 if (len(sys.argv) > 3 and sys.argv[3] == "f32") else torch.float16
    dev = torch.device("mps")
    T = np.load(HERE / f"mps_cov2_rank{rank:04d}.npz")
    plp = HERE / f"mps_planes_rank{rank:04d}.npz"
    if plp.exists():
        Pl = np.load(plp)
    else:
        class _NoPl:
            files = []
        Pl = _NoPl()          # truth planes absent: diagnostics skipped
    p1p = HERE / f"mps_planes_rank{rank:04d}_pre1.npz"
    P1 = np.load(p1p) if p1p.exists() else None
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    stl = HERE / f"states_rank{rank:04d}_lite.npz"
    ST = np.load(stl) if stl.exists() \
        else np.load(HERE / f"states_rank{rank:04d}.npz")
    d_host_k3 = ST["host_k3"]
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    weights = np.load((STORE / entry["path"]).parent / "weights.npy",
                      mmap_mode="r")

    SAVE = {}
    Tt = torch.zeros((W_, W_, W_), dtype=STOREDT, device=dev)
    preK21 = np.zeros((W_, W_))
    k4d_post = np.zeros(W_)
    K22_post = np.zeros((W_, W_))
    K31_post = np.zeros((W_, W_))
    out = {"rank": rank, "layers": {}}
    t00 = time.time()
    idx = torch.arange(W_, device=dev)
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
        elif os.environ.get("DEPLOY_FRAME") == "1":
            # closure-fed states: what the deployed estimator actually has
            m = ST["host_m"][l - 1].astype(np.float64)
            v = ST["host_v"][l - 1].astype(np.float64)
            C = z["cov"][l - 1].astype(np.float64)
            k3u = ST["host_k3"][l - 1].astype(np.float64)
            k4u = ST["host_k4"][l - 1].astype(np.float64)
        else:
            m = T["mc_pre_m"][l].astype(np.float64)
            v = T["mc_pre_v"][l].astype(np.float64)
            C = T["mc_pre_cov"][l].astype(np.float64)
            k3u = T["mc_pre_k3"][l].astype(np.float64)
            k4u = T["mc_pre_k4"][l].astype(np.float64)
            K22c = z["k22"][l - 1].astype(np.float64)
            K31c = z["k31"][l - 1].astype(np.float64)
        if l > 0:
            # SELF-SUPPLIED fourth order: push the previous post planes
            Wl_np = np.asarray(weights[l], dtype=np.float64)
            k4u, K22c, K31c = fourth_push_1024(
                dev, k4d_post, K22_post, K31_post, Wl_np)
            k3u = preK21_diag_prev if preK21_diag_prev is not None else k3u
        if os.environ.get("HOST_FOURTH") == "1" and l > 0:
            # G29 wedge: the CARD's own fourth-order pre-planes and k4 (13-63% low)
            # in place of the engine's exact self-supplied pushes
            k4u = np.asarray(ST["host_k4"][l - 1], dtype=np.float64)
            K22c = z["k22"][l - 1].astype(np.float64)
            K31c = z["k31"][l - 1].astype(np.float64)
        if os.environ.get("K3FIX") == "1" and l > 0:
            # IN-RECURSION collective-skewness correction (co-adaptation
            # law: the fix must be joint - all downstream objects
            # inherit it): Tt += (K3S_l - implied_l) w x w x w
            if not hasattr(push_mode, "_k3s"):
                from run_collective_patch import C_UNIV
                from math import erf as _e
                _ev = np.vectorize(_e)
                k3s_ = {1: 0.0}
                acc_ = 0.0
                for ll in range(1, 15):
                    m_l = T["mc_pre_m"][ll].astype(np.float64)
                    C_l = T["mc_pre_cov"][ll].astype(np.float64)
                    w_l = m_l / max(np.linalg.norm(m_l), 1e-30)
                    m_n2 = T["mc_pre_m"][ll + 1].astype(np.float64)
                    w_n2 = m_n2 / max(np.linalg.norm(m_n2), 1e-30)
                    u_ = np.asarray(weights[ll + 1], dtype=np.float64) @ w_n2
                    sig_ = np.sqrt(max(w_l @ C_l @ w_l, 1e-30))
                    b_ = (C_l @ w_l) / sig_
                    vp_ = np.maximum(np.diag(C_l) - b_ * b_, 1e-30)
                    sdp = np.sqrt(vp_)

                    def _F(t):
                        mu_ = m_l + t * b_
                        alp = mu_ / sdp
                        pd_ = np.exp(-0.5 * alp * alp) / np.sqrt(2 * np.pi)
                        cd_ = 0.5 * (1 + _ev(alp / np.sqrt(2)))
                        return float(u_ @ (sdp * pd_ + mu_ * cd_))

                    G_ = (_F(0.05) - _F(-0.05)) / 0.1
                    xq, wq = np.polynomial.hermite_e.hermegauss(60)
                    wq = wq / np.sqrt(2 * np.pi)
                    Sq = np.array([_F(t) for t in xq])
                    E1_, E2_, E3_ = wq @ Sq, wq @ Sq ** 2, wq @ Sq ** 3
                    fr_ = E3_ - 3 * E2_ * E1_ + 2 * E1_ ** 3
                    acc_ = (G_ / sig_) ** 3 * acc_ + C_UNIV * fr_
                    k3s_[ll + 1] = acc_
                push_mode._k3s = k3s_
            wv = m / max(np.linalg.norm(m), 1e-30)
            wt_ = torch.from_numpy(wv.astype(np.float32)).to(dev)
            Mv = torch.einsum("ijk,k->ij", Tt.to(torch.float32), wt_)
            impl_ = float(torch.einsum("ij,i,j->", Mv, wt_, wt_)
                          .cpu().item())
            del Mv
            dlt = float(push_mode._k3s.get(l, 0.0)) - impl_
            if l >= 5:          # tracker accurate from ~L5 (early wrong-signed)
                Tt += (dlt * (wt_[:, None, None] * wt_[None, :, None]
                              * wt_[None, None, :])).to(Tt.dtype)
        if os.environ.get("K3IMPL") == "1" and l > 0:
            # implied collective kappa3 per layer: einsum(T, w,w,w),
            # w = normalized mean direction (closure-known at deploy)
            wv = m / max(np.linalg.norm(m), 1e-30)
            wt_ = torch.from_numpy(wv.astype(np.float32)).to(dev)
            Mv = torch.einsum("ijk,k->ij", Tt.to(torch.float32), wt_)
            k3impl = float(torch.einsum("ij,i,j->", Mv, wt_, wt_)
                           .cpu().item())
            SAVE.setdefault("k3impl", {})[l] = np.float32(k3impl)
            del Mv
        tprobe = os.environ.get("TPROBE")
        if tprobe and l == int(tprobe):
            pz = np.load(HERE / f"probe_truth_rank{rank:04d}_L{l}.npz")
            names_p = list(pz["names"])
            Up = torch.from_numpy(np.stack(
                [pz[f"probe_{k}"] for k in names_p]
            ).astype(np.float32).T).to(dev)
            prs = pz["pairs"]
            Tf_p = Tt.to(torch.float32)
            JKQ = torch.einsum("ijk,iq->jkq", Tf_p,
                               Up[:, prs[:, 0]])
            Sest = torch.einsum("jkq,jq->qk", JKQ,
                                Up[:, prs[:, 1]])
            np.savez(HERE / f"probe_est_rank{rank:04d}_L{l}.npz",
                     S=Sest.cpu().numpy().astype(np.float64))
            del Tf_p, JKQ, Sest
        oe = os.environ.get("ORACLE_EARLY")
        if oe == "all":
            oe = str(l)
        if oe and l == int(oe) and f"K21_{l}" in Pl.files:
            # wedge: replace the carried pre-plane with TRUTH at layer l
            # (tests the early-seeded unit-gain-loop hypothesis)
            preK21 = Pl[f"K21_{l}"].astype(np.float64)
            k3u = np.diag(preK21).copy()
        if os.environ.get("KT_FROM_ENGINE") == "1" and l > 0:
            # residue-instrument: seed planes+tensor from the banked
            # FULL-run planes each layer -> next planes = E[K_eng_l]
            _tpf = np.load(HERE / f"tensor_planes_rank{rank:04d}.npz")
            if f"pre_{l}" in _tpf.files:
                preK21 = _tpf[f"pre_{l}"].astype(np.float64)
                k3u = np.diag(preK21).copy()
            Tt.zero_()
            Kp0 = torch.from_numpy(preK21.astype(np.float32)).to(dev).to(Tt.dtype)
            Tt[idx, idx, :] = Kp0
            Tt[idx, :, idx] = Kp0
            Tt[:, idx, idx] = Kp0.T
            Tt[idx, idx, idx] = torch.from_numpy(np.diag(
                preK21).astype(np.float32)).to(dev).to(Tt.dtype)
        if os.environ.get("HOST_SEED") == "1" and l > 0:
            # residue-instrument: seed planes+tensor from the HOST
            # trajectory each layer -> saved next planes = E[K_host_l]
            preK21 = z["k21"][l - 1].astype(np.float64)
            k3u = np.asarray(d_host_k3[l - 1], dtype=np.float64) \
                if True else k3u
            Tt.zero_()
            Kp0 = torch.from_numpy(preK21.astype(np.float32)).to(dev).to(Tt.dtype)
            Tt[idx, idx, :] = Kp0
            Tt[idx, :, idx] = Kp0
            Tt[:, idx, idx] = Kp0.T
            Tt[idx, idx, idx] = torch.from_numpy(np.diag(
                preK21).astype(np.float32)).to(dev).to(Tt.dtype)
        if os.environ.get("KILL_TENSOR") == "1" and l > 0:
            # wedge: drop all carried distinct content (keep plane/diag
            # slices only) - isolates the fresh single-step emission
            Tt.zero_()
            Kp0 = torch.from_numpy(preK21.astype(np.float32)).to(dev).to(
                Tt.dtype)
            Tt[idx, idx, :] = Kp0
            Tt[idx, :, idx] = Kp0
            Tt[:, idx, idx] = Kp0.T
            Tt[idx, idx, idx] = torch.from_numpy(
                np.diag(preK21).astype(np.float32)).to(dev).to(Tt.dtype)
        if os.environ.get("ORACLE_FOURTH") == "1" and l > 0:
            # wedge: true fourth-order supply into the emissions
            k4u = T["mc_pre_k4"][l].astype(np.float64)
            if f"K22_{l}" in Pl.files:
                K22c = Pl[f"K22_{l}"].astype(np.float64)
                K31c = Pl[f"K31_{l}"].astype(np.float64)
        sd = np.sqrt(np.maximum(v, 1e-30))
        al = m / sd
        pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
        cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
        rho = C / np.outer(sd, sd)
        np.fill_diagonal(rho, 0.0)
        qmax = 2 * DTOT + 1
        ma = -al
        He = [np.ones_like(al), ma]
        for k in range(2, qmax + 2):
            He.append(ma * He[-1] - (k - 1) * He[-2])
        c1l = [sd * pdf + m * cdf, sd * cdf]
        for q in range(2, qmax + 1):
            c1l.append(sd * He[q - 2] * pdf)
        a = np.stack(c1l)
        if os.environ.get("GC_COLUMNS") == "1" and l > 0:
            # GC marginal correction via Stein mean-shift (exact to 1st
            # GC order inside the connected Kibble series):
            # a_GC[t] = a[t] + (k3u/6) a[t+3]/sd^3 + (k4u/24) a[t+4]/sd^4
            need = a.shape[0] + 4
            while len(He) < need:
                He.append(ma * He[-1] - (len(He) - 1) * He[-2])
            c1x = list(c1l)
            for q in range(len(c1x), need):
                c1x.append(sd * He[q - 2] * pdf)
            ax = np.stack(c1x)
            nA = a.shape[0]
            a = (ax[:nA]
                 + (np.asarray(k3u) / 6.0) / sd ** 3 * ax[3:3 + nA]
                 + (np.asarray(k4u) / 24.0) / sd ** 4 * ax[4:4 + nA])
        g = cdf

        gt = torch.from_numpy(g.astype(np.float32)).to(dev)
        at = torch.from_numpy(a.astype(np.float32)).to(dev)
        rt = torch.from_numpy(rho.astype(np.float32)).to(dev)
        rpow = {t: rt ** t for t in range(1, max(DTOT, 4) + 1)}
        terms = []
        for tij in range(0, DTOT + 1):
            for tik in range(0, DTOT + 1 - tij):
                for tjk in range(0, DTOT + 1 - tij - tik):
                    di, dj, dk = tij + tik, tij + tjk, tik + tjk
                    if di == 0 or dj == 0 or dk == 0:
                        continue
                    coef = 1.0 / (factorial(tij) * factorial(tik)
                                  * factorial(tjk))
                    terms.append((tij, tik, tjk, di, dj, dk, coef))
        bcols = [cdf] + [He[t - 1] * pdf for t in range(1, DTOT + 1)]
        dcols = [He[t] * pdf / sd for t in range(0, DTOT + 1)]
        bt = torch.from_numpy(np.stack(bcols).astype(np.float32)).to(dev)
        d0 = torch.from_numpy(dcols[0].astype(np.float32)).to(dev)
        b0 = bt[0]
        a0 = at[0]
        preKt = torch.from_numpy(preK21.astype(np.float32)).to(dev)
        allterms = [(0, 0, 0, 0, 0, 0, 1.0)] + terms
        for i0 in range(0, W_, SLAB):
            sl = slice(i0, i0 + SLAB)
            blk = Tt[sl].to(torch.float32)
            F3 = torch.zeros_like(blk)
            for (tij, tik, tjk, di, dj, dk, coef) in allterms:
                t_ = (bt[di][sl, None, None] * bt[dj][None, :, None]
                      * bt[dk][None, None, :]) * coef
                if tij:
                    t_ = t_ * rpow[tij][sl, :][:, :, None]
                if tik:
                    t_ = t_ * rpow[tik][sl, :][:, None, :]
                if tjk:
                    t_ = t_ * rpow[tjk][None, :, :]
                F3 += t_
            blk = blk * F3
            del F3
            # plane-feed, CONNECTED response: K[u,w]=cum3pre[u,u,w]
            # feeds distinct [i,j,k] via E[f''_u f'_w f_spec] with the
            # spectator attached by >=1 rho bond (t=0 part is cancelled by
            # the disconnected subtraction). Bond patterns (tuw, tus, tws)
            # with tus + tws >= 1, total <= 4.
            def feed_factor(dcol_u, bcol_w, acol_s, r_uw, r_us, r_ws,
                            su, sw, ss):
                acc = None
                for tuw in range(0, 5):
                    for tus in range(0, 5 - tuw):
                        for tws in range(0, 5 - tuw - tus):
                            if tus + tws < 1:
                                continue
                            du, dw, ds = tuw + tus, tuw + tws, tus + tws
                            cf = 1.0 / (factorial(tuw) * factorial(tus)
                                        * factorial(tws))
                            t_ = (dcol_u[du][su] * bcol_w[dw][sw]
                                  * acol_s[ds][ss]) * cf
                            if tuw:
                                t_ = t_ * r_uw[tuw]
                            if tus:
                                t_ = t_ * r_us[tus]
                            if tws:
                                t_ = t_ * r_ws[tws]
                            acc = t_ if acc is None else acc + t_
                return acc
            while len(He) < 14:
                He.append(ma * He[-1] - (len(He) - 1) * He[-2])
            dt_np = np.stack([He[t] * pdf / sd for t in range(0, 13)])
            bt_np = np.stack([cdf] + [He[t - 1] * pdf
                                      for t in range(1, 13)])
            if os.environ.get("GC_TRANSPORT") == "1" and l > 0:
                # Stein/GC marginal shift on the TRANSPORT columns (the
                # F3 b-columns and feed d-columns) with the CARRIED
                # univariates; per-slot bias compounds x3 per layer
                sh3 = (np.asarray(k3u) / 6.0) / sd ** 3
                sh4 = (np.asarray(k4u) / 24.0) / sd ** 4
                bt_np = (bt_np[:9] + sh3 * bt_np[3:12] + sh4 * bt_np[4:13])
                dt_np = (dt_np[:9] + sh3 * dt_np[3:12] + sh4 * dt_np[4:13])
            else:
                bt_np = bt_np[:9]
                dt_np = dt_np[:9]
            dt_all = torch.from_numpy(dt_np.astype(np.float32)).to(dev)
            bt_all = torch.from_numpy(bt_np.astype(np.float32)).to(dev)
            at_all = at
            SL = (slice(None),)
            ii = (sl, None, None)
            jj = (None, slice(None), None)
            kk = (None, None, slice(None))
            def bshape(v, pos):
                if pos == 0:
                    return v[sl, None, None]
                if pos == 1:
                    return v[None, :, None]
                return v[None, None, :]
            def rshape(t, p1, p2):
                # rho^t between positions p1,p2 broadcast over slab block
                if {p1, p2} == {0, 1}:
                    return rpow[t][sl, :][:, :, None]
                if {p1, p2} == {0, 2}:
                    return rpow[t][sl, :][:, None, :]
                return rpow[t][None, :, :]
            def cshape(c, t, pos):
                return bshape(c[t], pos)
            # pattern list: (K-orient as [blk-axes], u-pos, w-pos, s-pos)
            pats = [
                (preKt[sl, :][:, :, None], 0, 1, 2),
                (preKt[sl, :][:, None, :], 0, 2, 1),
                (preKt.T[sl, :][:, :, None], 1, 0, 2),
                (preKt[None, :, :], 1, 2, 0),
                (preKt.T[sl, :][:, None, :], 2, 0, 1),
                (preKt.T[None, :, :], 2, 1, 0),
            ]
            fs = os.environ.get("FRESH_STOP")
            if fs is not None and l > int(fs):
                pats = []            # transport-only: no fresh feeds
            if os.environ.get("NOFEED") == "1":
                pats = []            # G29 ablation: drop ONLY the plane-to-distinct feed (Gaussian completion kept)
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
                            t_ = (cshape(dt_all, du, up)
                                  * cshape(bt_all, dw, wp)
                                  * cshape(at_all, ds, sp)) * cf
                            if tuw:
                                t_ = t_ * rshape(tuw, up, wp)
                            if tus:
                                t_ = t_ * rshape(tus, up, sp)
                            if tws:
                                t_ = t_ * rshape(tws, wp, sp)
                            acc = t_ if acc is None else acc + t_
                blk += Kb * acc
                del acc
            # Gaussian completion (zeroth order in cum3pre)
            for (tij, tik, tjk, di, dj, dk, coef) in (
                    [] if (fs is not None and l > int(fs)) else terms):
                t_ = (at[di][sl, None, None] * at[dj][None, :, None]
                      * at[dk][None, None, :]) * coef
                if tij:
                    t_ = t_ * rpow[tij][sl, :][:, :, None]
                if tik:
                    t_ = t_ * rpow[tik][sl, :][:, None, :]
                if tjk:
                    t_ = t_ * rpow[tjk][None, :, :]
                blk += t_
            if os.environ.get("TENSOR_K4") == "1" and l > 0:
                # FIRST-ORDER fourth-order-plane insertions into the
                # DISTINCT completion (k4 diag Q400-class 1/24, K31
                # Q310-class 1/6, K22 Q220-class 1/4; connected filter;
                # Q211/K4T not carried - omitted). Never previously
                # wired: the oracle-fourth wedge fed only pair channels.
                if not hasattr(push_mode, "_k4_terms"):
                    import itertools as _it

                    def _pedges(counts):
                        sup = [c for c in range(3) if counts[c] > 0]
                        return {frozenset(e) for e in
                                _it.combinations(sup, 2)}

                    def _conn(edge_sets):
                        par = list(range(3))

                        def f(x):
                            while par[x] != x:
                                par[x] = par[par[x]]
                                x = par[x]
                            return x
                        for es in edge_sets:
                            for e in es:
                                a2, b2 = tuple(e)
                                ra, rb = f(a2), f(b2)
                                if ra != rb:
                                    par[ra] = rb
                        return len({f(c) for c in range(3)}) == 1
                    K4P = [((4, 0, 0), "k4_0", 1 / 24.), ((0, 4, 0), "k4_1", 1 / 24.),
                           ((0, 0, 4), "k4_2", 1 / 24.),
                           ((3, 1, 0), "K31_01", 1 / 6.), ((1, 3, 0), "K31_10", 1 / 6.),
                           ((3, 0, 1), "K31_02", 1 / 6.), ((1, 0, 3), "K31_20", 1 / 6.),
                           ((0, 3, 1), "K31_12", 1 / 6.), ((0, 1, 3), "K31_21", 1 / 6.),
                           ((2, 2, 0), "K22_01", 1 / 4.), ((2, 0, 2), "K22_02", 1 / 4.),
                           ((0, 2, 2), "K22_12", 1 / 4.)]
                    tl = []
                    for (cnts, key, base) in K4P:
                        e1 = _pedges(cnts)
                        for t01 in range(0, 4):
                            for t02 in range(0, 4 - t01):
                                for t12 in range(0, 4 - t01 - t02):
                                    eb = set()
                                    if t01:
                                        eb.add(frozenset((0, 1)))
                                    if t02:
                                        eb.add(frozenset((0, 2)))
                                    if t12:
                                        eb.add(frozenset((1, 2)))
                                    if not _conn([e1, eb]):
                                        continue
                                    bc = base / (factorial(t01)
                                                 * factorial(t02)
                                                 * factorial(t12))
                                    tl.append((key, (t01, t02, t12),
                                               cnts, bc))
                    push_mode._k4_terms = tl
                fam4 = {0: at, 1: bt_all, 2: dt_all}
                for nn in (3, 4):
                    fam4[nn] = torch.from_numpy(np.stack(
                        [He[(nn - 2) + t] * pdf / sd ** (nn - 1)
                         for t in range(0, 7)]).astype(np.float32)).to(dev)
                k4t_v = torch.from_numpy(
                    np.asarray(k4u, dtype=np.float32)).to(dev)
                K22t = torch.from_numpy(
                    K22c.astype(np.float32)).to(dev)
                K31t = torch.from_numpy(
                    K31c.astype(np.float32)).to(dev)
                K31tt = K31t.T.contiguous()
                K22tt = K22t.T.contiguous()
                W4 = {
                    "k4_0": k4t_v[sl, None, None],
                    "k4_1": k4t_v[None, :, None],
                    "k4_2": k4t_v[None, None, :],
                    "K31_01": K31t[sl, :][:, :, None],
                    "K31_10": K31tt[sl, :][:, :, None],
                    "K31_02": K31t[sl, :][:, None, :],
                    "K31_20": K31tt[sl, :][:, None, :],
                    "K31_12": K31t[None, :, :],
                    "K31_21": K31tt[None, :, :],
                    "K22_01": K22t[sl, :][:, :, None],
                    "K22_02": K22t[sl, :][:, None, :],
                    "K22_12": K22t[None, :, :]}
                for (key4, (t01, t02, t12), (n0, n1, n2), cf4) in \
                        push_mode._k4_terms:
                    t_ = (cf4 * W4[key4]
                          * bshape(fam4[n0][t01 + t02], 0)
                          * bshape(fam4[n1][t01 + t12], 1)
                          * bshape(fam4[n2][t02 + t12], 2))
                    if t01:
                        t_ = t_ * rshape(t01, 0, 1)
                    if t02:
                        t_ = t_ * rshape(t02, 0, 2)
                    if t12:
                        t_ = t_ * rshape(t12, 1, 2)
                    blk += t_
                    del t_
            if os.environ.get("TENSOR_SO2") == "1" and l > 0:
                # distinct-tensor SECOND-ORDER pattern pairs (kappa3^2):
                # scalar framework second_order_triple.py (GH-validated);
                # broadcast form crosschecked exactly; w256 gate: deep
                # tensor improves (l31 K21_rel .500->.474)
                if not hasattr(push_mode, "_so2_terms"):
                    sys.path.insert(0, str(
                        HERE.parent.parent
                        / "x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1"))
                    from second_order_triple import second_order_terms
                    push_mode._so2_terms = [
                        tt for tt in second_order_terms(bond_cap=3)
                        if "T_012" not in (tt[0], tt[1])]
                fam_t = {0: at, 1: bt_all, 2: dt_all}
                for nn in (3, 4, 5, 6):
                    if nn not in fam_t:
                        fam_t[nn] = torch.from_numpy(np.stack(
                            [He[(nn - 2) + t] * pdf / sd ** (nn - 1)
                             for t in range(0, 7)]
                        ).astype(np.float32)).to(dev)
                k3t_v = torch.from_numpy(
                    np.asarray(k3u, dtype=np.float32)).to(dev)
                KTt = preKt.T.contiguous()
                WFAC = {
                    "k3_0": k3t_v[sl, None, None],
                    "k3_1": k3t_v[None, :, None],
                    "k3_2": k3t_v[None, None, :],
                    "K21_01": preKt[sl, :][:, :, None],
                    "K21_10": KTt[sl, :][:, :, None],
                    "K21_02": preKt[sl, :][:, None, :],
                    "K21_20": KTt[sl, :][:, None, :],
                    "K21_12": preKt[None, :, :],
                    "K21_21": KTt[None, :, :]}
                for (k1_, k2_, (t01, t02, t12), (n0, n1, n2), cf2) in \
                        push_mode._so2_terms:
                    t_ = (cf2 * WFAC[k1_] * WFAC[k2_]
                          * bshape(fam_t[n0][t01 + t02], 0)
                          * bshape(fam_t[n1][t01 + t12], 1)
                          * bshape(fam_t[n2][t02 + t12], 2))
                    if t01:
                        t_ = t_ * rshape(t01, 0, 1)
                    if t02:
                        t_ = t_ * rshape(t02, 0, 2)
                    if t12:
                        t_ = t_ * rshape(t12, 1, 2)
                    blk += t_
                    del t_
            Tt[sl] = blk.to(Tt.dtype)
            del blk
        torch.mps.empty_cache()

        K21post = emission_K21(m, v, k3u, k4u, C, preK21, K22c, K31c)
        r1 = sd * pdf + m * cdf
        r2 = (m * m + v) * cdf + m * sd * pdf
        r3 = (m ** 3 + 3 * m * v) * cdf + (m * m + 2 * v) * sd * pdf
        k3post = r3 - 3 * r2 * r1 + 2 * r1 ** 3
        SAVE.setdefault("post", {})[l] = K21post.astype(np.float32)
        Kp = torch.from_numpy(K21post.astype(np.float32)).to(dev).to(Tt.dtype)
        Tt[idx, idx, :] = Kp
        Tt[idx, :, idx] = Kp
        Tt[:, idx, idx] = Kp.T
        Tt[idx, idx, idx] = torch.from_numpy(
            k3post.astype(np.float32)).to(dev).to(Tt.dtype)

        M3P = int(os.environ.get("MODE3_PROBES", "0"))
        if M3P > 0 and l in (5, 9, 13):
            # h2h3-C3 pre-gate: dump X_p = sum_k T[i,j,k] y_p[k] for
            # M3P probe directions y_p = normalized columns of the
            # backward conjugation product (deploy-known family), plus
            # the off-diagonal mass needed for the randomized-read
            # variance analysis. Read-only wedge; ~n^2 per probe.
            rng_ = np.random.Generator(np.random.PCG64(1234 + l))
            cols = rng_.choice(W_, size=M3P, replace=False)
            Yp = np.zeros((W_, M3P))
            for a_, c_ in enumerate(cols):
                v0 = np.zeros(W_); v0[c_] = 1.0
                Yp[:, a_] = v0
            Ypt = torch.from_numpy(Yp.astype(np.float32)).to(dev)
            Xacc = torch.zeros((M3P, W_, W_), dtype=torch.float32,
                               device=dev)
            for i0 in range(0, W_, SLAB):
                sl_ = slice(i0, i0 + SLAB)
                Xacc[:, sl_, :] += torch.einsum(
                    "ijk,kp->pij", Tt[sl_].to(torch.float32), Ypt)
            xn = {}
            for a_ in range(M3P):
                Xp = Xacc[a_]
                dg = torch.diagonal(Xp)
                off2 = float((Xp * Xp).sum().item())                     - float((dg * dg).sum().item())
                xn[str(int(cols[a_]))] = {
                    "diag_sumsq": float((dg * dg).sum().item()),
                    "offdiag_sumsq": off2}
            SAVE.setdefault("m3probe", {})[l] = xn
            del Xacc
            torch.mps.empty_cache()
        RK = int(os.environ.get("RECON_K", "0"))
        if RK > 0 and l > 0:
            # AUX-PLANE WEDGE: replace carried DISTINCT content by its
            # symmetric reconstruction from K mode-1 contractions with
            # deploy-known coordinate probes ("aux planes"). Value
            # surviving this wedge = ceiling for a billable aux-plane
            # closure. Plane/diag slices preserved exactly.
            probes = [np.ones(W_), cdf, pdf,
                      m / max(np.linalg.norm(m), 1e-30)]
            if RK > 4:
                ew_, ev_ = np.linalg.eigh(rho)
                for r_ in range(min(RK - 4, 8)):
                    probes.append(ev_[:, -1 - r_])
            P_ = np.stack(probes[:RK], 1)
            Q_, _ = np.linalg.qr(P_)
            Phi = torch.from_numpy(Q_.astype(np.float32)).to(dev)
            K_ = Phi.shape[1]
            A = torch.zeros((K_, W_, W_), dtype=torch.float32,
                            device=dev)
            for i0 in range(0, W_, SLAB):
                sl = slice(i0, i0 + SLAB)
                A += torch.einsum("it,ijk->tjk", Phi[sl],
                                  Tt[sl].to(torch.float32))
            # remove plane-slice content along i (exact, incl. diag)
            Kp32 = Kp.to(torch.float32)
            k3p32 = torch.from_numpy(
                k3post.astype(np.float32)).to(dev)
            kdiag = torch.diagonal(Kp32)
            KQ = Kp32 @ Phi                       # (KQ)[j,t]
            for t_ in range(K_):
                ph = Phi[:, t_]
                A[t_] -= ph[:, None] * Kp32       # i=j: Kp[j,k]
                A[t_] -= Kp32.T * ph[None, :]     # i=k: Kp[k,j]
                dg = torch.diagonal(A[t_])
                dg += 3.0 * ph * kdiag - KQ[:, t_] - ph * k3p32
            # symmetric mode-averaged reconstruction, slab-wise
            for i0 in range(0, W_, SLAB):
                sl = slice(i0, i0 + SLAB)
                ns = min(SLAB, W_ - i0)
                blk = torch.zeros((ns, W_, W_), dtype=torch.float32,
                                  device=dev)
                for t_ in range(K_):
                    ph = Phi[:, t_]
                    At = A[t_]
                    blk += ph[sl, None, None] * At[None, :, :]
                    blk += ph[None, :, None] * At[sl][:, None, :]
                    blk += ph[None, None, :] * At[sl][:, :, None]
                blk /= 3.0
                Tt[sl] = blk.to(Tt.dtype)
                del blk
            Tt[idx, idx, :] = Kp
            Tt[idx, :, idx] = Kp
            Tt[:, idx, idx] = Kp.T
            Tt[idx, idx, idx] = torch.from_numpy(
                k3post.astype(np.float32)).to(dev).to(Tt.dtype)
            del A
            torch.mps.empty_cache()

        TB = int(os.environ.get("TUCKER_B", "0"))
        if TB > 0:
            # per-layer HOSVD truncation of the DISTINCT content to a
            # B-per-slot Tucker core (plane/diag slices preserved) -
            # the bounded-size long-memory carrier (w256-validated)
            keep_pl = (Tt[idx, idx, :].clone(), Tt[idx, :, idx].clone(),
                       Tt[:, idx, idx].clone())
            Tt[idx, idx, :] = 0
            Tt[idx, :, idx] = 0
            Tt[:, idx, idx] = 0
            Tf = Tt.to(torch.float32)
            bases = []
            for mode in range(3):
                M = Tf.permute(mode, *[x for x in range(3) if x != mode]
                               ).reshape(W_, -1)
                G = (M @ M.T).to(torch.float32)
                U_, S_, _ = torch.linalg.svd(G)
                if mode == 0:
                    # Tucker spectrum of the DISTINCT content: S_ are the
                    # mode-Gram eigenvalues (= squared singular values)
                    cs = torch.cumsum(S_, 0) / S_.sum()
                    print(json.dumps({"tucker_energy_frac": {
                        "L": l, "r64": round(float(cs[63]), 4),
                        "r128": round(float(cs[127]), 4),
                        "r256": round(float(cs[255]), 4),
                        "r512": round(float(cs[511]), 4)}}), flush=True)
                bases.append(U_[:, :TB])
                del M, G
            core = torch.einsum("ijk,ia->ajk", Tf, bases[0])
            core = torch.einsum("ajk,jb->abk", core, bases[1])
            core = torch.einsum("abk,kc->abc", core, bases[2])
            Tf = torch.einsum("abc,ia->ibc", core, bases[0])
            Tf = torch.einsum("ibc,jb->ijc", Tf, bases[1])
            Tt = torch.einsum("ijc,kc->ijk", Tf, bases[2]).to(Tt.dtype)
            del Tf, core, bases
            Tt[idx, idx, :] = keep_pl[0]
            Tt[idx, :, idx] = keep_pl[1]
            Tt[:, idx, idx] = keep_pl[2]
            del keep_pl
            torch.mps.empty_cache()

        Wn32 = torch.from_numpy(np.ascontiguousarray(
            weights[l + 1], dtype=np.float32)).to(dev)
        # mode-1 on i, then rotate axes so each mode takes a turn in front
        Tt = push_mode(Tt, Wn32, dev)                    # [a, j, k]
        Tt = Tt.permute(1, 2, 0).contiguous()            # [j, k, a]
        torch.mps.empty_cache()
        Tt = push_mode(Tt, Wn32, dev)                    # [b, k, a]
        Tt = Tt.permute(1, 2, 0).contiguous()            # [k, a, b]
        torch.mps.empty_cache()
        Tt = push_mode(Tt, Wn32, dev)                    # [c, a, b]
        Tt = Tt.permute(1, 2, 0).contiguous()            # [a, b, c]
        torch.mps.empty_cache()

        k4d_post, K22_post, K31_post = fourth_emission_1024(
            m, v, k3u, k4u, C, preK21, K22c, K31c)
        nl = l + 1
        K21new = Tt[idx, idx, :].to(torch.float32).cpu().numpy().astype(np.float64)
        # carry the tensor's own third univariate (was None -> oracle leak:
        # validation frame silently used TRUE k3, deploy frame host k3)
        preK21_diag_prev = np.diag(K21new).copy()
        SAVE.setdefault("pre", {})[l + 1] = K21new.astype(np.float32)
        SAVE.setdefault("k3u", {})[l] = np.asarray(k3u, dtype=np.float32)
        SAVE.setdefault("k4u", {})[l] = np.asarray(k4u, dtype=np.float32)
        SAVE.setdefault("K22", {})[l] = np.asarray(K22c, dtype=np.float32)
        SAVE.setdefault("K31", {})[l] = np.asarray(K31c, dtype=np.float32)
        preK21 = K21new
        rec = {"wall_s": round(time.time() - t0, 1)}
        truth = None
        if nl == 1 and P1 is not None and "K21_1" in P1.files:
            truth = P1["K21_1"].astype(np.float64)
        elif f"K21_{nl}" in Pl.files:
            truth = Pl[f"K21_{nl}"].astype(np.float64)
        if truth is not None:
            scale = float(np.sqrt(np.mean(truth ** 2)))
            rec["rel"] = round(float(np.sqrt(np.mean((K21new - truth) ** 2)))
                               / scale, 4)
            rec["corr"] = round(float(np.sum(K21new * truth)
                                / np.sqrt(np.sum(K21new ** 2)
                                          * np.sum(truth ** 2))), 4)
            if nl - 1 < z["k21"].shape[0]:
                carried = z["k21"][nl - 1].astype(np.float64)
                rec["closure_rel"] = round(float(np.sqrt(np.mean(
                    (carried - truth) ** 2))) / scale, 4)
        if l > 0:
            tk4 = T["mc_pre_k4"][l].astype(np.float64)
            s4 = float(np.sqrt(np.mean(tk4 ** 2)))
            if s4 > 0:
                rec["k4carried_rel"] = round(float(np.sqrt(np.mean(
                    (k4u - tk4) ** 2))) / s4, 3)
            if f"K22_{l}" in Pl.files:
                tk22 = Pl[f"K22_{l}"].astype(np.float64)
                np.fill_diagonal(tk22, 0.0)
                s22 = float(np.sqrt(np.mean(tk22 ** 2)))
                z22 = z["k22"][l - 1].astype(np.float64).copy()
                np.fill_diagonal(z22, 0.0)
                if s22 > 0:
                    rec["K22carried_rel"] = round(float(np.sqrt(np.mean(
                        (K22c - tk22) ** 2))) / s22, 3)
                    rec["K22closure_rel"] = round(float(np.sqrt(np.mean(
                        (z22 - tk22) ** 2))) / s22, 3)
        if "m3probe" in SAVE and l in SAVE["m3probe"]:
            rec["m3probe"] = SAVE["m3probe"][l]
        out["layers"][str(nl)] = rec
        print(json.dumps({f"L{nl}": rec}), flush=True)
        (HERE / f"TENSOR_RECURSION_rank{rank:04d}_v2b.json").write_text(
            json.dumps({k: v2 for k, v2 in out.items()}, indent=1) + "\n")
    o4 = ("_o4" if os.environ.get("ORACLE_FOURTH") == "1"
          else "_dep" if os.environ.get("DEPLOY_FRAME") == "1" else "")
    if os.environ.get("GC_COLUMNS") == "1":
        o4 += "_gc"
    if os.environ.get("KILL_TENSOR") == "1":
        o4 += "_kt"
    if DTOT != 6:
        o4 += f"_d{DTOT}"
    if os.environ.get("RECON_K"):
        o4 += "_aux" + os.environ["RECON_K"]
    if os.environ.get("HOST_SEED") == "1":
        o4 += "_hseed"
    if os.environ.get("KT_FROM_ENGINE") == "1":
        o4 += "_ktfe"
    if os.environ.get("TUCKER_B"):
        o4 += "_tb" + os.environ["TUCKER_B"]
    if os.environ.get("NOFEED") == "1":
        o4 += "_nofeed"
    if os.environ.get("HOST_FOURTH") == "1":
        o4 += "_hostk4"
    if os.environ.get("NOBIGPAIR") == "1":
        o4 += "_nobigpair"
    if os.environ.get("OUT_SUFFIX"):
        o4 += os.environ["OUT_SUFFIX"]
    np.savez(HERE / f"tensor_planes_rank{rank:04d}{o4}.npz",
             **{f"post_{k}": v2 for k, v2 in SAVE.get("post", {}).items()},
             **{f"pre_{k}": v2 for k, v2 in SAVE.get("pre", {}).items()},
             **{f"k3u_{k}": v2 for k, v2 in SAVE.get("k3u", {}).items()},
             **{f"k4u_{k}": v2 for k, v2 in SAVE.get("k4u", {}).items()},
             **{f"K22_{k}": v2 for k, v2 in SAVE.get("K22", {}).items()},
             **{f"K31_{k}": v2 for k, v2 in SAVE.get("K31", {}).items()},
             **{f"k3impl_{k}": v2
                for k, v2 in SAVE.get("k3impl", {}).items()})
    out["total_wall_s"] = round(time.time() - t00, 1)
    (HERE / f"TENSOR_RECURSION_rank{rank:04d}_v2b.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
