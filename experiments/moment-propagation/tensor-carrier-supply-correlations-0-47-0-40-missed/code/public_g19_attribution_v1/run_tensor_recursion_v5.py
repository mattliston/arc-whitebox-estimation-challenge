#!/usr/bin/env python3
"""TAG g19f-tensor-recursion-v1: full third-cumulant tensor recursion (MPS).
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
DTOT = 6
W_ = 1024
SLAB = 64
CHUNK = 131072


def columns(m, v, qmax=26):
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

    M21 = series(c2, c1, 0, 0) + dM(c2, c1)
    M11 = series(c1, c1, 0, 0) + dM(c1, c1)
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
    Pl = np.load(HERE / f"mps_planes_rank{rank:04d}.npz")
    P1 = np.load(HERE / f"mps_planes_rank{rank:04d}_pre1.npz")
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    weights = np.load((STORE / entry["path"]).parent / "weights.npy",
                      mmap_mode="r")

    SAVE = {}
    Tt = torch.zeros((W_, W_, W_), dtype=STOREDT, device=dev)
    preK21 = np.zeros((W_, W_))
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
        else:
            m = T["mc_pre_m"][l].astype(np.float64)
            v = T["mc_pre_v"][l].astype(np.float64)
            C = T["mc_pre_cov"][l].astype(np.float64)
            k3u = T["mc_pre_k3"][l].astype(np.float64)
            k4u = T["mc_pre_k4"][l].astype(np.float64)
            K22c = z["k22"][l - 1].astype(np.float64)
            K31c = z["k31"][l - 1].astype(np.float64)
        sd = np.sqrt(np.maximum(v, 1e-30))
        al = m / sd
        pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
        cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
        rho = C / np.outer(sd, sd)
        np.fill_diagonal(rho, 0.0)
        qmax = 2 * DTOT + 8
        ma = -al
        He = [np.ones_like(al), ma]
        for k in range(2, qmax + 2):
            He.append(ma * He[-1] - (k - 1) * He[-2])
        c1l = [sd * pdf + m * cdf, sd * cdf]
        for q in range(2, qmax + 1):
            c1l.append(sd * He[q - 2] * pdf)
        a_base = np.stack(c1l)
        b_base = np.stack([cdf] + [He[t - 1] * pdf for t in range(1, qmax + 1)])
        d_base = np.stack([He[t] * pdf / sd for t in range(0, qmax + 1)])
        g3v = k3u / np.maximum(sd ** 3, 1e-30)
        g4v = k4u / np.maximum(sd ** 4, 1e-30)

        def edgeworth_cols(base, tmax):
            from math import comb, factorial as fct
            out_c = []
            for t in range(tmax + 1):
                v_ = base[t].copy()
                for (n, w) in ((3, g3v / 6.0), (4, g4v / 24.0),
                               (6, g3v * g3v / 72.0)):
                    for kk in range(0, min(t, n) + 1):
                        idx2 = t + n - 2 * kk
                        if idx2 <= qmax:
                            v_ = v_ + w * (comb(t, kk) * comb(n, kk)
                                           * fct(kk)) * base[idx2]
                out_c.append(v_)
            return np.stack(out_c)

        a = edgeworth_cols(a_base, 2 * DTOT)
        bE = edgeworth_cols(b_base, 2 * DTOT)
        dE = edgeworth_cols(d_base, 2 * DTOT)
        g = bE[0]

        gt = torch.from_numpy(g.astype(np.float32)).to(dev)
        at = torch.from_numpy(a.astype(np.float32)).to(dev)
        rt = torch.from_numpy(rho.astype(np.float32)).to(dev)
        rpow = {t: rt ** t for t in range(1, DTOT + 1)}
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
        bt = torch.from_numpy(bE[:DTOT + 1].astype(np.float32)).to(dev)
        d0 = torch.from_numpy(dE[0].astype(np.float32)).to(dev)
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
            dt_all = torch.from_numpy(dE[:9].astype(np.float32)).to(dev)
            bt_all = torch.from_numpy(bE[:9].astype(np.float32)).to(dev)
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
            for (tij, tik, tjk, di, dj, dk, coef) in terms:
                t_ = (at[di][sl, None, None] * at[dj][None, :, None]
                      * at[dk][None, None, :]) * coef
                if tij:
                    t_ = t_ * rpow[tij][sl, :][:, :, None]
                if tik:
                    t_ = t_ * rpow[tik][sl, :][:, None, :]
                if tjk:
                    t_ = t_ * rpow[tjk][None, :, :]
                blk += t_
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

        nl = l + 1
        K21new = Tt[idx, idx, :].to(torch.float32).cpu().numpy().astype(np.float64)
        SAVE.setdefault("pre", {})[l + 1] = K21new.astype(np.float32)
        preK21 = K21new
        rec = {"wall_s": round(time.time() - t0, 1)}
        truth = None
        if nl == 1 and "K21_1" in P1.files:
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
        out["layers"][str(nl)] = rec
        print(json.dumps({f"L{nl}": rec}), flush=True)
        (HERE / f"TENSOR_RECURSION_rank{rank:04d}_v5.json").write_text(
            json.dumps({k: v2 for k, v2 in out.items()}, indent=1) + "\n")
    np.savez(HERE / f"tensor_planes_rank{rank:04d}_v5.npz",
             **{f"post_{k}": v2 for k, v2 in SAVE.get("post", {}).items()},
             **{f"pre_{k}": v2 for k, v2 in SAVE.get("pre", {}).items()})
    out["total_wall_s"] = round(time.time() - t00, 1)
    (HERE / f"TENSOR_RECURSION_rank{rank:04d}_v5.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
