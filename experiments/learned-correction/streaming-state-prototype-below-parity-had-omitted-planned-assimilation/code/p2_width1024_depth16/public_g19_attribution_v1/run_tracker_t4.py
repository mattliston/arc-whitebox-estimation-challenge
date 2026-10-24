#!/usr/bin/env python3
"""TAG tracker-T4: anchored difference-first supply from TRACKED state.

At layer l: my pair model (the T-3 readout: order<=2 Grams on the
tracked state + the certified >=3 completion + the tracked C_extra)
MINUS the closure on the tracker-implied stats (matched models) ->
pushed diag -> dv-hat candidate at L(l+1). Graded vs dv_true beside
the chain and the fitted-core parity (0.18). The tracked state is
exact-rooted (zero fitting); its carried error at l is the price of
depth; the C_extra cross-field is information no plane-fit can have.
Usage: run_tracker_t4.py <rank> <layer>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_chaos1024_shallow import hermite_relu_coefs, load_weights
from run_vresid_field import engine_post_cov
from run_tracker_t2 import pair_gram
from fast_engines import (mps_materialize, compress_modes,
                          batch_scalars_trunc)

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
L = int(sys.argv[2]) if len(sys.argv) > 2 else 5
n = 1024


def rebuild_dicts_to(l, W, sig, ahat, al, tr):
    """Re-run the (fast) T-3 recursion to layer l, returning the
    dictionary state + A + mv + C_extra (deterministic replay)."""
    W1 = np.asarray(W[1], dtype=np.float64).T
    Wsig = W1 * sig[None, :]
    mv = al[0] * Wsig.sum(1)
    A = al[1] * (Wsig @ ahat)
    dicts = [ahat]
    pairs = [(0, 0, np.sqrt(2.0) * al[2] * Wsig)]
    rho00 = ahat @ ahat.T
    Cz = np.zeros((n, n))
    rp = np.ones_like(rho00)
    for k in range(1, 13):
        rp = rp * rho00
        Cz += (al[k] ** 2) * (Wsig @ rp @ Wsig.T)
    Cexact = Cz
    C_extra = np.zeros((n, n))
    for cur in range(1, l):
        S = np.zeros((n, 4))
        E1 = np.zeros(n)
        VD = np.zeros(n)
        Ma_rows = np.zeros((n, n))
        K = 128
        for c0, Mch in mps_materialize(dicts, pairs, chunk=64):
            B = Mch.shape[0]
            lt = np.zeros((B, K + 1))
            att = np.zeros((B, K + 1))
            tails = []
            for b in range(B):
                c = c0 + b
                lam, U = np.linalg.eigh(Mch[b])
                at = U.T @ A[c]
                vr = max(float(Cexact[c, c]) - float(A[c] @ A[c])
                         - 0.5 * float((lam ** 2).sum()), 0.0)
                a_, b_, tl = compress_modes(
                    np.concatenate([lam, [0.0]]),
                    np.concatenate([at, [np.sqrt(vr)]]), K + 1)
                lt[b], att[b] = a_, b_
                tails.append(tl)
                Ma_rows[c] = Mch[b] @ A[c]
            e1, vd, pz, p0, p1, p2 = batch_scalars_trunc(
                mv[c0:c0 + B], lt, att, tails)
            E1[c0:c0 + B] = e1
            VD[c0:c0 + B] = vd
            S[c0:c0 + B] = np.stack([pz, p0, p1, p2], axis=1)
        alpha = S[:, 0:1] * A + S[:, 1:2] * Ma_rows
        aG = alpha @ alpha.T
        MMd = np.zeros(n)
        for (d1, d2, w) in pairs:
            G12 = dicts[d1] @ dicts[d2].T
            G11 = dicts[d1] @ dicts[d1].T
            G22 = dicts[d2] @ dicts[d2].T
            if d1 == d2:
                MMd += np.einsum('ci,ij,cj->c', w, G12 * G12, w)
            else:
                MMd += 0.5 * (np.einsum('ci,ij,cj->c', w, G12 * G12, w)
                              + np.einsum('ci,ij,cj->c', w,
                                          G11 * G22, w))
        an2 = np.linalg.norm(A, axis=1) ** 2
        diag2 = (np.einsum('ci,ci->c', alpha, alpha)
                 + 0.5 * (S[:, 0] ** 2) * MMd
                 + 0.5 * (S[:, 1] ** 2) * an2 ** 2)
        tail2 = np.maximum(VD - diag2, 0.0)
        Cpost2 = aG + 0.5 * ((S[:, 0][:, None] * S[:, 0][None, :])
                             * pair_gram(dicts, pairs))
        dpost = np.sqrt(np.maximum(np.diag(Cpost2), 1e-30))
        rt = np.clip(Cpost2 / np.outer(dpost, dpost), -0.999, 0.999)
        bt = np.sqrt(tail2)
        Ttail = np.outer(bt, bt) * rt ** 3
        np.fill_diagonal(Ttail, tail2)
        Wn = np.asarray(W[cur + 1], dtype=np.float64)
        A_prev = A
        Ma_prev = Ma_rows
        mv = Wn.T @ E1
        A = Wn.T @ alpha
        new_pairs = [(d1, d2, Wn.T @ (S[:, 0:1] * w))
                     for (d1, d2, w) in pairs]
        an = np.linalg.norm(A_prev, axis=1) + 1e-30
        Da = A_prev / an[:, None]
        ia = len(dicts)
        dicts.append(Da)
        new_pairs.append((ia, ia, Wn.T * (S[:, 1] * an ** 2)[None, :]))
        mn = np.linalg.norm(Ma_prev, axis=1) + 1e-30
        DMa = Ma_prev / mn[:, None]
        im = len(dicts)
        dicts.append(DMa)
        new_pairs.append((ia, im,
                          Wn.T * (-S[:, 2] * an * mn * 2.0)[None, :]))
        pairs = new_pairs
        C_extra = Wn.T @ Ttail @ Wn
        Cexact = tr["mc_pre_cov"][cur + 1].astype(np.float64)
    return dicts, pairs, A, mv, C_extra, Cexact


def main():
    W = load_weights(RANK)
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    v0 = tr["mc_pre_v"][0].astype(np.float64)
    W0 = np.asarray(W[0], dtype=np.float64)
    if np.corrcoef(np.sum(W0 ** 2, 0), v0)[0, 1] > \
            np.corrcoef(np.sum(W0 ** 2, 1), v0)[0, 1]:
        W0 = W0.T
    sig = np.linalg.norm(W0, axis=1)
    ahat = W0 / sig[:, None]
    al = hermite_relu_coefs(12)
    dicts, pairs, A, mv, C_extra, Cexact = rebuild_dicts_to(
        L, W, sig, ahat, al, tr)
    # ---- my POST pair model at layer L (T-3 readout, one step)
    S = np.zeros((n, 4))
    E1 = np.zeros(n)
    VD = np.zeros(n)
    Ma_rows = np.zeros((n, n))
    K = 128
    for c0, Mch in mps_materialize(dicts, pairs, chunk=64):
        B = Mch.shape[0]
        lt = np.zeros((B, K + 1))
        att = np.zeros((B, K + 1))
        tails = []
        for b in range(B):
            c = c0 + b
            lam, U = np.linalg.eigh(Mch[b])
            at = U.T @ A[c]
            vr = max(float(Cexact[c, c]) - float(A[c] @ A[c])
                     - 0.5 * float((lam ** 2).sum()), 0.0)
            a_, b_, tl = compress_modes(
                np.concatenate([lam, [0.0]]),
                np.concatenate([at, [np.sqrt(vr)]]), K + 1)
            lt[b], att[b] = a_, b_
            tails.append(tl)
            Ma_rows[c] = Mch[b] @ A[c]
        e1, vd, pz, p0, p1, p2 = batch_scalars_trunc(
            mv[c0:c0 + B], lt, att, tails)
        E1[c0:c0 + B] = e1
        VD[c0:c0 + B] = vd
        S[c0:c0 + B] = np.stack([pz, p0, p1, p2], axis=1)
    alpha = S[:, 0:1] * A + S[:, 1:2] * Ma_rows
    aG = alpha @ alpha.T
    PG = pair_gram(dicts, pairs)
    Cpost2 = aG + 0.5 * ((S[:, 0][:, None] * S[:, 0][None, :]) * PG)
    MMd = np.diag(PG).copy()
    an2 = np.linalg.norm(A, axis=1) ** 2
    diag2 = (np.einsum('ci,ci->c', alpha, alpha)
             + 0.5 * (S[:, 0] ** 2) * MMd
             + 0.5 * (S[:, 1] ** 2) * an2 ** 2)
    tail2 = np.maximum(VD - diag2, 0.0)
    dpost = np.sqrt(np.maximum(np.diag(Cpost2), 1e-30))
    rt = np.clip(Cpost2 / np.outer(dpost, dpost), -0.999, 0.999)
    bt = np.sqrt(tail2)
    Ttail = np.outer(bt, bt) * rt ** 3
    np.fill_diagonal(Ttail, tail2)
    # passthrough of the tracked >=3 PRE cross-field through phi:
    Cpost_mine = Cpost2 + Ttail \
        + (S[:, 0][:, None] * S[:, 0][None, :]) * C_extra
    np.fill_diagonal(Cpost_mine, VD)
    # ---- matched closure side on tracker-implied PRE stats
    C_impl = A @ A.T + 0.5 * PG + C_extra
    np.fill_diagonal(C_impl, np.diag(Cexact))
    v_impl = np.diag(C_impl).copy()
    z0 = np.zeros(n)
    Z0 = np.zeros((n, n))
    mu_eng = engine_post_cov(mv, v_impl, z0, z0, C_impl, Z0, Z0, Z0)
    dP = Cpost_mine - mu_eng
    Wp = np.asarray(W[L + 1], dtype=np.float64)
    dv_c = np.einsum("ac,ab,bc->c", Wp, dP, Wp, optimize=True)
    tv = tr["mc_pre_v"][L + 1].astype(np.float64)
    hv = st["host_v"][L].astype(np.float64)
    dv_true = tv - hv
    dvh = np.load(HERE / f"dvhat_rank{RANK:04d}.npz")[f"L{L + 1}"]
    beta = float(np.dot(dvh, dv_true) / np.dot(dvh, dvh))
    resid = dv_true - beta * dvh
    out = {
        "corr_vs_dvtrue": round(float(
            np.corrcoef(dv_c, dv_true)[0, 1]), 4),
        "corr_vs_resid": round(float(
            np.corrcoef(dv_c, resid)[0, 1]), 4),
        "chain_corr": round(float(
            np.corrcoef(dvh, dv_true)[0, 1]), 4),
        "parity_ref_fitted_cores": 0.18,
    }
    print(RANK, f"L{L}", json.dumps(out), flush=True)
    (HERE / f"TRACKER_T4_rank{RANK:04d}_L{L}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
