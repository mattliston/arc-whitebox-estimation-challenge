#!/usr/bin/env python3
"""TAG tracker-T5 (card-assimilated tracking)
Base: tracker-T4: anchored difference-first supply from TRACKED state.

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


def assimilate(A, C_impl, C_card, eps=1e-3):
    """One Gauss-Newton step attributing the C mismatch to A."""
    E = C_card - C_impl
    G = A.T @ A
    G[np.diag_indices_from(G)] += eps * np.trace(G) / G.shape[0]
    return A + 0.5 * E @ A @ np.linalg.inv(G)


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
        # deploy-legal anchor: the CARD's carried planes
        hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
        C_card = hc["cov"][cur].astype(np.float64)
        from run_tracker_t2 import pair_gram as _pg
        import os
        lamA = float(os.environ.get("LAMA", "0.5"))
        kspl = int(os.environ.get("KSPL", "0"))
        lamLo = float(os.environ.get("LAMLO", "0.3"))
        C_impl = A @ A.T + 0.5 * _pg(dicts, pairs) + C_extra
        E = C_card - C_impl
        if kspl > 0:
            Es = 0.5 * (E + E.T)
            w_, V_ = np.linalg.eigh(Es)
            idx = np.argsort(np.abs(w_))[::-1]
            top = idx[:kspl]
            Ebulk = (V_[:, top] * w_[top]) @ V_[:, top].T
            E = 1.0 * Ebulk + lamLo / lamA * (E - Ebulk)
        G = A.T @ A
        G[np.diag_indices_from(G)] += 1e-3 * np.trace(G) / G.shape[0]
        A = A + lamA * E @ A @ np.linalg.inv(G)
        Cexact = C_card
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
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
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
    # EXACT-regime H3 word-Gram on the assimilated state (certified
    # birth words, CF-exact scalars; replaces the copula Ttail):
    # T_c = E3 a3 + 3E4 sym(a,a,Ma) + 3E2 sym(M,a) + 3E3 sym(M,Ma)
    #       + 3E3 sym(M2,a); Gram via word-pair trace algebra.
    # Needed vectors: a (=A), Ma (=Ma_rows/TAU-consistent), M2a.
    M2a = np.zeros((n, n))
    for c0, Mch in mps_materialize(dicts, pairs, chunk=64):
        for b in range(Mch.shape[0]):
            M2a[c0 + b] = Mch[b] @ Ma_rows[c0 + b]
    E2v, E3v, E4v = S[:, 1], -S[:, 2], S[:, 3]
    # word coefficient vectors in the shared frame; Gram terms are
    # combinations of dots among {a, Ma, M2a} and PG entries.
    d_aa = A @ A.T
    d_aMa = A @ Ma_rows.T
    d_MaMa = Ma_rows @ Ma_rows.T
    d_aM2a = A @ M2a.T
    d_MaM2a = Ma_rows @ M2a.T
    d_M2aM2a = M2a @ M2a.T
    an2 = np.einsum('ci,ci->c', A, A)
    # leading Gram terms (dense-coef inner products, dominant set):
    # <a3,a3> = (a.a')^3 ; <sym(M,a),sym(M,a)> ~ PG*(a.a') + 2(Ma.M'a')
    # cross terms folded with their word scalars:
    G3w = (np.outer(E3v, E3v) * d_aa ** 3 / 6.0
           + np.outer(E2v, E2v) * (PG * d_aa + 2 * d_aMa * d_aMa.T)
           / 6.0
           + np.outer(E3v, E3v) * (PG * d_MaMa
                                   + 2 * d_MaM2a * 0
                                   + 2 * d_aM2a * d_aM2a.T) / 6.0
           + np.outer(E2v, E3v) * (PG * d_aMa.T
                                   + 2 * d_aMa * d_MaMa) / 6.0
           + np.outer(E3v, E2v) * (PG * d_aMa
                                   + 2 * d_aMa.T * d_MaMa) / 6.0)
    Cpost_mine = Cpost2 + G3w
    np.fill_diagonal(Cpost_mine, VD)
    XPASS = np.zeros((n, n))
    Ttail = G3w
    # ---- matched closure side on tracker-implied PRE stats
    C_card = hc["cov"][L - 1].astype(np.float64)
    K21c = hc["k21"][L - 1].astype(np.float64)
    K22c = hc["k22"][L - 1].astype(np.float64)
    K31c = hc["k31"][L - 1].astype(np.float64)
    mc = st["host_m"][L - 1].astype(np.float64)
    vc = st["host_v"][L - 1].astype(np.float64)
    k3c = st["host_k3"][L - 1].astype(np.float64)
    k4c = st["host_k4"][L - 1].astype(np.float64)
    mu_eng = engine_post_cov(mc, vc, k3c, k4c, C_card, K21c, K22c, K31c)
    dP = Cpost_mine - mu_eng
    Wp = np.asarray(W[L + 1], dtype=np.float64)
    dv_c = np.einsum("ac,ab,bc->c", Wp, dP, Wp, optimize=True)
    tv = tr["mc_pre_v"][L + 1].astype(np.float64)
    hv = st["host_v"][L].astype(np.float64)
    dv_true = tv - hv
    dvh = np.load(HERE / f"dvhat_rank{RANK:04d}.npz")[f"L{L + 1}"]
    beta = float(np.dot(dvh, dv_true) / np.dot(dvh, dvh))
    resid = dv_true - beta * dvh
    def pcorr(M):
        v = np.einsum("ac,ab,bc->c", Wp, M, Wp, optimize=True)
        return round(float(np.corrcoef(v, dv_true)[0, 1]), 4)
    base_part = Cpost2 - mu_eng
    np.fill_diagonal(base_part, 0.0)
    tt = Ttail.copy(); np.fill_diagonal(tt, 0.0)
    xp = XPASS.copy(); np.fill_diagonal(xp, 0.0)
    dg = np.diag(np.diag(Cpost_mine - mu_eng))
    def pvec(M):
        return np.einsum("ac,ab,bc->c", Wp, M, Wp, optimize=True)
    vb = pvec(base_part)
    vt = pvec(tt)
    # channel 3: the K21r five-family delta on the assimilated state.
    # word-T dense tensors are implicit; the五 families reduce to
    # vector/PG contractions with the same word set:
    # <T_b, a_a x B_a>-class terms with B ~ pz*M + p0*aa^T and
    # T ~ the word model -> all entries are combinations of
    # {d_aa, d_aMa, d_MaMa, d_aM2a, PG} with scalar weights.
    pzv = S[:, 0]
    p0v = S[:, 1]
    K21w = (np.outer(E3v * an2, pzv) * d_aMa
            + np.outer(E2v, pzv) * (PG * 0 + d_aMa.T * d_aa)
            + np.outer(E3v, p0v) * (d_aa ** 2) * d_aMa.T
            + np.outer(E2v * an2, p0v) * d_aa)
    K21e_base = (np.einsum('iab,jab->ij',
                           np.zeros((1, 1, 1)),
                           np.zeros((1, 1, 1)))
                 if False else 0.0)
    dK21 = K21w  # difference-first vs the closure's K21-emission is
    # implicit in the base channel; grade the raw shape:
    v3 = pvec(0.5 * (dK21 + dK21.T))
    X2 = np.stack([vb, vt, v3], 1)
    b2, *_ = np.linalg.lstsq(X2, dv_true, rcond=None)
    joint = X2 @ b2
    out = {
        "joint_corr": round(float(
            np.corrcoef(joint, dv_true)[0, 1]), 4),
        "base_x_G3_corr": round(float(
            np.corrcoef(vb, vt)[0, 1]), 4),
        "joint_betas": [round(float(x), 4) for x in b2],
        "comp_K21w": round(float(np.corrcoef(v3, dv_true)[0, 1]), 4),
        "comp_base": pcorr(base_part),
        "comp_Ttail": pcorr(tt),
        "comp_Cextra": pcorr(xp),
        "comp_diag": pcorr(dg),
        "corr_vs_dvtrue": round(float(
            np.corrcoef(dv_c, dv_true)[0, 1]), 4),
        "corr_vs_resid": round(float(
            np.corrcoef(dv_c, resid)[0, 1]), 4),
        "chain_corr": round(float(
            np.corrcoef(dvh, dv_true)[0, 1]), 4),
        "parity_ref_fitted_cores": 0.18,
    }
    print(RANK, f"L{L}", json.dumps(out), flush=True)
    (HERE / f"TRACKER_T5_rank{RANK:04d}_L{L}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
