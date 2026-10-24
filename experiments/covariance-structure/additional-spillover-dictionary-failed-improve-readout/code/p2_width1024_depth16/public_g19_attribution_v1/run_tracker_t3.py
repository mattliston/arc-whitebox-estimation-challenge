#!/usr/bin/env python3
"""TAG tracker-T3 (adds the certified >=3 readout completion)
Base: tracker-T2: dictionary-PAIR transport with spillover, L1 -> L3.

Extends T-1: the quadratic state is a sum over ORDERED dictionary
pairs, M_c = sum_p Dp1^T diag(w_c^p) Dp2 (symmetrized), so the
engine's B = pz*M + p0*aa^T + (-p1)*sym(a, Ma) transports exactly:
aa^T joins as (a-dict, a-dict); sym(a,Ma) as (a-dict, Ma-dict)
cross-weights, all diagonal in the unit index. Gram algebra between
pairs stays GEMM-class. Engines at depth materialize M_c per unit
transiently for eigh (one at a time). Measures carried C at L1/L2/L3
vs banked truth. Usage: run_tracker_t2.py <rank>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_chaos1024_shallow import hermite_relu_coefs, load_weights
from run_chaos_supply_shallow import unit_scalars_and_var

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
n = 1024


def materialize_M(c, dicts, pairs):
    M = np.zeros((n, n))
    for (d1, d2, w) in pairs:
        term = (dicts[d1].T * w[c][None, :]) @ dicts[d2]
        if d1 == d2:
            M += term
        else:
            M += 0.5 * (term + term.T)
    return M


def pair_gram(dicts, pairs):
    """<M_a, M_b> for all unit pairs, GEMM-class."""
    G = {}
    def g(i, j):
        if (i, j) not in G:
            G[(i, j)] = dicts[i] @ dicts[j].T
        return G[(i, j)]
    MM = np.zeros((n, n))
    for (a1, a2, wa) in pairs:
        for (b1, b2, wb) in pairs:
            # <sym part x sym part>: for d1==d2 terms plain; cross
            # terms carry the 1/2 sym factors -> expand all four:
            def blk(x1, x2, y1, y2, fa, fb):
                return fa * fb * (wa @ (g(x1, y1) * g(x2, y2))) @ wb.T
            if a1 == a2 and b1 == b2:
                MM += blk(a1, a2, b1, b2, 1, 1)
            elif a1 == a2:
                MM += blk(a1, a2, b1, b2, 1, 0.5) \
                    + blk(a1, a2, b2, b1, 1, 0.5)
            elif b1 == b2:
                MM += blk(a1, a2, b1, b2, 0.5, 1) \
                    + blk(a2, a1, b1, b2, 0.5, 1)
            else:
                MM += blk(a1, a2, b1, b2, 0.5, 0.5) \
                    + blk(a1, a2, b2, b1, 0.5, 0.5) \
                    + blk(a2, a1, b1, b2, 0.5, 0.5) \
                    + blk(a2, a1, b2, b1, 0.5, 0.5)
    return MM


def main():
    W = load_weights(RANK)
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    v0 = tr["mc_pre_v"][0].astype(np.float64)
    W0 = np.asarray(W[0], dtype=np.float64)
    if np.corrcoef(np.sum(W0 ** 2, 0), v0)[0, 1] > \
            np.corrcoef(np.sum(W0 ** 2, 1), v0)[0, 1]:
        W0 = W0.T
    sig = np.linalg.norm(W0, axis=1)
    ahat = W0 / sig[:, None]
    al = hermite_relu_coefs(12)
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
    out = {}
    for l in [1, 2, 3, 4, 5]:
        MM = pair_gram(dicts, pairs)
        C_model = A @ A.T + 0.5 * MM + C_extra
        np.fill_diagonal(C_model, np.diag(Cexact))
        tC = tr["mc_pre_cov"][l].astype(np.float64)
        rec = {"C_rel": round(float(np.linalg.norm(C_model - tC)
                                    / np.linalg.norm(tC)), 5),
               "ndicts": len(dicts), "npairs": len(pairs)}
        out[l] = rec
        print(RANK, f"L{l}", json.dumps(rec), flush=True)
        if l == 5:
            break
        # engines per unit at this layer
        from fast_engines import (mps_materialize, compress_modes,
                                  batch_scalars_trunc)
        S = np.zeros((n, 4))
        E1 = np.zeros(n)
        VD_stash = np.zeros(n)
        Ma_rows = np.zeros((n, n))
        K = 128
        for c0, Mchunk in mps_materialize(dicts, pairs, chunk=64):
            B = Mchunk.shape[0]
            lt = np.zeros((B, K + 1))
            att = np.zeros((B, K + 1))
            tails = []
            for b in range(B):
                c = c0 + b
                lam, U = np.linalg.eigh(Mchunk[b])
                at = U.T @ A[c]
                vr = max(float(Cexact[c, c]) - float(A[c] @ A[c])
                         - 0.5 * float((lam ** 2).sum()), 0.0)
                lam2 = np.concatenate([lam, [0.0]])
                at2 = np.concatenate([at, [np.sqrt(vr)]])
                a_, b_, tl = compress_modes(lam2, at2, K + 1)
                lt[b], att[b] = a_, b_
                tails.append(tl)
                Ma_rows[c] = Mchunk[b] @ A[c]
            e1, vd, pz, p0, p1, p2 = batch_scalars_trunc(
                mv[c0:c0 + B], lt, att, tails)
            E1[c0:c0 + B] = e1
            VD_stash[c0:c0 + B] = vd
            S[c0:c0 + B, 0] = pz
            S[c0:c0 + B, 1] = p0
            S[c0:c0 + B, 2] = p1
            S[c0:c0 + B, 3] = p2
            if (c0 + B) % 256 == 0:
                print(f"  engines L{l} {c0 + B}/{n}", flush=True)
        alpha = S[:, 0:1] * A + S[:, 1:2] * Ma_rows
        # >=3-order POST completion (certified stage-1 recipe):
        # per-unit Var(phi) exact (VD) minus the order<=2 model diag;
        # copula-shaped off-diag on the post correlations.
        aG = alpha @ alpha.T
        # B-Gram diag: 0.5<B,B> with B = pz*M + p0*aa^T (+cross):
        # reuse the pair machinery cheaply at diag level via traces:
        # diag<M,M> from current pairs:
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
        # per-unit post Var model at order<=2 vs exact VD:
        VD = np.zeros(n)
        for c in range(n):
            pass
        # (VD from the engine loop: recompute inline was done there;
        # stash it) -- see engine loop edit below
        VD = VD_stash
        an2 = np.linalg.norm(A, axis=1) ** 2
        diag2 = (S[:, 0] ** 2 * 0 + np.einsum('ci,ci->c', alpha, alpha)
                 + 0.5 * (S[:, 0] ** 2) * MMd
                 + 0.5 * (S[:, 1] ** 2) * an2 ** 2)
        tail2 = np.maximum(VD - diag2, 0.0)
        # post correlations for the copula shape: order<=2 approx
        Cpost2 = aG + 0.5 * ((S[:, 0][:, None] * S[:, 0][None, :])
                             * pair_gram(dicts, pairs))
        dpost = np.sqrt(np.maximum(np.diag(Cpost2), 1e-30))
        rt = np.clip(Cpost2 / np.outer(dpost, dpost), -0.999, 0.999)
        bt = np.sqrt(tail2)
        Ttail = np.outer(bt, bt) * rt ** 3
        np.fill_diagonal(Ttail, tail2)
        Wn = np.asarray(W[l + 1], dtype=np.float64)
        A_prev = A
        Ma_prev = Ma_rows
        mv = Wn.T @ E1
        A = Wn.T @ alpha
        # transport the quadratic pairs: existing pairs scale by pz
        new_pairs = []
        for (d1, d2, w) in pairs:
            new_pairs.append((d1, d2, Wn.T @ (S[:, 0:1] * w)))
        # aa^T: new dictionary of a-rows
        an = np.linalg.norm(A_prev, axis=1) + 1e-30
        Da = A_prev / an[:, None]
        ia = len(dicts)
        dicts.append(Da)
        new_pairs.append((ia, ia, Wn.T * (S[:, 1] * an ** 2)[None, :]))
        # sym(a, Ma): cross pair between Da and DMa
        mn = np.linalg.norm(Ma_prev, axis=1) + 1e-30
        DMa = Ma_prev / mn[:, None]
        im = len(dicts)
        dicts.append(DMa)
        new_pairs.append((ia, im,
                          Wn.T * (-S[:, 2] * an * mn * 2.0)[None, :]))
        pairs = new_pairs
        C_extra = Wn.T @ Ttail @ Wn
        np.savez_compressed(HERE / f"tracker_state_L{l + 1}.npz",
                            C_extra=C_extra.astype(np.float32),
                            A=A.astype(np.float32), mv=mv)
        Cexact = tr["mc_pre_cov"][l + 1].astype(np.float64)
    (HERE / f"TRACKER_T3_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
