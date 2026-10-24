#!/usr/bin/env python3
"""TAG tracker-T2: dictionary-PAIR transport with spillover, L1 -> L3.

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
    out = {}
    for l in [1, 2, 3]:
        MM = pair_gram(dicts, pairs)
        C_model = A @ A.T + 0.5 * MM
        np.fill_diagonal(C_model, np.diag(Cexact))
        tC = tr["mc_pre_cov"][l].astype(np.float64)
        rec = {"C_rel": round(float(np.linalg.norm(C_model - tC)
                                    / np.linalg.norm(tC)), 5),
               "ndicts": len(dicts), "npairs": len(pairs)}
        out[l] = rec
        print(RANK, f"L{l}", json.dumps(rec), flush=True)
        if l == 3:
            break
        # engines per unit at this layer
        S = np.zeros((n, 4))
        E1 = np.zeros(n)
        Ma_rows = np.zeros((n, n))
        for c in range(n):
            Mc = materialize_M(c, dicts, pairs)
            lam, U = np.linalg.eigh(Mc)
            at = U.T @ A[c]
            vr = max(float(Cexact[c, c]) - float(A[c] @ A[c])
                     - 0.5 * float((lam ** 2).sum()), 0.0)
            at2 = np.concatenate([at, [np.sqrt(vr)]])
            lam2 = np.concatenate([lam, [0.0]])
            e1, vd, pz, p0, p1, p2 = unit_scalars_and_var(
                float(mv[c]), lam2, at2)
            E1[c] = e1
            S[c] = (pz, p0, p1, p2)
            Ma_rows[c] = Mc @ A[c]
            if (c + 1) % 256 == 0:
                print(f"  engines L{l} {c + 1}/{n}", flush=True)
        # alpha (Stein words, using materlz-free forms):
        # alpha = pz*a + p0*Ma + (-p1)*M2a ~ keep first two (M2a via
        # a second materialization pass would double cost; measured
        # small at these spectra)
        alpha = S[:, 0:1] * A + S[:, 1:2] * Ma_rows
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
        Cexact = tr["mc_pre_cov"][l + 1].astype(np.float64)
    (HERE / f"TRACKER_T2_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
