#!/usr/bin/env python3
"""TAG pair-exact-probe: information-limit fork for the pair closure.

Per probe pair (a,b) at layer l: E[relu(z_a) relu(z_b)] three ways:
  (i)   truth (banked mc_post_cov)
  (ii)  the closure's Mehler-series entry (engine_post_cov)
  (iii) EXACT 2D quadrature over the bivariate Gram-Charlier density
        built from the SAME carried information (pair-restricted
        cumulants: m, v, C_ab, k3, K21 entries, k4, K22, K31 entries)
If (iii)~(ii): the deficit is INFORMATION-limited -> state must grow
(the tracker). If (iii) closes toward (i): richer consumption series
on the EXISTING carried state pays (cheap build). Probes: top-|R|
pairs + random off-diag pairs. Usage: run_pair_exact_probe.py <rank>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_vresid_field import engine_post_cov

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
LAYERS = [5, 8, 11, 13]
NTOP, NRAND = 80, 80
GH = 120


def pair_exact(ma, mb, Cab2, k3t, k4t):
    """E[relu(z_a)relu(z_b)] under 2D GC density with given cumulants.

    Cab2: 2x2 cov; k3t: (2,2,2) sym tensor; k4t: (2,2,2,2) sym."""
    L = np.linalg.cholesky(Cab2)
    Li = np.linalg.inv(L)
    c3 = np.einsum("ia,jb,kc,abc->ijk", Li, Li, Li, k3t)
    c4 = np.einsum("ia,jb,kc,ld,abcd->ijkl", Li, Li, Li, Li, k4t)
    x, w = np.polynomial.hermite_e.hermegauss(GH)
    w = w / np.sqrt(2 * np.pi)          # so sum w f = E under N(0,1)
    w = w / w.sum()
    Y1, Y2 = np.meshgrid(x, x, indexing="ij")
    W2 = np.outer(w, w)
    He = {}
    for d, Y in ((0, Y1), (1, Y2)):
        He[(d, 0)] = np.ones_like(Y)
        He[(d, 1)] = Y
        He[(d, 2)] = Y * Y - 1
        He[(d, 3)] = Y ** 3 - 3 * Y
        He[(d, 4)] = Y ** 4 - 6 * Y * Y + 3
    # GC correction: sum_{|alpha|=3} c3[alpha]/alpha! He_alpha +
    # order-4 analog. alpha over 2 dims: (i,j) with i+j=3, 4.
    from math import factorial
    corr = np.ones_like(Y1)
    for order, ct in ((3, c3), (4, c4)):
        for i in range(order + 1):
            j = order - i
            # coefficient = multinomial count * tensor entry /order!
            # tensor is symmetric; number of index-arrangements with
            # i of dim0, j of dim1 = C(order, i)
            from math import comb
            idx = tuple([0] * i + [1] * j)
            corr = corr + (comb(order, i) * float(ct[idx])
                           / factorial(order)) \
                * He[(0, i)] * He[(1, j)]
    Z1 = L[0, 0] * Y1 + ma              # L lower-tri: z1 = L00 y1
    Z2 = L[1, 0] * Y1 + L[1, 1] * Y2 + mb
    val = np.sum(W2 * np.maximum(Z1, 0) * np.maximum(Z2, 0) * corr)
    m1 = np.sum(W2 * np.maximum(Z1, 0) * corr)
    m2 = np.sum(W2 * np.maximum(Z2, 0) * corr)
    return float(val - m1 * m2)


def main():
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    RF = np.load(HERE / f"vresid_field_rank{RANK:04d}.npz")
    rng = np.random.default_rng(5)
    out = {}
    for l in LAYERS:
        i = l - 1
        m_ = st["host_m"][i].astype(np.float64)
        v_ = st["host_v"][i].astype(np.float64)
        k3 = st["host_k3"][i].astype(np.float64)
        k4 = st["host_k4"][i].astype(np.float64)
        C = hc["cov"][i].astype(np.float64)
        K21 = hc["k21"][i].astype(np.float64)
        K22 = hc["k22"][i].astype(np.float64)
        K31 = hc["k31"][i].astype(np.float64)
        mu11 = engine_post_cov(m_, v_, k3, k4, C, K21, K22, K31)
        Tp = tr["mc_post_cov"][l].astype(np.float64)
        R = RF[f"R_{l}"].astype(np.float64)
        iu = np.triu_indices(1024, k=1)
        rmag = np.abs(R[iu])
        top = np.argsort(rmag)[::-1][:NTOP]
        rnd = rng.choice(len(rmag), NRAND, replace=False)
        sel = np.unique(np.concatenate([top, rnd]))
        errs_ser, errs_ex = [], []
        for s in sel:
            a, b = iu[0][s], iu[1][s]
            C2 = np.array([[C[a, a], C[a, b]], [C[a, b], C[b, b]]])
            k3t = np.zeros((2, 2, 2))
            k3t[0, 0, 0] = k3[a]
            k3t[1, 1, 1] = k3[b]
            for p in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
                k3t[p] = K21[a, b]
            for p in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
                k3t[p] = K21[b, a]
            k4t = np.zeros((2, 2, 2, 2))
            k4t[0, 0, 0, 0] = k4[a]
            k4t[1, 1, 1, 1] = k4[b]
            import itertools as it
            for p in set(it.permutations((0, 0, 0, 1))):
                k4t[p] = K31[a, b]
            for p in set(it.permutations((0, 0, 1, 1))):
                k4t[p] = K22[a, b]
            for p in set(it.permutations((0, 1, 1, 1))):
                k4t[p] = K31[b, a]
            ex = pair_exact(m_[a], m_[b], C2, k3t, k4t)
            t = Tp[a, b]
            errs_ser.append(mu11[a, b] - t)
            errs_ex.append(ex - t)
        es = np.array(errs_ser)
        ee = np.array(errs_ex)
        rec = {
            "n_pairs": int(len(sel)),
            "rms_series_err": round(float(np.sqrt(np.mean(es ** 2))), 6),
            "rms_exactGC_err": round(float(np.sqrt(np.mean(ee ** 2))), 6),
            "ratio_exact_over_series": round(float(
                np.sqrt(np.mean(ee ** 2)) / np.sqrt(np.mean(es ** 2))), 3),
            "corr_errs": round(float(np.corrcoef(es, ee)[0, 1]), 3),
        }
        out[l] = rec
        print(RANK, f"L{l}", json.dumps(rec), flush=True)
    (HERE / f"PAIR_EXACT_PROBE_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
