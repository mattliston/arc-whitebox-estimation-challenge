#!/usr/bin/env python3
"""TAG fable-physicist (FO-1): can the card's carried third-order columns also carry the missing FOURTH-order content?
Independent-source picture: K3 = sum_j lam_j p_j^(x)3 (the card's bank), consistent K4 = sum_j lam4_j p_j^(x)4 (same columns, free transport).
Fit lam4 (one free weight per column, oracle) to the 8M truth errors of the pair map's fourth-order inputs at layer L (r35 bank dump,
fable_oldtier_economics_v1/bank_r35/layer_LL.npz pp/lam/response/live; state_LL.npz Cpre/radial), off-diagonal, response rows x live cols:
  E31 = K31_truth - radial v_a C_ac,  E22 = K22_truth - radial (v v + 2 C^2)/3,
weighted by what the covariance output reads at r=0: w31_ac = |alpha_a| pdf_a g_c / v_a, w22_ac = pdf_a pdf_c / (s_a s_c).
Model: (A31 lam4)_ac = sum_j lam4_j p_aj^3 p_cj, (A22 lam4)_ac = sum_j lam4_j p_aj^2 p_cj^2. Conjugate gradients on the normal equations.
Reports the weighted error energy explained for K31 and K22 by a JOINT lam4 (consistency), and by separate fits, for column subsets
young / old / all. usage: col_lambda4_fit.py <layers> <subsets e.g. young,all> <iters>"""
import sys, json
from pathlib import Path
import numpy as np
from math import erf
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]; G19 = P2 / 'public_g19_attribution_v1'
D = P2 / 'fable_oldtier_economics_v1' / 'bank_r35'
MP = np.load(G19 / 'mps_planes_rank0035.npz')
verf = np.vectorize(erf)
layers = [int(x) for x in sys.argv[1].split(',')]; subsets = sys.argv[2].split(','); iters = int(sys.argv[3])
def cg(apply_N, b, x0, iters):
    x = x0.copy(); r = b - apply_N(x); pvec = r.copy(); rs = r @ r
    for _ in range(iters):
        Ap = apply_N(pvec); alpha = rs / (pvec @ Ap); x += alpha * pvec; r -= alpha * Ap
        rs_new = r @ r
        if rs_new < 1e-30: break
        pvec = r + (rs_new / rs) * pvec; rs = rs_new
    return x
for L in layers:
    z = np.load(D / f'layer_{L:02d}.npz'); st = np.load(D / f'state_{L:02d}.npz')
    resp, live = z['response'], z['live']; no = int(z['old_columns'])
    C = st['Cpre'].astype(np.float64); v = np.diag(C).copy(); s = np.sqrt(v); mu = st['mu'].astype(np.float64); rad = float(st['radial'])
    al = mu / s; pdf = np.exp(-al * al / 2) / np.sqrt(2 * np.pi); g = 0.5 * (1 + verf(al / np.sqrt(2)))
    Cb = C[np.ix_(resp, live)]
    off = (resp[:, None] != live[None, :])
    E31 = (MP[f'K31_{L}'].astype(np.float64)[np.ix_(resp, live)] - rad * v[resp][:, None] * Cb) * off
    E22 = (MP[f'K22_{L}'].astype(np.float64)[np.ix_(resp, live)] - rad * (np.outer(v[resp], v[live]) + 2 * Cb * Cb) / 3) * off
    W31 = (np.abs(al[resp]) * pdf[resp] / v[resp])[:, None] * g[live][None, :] * off
    W22 = np.outer(pdf[resp] / s[resp], pdf[live] / s[live]) * off
    e31, e22 = W31 * E31, W22 * E22
    for sub in subsets:
        cols = slice(0, no) if sub == 'old' else (slice(no, None) if sub == 'young' else slice(None))
        P = z['pp'][:, cols].astype(np.float64); Pr, Pl = P[resp], P[live]
        Pr3, Pr2, Pl2 = Pr ** 3, Pr ** 2, Pl ** 2
        def A31(x): return W31 * ((Pr3 * x) @ Pl.T)
        def A22(x): return W22 * ((Pr2 * x) @ Pl2.T)
        def A31T(R): return ((W31 * R) @ Pl * Pr3).sum(0)
        def A22T(R): return ((W22 * R) @ Pl2 * Pr2).sum(0)
        res = dict(layer=L, subset=sub, n_cols=int(P.shape[1]))
        for mode in ('31', '22', 'joint'):
            use31, use22 = mode in ('31', 'joint'), mode in ('22', 'joint')
            def apply_N(x):
                out = np.zeros_like(x)
                if use31: out += A31T(A31(x))
                if use22: out += A22T(A22(x))
                return out
            b = (A31T(e31) if use31 else 0) + (A22T(e22) if use22 else 0)
            lam4 = cg(apply_N, b, np.zeros(P.shape[1]), iters)
            r31 = e31 - A31(lam4); r22 = e22 - A22(lam4)
            res[mode] = dict(expl31=round(1 - float((r31 ** 2).sum() / (e31 ** 2).sum()), 4), expl22=round(1 - float((r22 ** 2).sum() / (e22 ** 2).sum()), 4),
                             lam4_over_lam_median=float(np.median(lam4 / np.where(np.abs(z['lam'][cols]) > 0, z['lam'][cols], 1))))
        print(json.dumps(res), flush=True)
