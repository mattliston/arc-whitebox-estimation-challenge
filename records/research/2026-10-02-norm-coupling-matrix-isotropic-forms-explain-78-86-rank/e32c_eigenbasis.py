#!/usr/bin/env python3
"""TAG opus-reexam E32c (network 35, numpy): is the norm-coupling matrix Ntilde a SPECTRAL FUNCTION of the covariance?
In the eigenbasis of C (pre-activation covariance of the layer): share of Ntilde's off-diagonal cross-half energy held by
 (a) the diagonal of V' Ntilde V (= any matrix function F(C)), (b) polynomials a tr C x C + b C^2 + c C^3 + d C^4,
 (c) the block of the top-k eigenvectors (two-sided) and the top-k rows (one-sided: V_k V_k' N + N V_k V_k' - two-sided),
and the fitted f(lambda_m) = diag(V' N V)_m against lambda_m for the top modes. Layers 9, 12, 13."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; rank = 35; n = 1024
D = np.load(HERE / f'e32_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); off = ~np.eye(n, dtype=bool)
for l in [int(x) for x in D['layers']]:
    C = MC[f'cov_pre_{l}'].astype(np.float64); mu = MC[f'mean_pre_{l}'].astype(np.float64); C2 = C @ C
    N = [(lambda x: (x + x.T) / 2)(D[f'M_{l}_{h}'] - 2 * C2) for h in (0, 1)]
    tot = float((N[0][off] * N[1][off]).sum())
    lam, V = np.linalg.eigh(C); lam, V = lam[::-1], V[:, ::-1]
    A = [V.T @ x @ V for x in N]
    def share(recon):            # recon: list of two matrices (from half 0 and half 1), cross-half explained share of the off-diagonal energy
        return round(float(np.mean([(2 * (recon[h][off] * N[1 - h][off]).sum() - (recon[h][off] ** 2).sum()) / tot for h in (0, 1)])), 4)
    out = dict(layer=l, top_eigenvalues_of_C=[float('%.3g' % x) for x in lam[:6]], trC=float('%.4g' % lam.sum()))
    out['diagonal_in_eigenbasis_of_C'] = share([(V * np.diag(a)) @ V.T for a in A])
    for name, pows in (('trC_C + C2', (1, 2)), ('C..C3', (1, 2, 3)), ('C..C4', (1, 2, 3, 4))):
        X = np.stack([np.linalg.matrix_power(C, p)[off] for p in pows], 1); rec = []
        for h in (0, 1):
            co = np.linalg.lstsq(X, N[h][off], rcond=None)[0]; r = np.zeros((n, n)); r[off] = X @ co; rec.append(r)
        out[name] = share(rec)
    mumu = np.outer(mu, mu); Cm = C @ mu; mc = np.outer(mu, Cm) + np.outer(Cm, mu)
    for name, extra in (('diag_eig + mean forms', True),):
        rec = []
        for h in (0, 1):
            base = (V * np.diag(A[h])) @ V.T; X = np.stack([mumu[off], mc[off]], 1); co = np.linalg.lstsq(X, (N[h] - base)[off], rcond=None)[0]; r = base.copy(); r[off] += X @ co; rec.append(r)
        out[name] = share(rec)
    for k in (1, 2, 4, 8, 16, 32, 64):
        two = [V[:, :k] @ a[:k, :k] @ V[:, :k].T for a in A]
        one = [(V[:, :k] @ a[:k, :]) @ V.T + V @ a[:, :k] @ V[:, :k].T - t for a, t in zip(A, two)]
        dg = [(V * np.diag(a)) @ V.T for a in A]
        mix = [o + d_ - (V[:, :k] * np.diag(a)[:k]) @ V[:, :k].T for o, d_, a in zip(one, dg, A)]
        out[f'top{k}'] = dict(two_sided=share(two), one_sided=share(one), one_sided_plus_diagonal=share(mix))
    f = (np.diag(A[0]) + np.diag(A[1])) / 2
    out['f_over_lambda_trC (isotropic model: constant radial/3)'] = [float('%.3g' % (f[m] / (lam[m] * lam.sum()))) for m in (0, 1, 2, 3, 5, 10, 20, 50, 100, 300, 600, 900)]
    print(json.dumps(out))
