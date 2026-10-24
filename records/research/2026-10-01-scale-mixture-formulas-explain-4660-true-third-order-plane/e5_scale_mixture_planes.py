#!/usr/bin/env python3
"""TAG opus-reexam E5: do the scale-mixture (collective gain) identities explain the TRUE fourth-order planes better than the card's forms?
Model: z = s * zeta, s a scalar gain independent of zeta. Then (to leading order in Var s)
  K31[a,c] = kappa4(a,a,a,c) = R v_a C_ac + (R/4) (3 mu_a K21b[a,c] + mu_c k3b_a) + g3 (v_a mu_a mu_c + C_ac mu_a^2) + g4 mu_a^3 mu_c + bulk
  K22[a,c] = kappa4(a,a,c,c) = (R/3)(v_a v_c + 2 C_ac^2) + (R/2)(mu_c K21b[a,c] + mu_a K21b[c,a]) + g3'(...) + bulk
with R the isotropic excess kurtosis, K21b / k3b the BULK third-order plane = total minus the gain part Sym(C x b), b = beta mu.
Truth: 8M-sample planes (two independent draws A, B at layers 5, 9, 12), 32M means/variances/covariances. Features are built from draw A
and regressed on draw B's plane (independent noise), rows = kink units (|mu/sigma| < 2.33), off-diagonal entries only.
Compares: (i) the card's form [v_a C_ac, v_a C_ac * mu_a k3_a / v_a^2], (ii) total-K21 form (the 09-22 'derived term'), (iii) bulk form,
(iv) bulk form + gain-skew terms. Reports R^2 against draw B, the ceiling (draw A vs draw B), and fitted coefficients vs theory."""
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; G = P2 / 'public_g19_attribution_v1'
A = np.load(G / 'mps_planes_rank0035.npz'); B = np.load(G / 'mps_planes_rank0035_noise2.npz')
t = np.load(G / 'mps_truth_rank0035_P32M.npz'); mc = np.load(P2 / 'fable_covhunter_v1/mcall_rank0035_P32M.npz')
def fit(y, X):
    c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c
    return c, 1 - (r ** 2).sum() / (y ** 2).sum()
for l in (5, 9, 12):
    mu = t['mc_pre_m'][l].astype(np.float64); v = t['mc_pre_v'][l].astype(np.float64); s = np.sqrt(v)
    C = mc[f'cov_pre_{l}'].astype(np.float64)
    k4u = t['mc_pre_k4'][l].astype(np.float64); R = float(np.mean(k4u / v ** 2))
    K21a = A[f'K21_{l}'].astype(np.float64); K31a = A[f'K31_{l}'].astype(np.float64); K22a = A[f'K22_{l}'].astype(np.float64)
    K21b_ = B[f'K21_{l}'].astype(np.float64); K31b_ = B[f'K31_{l}'].astype(np.float64); K22b_ = B[f'K22_{l}'].astype(np.float64)
    al = mu / s; kink = np.abs(al) < 2.33
    off = ~np.eye(1024, dtype=bool); M = off & kink[:, None]
    sc31 = (s ** 3)[:, None] * s[None, :]; sc22 = v[:, None] * v[None, :]; sc21 = v[:, None] * s[None, :]
    # gain carrier: fit beta in K21 ~ beta (v_a mu_c + 2 C_ac mu_a) + rest, on all rows (plane fit)
    Gp = v[:, None] * mu[None, :] + 2 * C * mu[:, None]
    beta = float((K21a[off] * Gp[off]).sum() / (Gp[off] ** 2).sum())
    K21bulk = K21a - beta * Gp
    k3 = t['mc_pre_k3'][l].astype(np.float64); k3bulk = k3 - 3 * beta * v * mu
    print(f'\n===== layer {l}: R (mean excess kurtosis) {R:.4f}, R/6 {R / 6:.5f}, fitted beta {beta:.5f} (ratio to R/6: {beta / (R / 6):.2f}); kink rows {int(kink.sum())}')
    print(f'      share of K21 (kink rows, off-diag, draw A) energy in the gain part: {1 - ((K21bulk / sc21)[M] ** 2).sum() / ((K21a / sc21)[M] ** 2).sum():.3f}')
    y31 = (K31b_ / sc31)[M]; ceil31 = fit(y31, (K31a / sc31)[M][:, None])
    f0 = (v[:, None] * C / sc31)[M]
    f_card = (v[:, None] * C * (mu * k3 / v ** 2)[:, None] / sc31)[M]
    f_tot1 = (mu[:, None] * K21a / sc31)[M]; f_tot2 = (k3[:, None] * mu[None, :] / sc31)[M]
    f_b1 = (mu[:, None] * K21bulk / sc31)[M]; f_b2 = (k3bulk[:, None] * mu[None, :] / sc31)[M]
    f_g1 = ((v * mu)[:, None] * mu[None, :] / sc31)[M]; f_g2 = (C * (mu ** 2)[:, None] / sc31)[M]; f_g3 = ((mu ** 3)[:, None] * mu[None, :] / sc31)[M]
    print(f'  K31 (kink rows): ceiling R2 (draw A plane predicting draw B) {ceil31[1]:.3f}')
    for name, X in (('iso only [v C]', [f0]), ('card form [v C, v C mu k3/v^2]', [f0, f_card]),
                    ('total-K21 form [v C, mu_a K21, mu_c k3]', [f0, f_tot1, f_tot2]),
                    ('bulk form [v C, mu_a K21bulk, mu_c k3bulk]', [f0, f_b1, f_b2]),
                    ('bulk + gain-skew [.., v mu mu, C mu^2, mu^3 mu]', [f0, f_b1, f_b2, f_g1, f_g2, f_g3])):
        c, r2 = fit(y31, np.stack(X, 1))
        print(f'    {name:52s} R2 {r2:.3f} ({r2 / ceil31[1]:.2f} of ceiling)  coef ' + ' '.join(f'{x:+.4f}' for x in c))
    print(f'    theory: v C -> R = {R:.4f}; mu_a K21bulk -> 3R/4 = {0.75 * R:.4f}; mu_c k3bulk -> R/4 = {R / 4:.4f}')
    # K22
    M2 = off & kink[:, None] & kink[None, :]
    y22 = (K22b_ / sc22)[M2]; ceil22 = fit(y22, (K22a / sc22)[M2][:, None])
    g0 = ((v[:, None] * v[None, :] + 2 * C * C) / sc22)[M2]
    g_b = ((mu[None, :] * K21bulk + mu[:, None] * K21bulk.T) / sc22)[M2]
    g_t = ((mu[None, :] * K21a + mu[:, None] * K21a.T) / sc22)[M2]
    g_s1 = ((v[:, None] * (mu ** 2)[None, :] + (mu ** 2)[:, None] * v[None, :]) / sc22)[M2]; g_s2 = (C * mu[:, None] * mu[None, :] / sc22)[M2]
    g_s3 = ((mu ** 2)[:, None] * (mu ** 2)[None, :] / sc22)[M2]
    print(f'  K22 (kink x kink): ceiling R2 {ceil22[1]:.3f}')
    for name, X in (('iso only [(v v + 2 C^2)]', [g0]), ('+ total K21 term', [g0, g_t]), ('+ bulk K21 term', [g0, g_b]),
                    ('+ bulk K21 + gain-skew', [g0, g_b, g_s1, g_s2, g_s3])):
        c, r2 = fit(y22, np.stack(X, 1))
        print(f'    {name:52s} R2 {r2:.3f} ({r2 / ceil22[1]:.2f} of ceiling)  coef ' + ' '.join(f'{x:+.4f}' for x in c))
    print(f'    theory: iso -> R/3 = {R / 3:.4f}; K21bulk term -> R/2 = {R / 2:.4f}')
