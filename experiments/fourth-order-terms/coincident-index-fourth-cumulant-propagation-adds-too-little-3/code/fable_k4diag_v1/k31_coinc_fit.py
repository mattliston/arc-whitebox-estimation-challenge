#!/usr/bin/env python3
"""TAG fable-k4diag (KD-1): is the COINCIDENT-INDEX fourth-cumulant push missing from the card's isotropic K31/K22/k4?
X_iso[a,c] = v_a C_ac (card form, radial absorbed in the fit), X_diag[a,c] = sum_b W_ba^3 W_bc kappa4(y_b) where kappa4(y_b) is the
per-unit post-activation fourth cumulant at l-1 (rectified Gaussian from the truth pre mean/variance at l-1; certified by MC below).
Fit truth K31 (8M, public_g19 mps_planes) off-diagonal: R2 of alpha*X_iso, beta*X_diag, and both. Same for K22 (X_iso = v v^T + 2 C^2,
X_diag = sum_b W_ba^2 W_bc^2 kappa4(y_b)) and per-unit k4 (iso v^2, diag sum_b W_ba^4 kappa4(y_b)) vs indep32M truth.
usage: k31_coinc_fit.py <rank> <layers e.g. 4,7,10,13>"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
rank = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(',')]
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy").astype(np.float64)
pl = np.load(P2 / 'public_g19_attribution_v1' / f'mps_planes_rank{rank:04d}.npz')
tr = np.load(P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{rank:04d}_indep32M.npz')
mc2 = np.load(P2 / 'fable_mc_v1' / f'mc2_rank{rank:04d}_P32M.npz')
xg, wg = np.polynomial.hermite_e.hermegauss(120); wg = wg / wg.sum()
def relu_k4(mu, v):
    s = np.sqrt(np.maximum(v, 1e-30)); y = np.maximum(mu[:, None] + s[:, None] * xg[None, :], 0.0)
    m1 = (y * wg).sum(1); d = y - m1[:, None]; m2 = (d ** 2 * wg).sum(1); m4 = (d ** 4 * wg).sum(1)
    return m4 - 3 * m2 ** 2
# certificate: rectified-Gaussian k4 by MC on 3 units
rng = np.random.default_rng(0); cert = []
for mu, v in ((0.0, 1.0), (0.7, 0.5), (-0.4, 2.0)):
    z = mu + np.sqrt(v) * rng.standard_normal(4_000_000); y = np.maximum(z, 0); d = y - y.mean()
    cert.append((round(float(relu_k4(np.array([mu]), np.array([v]))[0]), 5), round(float((d ** 4).mean() - 3 * (d ** 2).mean() ** 2), 5)))
print(json.dumps(dict(certificate_quadrature_vs_mc=cert)), flush=True)
def off(X): return X - np.diag(np.diag(X))
def fit(T, Xs):
    A = np.stack([off(X).ravel() for X in Xs], 1); b = off(T).ravel()
    coef, *_ = np.linalg.lstsq(A, b, rcond=None); r = b - A @ coef
    return [round(float(c), 5) for c in coef], round(float(1 - (r @ r) / (b @ b)), 4)
for l in layers:
    Wl = W[l]                                      # z_l[a] = sum_b y_{l-1}[b] W_l[b, a]
    k4y = relu_k4(tr['mc_pre_m'][l - 1], tr['mc_pre_v'][l - 1])
    Cpre = mc2[f'cov_pre_{l}'] if f'cov_pre_{l}' in mc2.files else None
    v = tr['mc_pre_v'][l]
    X31_iso = v[:, None] * Cpre
    X31_diag = (Wl ** 3).T @ (k4y[:, None] * Wl)
    X22_iso = np.outer(v, v) + 2 * Cpre * Cpre
    X22_diag = (Wl ** 2).T @ (k4y[:, None] * Wl ** 2)
    T31, T22 = pl[f'K31_{l}'].astype(np.float64), pl[f'K22_{l}'].astype(np.float64)
    k4t = tr['mc_pre_k4'][l]; k4_iso = v * v; k4_diag = (Wl ** 4).T @ k4y
    A = np.stack([k4_iso, k4_diag], 1); c4, *_ = np.linalg.lstsq(A, k4t, rcond=None); rk = k4t - A @ c4
    c4i = (k4_iso @ k4t) / (k4_iso @ k4_iso); rki = k4t - c4i * k4_iso
    out = dict(rank=rank, layer=l,
               K31=dict(iso=fit(T31, [X31_iso]), diag=fit(T31, [X31_diag]), both=fit(T31, [X31_iso, X31_diag])),
               K22=dict(iso=fit(T22, [X22_iso]), diag=fit(T22, [X22_diag]), both=fit(T22, [X22_iso, X22_diag])),
               k4=dict(iso_R2=round(float(1 - rki @ rki / (k4t @ k4t)), 4), both_coef=[round(float(x), 4) for x in c4],
                       both_R2=round(float(1 - rk @ rk / (k4t @ k4t)), 4)),
               scale=dict(rms_T31=float(np.sqrt((off(T31) ** 2).mean())), rms_diag31=float(np.sqrt((off(X31_diag) ** 2).mean())),
                          mean_k4y_kink=float(np.median(k4y[(tr['mc_pre_m'][l-1] / np.sqrt(tr['mc_pre_v'][l-1]) > -1.5) & (tr['mc_pre_m'][l-1] / np.sqrt(tr['mc_pre_v'][l-1]) < 1.5)]))))
    print(json.dumps(out), flush=True)
