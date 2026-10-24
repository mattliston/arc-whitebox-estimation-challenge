#!/usr/bin/env python3
"""TAG opus-physicist T32 (numpy, 2 threads): WHAT THE TOP DIRECTIONS OF THE MISSING K31 CONTENT ARE. Network 35 (two truth draws at layers 5, 9, 12). Residual = truth K31 (standardised,
row-by-row multiple of rho out, weighted as the covariance uses it: rows x alpha pdf / 6, columns x gate) minus the card's plane (tkd dump). Top singular triplets of the two-draw mean.
For the top m = 1..4: sigma^2 share; the RIGHT vector divided by its weight (gate_j / sigma_j -> a column vector H_j in raw units) against candidate column vectors
(mu, C mu, bpre, 1, sigma, k3 / v): weighted correlation; the LEFT vector divided by its weight (alpha pdf / (6 v sigma) -> a row vector F_i in raw units) against candidate row vectors
(mu v, mu^3, k3, v sigma, mu^2 sigma, k4, sigma^3 alpha, mu); and the best joint fit R2 of each side on its candidates.
usage: t32_top_direction.py <tag> <arm> [rank = 35]"""
import sys, os, math
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; tag, arm = sys.argv[1], sys.argv[2]; rank = int(sys.argv[3]) if len(sys.argv) > 3 else 35
T1 = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz'); p2 = P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}_noise2.npz'; T2 = np.load(p2) if p2.exists() else None
D = np.load(HERE / f'tkd_rank{rank:04d}_{tag}_{arm}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); n = 1024; eye = np.eye(n, dtype=bool)
def wcorr(x, y, w): 
    x = x - (w * x).sum() / w.sum(); y = y - (w * y).sum() / w.sum(); return float((w * x * y).sum() / np.sqrt((w * x * x).sum() * (w * y * y).sum() + 1e-300))
def wr2(y, X, w):
    Xw = X * np.sqrt(w)[:, None]; yw = y * np.sqrt(w); co = np.linalg.lstsq(Xw, yw, rcond=None)[0]; return 1 - float(((yw - Xw @ co) ** 2).sum() / (yw ** 2).sum()), co
for l in ((5, 9, 12) if T2 is not None else (5, 7, 9, 11, 12, 13, 14)):
    C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); s = np.sqrt(v); rho = C / np.outer(s, s); rho[eye] = 0.0; mu = MC[f'mean_pre_{l}'].astype(np.float64); al = mu / s
    ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); gt = np.array([0.5 * math.erfc(-a / math.sqrt(2)) for a in al]); rg = rho * gt[None, :]; S31 = v[:, None] * s[:, None] * s[None, :]
    def q(X, std=True):
        Y = np.where(eye, 0.0, (X.astype(np.float64) / S31) if std else X.astype(np.float64)) * gt[None, :]; Y = Y - ((Y * rg).sum(1) / (rg * rg).sum(1))[:, None] * rg; return Y * (al * ph / 6)[:, None]
    card = q(0.788 * D[f'T31int_{l}'].astype(np.float64) + D[f'T31rem_{l}'].astype(np.float64), std=False)
    tr = q(T1[f'K31_{l}']) if (T2 is None or f'K31_{l}' not in T2.files) else (q(T1[f'K31_{l}']) + q(T2[f'K31_{l}'])) / 2
    res = tr - card; U, sg, Vt = np.linalg.svd(res, full_matrices=False); tot = float((sg ** 2).sum())
    kc = D[f'K21c_{l}'].astype(np.float64); k3 = np.diag(kc).copy(); bp = D[f'bpre_{l}']; k4 = D[f'k4_{l}']; Cmu = C @ mu
    wr_ = gt / s; wl_ = al * ph / (6 * v * s)      # weights that turn raw forms F_i H_j into the weighted plane
    colc = dict(mu=mu, Cmu=Cmu, bpre=bp, one=np.ones(n), sigma=s, k3_over_v=k3 / v, g_mu=gt * mu); rowc = dict(mu_v=mu * v, mu3=mu ** 3, k3=k3, v_sigma=v * s, mu2_sigma=mu * mu * s, k4_over_s=k4 / s, mu=mu, sigma3=s ** 3, k3_mu_over_s=k3 * mu / s)
    print('\nlayer %d rank %d: residual energy %.4e; shares of the top 8 singular directions: %s' % (l, rank, tot, ' '.join('%.3f' % (x * x / tot) for x in sg[:8])))
    for m in range(3):
        H = Vt[m] / np.where(np.abs(wr_) > 1e-6, wr_, np.inf); F = U[:, m] / np.where(np.abs(wl_) > 1e-9, wl_, np.inf); ww_r = wr_ ** 2; ww_l = wl_ ** 2
        rc = '  '.join('%s %+.2f' % (k, wcorr(H, x, ww_r)) for k, x in colc.items()); lc = '  '.join('%s %+.2f' % (k, wcorr(F, x, ww_l)) for k, x in rowc.items())
        r2r, _ = wr2(H, np.stack(list(colc.values()), 1), ww_r); r2l, _ = wr2(F, np.stack(list(rowc.values()), 1), ww_l)
        print('  direction %d (share %.3f):\n     column vector against: %s   | joint R2 %.3f\n     row vector against:    %s   | joint R2 %.3f' % (m + 1, sg[m] ** 2 / tot, rc, r2r, lc, r2l))
