#!/usr/bin/env python3
"""TAG opus-physicist T55b (numpy, 2 threads): THE FORM OF THE ROW GAIN THE TRUE K21 PLANE WANTS FROM THE BANK READ. r_i = sum_j res_ij cp_ij / sum_j cp_ij^2 on the response rows
(res = truth - card K21 without knobs, columns x gate / sigma, off the diagonal; mean of the truth draws where two exist). Weighted least squares (row weight = the row's weight in
the residual) of r_i on  (1, ls = log sigma_i - layer mean, t4 = the card's k4_i / v_i^2 - layer mean, qb = the bank's own response to the scale mode / v_i - layer mean):
coefficients alone and jointly, R^2; all layers 3-14.  A stable coefficient across layers and networks = a physics dial, not a learned readout.
usage: t55b_row_gain_form.py <rank> <tag> <arm>"""
import sys, os, math
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; rank, tag, arm = int(sys.argv[1]), sys.argv[2], sys.argv[3]
T1 = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz'); p2 = P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}_noise2.npz'; T2 = np.load(p2) if p2.exists() else None
D = np.load(HERE / f'tkd_rank{rank:04d}_{tag}_{arm}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); n = 1024; eye = np.eye(n, dtype=bool)
print('rank %d   layer | mean gain, rms gain | alone: coefficient (R^2) of ls, t4, qb | joint coefficients ls t4 qb -> R^2 | the gain of rows above / below the median sigma' % rank)
for l in range(3, 15):
    if f'K21_{l}' not in T1.files: continue
    C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); s = np.sqrt(v); mu = MC[f'mean_pre_{l}'].astype(np.float64); al = mu / s
    ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); gt = np.array([0.5 * math.erfc(-a / math.sqrt(2)) for a in al]); wr = ph / (2 * v); wc = gt / s; Wc = np.tile(wc[None, :], (n, 1)); Wc[eye] = 0.0
    kc = D[f'K21c_{l}'].astype(np.float64); cpr = D[f'cp_{l}'].astype(np.float64); cp = cpr * Wc; resp = D[f'resp_{l}'].astype(bool); vc = D[f'v_{l}'].astype(np.float64); k4 = D[f'k4_{l}'].astype(np.float64)
    two = T2 is not None and f'K21_{l}' in T2.files; T = (T1[f'K21_{l}'].astype(np.float64) + T2[f'K21_{l}'].astype(np.float64)) / 2 if two else T1[f'K21_{l}'].astype(np.float64)
    res = (T - kc) * Wc; den = (cp * cp).sum(1); g = np.where(den > 0, (res * cp).sum(1) / np.where(den > 0, den, 1), 0.0)
    c = wc * wc * mu; dn = (c * mu).sum() - c * mu; qb = ((np.where(eye, 0.0, cpr) * c[None, :]).sum(1) / dn) / vc
    w = (wr ** 2 * den)[resp]; y = g[resp]; ctr = lambda x: x[resp] - (w * x[resp]).sum() / w.sum()
    ls, t4, q = ctr(np.log(s)), ctr(k4 / vc ** 2), ctr(qb); ym = (w * y).sum() / w.sum()
    def fit(cols):
        X = np.stack([np.ones(len(y))] + cols, 1) * np.sqrt(w)[:, None]; yy = y * np.sqrt(w); co = np.linalg.lstsq(X, yy, rcond=None)[0]; r = yy - X @ co; return co[1:], 1 - (r @ r) / (w * (y - ym) ** 2).sum()
    a1, r1 = fit([ls]); a2, r2 = fit([t4]); a3, r3 = fit([q]); cj, rj = fit([ls, t4, q]); med = np.median(s[resp]); hi = s[resp] > med
    print('%3d%s | %+.4f %.4f | %+.3f (%.2f)  %+.2f (%.2f)  %+.1f (%.2f) | %+.3f %+.2f %+.1f -> %.2f | %+.4f / %+.4f' % (l, '' if two else '*', ym, math.sqrt((w * y * y).sum() / w.sum()), a1[0], r1, a2[0], r2, a3[0], r3, cj[0], cj[1], cj[2], rj,
          (w[hi] * y[hi]).sum() / w[hi].sum(), (w[~hi] * y[~hi]).sum() / w[~hi].sum()))
