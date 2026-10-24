#!/usr/bin/env python3
"""TAG opus-physicist T33c (numpy, 2 threads): the cross form of T33 split into its parts. K21 = specific part (bank + star: cp + sp) + the b-form v_i b_j + 2 C_ij b_i, and
k3_i = k3spec_i + 3 v_i b_i. Forms (raw units, then standardised, weighted and row-projected as in T33):
   F1s = k3spec_i b_j     F2s = b_i K21spec_ij     Fbb = v_i b_i b_j  (second order in b; the other second-order piece b_i^2 C_ij is a row form and drops out)
Per layer: joint least-squares coefficients of the residual truth - card on (F1s, F2s, Fbb), share explained; each alone. Theory for Sym(b (x) kappa3) at 3/2: 1.5, 4.5, 9.
usage: t33c_bk3_split.py <rank> <tag> <arm>"""
import sys, os, math
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; rank, tag, arm = int(sys.argv[1]), sys.argv[2], sys.argv[3]
T1 = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz'); p2 = P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}_noise2.npz'; T2 = np.load(p2) if p2.exists() else None
D = np.load(HERE / f'tkd_rank{rank:04d}_{tag}_{arm}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); n = 1024; eye = np.eye(n, dtype=bool); dot = lambda x, y: float((x * y).sum())
print('rank %d %s %s' % (rank, tag, arm)); print('layer | joint coefficients F1s F2s Fbb -> share of the residual | alone: F1s coef share | F2s coef share | Fbb coef share | energy of the forms at theory strength relative to the residual: F1s F2s Fbb')
for l in range(2, 15):
    C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); s = np.sqrt(v); rho = C / np.outer(s, s); rho[eye] = 0.0; mu = MC[f'mean_pre_{l}'].astype(np.float64); al = mu / s
    ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); gt = np.array([0.5 * math.erfc(-a / math.sqrt(2)) for a in al]); rg = rho * gt[None, :]; S31 = v[:, None] * s[:, None] * s[None, :]
    def q(X, std=True):
        Y = np.where(eye, 0.0, (X.astype(np.float64) / S31) if std else X.astype(np.float64)) * gt[None, :]; Y = Y - ((Y * rg).sum(1) / (rg * rg).sum(1))[:, None] * rg; return Y * (al * ph / 6)[:, None]
    card = q(0.788 * D[f'T31int_{l}'].astype(np.float64) + D[f'T31rem_{l}'].astype(np.float64), std=False)
    two = T2 is not None and f'K31_{l}' in T2.files; t1 = q(T1[f'K31_{l}']); t2 = q(T2[f'K31_{l}']) if two else t1; res = (t1 + t2) / 2 - card; rr = dot(t1 - card, t2 - card)
    bp = D[f'bpre_{l}']; vc = D[f'v_{l}']; Cc = D[f'C_{l}'].astype(np.float64); Ks = D[f'cp_{l}'].astype(np.float64) + D[f'sp_{l}'].astype(np.float64); k3s = np.diag(Ks).copy()
    F = [q(np.outer(k3s, bp)), q(bp[:, None] * Ks), q(np.outer(vc * bp, bp))]
    def fit(cols):
        X = np.stack([c.ravel() for c in cols], 1); co = np.linalg.lstsq(X, res.ravel(), rcond=None)[0]; return co, float(co @ (X.T @ res.ravel())) / rr
    cj, sj = fit(F); al_ = [fit([f]) for f in F]; th = (1.5, 4.5, 9.0)
    print('%3d | %6.2f %6.2f %7.2f -> %.3f | %5.2f %.3f | %5.2f %.3f | %6.2f %.3f | %.3f %.3f %.3f%s' % (l, cj[0], cj[1], cj[2], sj, al_[0][0][0], al_[0][1], al_[1][0][0], al_[1][1], al_[2][0][0], al_[2][1],
          th[0] ** 2 * dot(F[0], F[0]) / rr, th[1] ** 2 * dot(F[1], F[1]) / rr, th[2] ** 2 * dot(F[2], F[2]) / rr, '' if two else ' *'))
