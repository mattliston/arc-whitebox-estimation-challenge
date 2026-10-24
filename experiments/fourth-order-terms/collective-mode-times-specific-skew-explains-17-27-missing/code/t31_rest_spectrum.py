#!/usr/bin/env python3
"""TAG opus-physicist T31 (numpy, 2 threads): IS THE MISSING K31 CONTENT LOW-RANK? Network 35, layers with two truth draws (5, 9, 12). The truth's K31 plane as the covariance uses it
(standardised, rows x alpha pdf / 6, columns x gate, row-by-row multiple of rho out) minus the card's plane (dump tkd_rank0035_<tag>_<arm>.npz: internal x 0.788 + carried plane).
The singular vectors of that residual are taken from draw 1 and scored on draw 2: cross-validated share of the residual's energy in the top-k directions
(sum_m sigma1_m (u_m' T2 v_m) / <T1, T2>). A random-like plane gives about k / n for small k; structure shows as a share far above that.
Also: the same for the symmetric part and the antisymmetric part, and the share of the residual that is symmetric.
usage: t31_rest_spectrum.py <tag> <arm>"""
import sys, os, math
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; rank = 35; tag, arm = sys.argv[1], sys.argv[2]
T1 = np.load(P2 / 'public_g19_attribution_v1/mps_planes_rank0035.npz'); T2 = np.load(P2 / 'public_g19_attribution_v1/mps_planes_rank0035_noise2.npz'); D = np.load(HERE / f'tkd_rank0035_{tag}_{arm}.npz')
MC = np.load(P2 / 'fable_covhunter_v1/mcall_rank0035_P32M.npz'); n = 1024; eye = np.eye(n, dtype=bool); dot = lambda x, y: float((x * y).sum()); KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
for l in (5, 9, 12):
    C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); s = np.sqrt(v); rho = C / np.outer(s, s); rho[eye] = 0.0; al = MC[f'mean_pre_{l}'].astype(np.float64) / s
    ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); gt = np.array([0.5 * math.erfc(-a / math.sqrt(2)) for a in al]); rg = rho * gt[None, :]; S31 = v[:, None] * s[:, None] * s[None, :]
    def q(X, std=True):
        Y = np.where(eye, 0.0, (X.astype(np.float64) / S31) if std else X.astype(np.float64)) * gt[None, :]; Y = Y - ((Y * rg).sum(1) / (rg * rg).sum(1))[:, None] * rg; return Y * (al * ph / 6)[:, None]
    card = q(0.788 * D[f'T31int_{l}'].astype(np.float64) + D[f'T31rem_{l}'].astype(np.float64), std=False); r1 = q(T1[f'K31_{l}']) - card; r2 = q(T2[f'K31_{l}']) - card; rr = dot(r1, r2)
    U, sg, Vt = np.linalg.svd(r1, full_matrices=False); proj = np.einsum('im,ij,mj->m', U, r2, Vt); cum = np.cumsum(sg * proj) / rr
    sy1, sy2 = (r1 + r1.T) / 2, (r2 + r2.T) / 2; print('layer %d: residual energy (cross-draw) / truth energy %.3f; symmetric part holds %.3f of the residual' % (l, rr / dot(q(T1[f'K31_{l}']), q(T2[f'K31_{l}'])), dot(sy1, sy2) / rr))
    print('   top-k share, cross-validated: ' + '  '.join('%d: %.3f' % (k, cum[k - 1]) for k in KS) + '   (k / n: ' + ' '.join('%.3f' % (k / n) for k in KS) + ')')
    ev, Q = np.linalg.eigh(sy1); o = np.argsort(-np.abs(ev)); pr = np.einsum('im,ij,jm->m', Q[:, o], sy2, Q[:, o]); cs = np.cumsum(ev[o] * pr) / dot(sy1, sy2)
    print('   symmetric part, top-k eigen-directions: ' + '  '.join('%d: %.3f' % (k, cs[k - 1]) for k in KS))
    # the same for the truth's own rest (no card subtracted) and for the card's plane, as references
    t1, t2 = q(T1[f'K31_{l}']), q(T2[f'K31_{l}']); U, sg, Vt = np.linalg.svd(t1, full_matrices=False); proj = np.einsum('im,ij,mj->m', U, t2, Vt); cum = np.cumsum(sg * proj) / dot(t1, t2)
    print('   truth rest itself, top-k share: ' + '  '.join('%d: %.3f' % (k, cum[k - 1]) for k in KS))
    sgc = np.linalg.svd(card, compute_uv=False); print('   card plane, top-k share of its own energy: ' + '  '.join('%d: %.3f' % (k, float((sgc[:k] ** 2).sum() / (sgc ** 2).sum())) for k in KS))
