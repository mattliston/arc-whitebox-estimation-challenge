"""TAG opus-reexam E66 (lead, 2026-10-02): WHAT IS IN THE TRUE (3,1) PLANE. Candidate planes dumped by k31diag.py (and the pair law's plane from ncar6's dump) against the
8M-sample Monte Carlo truth planes, numpy, 2 threads. All planes standardised (K31_ij / (v_i s_i s_j)), off the diagonal, with the part proportional row by row to the
correlation taken out (the physicist's 'rest', t23_k31l_vs_truth.py). Per layer: rms of the truth's rest [noise share where a second draw exists]; for every candidate alone:
share of the truth's energy explained at strength 1, cosine, best scalar; joint least-squares fits of the truth's rest on growing lists of candidates (share explained).
usage: e66_k31_truth.py <rank> <arm name of the dump>"""
import sys, os
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; rank, arm = int(sys.argv[1]), sys.argv[2]; n = 1024; eye = np.eye(n, dtype=bool)
D = np.load(HERE / f'k31diag_rank{rank:04d}_{arm}.npz'); L = np.load(HERE / f'ncar_dump_rank{rank:04d}_{arm}.npz'); T = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz')
p2 = P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}_noise2.npz'; T2 = np.load(p2) if p2.exists() else None
def rest(X, rho):
    X = X.copy(); X[eye] = 0.0; return X - ((X * rho).sum(1) / (rho * rho).sum(1))[:, None] * rho
dot = lambda x, y: float((x * y).sum())
layers = sorted({int(k.split('_')[-1]) for k in D.files if k.startswith('mu_')})
for l in layers:
    v = D[f'v_{l}'].astype(np.float64); s = np.sqrt(v); C = D[f'C_{l}'].astype(np.float64); rho = C / np.outer(s, s); rho[eye] = 0.0; std = v[:, None] * s[:, None] * s[None, :]
    tr = rest(T[f'K31_{l}'].astype(np.float64) / std, rho); tt = dot(tr, tr); ns = ''
    if T2 is not None and f'K31_{l}' in T2.files:
        t2 = rest(T2[f'K31_{l}'].astype(np.float64) / std, rho); ns = ' [noise share %.2f]' % (1 - dot(tr, t2) / np.sqrt(tt * dot(t2, t2)))
    cand = {}
    if f'R_{l}' in L.files: cand['law'] = rest((3.0 / n) * L[f'R_{l}'].astype(np.float64) / std, rho)
    for k in ('QL4_1', 'QL4_2', 'QL4_3', 'QL4_4', 'QLf_1', 'QLf_2', 'CH4', 'CHf', 'MX1', 'MX2', 'MX3'):
        if f'{k}_{l}' in D.files: cand[k] = rest(D[f'{k}_{l}'].astype(np.float64) / std, rho)
    print('layer %2d: truth rest rms %.3e%s' % (l, np.sqrt(tt / (n * n)), ns))
    for k, c in cand.items():
        cc = dot(c, c); ct = dot(c, tr); print('    %-6s alone: share %+.3f  cos %+.3f  best scalar %+.2f   (own energy / truth %.3f)' % (k, (2 * ct - cc) / tt, ct / np.sqrt(cc * tt + 1e-300), ct / (cc + 1e-300), cc / tt))
    def joint(names):
        names = [k for k in names if k in cand]
        X = np.stack([cand[k].ravel() for k in names], 1); co = np.linalg.lstsq(X, tr.ravel(), rcond=None)[0]; fit = X @ co
        return names, co, 1 - float(((tr.ravel() - fit) ** 2).sum()) / tt
    for names in (('law',), ('law', 'QL4_1', 'QL4_2'), ('law', 'QL4_1', 'QL4_2', 'QL4_3', 'QL4_4'), ('law', 'QL4_1', 'QL4_2', 'CH4'), ('law', 'QL4_1', 'QL4_2', 'MX1', 'MX2', 'MX3'),
                  ('law', 'QL4_1', 'QL4_2', 'QL4_3', 'QL4_4', 'CH4', 'MX1', 'MX2', 'MX3')):
        nm, co, sh = joint(names); print('    joint %-55s share %.3f   coefficients %s' % ('+'.join(nm), sh, ' '.join('%+.2f' % x for x in co)))
