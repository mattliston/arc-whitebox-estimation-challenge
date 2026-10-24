#!/usr/bin/env python3
"""TAG opus-physicist T55 (numpy, 2 threads): WHAT A PER-ROW GAIN OF THE BANK READ CAN REACH, AGAINST THE TRUE K21 PLANE (for the lead's readout o4 of 19:53Z: the read's plane on
the response rows times (1 + r_i)). Residual = truth K21 - card K21 (no knobs), weighted as the covariance uses it (columns x gate / sigma; the row weight drops out of a per-row fit).
The best gain per row, r_i = sum_j res_ij cp_ij / sum_j cp_ij^2 (cp = the bank read's plane, response rows, off the diagonal): fitted on truth draw 1, scored on draw 2 (network 35,
layers 2 / 5 / 9 / 12): the share of the residual a per-row gain removes; the rms of r_i; and how much of r_i per-unit candidates predict (R^2, plain and weighted by the row's weight
in the residual): alpha, alpha^2, kink flag, log sigma, t3, t4, the bank's own response to the scale mode qb, the b-form's qf, the row's bank norm.
Also the same for a gain on the whole plane (bank + star + b-form) and for a per-row OFFSET along the scale mode's column (q_i mu_j, T43) for comparison.
usage: t55_row_gain.py <rank> <tag> <arm> [layers]"""
import sys, os, math
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; rank, tag, arm = int(sys.argv[1]), sys.argv[2], sys.argv[3]
LAYERS = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [2, 5, 9, 12]
T1 = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz'); T2 = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}_noise2.npz')
D = np.load(HERE / f'tkd_rank{rank:04d}_{tag}_{arm}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
n = 1024; eye = np.eye(n, dtype=bool); dot = lambda x, y: float((x * y).sum())
print('rank %d   layer | residual / truth | share of the residual removed (fit draw 1, scored draw 2): row gain on the bank read, on the whole plane, row offset along mu | rms of the bank row gain (response rows) | R^2 of the gain on candidates, weighted by the row weight: %s | all together' % (rank, 'alpha alpha2 kink logs t3 t4 qb qf rb'))
for l in LAYERS:
    C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); s = np.sqrt(v); mu = MC[f'mean_pre_{l}'].astype(np.float64); al = mu / s
    ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); gt = np.array([0.5 * math.erfc(-a / math.sqrt(2)) for a in al]); wr = ph / (2 * v); wc = gt / s; Wc = np.tile(wc[None, :], (n, 1)); Wc[eye] = 0.0
    kc = D[f'K21c_{l}'].astype(np.float64); cp = D[f'cp_{l}'].astype(np.float64) * Wc; kw = kc * Wc; resp = D[f'resp_{l}'].astype(bool); vc = D[f'v_{l}'].astype(np.float64); bp = D[f'bpre_{l}'].astype(np.float64); Cc = D[f'C_{l}'].astype(np.float64)
    r1 = (T1[f'K21_{l}'].astype(np.float64) - kc) * Wc; r2 = (T2[f'K21_{l}'].astype(np.float64) - kc) * Wc; rr = dot(wr[:, None] * r1, wr[:, None] * r2)
    def gain(P):      # per-row gain on plane P: fitted on r1, the removed share scored on r2 with the row weights
        den = (P * P).sum(1); g = np.where(den > 0, (r1 * P).sum(1) / np.where(den > 0, den, 1), 0.0); fit = g[:, None] * P
        return g, (2 * dot(wr[:, None] * fit, wr[:, None] * r2) - dot(wr[:, None] * fit, wr[:, None] * fit)) / rr
    gb, sb = gain(cp); gw, sw = gain(kw); M = np.tile((mu * wc)[None, :], (n, 1)); M[eye] = 0.0; go, so = gain(M)
    k4 = D[f'k4_{l}'].astype(np.float64); c = wc * wc * mu; den = (c * mu).sum() - c * mu; col = lambda X: ((np.where(eye, 0.0, X) * c[None, :]).sum(1) / den) / vc
    cand = [al, al * al, (np.abs(al) < 1.5).astype(float), np.log(s) - np.log(s).mean(), np.diag(kc) / (vc * np.sqrt(vc)), k4 / vc ** 2, col(D[f'cp_{l}'].astype(np.float64)), col(np.outer(vc, bp) + 2 * Cc * bp[:, None]), np.sqrt((cp * cp).sum(1)) / (vc * np.sqrt(vc))]
    wrow = (wr ** 2 * (cp * cp).sum(1))[resp]; y = gb[resp]
    def r2(cols):
        X = np.stack([np.ones(resp.sum())] + [x[resp] for x in cols], 1) * np.sqrt(wrow)[:, None]; yy = y * np.sqrt(wrow); res = yy - X @ np.linalg.lstsq(X, yy, rcond=None)[0]; ym = (wrow * y).sum() / wrow.sum(); return 1 - (res @ res) / (wrow * (y - ym) ** 2).sum()
    print('%3d | %.5f | %.3f %.3f %.3f | %.4f (weighted %.4f) | %s | %.3f' % (l, rr / dot(wr[:, None] * T1[f'K21_{l}'] * Wc, wr[:, None] * T2[f'K21_{l}'] * Wc), sb, sw, so, math.sqrt(np.mean(y ** 2)), math.sqrt((wrow * y * y).sum() / wrow.sum()), ' '.join('%.3f' % r2([x]) for x in cand), r2(cand)))
