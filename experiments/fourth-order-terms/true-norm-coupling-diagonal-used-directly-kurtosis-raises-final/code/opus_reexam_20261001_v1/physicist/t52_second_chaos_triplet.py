#!/usr/bin/env python3
"""TAG opus-physicist T52 (numpy): THE UNIT-SPECIFIC SECOND CHAOS IS A PER-UNIT SCALE MODULATION, AND ITS THREE PER-UNIT SIGNATURES CANCEL IN THE MEAN. For z_i = (1 + S_i) y_i with a
small unit-specific scale variance s_i^2 (independent of y_i):   dv = s_i^2 (v + mu^2),   dk3 = 6 s_i^2 mu v,   dk4 = 12 s_i^2 v^2,   and the mean of relu does not move:
   + sigma s_i^2 pdf (1 + alpha^2) / 2   (variance)   - sigma s_i^2 pdf alpha^2   (skew)   + sigma s_i^2 pdf (alpha^2 - 1) / 2   (kurtosis)   = 0.
With the true matrix (t19 oracle) s_i^2 = R_ii / (2 n v_i^2), so the three forms are  Fv = R_ii (v + mu^2) / (2 n v^2),  Fs = 3 R_ii mu / (n v),  Fk = 6 R_ii / n.
Per layer (network <rank>; truth = two 32M per-unit draws; card = e16b dump):
  the TRUTH's unit-specific skew and kurtosis on Fs, Fk (does the truth carry the triplet: coefficient 1);
  the CARD's error (truth - card) of v, k3, k4 on Fv, Fs, Fk: coefficient 1 = the card lacks that part, 0 = it has it;
  the first-order error of the card's mean that those missing parts make, rms (x1e-5), and its size against the card's whole first-order mean error from the three inputs.
usage: t52_second_chaos_triplet.py <rank> <tag> <arm>"""
import sys, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; rank, tag, arm = int(sys.argv[1]), sys.argv[2], sys.argv[3]
A = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz'); B = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_indep32M.npz'); D = np.load(HERE / f'tkd_rank{rank:04d}_{tag}_{arm}.npz')
RO = np.load(HERE / f't19_oracle_rank{rank:04d}.npz'); n = 1024
def reg(y, x, w=None):
    w = np.ones_like(x) if w is None else w; xm, ym = (w * x).sum() / w.sum(), (w * y).sum() / w.sum(); xc, yc = x - xm, y - ym; c = (w * xc * yc).sum() / (w * xc * xc).sum(); return c, c * c * (w * xc * xc).sum() / (w * yc * yc).sum()
rms = lambda x: math.sqrt(np.mean(x * x)); cor = lambda x, y: float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))
print('rank %d   layer | truth on the forms: skew residual (truth - its line in alpha) on Fs coef R2, kurtosis on Fk coef R2 | card error on the forms: variance coef R2, skew coef R2, kurtosis coef R2 | mean error (x1e-5): from the three missing parts with those coefficients, the whole first-order error e_v + e_s + e_k, their correlation' % rank)
for l in range(3, 15):
    if f'v_{l}' not in D.files: continue
    vt = (A['mc_pre_v'][l].astype(np.float64) + B['mc_pre_v'][l]) / 2; mu = A['mc_pre_m'][l].astype(np.float64); s = np.sqrt(vt); al = mu / s; ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi)
    k3t = (A['mc_pre_k3'][l].astype(np.float64) + B['mc_pre_k3'][l]) / 2; k4t = (A['mc_pre_k4'][l].astype(np.float64) + B['mc_pre_k4'][l]) / 2
    vc = D[f'v_{l}'].astype(np.float64); k3c = np.diag(D[f'K21c_{l}']).astype(np.float64); k4c = D[f'k4_{l}'].astype(np.float64)
    R = np.diag(RO[f'R_L{l}']).astype(np.float64); Fv = R * (vt + mu * mu) / (2 * n * vt * vt); Fs = 3 * R * mu / (n * vt); Fk = 6 * R / n
    # the truth's unit-specific skew: take out its collective line in alpha (k3 / sigma^3 = c alpha) before the regression
    t3 = k3t / (vt * s); c_line = (al * t3).sum() / (al * al).sum(); k3u = (t3 - c_line * al) * vt * s
    a1, r1 = reg(k3u, Fs); a2, r2 = reg(k4t, Fk)
    cv, rv = reg(vt - vc, Fv); cs, rs_ = reg(k3t - k3c, Fs); ck, rk = reg(k4t - k4c, Fk)
    ev = (ph / (2 * s)) * (vc - vt); es = (k3c - k3t) * (-al * ph / (6 * vt)); ek = s * ((k4c - k4t) / vt ** 2) * (al * al - 1) * ph / 24
    miss = -(ph / (2 * s)) * cv * Fv - cs * Fs * (-al * ph / (6 * vt)) - s * (ck * Fk / vt ** 2) * (al * al - 1) * ph / 24      # the card's mean error from the missing parts (card - truth)
    print('%3d | %+.2f %.3f, %+.2f %.3f | %+.2f %.3f, %+.2f %.3f, %+.2f %.3f | %5.2f %5.2f %+.2f' % (l, a1, r1, a2, r2, cv, rv, cs, rs_, ck, rk, 1e5 * rms(miss), 1e5 * rms(ev + es + ek), cor(miss, ev + es + ek)))
