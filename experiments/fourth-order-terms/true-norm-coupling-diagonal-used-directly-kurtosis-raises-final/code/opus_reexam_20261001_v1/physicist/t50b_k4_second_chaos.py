#!/usr/bin/env python3
"""TAG opus-physicist T50b (numpy): THE CARD'S PER-UNIT KURTOSIS ERROR AGAINST THE SECOND-CHAOS FORM WITH THE TRUE MATRIX OF THE SAME LAYER. kappa4(i, i, i, i) = 6 v_i R_ii / tr C
with R = the network-specific part of the law matrix of layer l (Monte Carlo oracle t19, key R_L<l>). Error = truth - card of k4 / v^2 (two 32M truth draws; card = e16b dump: radial
+ the young pairs' local terms). Per layer: coefficient and R^2 of the error on 6 R_ii / (tr C v_i), plain and weighted by the mean's sensitivity ((alpha^2 - 1) pdf / 24)^2; the same
for the truth's own unit-specific kurtosis (truth minus its mean): how much of the TRUE unit-specific kurtosis is second chaos; and the card's local term against the truth.
usage: t50b_k4_second_chaos.py <rank> <tag> <arm>"""
import sys, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; rank, tag, arm = int(sys.argv[1]), sys.argv[2], sys.argv[3]
A = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz'); B = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_indep32M.npz'); D = np.load(HERE / f'tkd_rank{rank:04d}_{tag}_{arm}.npz')
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); RO = np.load(HERE / f't19_oracle_rank{rank:04d}.npz')
def reg(y, x, w):
    xm, ym = (w * x).sum() / w.sum(), (w * y).sum() / w.sum(); xc, yc = x - xm, y - ym; c = (w * xc * yc).sum() / (w * xc * xc).sum(); return c, c * c * (w * xc * xc).sum() / (w * yc * yc).sum()
print('rank %d   layer | sd of the true unit-specific k4 / v^2, rms card error, truth noise | card error on 6 R_ii / (tr C v): coefficient R^2 plain, weighted | TRUE unit-specific kurtosis on the same: coefficient R^2 plain, weighted | card unit-specific part against the truth unit-specific: coefficient R^2' % rank)
for l in range(2, 15):
    if f'v_{l}' not in D.files or f'R_L{l}' not in RO.files: continue
    vA, vB = A['mc_pre_v'][l].astype(np.float64), B['mc_pre_v'][l].astype(np.float64); v = (vA + vB) / 2; mu = A['mc_pre_m'][l].astype(np.float64); s = np.sqrt(v); al = mu / s; ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi)
    kA, kB = A['mc_pre_k4'][l] / vA ** 2, B['mc_pre_k4'][l] / vB ** 2; kt = (kA + kB) / 2; vc = D[f'v_{l}'].astype(np.float64); kc = D[f'k4_{l}'].astype(np.float64) / vc ** 2
    trC = np.diag(MC[f'cov_pre_{l}']).sum(); f = 6 * np.diag(RO[f'R_L{l}']).astype(np.float64) / (trC * v); one = np.ones_like(v); w4 = ((al * al - 1) * ph / 24) ** 2
    e = kt - kc; c1, r1 = reg(e, f, one); c2, r2 = reg(e, f, w4); c3, r3 = reg(kt, f, one); c4, r4 = reg(kt, f, w4); c5, r5 = reg(kt, kc, one)
    print('%3d | %.4f %.4f %.4f | %+.2f %.3f, %+.2f %.3f | %+.2f %.3f, %+.2f %.3f | %+.2f %.3f' % (l, kt.std(), math.sqrt(np.mean(e ** 2)), math.sqrt(np.mean((kA - kB) ** 2)) / 2, c1, r1, c2, r2, c3, r3, c4, r4, c5, r5))
