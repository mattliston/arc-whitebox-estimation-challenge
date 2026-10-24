#!/usr/bin/env python3
"""TAG opus-physicist T51 (numpy): DO THE CARD'S PER-UNIT VARIANCE ERROR AND KURTOSIS ERROR CANCEL IN THE MEAN? First-order effect of each input error on the layer's own mean:
   e_v = (pdf / (2 sigma)) (v_card - v_true)                               (variance)
   e_k = sigma ((k4_card - k4_true) / v^2) He2(alpha) pdf / 24              (kurtosis)
   e_s = -sigma^-2 ... : (k3_card - k3_true) (-alpha pdf / (6 sigma^2))     (skew)
per unit and layer (card = e16b dump; truth = two 32M draws, their product gives noise-free second moments). Reported: rms of each (x1e-5), the correlation between e_v and e_k and
between e_v and e_s, rms of the sum against the root of the summed squares; and for the SECOND-CHAOS part of the kurtosis error alone, e_k2 = -sigma ((6 / n) R_ii / v^2) He2 pdf / 24
(true matrix, t19 oracle): its correlation with e_v and with e_k, and what adding the true diagonal to the card's kurtosis does to rms(e_v + e_k) (the mean's error from both inputs).
usage: t51_error_cancellation.py <rank> <tag> <arm>"""
import sys, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; rank, tag, arm = int(sys.argv[1]), sys.argv[2], sys.argv[3]
A = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz'); B = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_indep32M.npz'); D = np.load(HERE / f'tkd_rank{rank:04d}_{tag}_{arm}.npz')
RO = np.load(HERE / f't19_oracle_rank{rank:04d}.npz')
cor = lambda x, y: float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))
rms = lambda x: math.sqrt(np.mean(x * x))
print('rank %d   layer | rms x1e-5: e_v e_k e_s | corr(e_v, e_k) corr(e_v, e_s) corr(e_k, e_s) | rms of the sum / root of summed squares | second-chaos part: rms, corr with e_v, corr with e_k | rms(e_v + e_k) before -> with the true diagonal added (x1e-5)' % rank)
for l in range(3, 15):
    if f'v_{l}' not in D.files: continue
    vt = (A['mc_pre_v'][l].astype(np.float64) + B['mc_pre_v'][l]) / 2; mu = A['mc_pre_m'][l].astype(np.float64); s = np.sqrt(vt); al = mu / s; ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi)
    k3t = (A['mc_pre_k3'][l].astype(np.float64) + B['mc_pre_k3'][l]) / 2; k4t = (A['mc_pre_k4'][l].astype(np.float64) + B['mc_pre_k4'][l]) / 2
    vc = D[f'v_{l}'].astype(np.float64); k3c = np.diag(D[f'K21c_{l}']).astype(np.float64); k4c = D[f'k4_{l}'].astype(np.float64)
    ev = (ph / (2 * s)) * (vc - vt); ek = s * ((k4c - k4t) / vt ** 2) * (al * al - 1) * ph / 24; es = (k3c - k3t) * (-al * ph / (6 * vt))
    f = (6 / 1024) * np.diag(RO[f'R_L{l}']).astype(np.float64); ek2 = -s * (f / vt ** 2) * (al * al - 1) * ph / 24      # the card lacks +f in k4: its kurtosis error contains -f
    tot = ev + ek + es
    print('%3d | %5.2f %5.2f %5.2f | %+.2f %+.2f %+.2f | %.2f | %5.2f %+.2f %+.2f | %5.2f -> %5.2f' % (l, 1e5 * rms(ev), 1e5 * rms(ek), 1e5 * rms(es), cor(ev, ek), cor(ev, es), cor(ek, es), rms(tot) / math.sqrt(rms(ev) ** 2 + rms(ek) ** 2 + rms(es) ** 2),
          1e5 * rms(ek2), cor(ek2, ev), cor(ek2, ek), 1e5 * rms(ev + ek), 1e5 * rms(ev + ek - ek2)))
