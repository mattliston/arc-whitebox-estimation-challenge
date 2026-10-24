#!/usr/bin/env python3
"""TAG opus-physicist T17 (truth inputs, no card): what the second-order list says about the constant k21n of the covariance knob hook.
The card's K21 term of the covariance is (1/2) K21[i, j] E_G relu''(x_i) E_G relu'(x_j). The second-order parts a and b' of so2.py
replace the two Gaussian expectations by the skew-corrected ones:
    E relu''(x_i) = (phi_i / sigma_i) (1 - (t3_i / 6) He3(alpha_i)),     E relu'(x_j) = gate_j + (t3_j / 6) He2(alpha_j) phi_j.
The knob multiplies K21 by (1 + k21n) on the rows with |alpha| >= 1.5 (parent -0.028, the trainer's line search -0.048) and leaves the
kink rows alone. Printed per layer, from the 32M-sample per-unit truth: the row factor -(t3/6) He3(alpha), its mean and spread on the two
row classes, weighted by the size of the row's K21 term (phi / sigma), and the column factor's mean and spread weighted by the gate.
usage: t17_k21n_reading.py <rank> [<rank> ...]"""
import sys, math, json
import numpy as np
import pcard_light as pl
for rank in (int(x) for x in sys.argv[1:]):
    z = pl.pu(rank, 'P32M'); m, v, k3 = (z[k].astype(np.float64) for k in ('mc_pre_m', 'mc_pre_v', 'mc_pre_k3'))
    for l in range(1, 16):
        s = np.sqrt(v[l]); a = m[l] / s; t3 = k3[l] / s ** 3; ph = np.exp(-a * a / 2) / math.sqrt(2 * math.pi); g = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in a])
        row = -(t3 / 6) * (a ** 3 - 3 * a); col = (t3 / 6) * (a * a - 1) * ph / np.maximum(g, 1e-12); w = ph / s; nk = np.abs(a) >= 1.5
        st = lambda x, w_: (float((w_ * x).sum() / w_.sum()), float(np.sqrt((w_ * (x - (w_ * x).sum() / w_.sum()) ** 2).sum() / w_.sum())))
        r_n, r_k, c_all = st(row[nk], w[nk]), st(row[~nk], w[~nk]), st(col, g)
        print(json.dumps(dict(rank=rank, layer=l, t3_rms=round(float(np.sqrt((t3 ** 2).mean())), 4), corr_t3_alpha=round(float(np.corrcoef(t3, a)[0, 1]), 3),
                              nonkink_rows=int(nk.sum()), nonkink_share_of_row_weight=round(float(w[nk].sum() / w.sum()), 3),
                              row_factor_nonkink_mean_pct=round(100 * r_n[0], 2), row_factor_nonkink_spread_pct=round(100 * r_n[1], 2),
                              row_factor_kink_mean_pct=round(100 * r_k[0], 2), row_factor_kink_spread_pct=round(100 * r_k[1], 2),
                              col_factor_mean_pct=round(100 * c_all[0], 2), col_factor_spread_pct=round(100 * c_all[1], 2))), flush=True)
