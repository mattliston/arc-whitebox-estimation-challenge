"""TAG fable-lead 2026-10-19: fit the PER-UNIT k4 shape correction against Monte Carlo truth. For each layer l (1-14) and each fresh seed
with a dump (probe_k4dump_v1) and a truth file (mc_cumulants_v1): err_i = k4_total_i - k4_true_i; regressors (centered per layer per net)
x1 = vv^2 (k3n^2 - mean), x2 = vv^2 (z^2 - mean) with k3n = 3 bpre / sqrt(vv) (the card's carried skew proxy), z = mu / sqrt(vv).
The correction is dk4_i = -(b1 x1 + b2 x2) (centered, so the layer mean k4 and the band dials are untouched). Fits per layer on all
units of all seeds, reports per-seed coefficients (stability) and the fraction of per-unit error variance removed.
usage: k4shape_fit_v1.py <out json> <seed> [<seed> ...]   (files k4dump_seed<seed>_v2.npz, mc_cum_v1/mc_cum_seed<seed>.npz)"""
import json
import sys

import numpy as np

out, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
data = {}
for s in seeds:
    D = np.load('k4dump_seed%d_v2.npz' % s); M = np.load('mc_cum_v1/mc_cum_seed%d.npz' % s)
    for l in range(1, 15):
        vv, mu, bp = D['L%d_pre_vv' % l], D['L%d_pre_mu' % l], D['L%d_pre_bpre' % l]
        k3n = 3 * bp / np.sqrt(vv); z2 = (mu / np.sqrt(vv)) ** 2
        x1 = vv ** 2 * (k3n ** 2 - np.mean(k3n ** 2)); x2 = vv ** 2 * (z2 - np.mean(z2))
        err = D['L%d_total_k4' % l] - M['k4'][l]
        data.setdefault(l, []).append((s, x1, x2, err - err.mean()))
coef = {}
print('layer | pooled b1 b2 | per-seed b1 .. | per-seed b2 .. | err var removed (pooled fit, per seed)')
for l in range(1, 15):
    rows = data[l]
    X = np.concatenate([np.column_stack([r[1], r[2]]) for r in rows]); y = np.concatenate([r[3] for r in rows])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    per = [np.linalg.lstsq(np.column_stack([r[1], r[2]]), r[3], rcond=None)[0] for r in rows]
    removed = [1 - np.var(r[3] - np.column_stack([r[1], r[2]]) @ b) / np.var(r[3]) for r in rows]
    coef[l] = [float(b[0]), float(b[1])]
    print('L%2d | %+.3f %+.3f | %s | %s | %s' % (l, b[0], b[1], ' '.join('%+.3f' % p[0] for p in per), ' '.join('%+.3f' % p[1] for p in per), ' '.join('%.2f' % r for r in removed)))
json.dump(dict(seeds=seeds, coef=coef, note='dk4 = -(b1 vv^2 (k3n^2 - mean) + b2 vv^2 (z^2 - mean)), k3n = 3 bpre / sqrt(vv)'), open(out, 'w'), indent=1)
print('written', out)
