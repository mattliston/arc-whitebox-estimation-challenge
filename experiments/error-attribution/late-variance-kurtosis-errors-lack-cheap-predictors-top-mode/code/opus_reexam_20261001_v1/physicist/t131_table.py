#!/usr/bin/env python3
"""TAG opus-physicist T131 table: star rank 8 at L11-14 only (t131_ss8late) vs the S50 frozen point's base (T129 base runs), per network paired; also the
T132 top-mode variance dial arms (t132_plus / t132_minus, d = +-0.02 at L13-15) with the finite-difference slope / curvature, the best d and its predicted gain.
usage: t131_table.py"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
def load(run, key):
    out = {}
    for f in (HERE / run / 'results').glob('*.json'):
        d = json.loads(f.read_text()); out[int(d['id'].split('_')[key])] = np.array(d['mse_layers'])
    return out
B = {int(json.loads(f.read_text())['id'].split('_')[1]): np.array(json.loads(f.read_text())['mse_layers']) for f in (HERE / 't129_floor_raise' / 'results').glob('t129_*_base.json')}
def line(name, A):
    nets = sorted(s for s in A if s in B)
    if len(nets) < 2:
        return None
    a = np.array([A[s] for s in nets]); b = np.array([B[s] for s in nets]); lr = np.log(a[:, -1] / b[:, -1])
    return nets, a, b, '%-12s on %2d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%; L11-15 %s' % (
        name, len(nets), 100 * (a[:, -1].mean() / b[:, -1].mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a[:, -1] < b[:, -1]).sum()),
        len(nets), 100 * (a.mean(1).mean() / b.mean(1).mean() - 1), ' '.join('%+.2f' % (100 * (a[:, l].mean() / b[:, l].mean() - 1)) for l in range(11, 16)))
r = line('ss8_L11-14', load('t131_ss8late', -1))
if r: print(r[3])
P, M = load('t132_plus', -1), load('t132_minus', -1)
rp, rm = line('dial +0.02', P), line('dial -0.02', M)
for x in (rp, rm):
    if x: print(x[3])
if rp and rm:
    nets = sorted(set(rp[0]) & set(rm[0]))
    fp = np.array([P[s][-1] for s in nets]).mean(); fm = np.array([M[s][-1] for s in nets]).mean(); f0 = np.array([B[s][-1] for s in nets]).mean()
    d = 0.02; g = (fp - fm) / (2 * d); h = (fp + fm - 2 * f0) / d ** 2
    if h > 0:
        ds = -g / h; gain = -g * g / (2 * h)
        print('dial finite differences on %d networks: slope %+.4e per unit d, curvature %+.4e: best d %+.4f, predicted final-layer change %+.3f%%' % (len(nets), g, h, ds, 100 * gain / f0))
    else:
        print('dial finite differences on %d networks: slope %+.4e, curvature %+.4e (not convex: no interior optimum)' % (len(nets), g, h))
