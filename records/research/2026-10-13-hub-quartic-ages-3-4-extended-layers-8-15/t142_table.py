#!/usr/bin/env python3
"""TAG opus-physicist T142 table: each scan arm vs T141's C on the same networks (paired, final-layer MSE), with per-layer L8-15.
Pre-declared: mean log <= -0.15% vs C on 8 -> extend to 24 (the 16 new networks must agree in sign). usage: t142_table.py"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
C = {}
for f in (HERE / 't141_k4l_ext' / 'results').glob('ph141_r*_c.json'):
    d = json.loads(f.read_text()); C[int(d['id'].split('_r')[1].split('_')[0])] = d['mse_layers']
X = {}
for f in (HERE / 't142_k4l_scan' / 'results').glob('ph142_*.json'):
    d = json.loads(f.read_text()); _, r, arm = d['id'].split('_'); X.setdefault(arm, {})[int(r[1:])] = d['mse_layers']
names = {'i': 'inner 0.75 at L8-14', 'ii': 'inner 1.0 at L8-14', 'iii': '+ L4-7 at 0.5', 'iv': 'final raised to 0.75'}
for arm in ('i', 'ii', 'iii', 'iv'):
    A = X.get(arm, {}); s = sorted(set(A) & set(C))
    if len(s) < 2:
        print('%s (%s): %d networks' % (arm, names[arm], len(s))); continue
    a = np.array([A[x][-1] for x in s]); b = np.array([C[x][-1] for x in s]); lr = np.log(a / b)
    la = np.array([A[x] for x in s]); lb = np.array([C[x] for x in s])
    print('%-4s %-22s vs C on %d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f), lower on %d of %d [noise-free x1.36 %+.3f%%]; L8-15 %s' % (
        arm, names[arm], len(s), 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(s)), int((a < b).sum()), len(s),
        136 * lr.mean(), ' '.join('%+.2f' % (100 * (la[:, l].mean() / lb[:, l].mean() - 1)) for l in range(8, 16))))
