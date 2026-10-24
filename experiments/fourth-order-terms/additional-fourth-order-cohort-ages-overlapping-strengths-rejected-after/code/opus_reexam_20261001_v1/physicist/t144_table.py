#!/usr/bin/env python3
"""TAG opus-physicist T144 table: ARM P1 (K4L ages 5-6 at 0.5 where unjoined, on top of C) vs T141's C, paired; the control must equal C bit for bit.
Pre-declared: mean log <= -0.15% vs C on 8 -> extend to 24 (the 16 new networks must agree in sign). usage: t144_table.py"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
C = {int(json.loads(f.read_text())['id'].split('_r')[1].split('_')[0]): json.loads(f.read_text())['mse_layers'] for f in (HERE / 't141_k4l_ext' / 'results').glob('ph141_r*_c.json')}
X = {json.loads(f.read_text())['id']: json.loads(f.read_text())['mse_layers'] for f in (HERE / 't144_k4l56_inner' / 'results').glob('ph144*.json')}
c = X.get('ph144c_r7100584')
if c: print('control (ages 5-6 off) vs C on 7100584: %s (worst rel %.2e)' % ('BIT FOR BIT' if c == C[7100584] else 'DIFFERS', max(abs(a / b - 1) for a, b in zip(c, C[7100584]))))
A = {int(k.split('_r')[1]): v for k, v in X.items() if k.startswith('ph144a_')}
s = sorted(set(A) & set(C))
if len(s) >= 2:
    a = np.array([A[x][-1] for x in s]); b = np.array([C[x][-1] for x in s]); lr = np.log(a / b)
    la = np.array([A[x] for x in s]); lb = np.array([C[x] for x in s])
    print('P1 K4L ages 5-6 at 0.5 where unjoined (on C) vs C on %d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f), lower on %d of %d [noise-free x1.36 %+.3f%%]; L8-15 %s; per network %s'
          % (len(s), 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(s)), int((a < b).sum()), len(s), 136 * lr.mean(),
             ' '.join('%+.2f' % (100 * (la[:, l].mean() / lb[:, l].mean() - 1)) for l in range(8, 16)), ' '.join('%+.2f' % (100 * x) for x in lr)))
