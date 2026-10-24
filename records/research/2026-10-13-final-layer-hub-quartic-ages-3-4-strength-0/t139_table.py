#!/usr/bin/env python3
"""TAG opus-physicist T139 table: arms ph139a (K4L ages 3-4 at 0.5 at L12-15) / ph139b (at L8-11 + 15) vs scout-card's s51k4lf34h (ages 3-4 at 0.5 at
L15 only) on the same seeds, and vs the physicist's t134 controls (S51 + gk, no k4lf34h) for reference. Pre-declared rule: mean log <= -0.15% vs
s51k4lf34h on 8 -> extend to 24 (the 16 new networks must agree in sign).
usage: t139_table.py"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SC = HERE.parent / 'scout_card'
mine = {json.loads(f.read_text())['id']: json.loads(f.read_text())['mse_layers'] for f in (HERE / 't139_k4l_inner' / 'results').glob('ph139*.json')}
ref = {}
for f in SC.glob('run_k7/results/sc_k*_r*_s51k4lf34h.json'):
    d = json.loads(f.read_text()); ref[int(d['id'].split('_r')[1].split('_')[0])] = d['mse_layers']
ctl = {s: json.loads((HERE / 't134_refit_screen' / 'results' / ('t134_a_ctl_on_%d.json' % s)).read_text())['mse_layers'] for s in range(7100584, 7100592)}
def line(name, A, B):
    s = sorted(set(A) & set(B))
    if len(s) < 2: return '%s: %d networks' % (name, len(s))
    a = np.array([A[x][-1] for x in s]); b = np.array([B[x][-1] for x in s]); lr = np.log(a / b)
    la = np.array([A[x] for x in s]); lb = np.array([B[x] for x in s])
    return '%s on %d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f), lower on %d of %d [noise-free x1.36 mean log %+.3f%%]; L11-15 %s; per network %s' % (
        name, len(s), 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(s)), int((a < b).sum()), len(s), 136 * lr.mean(),
        ' '.join('%+.2f' % (100 * (la[:, l].mean() / lb[:, l].mean() - 1)) for l in range(11, 16)), ' '.join('%+.2f' % (100 * x) for x in lr))
for tag, name in (('a', 'K4L 3-4 at 0.5 inner L12-14 (+final)'), ('b', 'K4L 3-4 at 0.5 inner L8-11 (+final)')):
    A = {int(k.split('_r')[1]): v for k, v in mine.items() if k.startswith('ph139%s_' % tag)}
    print(line(name + ' vs s51k4lf34h', A, ref))
    print(line(name + ' vs t134 control (S51 + gk)', A, ctl))
print(line('reference: s51k4lf34h vs t134 control', ref, ctl))
