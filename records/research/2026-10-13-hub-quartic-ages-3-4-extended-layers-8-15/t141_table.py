#!/usr/bin/env python3
"""TAG opus-physicist T141 table: arms A / B (T139's 7100584-591 + T141's 7100592-607) and C (T141, 7100584-607) vs the S51 reference s51k4lf34h
(scout-card's on 584-591, T141's on 592-607), paired per network; all 24 and the out-of-selection 16 (592-607) separately; C vs A and C vs B.
Pre-declared: a 24-network pass needs the out-of-selection 16 to agree in sign (mean log < 0). usage: t141_table.py"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SC = HERE.parent / 'scout_card'
def load(glob, key):
    out = {}
    for f in glob:
        d = json.loads(f.read_text()); out[key(d['id'])] = d['mse_layers']
    return out
ref = load(SC.glob('run_k7/results/sc_k*_r*_s51k4lf34h.json'), lambda i: int(i.split('_r')[1].split('_')[0]))
t141 = {}
for f in (HERE / 't141_k4l_ext' / 'results').glob('ph141_*.json'):
    d = json.loads(f.read_text()); _, r, arm = d['id'].split('_'); t141.setdefault(arm, {})[int(r[1:])] = d['mse_layers']
ref.update(t141.get('ref', {}))
A = load((HERE / 't139_k4l_inner' / 'results').glob('ph139a_*.json'), lambda i: int(i.split('_r')[1])); A.update(t141.get('a', {}))
B = load((HERE / 't139_k4l_inner' / 'results').glob('ph139b_*.json'), lambda i: int(i.split('_r')[1])); B.update(t141.get('b', {}))
C = t141.get('c', {})
def line(name, X, Y, seeds):
    s = [x for x in seeds if x in X and x in Y]
    if len(s) < 2: return '%s: %d networks' % (name, len(s))
    a = np.array([X[x][-1] for x in s]); b = np.array([Y[x][-1] for x in s]); lr = np.log(a / b)
    la = np.array([X[x] for x in s]); lb = np.array([Y[x] for x in s])
    return '%s on %d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f), lower on %d of %d [noise-free x1.36 mean log %+.3f%%]; L11-15 %s' % (
        name, len(s), 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(s)), int((a < b).sum()), len(s), 136 * lr.mean(),
        ' '.join('%+.2f' % (100 * (la[:, l].mean() / lb[:, l].mean() - 1)) for l in range(11, 16)))
ALL, OUT = range(7100584, 7100608), range(7100592, 7100608)
for nm, X in (('A (L12-15)', A), ('B (L8-11 + 15)', B), ('C (L8-15)', C)):
    print(line(nm + ' vs s51k4lf34h, all', X, ref, ALL)); print(line(nm + ' vs s51k4lf34h, out-of-selection 592-607', X, ref, OUT))
print(line('C vs A', C, A, ALL)); print(line('C vs B', C, B, ALL))
