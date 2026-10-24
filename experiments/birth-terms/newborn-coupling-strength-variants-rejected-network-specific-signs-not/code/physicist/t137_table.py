#!/usr/bin/env python3
"""TAG opus-physicist T137 table: the signed gated beta ('nets', c = 1, L8-14) on 7100584-607 vs scout-card's gk and ungated gbeta2 results (same
worker family, same point), paired per network; the control (gbg 'none' on 7100584) vs scout-card's gk must be bit for bit.
usage: t137_table.py"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SC = HERE.parent / 'scout_card'
mine = {json.loads(f.read_text())['id']: json.loads(f.read_text())['mse_layers'] for f in (HERE / 't137_gbgnets' / 'results').glob('ph137_*.json')}
def scout(seed, arm):
    vals = [json.loads(f.read_text())['mse_layers'] for f in SC.glob('run_*/results/*_r%d_%s.json' % (seed, arm))]
    if not vals: return None
    assert all(v == vals[0] for v in vals), ('scout copies differ', seed, arm)
    return vals[0]
c = mine.get('ph137_a_r7100584_gk+gbgnone'); g = scout(7100584, 'gk')
if c and g: print('control gbg none vs gk on 7100584: %s (worst rel %.2e)' % ('BIT FOR BIT' if c == g else 'DIFFERS', max(abs(a / b - 1) for a, b in zip(c, g))))
def line(name, A, B):
    s = sorted(set(A) & set(B))
    if len(s) < 2: return '%s: %d networks' % (name, len(s))
    a = np.array([A[x][-1] for x in s]); b = np.array([B[x][-1] for x in s]); lr = np.log(a / b)
    return '%s on %d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f), lower on %d of %d [noise-free x1.36: mean log %+.3f%%]; per network %s' % (
        name, len(s), 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(s)), int((a < b).sum()), len(s),
        136 * lr.mean(), ' '.join('%+.2f' % (100 * x) for x in lr))
seeds = range(7100584, 7100608)
N = {s: mine['ph137_r%d_gk+gbgnets' % s] for s in seeds if 'ph137_r%d_gk+gbgnets' % s in mine}
GK = {s: scout(s, 'gk') for s in seeds if scout(s, 'gk')}
B2 = {s: scout(s, 'gk+gbeta2') for s in seeds if scout(s, 'gk+gbeta2')}
print(line('gk+gbgnets (signed, c 1, L8-14) vs gk', N, GK))
print(line('gk+gbgnets vs gk+gbeta2 (ungated)', N, B2))
print(line('reference: gk+gbeta2 vs gk', B2, GK))
