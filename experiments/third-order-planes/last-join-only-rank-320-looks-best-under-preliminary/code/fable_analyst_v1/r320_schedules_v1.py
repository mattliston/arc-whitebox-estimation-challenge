#!/usr/bin/env python3
"""TAG fable-analyst (AN-69g): fixed per-join old-tier rank schedules, priced per net. raw = torch MSE(s) / MSE(384) - 1 (the torch card
matches native R320 penalties at slope .996); bill = the net's native all-320 bill change (VALID40; the VALID40 mean on other nets) x the
schedule's saved fraction: segment shares of the rank-dependent cost (torch logs, porter unit costs) x (384 - r) / 64 per segment.
Pooled adjusted effect vs all-384, weighted by the no-R320 adjusted (VALID40 native; elsewhere torch MSE(384) x mean util).
usage: r320_schedules.py <valid40|heldout40>"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE = Path(__file__).resolve().parent; PO = HERE.parent / 'fable_porter_v1'
exec(open(HERE / 'r320_perjoin.py').read().split('# VALID40')[0].split('def effects')[0])
A = json.load(open(PO / 'paired_valid40_stack_S25r_mem.json')); B = json.load(open(PO / 'paired_valid40_s25r_noR320_mem.json'))
g = lambda D, k: np.array([D['results'][f'cand:{r}'][k] for r in A['ranks']], float)
bill_v = dict(zip(A['ranks'], g(A, 'util') / g(B, 'util') - 1)); w_v = dict(zip(A['ranks'], g(B, 'adjusted'))); mu = g(B, 'util').mean()
which = sys.argv[1]
files = ['r320_features_v40.jsonl', 'r320_features_v40_sched.jsonl', 'r320_features_v40_sched2.jsonl', 'r320_features_v40_sched3.jsonl', 'r320_features_v40_sched4.jsonl'] if which == 'valid40' else ['r320_features_ho40.jsonl', 'r320_features_ho40_sched2.jsonl', 'r320_features_ho40_sched3.jsonl']
M, L = load([f for f in files if (HERE / f).exists()])
nets = sorted(M); arms = sorted({a for n in nets for a in M[n]} - {'fixed384'})
print(f'{which}: {len(nets)} nets')
for s in arms:
    ok = [n for n in nets if s in M[n]]
    if len(ok) < len(nets): continue
    r = [384] * 3 if s == 'fixed384' else ([int(s[5:])] * 3 if s.startswith('fixed') else [int(x) for x in s[5:].split('-')])
    raw = np.array([M[n][s] / M[n]['fixed384'] - 1 for n in nets]); eff = np.zeros(len(nets)); bl = np.zeros(len(nets)); w = np.zeros(len(nets))
    for i, n in enumerate(nets):
        sh = seg_shares(L[n]); frac = sum(sh[j] * (384 - rr) / 64 for j, rr in zip((7, 10, 13), r))
        b = bill_v.get(n, np.mean(list(bill_v.values()))) * frac; bl[i] = b; eff[i] = (1 + raw[i]) * (1 + b) - 1
        w[i] = w_v[n] if n in w_v else M[n]['fixed384'] * mu
    print(f'  {s:18s} raw {100 * raw.mean():+.2f}% (se {100 * raw.std(ddof=1) / np.sqrt(len(raw)):.2f})  bill {100 * bl.mean():+.2f}%  '
          f'pooled adjusted {100 * ((w * (1 + eff)).sum() / w.sum() - 1):+.3f}%  better {int((eff < 0).sum())}/{len(nets)}')
