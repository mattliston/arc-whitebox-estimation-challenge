#!/usr/bin/env python3
"""TAG fable-porter (PT-40b): report for join_sketch_test.py: control A vs covhunter's r1 fit rows (reproduction), and per arm pooled raw
vs A over the nets present, paired bootstrap 95% CI (20000, seed 0), nets better, per-net range; bar raw <= +0.1% (adjusted <= -1.66% at
-1.76% bill). usage: join_sketch_report.py <in.jsonl>"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(sys.argv[1])]
ref = json.load(open(HERE.parent / 'fable_covhunter_v1/r1_ep2dials_vs_S19.json'))['rows']
A = {r['rank']: r['mse'] for r in rows if r['arm'] == 'A'}
rep = [A[k] / ref[str(k)]['fit_s18'] - 1 for k in A]
print(f"TAG fable-porter PT-40: join sketch, S21-physics torch card (ep2 + round-1 dials, shared, lrstar), no refit, VALID16; n(A) = {len(A)}")
print(f"  control A vs covhunter r1 fit rows: max |rel| {max(abs(x) for x in rep):.2e} over {len(rep)} nets (must be ~0)")
for arm in ('B_lit', 'B', 'C'):
    R = {r['rank']: r['mse'] for r in rows if r['arm'] == arm and r['rank'] in A}; ks = sorted(R)
    if not ks: continue
    a = np.array([R[k] for k in ks]); b = np.array([A[k] for k in ks]); n = len(ks)
    pooled = a.sum() / b.sum() - 1; idx = np.random.default_rng(0).integers(0, n, (20000, n)); boot = a[idx].sum(1) / b[idx].sum(1) - 1
    lo, hi = np.percentile(boot, [2.5, 97.5]); per = a / b - 1
    print(f"  {arm:5s}: n={n:2d} pooled raw vs A {100*pooled:+.3f}% [{100*lo:+.3f}, {100*hi:+.3f}]  better {int((per < 0).sum())}/{n}  per-net {100*per.min():+.2f}..{100*per.max():+.2f}%"
          f"  -> adjusted at -1.76% bill {100*((1+pooled)*(1-0.0176)-1):+.2f}%  bar (raw <= +0.1%): {'PASS' if pooled <= 0.001 else 'FAIL'}")
