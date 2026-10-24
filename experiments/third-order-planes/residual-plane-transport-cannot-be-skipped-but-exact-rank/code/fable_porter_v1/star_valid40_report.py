#!/usr/bin/env python3
"""TAG fable-porter (PT-29b): report for star_valid40_s6b.py. Per (card, arm): pooled raw = sum(arm)/sum(base) - 1 over the nets present, paired
bootstrap 95% CI over nets (20000 draws, seed 0), nets better, per-net range; lead bar (18:25): pooled raw <= +0.8% AND CI upper <= +1.5%.
Base = physicist eval_n40.json rows (b2: s6b_B2_ep1_valid40; j05: s6b_B2ep1_j05_1ep_valid40). Also: j05 base reproduction and PT-25 overlap.
usage: star_valid40_report.py <in.jsonl>"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; RF = HERE.parent / 'fable_physicist_v1/refit6b'
BASE = {c: {int(k): v for k, v in json.load(open(RF / f / 'eval_n40.json'))['valid'].items()}
        for c, f in (('b2', 's6b_B2_ep1_valid40'), ('j05', 's6b_B2ep1_j05_1ep_valid40'))}
rows = [json.loads(l) for l in open(sys.argv[1])]
print('TAG fable-porter PT-29: star-plane rank truncation, torch, no refit, VALID40 (valid[:40]); raw vs the card\'s own base rows')
for card, arm in sorted({(r['card'], r['arm']) for r in rows if r['arm'] != 'base'}):
    R = {r['rank']: r['mse'] for r in rows if r['card'] == card and r['arm'] == arm}; ks = sorted(R)
    a = np.array([R[k] for k in ks]); b = np.array([BASE[card][k] for k in ks]); n = len(ks)
    pooled = a.sum() / b.sum() - 1; rng = np.random.default_rng(0); idx = rng.integers(0, n, (20000, n))
    boot = a[idx].sum(1) / b[idx].sum(1) - 1; lo, hi = np.percentile(boot, [2.5, 97.5]); per = a / b - 1
    verdict = 'PASS' if (pooled <= 0.008 and hi <= 0.015 and n == 40) else ('FAIL' if n == 40 else f'partial n={n}')
    print(f'  {card:4s} {arm:4s}: n={n:2d} pooled raw {100*pooled:+.3f}% [{100*lo:+.3f}, {100*hi:+.3f}]  better {int((per < 0).sum())}/{n}  '
          f'per-net {100*per.min():+.2f}..{100*per.max():+.2f}%  bar (<=+0.8%, CI hi <=+1.5%): {verdict}')
for r in rows:
    if r['arm'] == 'base': print(f"  base reproduction {r['card']} r{r['rank']}: {r['mse']:.17g} vs eval_n40 {BASE[r['card']][r['rank']]:.17g} rel {r['mse']/BASE[r['card']][r['rank']]-1:+.2e}")
old = {(r['arm'], r['rank']): r['mse'] for r in map(json.loads, open(HERE / 'stack/star_ablation.jsonl'))}
ov = [(r['rank'], r['mse'] / old[(r['arm'], r['rank'])] - 1) for r in rows if r['card'] == 'b2' and (r['arm'], r['rank']) in old]
if ov: print(f"  PT-25 overlap (b2, same arm/net): {len(ov)} rows, max |rel| {max(abs(x) for _, x in ov):.2e}")
