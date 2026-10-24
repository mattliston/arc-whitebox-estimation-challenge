#!/usr/bin/env python3
"""TAG fable-analyst (AN-69m, v2; v1 = r320_schedules_v1.py with the unit-cost bill model, WRONG for L13: the native L13 join is lazy, see
r320_lazy_model.py): fixed per-join old-tier rank schedules, priced per net. raw = torch MSE(s) / MSE(384) - 1 (the torch card matches native
R320 penalties at slope .996). bill = NATIVE (lead 11:22: price bills natively): per net from the VALID40 paired rows where the schedule was
built (384/384/320 = S27s vs S27, 384/320/320 = S27r vs S27, all-320 = S25r vs S25 no-R320), else the porter's native bill map (mean of nets
35/100 vs S27, no setup = the VALID40 util definition; fable_porter_v1/stack/billmap); nets without a paired row get the VALID40 mean.
Pooled adjusted effect vs all-384, weighted by the no-R320 adjusted (VALID40 native; elsewhere torch MSE(384) x mean util).
usage: r320_schedules.py <valid40|heldout40|pool80>  (v3: pool80 = both pools, bootstrap 95% CI over nets, stratified by pool)"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE = Path(__file__).resolve().parent; PO = HERE.parent / 'fable_porter_v1'
exec(open(HERE / 'r320_perjoin.py').read().split('# VALID40')[0].split('def effects')[0])
A = json.load(open(PO / 'paired_valid40_stack_S25r_mem.json')); B = json.load(open(PO / 'paired_valid40_s25r_noR320_mem.json'))
g = lambda D, k: np.array([D['results'][f'cand:{r}'][k] for r in A['ranks']], float)
bill_v = dict(zip(A['ranks'], g(A, 'util') / g(B, 'util') - 1)); w_v = dict(zip(A['ranks'], g(B, 'adjusted'))); mu = g(B, 'util').mean()
def paired(cand, parent='stack_S27_mem'):
    C = json.load(open(PO / f'paired_valid40_{cand}.json')); P = json.load(open(PO / f'paired_valid40_{parent}.json'))
    return {r: C['results'][f'cand:{r}']['util'] / P['results'][f'cand:{r}']['util'] - 1 for r in C['ranks']}
NATIVE = {'320-320-320': bill_v, '384-384-320': paired('stack_S27s_mem'), '384-320-320': paired('stack_S27r_mem')}
BM = PO / 'stack' / 'billmap'; T0 = json.load(open(BM / 'census_S27.json'))['totals']
for f in sorted(BM.glob('census_*-*-*.json')):
    k = f.stem[7:]
    if k not in NATIVE:
        T = json.load(open(f))['totals']; NATIVE[k] = {'map': np.mean([T[n] / T0[n] - 1 for n in T0])}
FILES = {'valid40': ['r320_features_v40.jsonl', 'r320_features_v40_sched.jsonl', 'r320_features_v40_sched2.jsonl', 'r320_features_v40_sched3.jsonl', 'r320_features_v40_sched4.jsonl'],
         'heldout40': ['r320_features_ho40.jsonl', 'r320_features_ho40_sched2.jsonl', 'r320_features_ho40_sched3.jsonl']}


def price(which):
    """{schedule: (raw, bill, eff, w)} per net; None bill if no native bill for the schedule."""
    M, L = load([f for f in FILES[which] if (HERE / f).exists()])
    nets = sorted(M); out = {}
    for s in sorted({a for n in nets for a in M[n]} - {'fixed384'}):
        if any(s not in M[n] for n in nets): continue
        r = [int(s[5:])] * 3 if s.startswith('fixed') else [int(x) for x in s[5:].split('-')]
        raw = np.array([M[n][s] / M[n]['fixed384'] - 1 for n in nets]); tab = NATIVE.get('-'.join(map(str, r)))
        if tab is None: out[s] = (raw, None, None, None); continue
        bl = np.array([tab.get(n, np.mean(list(tab.values()))) for n in nets]); eff = (1 + raw) * (1 + bl) - 1
        w = np.array([w_v[n] if n in w_v else M[n]['fixed384'] * mu for n in nets]); out[s] = (raw, bl, eff, w)
    return out


def pooled(parts, boot=2000, seed=0):
    val = lambda ps: sum((w * (1 + e)).sum() for e, w in ps) / sum(w.sum() for e, w in ps) - 1
    rng = np.random.default_rng(seed); bs = []
    for _ in range(boot):
        ix = [rng.integers(0, len(e), len(e)) for e, w in parts]; bs.append(val([(e[i], w[i]) for (e, w), i in zip(parts, ix)]))
    return val(parts), np.percentile(bs, [2.5, 97.5])


which = sys.argv[1]
print('native bills: ' + ', '.join(f"{k} {100 * np.mean(list(v.values())):+.3f}%" for k, v in sorted(NATIVE.items())))
pools = ['valid40', 'heldout40'] if which == 'pool80' else [which]
P = {q: price(q) for q in pools}
print(f"{which}: " + ', '.join(f'{q} {len(next(iter(P[q].values()))[0])} nets' for q in pools))
for s in sorted(set.intersection(*[set(P[q]) for q in pools])):
    rows = [P[q][s] for q in pools]
    raw = np.concatenate([x[0] for x in rows])
    if any(x[1] is None for x in rows):
        print(f'  {s:18s} raw {100 * raw.mean():+.2f}% (se {100 * raw.std(ddof=1) / np.sqrt(len(raw)):.2f})  no native bill yet'); continue
    v, ci = pooled([(x[2], x[3]) for x in rows]); eff = np.concatenate([x[2] for x in rows])
    print(f'  {s:18s} raw {100 * raw.mean():+.2f}% (se {100 * raw.std(ddof=1) / np.sqrt(len(raw)):.2f})  bill {100 * np.concatenate([x[1] for x in rows]).mean():+.2f}%  '
          f'pooled adjusted {100 * v:+.3f}% [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]  better {int((eff < 0).sum())}/{len(eff)}')
