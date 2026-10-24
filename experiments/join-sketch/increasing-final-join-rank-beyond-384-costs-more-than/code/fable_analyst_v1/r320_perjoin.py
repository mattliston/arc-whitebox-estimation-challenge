#!/usr/bin/env python3
"""TAG fable-analyst (AN-69e): CAUSAL per-join R320 router. Per net and schedule s in {384-384-384, 320-320-320, 384-320-320, 320-384-384}
(ranks at the L7 / L10 / L13 joins): effect_s = (1 + raw_s)(1 + bill_s) - 1 relative to all-384, where raw_s = torch MSE(s)/MSE(384) - 1
(the torch card matches native per-net R320 penalties at slope .996, corr .995) and bill_s = the net's all-320 bill change x the share of
the rank-dependent cost in s's rank-320 segments (torch logs: segment L7-9 + L7 join, L10-12 + L10 join, L13-15 + L13 join; porter unit
costs). The all-320 bill change is the native per-net value on VALID40 (S25 vs no-R320 rows) and the VALID40 mean on the held-out nets.
Routers (the decision for L10 and L13 uses resp_L10, known at L10; L7 must be decided with L7-time information):
  nonc   all-320 iff resp_L10 >= T (NOT causal: reference only)
  c384   L7 at 384, L10+L13 at 320 iff resp_L10 >= T
  c320   L7 at 320, L10+L13 at 320 iff resp_L10 >= T (else 384 there)
  c7x    L7 at 320 iff <L7 feature> >= T7 (either direction), L10+L13 as c384/c320
Scores: pooled adjusted effect sum_i adj_off_i * (1 + effect_i) / sum_i adj_off_i - 1 (adj_off = the no-R320 native adjusted; on the held-out
nets, torch MSE(384) x the VALID40 mean util). VALID40: leave-one-out thresholds; held-out: the thresholds fitted on all of VALID40.
usage: r320_perjoin.py"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
HERE = Path(__file__).resolve().parent; PO = HERE.parent / 'fable_porter_v1'
SCH = ('fixed384', 'fixed320', 'sched384-320-320', 'sched320-384-384')
def load(files):
    M, L = defaultdict(dict), {}
    for f in files:
        for l in open(HERE / f):
            r = json.loads(l); M[r['rank']][r['arm']] = r['mse']
            if r['arm'] == 'fixed384': L[r['rank']] = r
    return M, L
def seg_shares(r):
    c = {7: 0.0, 10: 0.0, 13: 0.0}
    for x in r['layers']:
        if x['rank'] > 0 and x['layer'] >= 7: c[7 if x['layer'] < 10 else (10 if x['layer'] < 13 else 13)] += 0.174 + 0.323 * x['old'] / 1000
    for j in r['joins']: c[j['layer']] += 0.718 * j['joined'] / 1000
    tot = sum(c.values()); return {k: v / tot for k, v in c.items()}
def effects(M, L, nets, bill_all):
    E = {s: np.zeros(len(nets)) for s in SCH}; feats = defaultdict(lambda: np.zeros(len(nets)))
    for i, n in enumerate(nets):
        sh = seg_shares(L[n]); b = bill_all[i]
        frac = {'fixed384': 0, 'fixed320': 1, 'sched384-320-320': sh[10] + sh[13], 'sched320-384-384': sh[7]}
        for s in SCH: E[s][i] = (M[n][s] / M[n]['fixed384']) * (1 + b * frac[s]) - 1
        C = L[n]['counts']
        for l in (7, 10): feats[f'resp_L{l}'][i] = C[str(l)]['resp']; feats[f'kinkfrac_L{l}'][i] = C[str(l)]['resp'] / C[str(l)]['alive']; feats[f'alive_L{l}'][i] = C[str(l)]['alive']
        feats['live_L7'][i] = next(j['live'] for j in L[n]['joins'] if j['layer'] == 7); feats['miss320_L7'][i] = next(1 - j['E']['320'] for j in L[n]['joins'] if j['layer'] == 7)
    return E, feats
def choose(E, rule, feats, T10, T7=None):
    x10 = feats['resp_L10'] >= T10
    if rule == 'nonc': return np.where(x10, E['fixed320'], 0.0)
    if rule == 'c384': return np.where(x10, E['sched384-320-320'], 0.0)
    if rule == 'c320': return np.where(x10, E['fixed320'], E['sched320-384-384'])
    k7, t7, sg = T7; on7 = sg * (feats[k7] - t7) >= 0
    return np.where(on7, np.where(x10, E['fixed320'], E['sched320-384-384']), np.where(x10, E['sched384-320-320'], 0.0))
def pooled(eff, w): return float((w * (1 + eff)).sum() / w.sum() - 1)
def fit(E, feats, w, m, rule):
    best = (np.inf, None, None)
    t10s = np.unique(feats['resp_L10'][m]); cands7 = [None] if rule != 'c7x' else [(k, t, sg) for k in ('resp_L7', 'kinkfrac_L7', 'alive_L7', 'live_L7', 'miss320_L7') for t in np.unique(feats[k][m]) for sg in (1, -1)]
    for T10 in list(t10s) + [1e9]:
        for T7 in cands7:
            v = pooled(choose(E, rule, feats, T10, T7)[m], w[m])
            if v < best[0]: best = (v, T10, T7)
    return best
# VALID40
A = json.load(open(PO / 'paired_valid40_stack_S25r_mem.json')); B = json.load(open(PO / 'paired_valid40_s25r_noR320_mem.json')); nets = A['ranks']
g = lambda D, k: np.array([D['results'][f'cand:{r}'][k] for r in nets], float)
bill_v = g(A, 'util') / g(B, 'util') - 1; w_v = g(B, 'adjusted'); nat = g(A, 'adjusted') / w_v - 1
Mv, Lv = load(['r320_features_v40.jsonl', 'r320_features_v40_sched.jsonl'])
Ev, Fv = effects(Mv, Lv, nets, bill_v)
print(f"VALID40 check: torch-derived all-320 effect {100 * pooled(Ev['fixed320'], w_v):+.3f}% vs native {100 * pooled(nat, w_v):+.3f}% (corr per net {np.corrcoef(Ev['fixed320'], nat)[0, 1]:+.3f})")
orc = np.minimum.reduce([np.zeros(40), Ev['fixed320'], Ev['sched384-320-320'], Ev['sched320-384-384']])
print(f"VALID40 oracles: all-joins on/off {100 * pooled(np.minimum(0, Ev['fixed320']), w_v):+.3f}%, per-net best of 4 schedules {100 * pooled(orc, w_v):+.3f}%; "
      f"always 384-320-320 {100 * pooled(Ev['sched384-320-320'], w_v):+.3f}%, always 320-384-384 {100 * pooled(Ev['sched320-384-384'], w_v):+.3f}%, always all-320 {100 * pooled(Ev['fixed320'], w_v):+.3f}%")
fits = {}
for rule in ('nonc', 'c384', 'c320', 'c7x'):
    picks = np.zeros(40)
    for i in range(40):
        m = np.ones(40, bool); m[i] = False; _, T10, T7 = fit(Ev, Fv, w_v, m, rule); picks[i] = choose(Ev, rule, Fv, T10, T7)[i]
    fits[rule] = fit(Ev, Fv, w_v, np.ones(40, bool), rule)
    print(f"VALID40 LOO {rule}: {100 * pooled(picks, w_v):+.3f}%   (full-data fit: T10 {fits[rule][1]}, T7 {fits[rule][2]}, in-sample {100 * fits[rule][0]:+.3f}%)")
# held-out
Mh, Lh = load(['r320_features_ho40.jsonl']); hn = sorted(k for k in Mh if all(s in Mh[k] for s in SCH))
if hn:
    bill_h = np.full(len(hn), bill_v.mean()); Eh, Fh = effects(Mh, Lh, hn, bill_h); w_h = np.array([Mh[n]['fixed384'] for n in hn]) * g(B, 'util').mean()
    orc_h = np.minimum(0, Eh['fixed320']); orc4 = np.minimum.reduce([np.zeros(len(hn)), Eh['fixed320'], Eh['sched384-320-320'], Eh['sched320-384-384']])
    print(f"HELD-OUT {len(hn)} nets: always all-320 {100 * pooled(Eh['fixed320'], w_h):+.3f}%, always 384-320-320 {100 * pooled(Eh['sched384-320-320'], w_h):+.3f}%, "
          f"oracle all-joins {100 * pooled(orc_h, w_h):+.3f}%, oracle of 4 schedules {100 * pooled(orc4, w_h):+.3f}%")
    for rule, (_, T10, T7) in fits.items():
        e = pooled(choose(Eh, rule, Fh, T10, T7), w_h)
        print(f"HELD-OUT rule {rule} (VALID40 thresholds): {100 * e:+.3f}%  ({e / pooled(orc_h, w_h):.2f} of the all-joins oracle, {e / pooled(orc4, w_h):.2f} of the 4-schedule oracle)")
else:
    print('held-out rows not ready')
