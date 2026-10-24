#!/usr/bin/env python3
"""TAG fable-analyst (AN-69): PER-NET R320 ROUTER search (lead's top priority). Target: R320's exact per-net effect from the porter's native
VALID40 rows, S25 (paired_valid40_stack_S25r_mem.json) vs the same card without R320 (paired_valid40_s25r_noR320_mem.json):
  d_i = adj_S25_i - adj_off_i;  routing a set S gives the pooled adjusted effect sum_{i in S} d_i / sum_i adj_off_i.
Oracle: S = {d_i < 0}. Routers (leave-one-out over the 40 nets): (1) a threshold on one feature (either direction, fit on 39 nets to
minimize the summed d on the routed set); (2) two features: a threshold on the sum of their standardized values with a weight grid;
(3) logistic regression on 1-3 standardized features (ridge), routing where P(d < 0) > 1/2. Features come from a JSON file
{feature name: {rank: value}} (merged from several files) plus the built-in 'bill_share' = R320's own per-net bill change (from the rows).
The lead's bar: LOO captures >= 50% of the oracle (i.e. <= half of the oracle's pooled effect), then a held-out check on non-VALID nets.
usage: r320_router.py [features.json ...]"""
import sys, json, itertools
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; PO = HERE.parent / 'fable_porter_v1'
A = json.load(open(PO / 'paired_valid40_stack_S25r_mem.json')); B = json.load(open(PO / 'paired_valid40_s25r_noR320_mem.json'))
assert A['ranks'] == B['ranks']; nets = A['ranks']
g = lambda D, k: np.array([D['results'][f'cand:{r}'][k] for r in nets], float)
aon, aoff = g(A, 'adjusted'), g(B, 'adjusted'); d = aon - aoff; den = aoff.sum()
F = {'bill_share': g(A, 'util') / g(B, 'util') - 1, 'raw_off': g(B, 'mse'), 'util_off': g(B, 'util')}
for f in sys.argv[1:]:
    for k, v in json.load(open(f)).items(): F[k] = np.array([float(v[str(r)]) for r in nets])
eff = lambda S: float(d[S].sum() / den)
oracle, always = eff(d < 0), eff(np.ones(40, bool))
print(f'R320 per-net effect (exact off arm): always-on {100 * always:+.3f}%, oracle {100 * oracle:+.3f}% ({int((d < 0).sum())}/40), '
      f'headroom over the better of always/never {100 * (oracle - min(always, 0)):+.3f} pt; bar = LOO <= {100 * oracle / 2:+.3f}%')
def fit_thr(x, m):
    best = (0.0, None, 1)
    for t in np.unique(x[m]):
        for sgn in (1, -1):
            on = m & (sgn * (x - t) >= 0); val = d[on].sum()
            if val < best[0]: best = (val, t, sgn)
    return best
def loo_thr(x):
    picks = np.zeros(40, bool)
    for i in range(40):
        m = np.ones(40, bool); m[i] = False; _, t, sgn = fit_thr(x, m)
        picks[i] = t is not None and sgn * (x[i] - t) >= 0
    return picks
def z(x): return (x - x.mean()) / (x.std() + 1e-300)
def loo_logit(X, lam=1.0):
    picks = np.zeros(40, bool); yb = (d < 0).astype(float)
    for i in range(40):
        m = np.ones(40, bool); m[i] = False; Xm = np.column_stack([np.ones(39), X[m]]); w = np.zeros(Xm.shape[1])
        for _ in range(200):                                    # Newton steps, ridge on the slopes
            p = 1 / (1 + np.exp(-Xm @ w)); G = Xm.T @ (p - yb[m]) + lam * np.r_[0, w[1:]]
            H = Xm.T @ (Xm * (p * (1 - p))[:, None]) + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
            w -= np.linalg.solve(H + 1e-9 * np.eye(len(w)), G)
        picks[i] = (np.r_[1, X[i]] @ w) > 0
    return picks
DIAG = [k for k in F if k.endswith('_DIAG') or k == 'raw_off']          # not observable at runtime
for k in DIAG: print(f'diagnostic {k}: corr with the native relative effect {np.corrcoef(F[k], d / aoff)[0, 1]:+.2f}')
F = {k: v for k, v in F.items() if k not in DIAG and np.ptp(v) > 0}
# honest estimate: NESTED LOO = pick the single feature (and its threshold) inside each fold, predict the held-out net
nested = np.zeros(40, bool); chosen = []
for i in range(40):
    m = np.ones(40, bool); m[i] = False; best = (0.0, None, None, 1)
    for k, x in F.items():
        v, t, sgn = fit_thr(x, m)
        if v < best[0]: best = (v, k, t, sgn)
    _, k, t, sgn = best; chosen.append(k); nested[i] = k is not None and sgn * (F[k][i] - t) >= 0
from collections import Counter
print(f'NESTED LOO over all {len(F)} observable features (feature + threshold chosen inside each fold): {100 * eff(nested):+.3f}% '
      f'({int(nested.sum())}/40 routed), captured share {eff(nested) / oracle:.2f}; features chosen: {Counter(chosen).most_common(4)}')
rows = []
for k, x in F.items():
    p = loo_thr(x); rows.append((f'threshold on {k}', eff(p), int(p.sum()), float(np.corrcoef(x, d / aoff)[0, 1])))
    p = loo_logit(z(x)[:, None]); rows.append((f'logistic on {k}', eff(p), int(p.sum()), None))
obs = ['bill_share'] + [k for k in (sys.argv[2:] and []) ]  # pairs: bill_share with each feature (below)
obs = list(F)
for a_, b_ in [('bill_share', k) for k in obs if k != 'bill_share']:
    X = np.column_stack([z(F[a_]), z(F[b_])])
    p = loo_logit(X); rows.append((f'logistic on {a_} + {b_}', eff(p), int(p.sum()), None))
    best = None
    for wgt in np.linspace(-1, 1, 21):                          # threshold on a weighted sum, weight chosen inside LOO
        picks = np.zeros(40, bool)
        for i in range(40):
            m = np.ones(40, bool); m[i] = False; bw = (0.0, None, None, 1)
            for w2 in np.linspace(-1, 1, 21):
                s = z(F[a_]) + w2 * z(F[b_]); v, t, sgn = fit_thr(s, m)
                if v < bw[0]: bw = (v, w2, t, sgn)
            _, w2, t, sgn = bw
            picks[i] = t is not None and sgn * ((z(F[a_]) + w2 * z(F[b_]))[i] - t) >= 0
        best = picks; break
    rows.append((f'threshold on {a_} + w x {b_} (w fit in LOO)', eff(best), int(best.sum()), None))
print('router (LOO over 40 nets) | pooled adjusted effect | nets routed | captured share of the oracle | corr(feature, relative effect)')
for name, e, n, c in sorted(rows, key=lambda r: r[1]):
    print(f'  {name:60s} {100 * e:+.3f}%  {n:2d}/40  {e / oracle:5.2f}  ' + (f'{c:+.2f}' if c is not None else ''))
json.dump(dict(always=always, oracle=oracle, rows=[dict(router=r[0], effect=r[1], routed=r[2]) for r in rows]), open(HERE / 'r320_router.json', 'w'), indent=1)
