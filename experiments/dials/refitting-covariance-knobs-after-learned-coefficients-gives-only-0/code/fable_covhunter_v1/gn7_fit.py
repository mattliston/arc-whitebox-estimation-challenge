#!/usr/bin/env python3
"""TAG fable-covhunter (CH-33): ridge-pulled Gauss-Newton fit of the 5 knob deviations from v3 on the S6b card (gn7_s6b.py output):
penalty lam * |dev/step|^2 toward v3, lam by leave-one-net-out; all knobs and every subset; prints the best settings as JSON.
usage: gn7_fit.py <gn7.npz>"""
import sys, json, itertools
import numpy as np
d = np.load(sys.argv[1]); knobs = list(d['knobs']); ranks = list(d['ranks']); steps = d['steps']
E0 = {r: d[f'e0_{r}'] for r in ranks}; U = {r: d[f'U_{r}'] * steps[None, :] for r in ranks}
V3 = {'iso': -0.094, 'k21n': -0.028, 'k31': -0.212, 'edge': 0.164, 'rc': -0.080}


def fit(rs, sel, lam):
    A = sum(U[r][:, sel].T @ U[r][:, sel] / (E0[r] @ E0[r]) for r in rs); b = sum(U[r][:, sel].T @ E0[r] / (E0[r] @ E0[r]) for r in rs)
    return -np.linalg.solve(A + lam * len(rs) * np.eye(len(sel)), b)


def ratio(r, sel, th):
    return float(((E0[r] + U[r][:, sel] @ th) ** 2).sum() / (E0[r] @ E0[r]))


rows = []
for k in range(1, len(knobs) + 1):
    for sub in itertools.combinations(knobs, k):
        sel = [knobs.index(x) for x in sub]; best = None
        for lam in (0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
            loo = np.array([ratio(r, sel, fit([x for x in ranks if x != r], sel, lam)) for r in ranks])
            if best is None or loo.mean() < best[1].mean(): best = (lam, loo)
        lam, loo = best; th = fit(ranks, sel, lam)
        rows.append((float(loo.mean()), sub, lam, loo, {kk: round(float(t * steps[knobs.index(kk)]), 4) for kk, t in zip(sub, th)}))
rows.sort(key=lambda x: x[0])
print(f'{len(ranks)} nets; S6b base mean MSE {np.mean([np.mean(E0[r] ** 2) for r in ranks]):.4e}')
for m, sub, lam, loo, dev in rows[:10]:
    print(f'LOO {m:.4f} wins {(loo < 1).sum()}/{len(loo)} worst {loo.max():.4f} lam {lam:g} | dev {json.dumps(dev)} | abs '
          + ', '.join(f'{k} {V3[k] + v:+.4f}' for k, v in dev.items()))
for k in knobs:
    r_ = [x for x in rows if x[1] == (k,)][0]
    print(f'single {k:5s} LOO {r_[0]:.4f} wins {(r_[3] < 1).sum()}/{len(ranks)} dev {r_[4][k]:+.4f}')
print('BEST_DEV ' + json.dumps(rows[0][4]))
