#!/usr/bin/env python3
"""TAG fable-covhunter (CH-31): ridge-pulled Gauss-Newton fit of the band knobs (gn6_bands.py output). Deviations from v3 are measured
in finite-difference step units; penalty lam * |th/step|^2 pulls toward v3. lam is chosen by leave-one-net-out (mean held-out ratio);
also reported for knob subsets. Prints the deviations and the absolute per-band constants.
usage: gn6_fit.py <gn6.npz>"""
import sys, json
import numpy as np
d = np.load(sys.argv[1]); knobs = list(d['knobs']); ranks = list(d['ranks']); steps = d['steps']
E0 = {r: d[f'e0_{r}'] for r in ranks}; U = {r: d[f'U_{r}'] * steps[None, :] for r in ranks}      # response per step unit
V3 = {'iso': -0.094, 'k21n': -0.028, 'k31': -0.212, 'edge': 0.164, 'rc': -0.080}


def fit(rs, sel, lam):
    A = sum(U[r][:, sel].T @ U[r][:, sel] / (E0[r] @ E0[r]) for r in rs); b = sum(U[r][:, sel].T @ E0[r] / (E0[r] @ E0[r]) for r in rs)
    return -np.linalg.solve(A + lam * len(rs) * np.eye(len(sel)), b)


def ratio(r, sel, th):
    return float(((E0[r] + U[r][:, sel] @ th) ** 2).sum() / (E0[r] @ E0[r]))


def report(sel_names, label):
    sel = [knobs.index(k) for k in sel_names]; best = None
    for lam in (0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3):
        loo = np.array([ratio(r, sel, fit([x for x in ranks if x != r], sel, lam)) for r in ranks])
        if best is None or loo.mean() < best[1].mean(): best = (lam, loo)
    lam, loo = best; th = fit(ranks, sel, lam) * steps[sel]
    ins = np.array([ratio(r, sel, th / steps[sel]) for r in ranks])
    print(f'{label}: lam {lam:g} LOO mean {loo.mean():.4f} median {np.median(loo):.4f} wins {(loo < 1).sum()}/{len(loo)} worst {loo.max():.4f} | in-sample mean {ins.mean():.4f}')
    print('   deviations from v3:', json.dumps({k: round(float(t), 4) for k, t in zip(sel_names, th)}))
    return {k: round(float(t), 4) for k, t in zip(sel_names, th)}


allk = knobs
dev = report(allk, 'all 14 knobs')
report([k for k in knobs if k not in ('edge', 'rc')], '12 band knobs')
report([k for k in knobs if k.endswith('_b3')] + ['edge', 'rc'], 'band L10-12 only + edge/rc')
report([k for k in knobs if k.endswith('_b3') or k.endswith('_b4')], 'bands L10-14 only')
print('absolute constants (all-14 fit):')
for b in ('b1', 'b2', 'b3', 'b4'):
    print(f'  {b}: ' + ', '.join(f"{k} {V3[k] + dev.get(f'{k}_{b}', 0.0):+.4f}" for k in ('iso', 'k21n', 'k31')))
print(f"  edge x{1 + V3['edge'] + dev.get('edge', 0):.4f}, recentering x{1 + V3['rc'] + dev.get('rc', 0):.4f}")
print('EVAL_THETA ' + json.dumps(dev))
