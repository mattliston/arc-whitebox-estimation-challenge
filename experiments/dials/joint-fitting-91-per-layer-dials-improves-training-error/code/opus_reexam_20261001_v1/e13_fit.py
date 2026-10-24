#!/usr/bin/env python3
"""TAG opus-reexam E13 fit: Gauss-Newton fit of per-layer physics scalars from the finite-difference Jacobians (e13_layer_knob_fd.py).
Model per network: e(theta) = e0 + U theta (theta = relative change of each scalar). Fit theta on a set of networks by ridge least
squares (minimise the pooled sum of squares), report the predicted change of the pooled final-layer MSE on the fit networks and on
held-out networks (their own e0 and U, the same theta), for the full per-layer model and for restricted models (one scalar per
family = what global dials can do; single families).
usage: e13_fit.py <npz> [<npz> ...]"""
import sys, itertools, numpy as np
E0, U, knobs = {}, {}, None
for f in sys.argv[1:]:
    d = np.load(f, allow_pickle=True); knobs = [str(k) for k in d['knobs']]
    for r in d['ranks']:
        if f'U_{int(r)}' in d.files: E0[int(r)] = d[f'e0_{int(r)}']; U[int(r)] = d[f'U_{int(r)}']
ranks = sorted(E0); fam = [k.split(':')[0] for k in knobs]; fams = list(dict.fromkeys(fam))
print(f'{len(ranks)} networks {ranks}; {len(knobs)} scalars in families {fams}')
def design(mode, only=None):
    """columns of a restriction matrix R: theta = R phi."""
    idx = [i for i, f in enumerate(fam) if only is None or f in only]
    if mode == 'layer': R = np.zeros((len(knobs), len(idx))); R[idx, range(len(idx))] = 1
    else:
        fs = [f for f in fams if only is None or f in only]; R = np.zeros((len(knobs), len(fs)))
        for j, f in enumerate(fs): R[[i for i, x in enumerate(fam) if x == f], j] = 1
    return R
def fit(train, R, lam):
    A = sum((U[r] @ R).T @ (U[r] @ R) for r in train); b = -sum((U[r] @ R).T @ E0[r] for r in train)
    scale = np.sqrt(np.diag(A)) + 1e-30
    phi = np.linalg.solve(A / np.outer(scale, scale) + lam * np.eye(len(A)), b / scale) / scale
    return R @ phi
def gain(test, theta):
    a = sum(((E0[r] + U[r] @ theta) ** 2).sum() for r in test); b = sum((E0[r] ** 2).sum() for r in test)
    return 100 * (a / b - 1)
for name, mode, only in [('one scalar per family (global dials)', 'family', None), ('per layer, all families', 'layer', None)] + [(f'per layer, {f} only', 'layer', [f]) for f in fams]:
    R = design(mode, only); res = []
    for lam in (1e-3, 1e-2, 1e-1, 0.3, 1.0):
        th = fit(ranks, R, lam); ins = gain(ranks, th)
        loo = []
        for r in ranks:
            tr = [x for x in ranks if x != r]; loo.append(gain([r], fit(tr, R, lam)))
        res.append((lam, ins, float(np.mean(loo)), int(sum(x < 0 for x in loo)), float(np.abs(th).max())))
    print(f'\n{name}: {R.shape[1]} parameters')
    for lam, ins, loo, wins, mx in res:
        print(f'   ridge {lam:5.3f}: predicted change of final-layer MSE, in sample {ins:+.2f}% | leave-one-network-out {loo:+.2f}% ({wins}/{len(ranks)} better) | largest |theta| {mx:.2f}')
