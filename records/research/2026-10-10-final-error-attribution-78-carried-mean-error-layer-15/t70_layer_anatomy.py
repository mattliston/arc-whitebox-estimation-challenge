#!/usr/bin/env python3
"""TAG opus-physicist T70 (numpy): ANATOMY OF THE POST-MEAN ERROR AT EVERY LAYER of the S42 card (T60 dumps), against two independent per-unit truths (analyst A13b,
2 x 32,047,104 samples, O/analyst/fresh/pu/), the 1e9 truth for the post means. Per layer l and unit: e = post mean after the head - 1e9 truth, and its first-order channels
mean Phi (mu_card - mu_true) (carried in from layer l-1), var pdf / (2 sigma) dv, skew -alpha pdf / (6 v) dk3, kurt sigma dk4 / v^2 (alpha^2 - 1) pdf / 24; share of e's energy
each explains (noise-free with the two halves), all four jointly, and the joint coefficients. Same arithmetic as T69 (layer 15 on the T66 dumps).
usage: t70_layer_anatomy.py"""
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
erf = np.vectorize(math.erf)
R = {}
for f in sorted(HERE.glob('s42dump_rank*.npz')):
    if '_v1' in f.name or '_cert' in f.name: continue
    r = int(f.name[12:16]); a_, b_ = O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_A32M.npz', O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_B32M.npz'
    if not (a_.exists() and b_.exists()): continue
    D = np.load(f); TA, TB = np.load(a_), np.load(b_); tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{r:04d}.npz')['mc_post'].astype(np.float64)
    for l in range(2, 16):
        if f'v_{l}' not in D.files or f'm_{l}' not in D.files: continue
        mu, v, k3, k4 = D[f'mu_{l}'], D[f'v_{l}'].clip(1e-12), D[f'k3_{l}'], D[f'k4_{l}']; s = np.sqrt(v); al = mu / s; ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + erf(al / math.sqrt(2))); e = D[f'm_{l}'] - tp[l]; ch = {}
        for h, T in (('A', TA), ('B', TB)):
            ch['mean' + h] = Phi * (mu - T['mc_pre_m'][l]); ch['var' + h] = ph / (2 * s) * (v - T['mc_pre_v'][l])
            ch['skew' + h] = -al * ph / (6 * v) * (k3 - T['mc_pre_k3'][l]); ch['kurt' + h] = s * (k4 - T['mc_pre_k4'][l]) / v ** 2 * (al * al - 1) * ph / 24
        E = (e * e).sum(); o = {}
        for c in ('mean', 'var', 'skew', 'kurt'):
            xa, xb = ch[c + 'A'], ch[c + 'B']; vx = (xa * xb).sum(); o[c] = ((xa * e).sum() + (xb * e).sum()) ** 2 / 4 / (vx * E) if vx > 0 else np.nan
        XA = np.column_stack([ch[c + 'A'] for c in ('mean', 'var', 'skew', 'kurt')]); XB = np.column_stack([ch[c + 'B'] for c in ('mean', 'var', 'skew', 'kurt')])
        G = 0.5 * (XA.T @ XB + XB.T @ XA); c_ = 0.5 * (XA.T @ e + XB.T @ e); beta = np.linalg.solve(G, c_); o['joint'] = float(c_ @ beta) / E; o['beta'] = beta; o['mse'] = float((e * e).mean())
        R.setdefault(l, []).append((r, o))
nets = sorted({r for L in R.values() for r, _ in L})
print('T70 post-mean error anatomy by layer, S42 card, networks %s (two 32M halves each); share of the error energy explained by each first-order channel, all four jointly, joint coefficients' % nets)
for l in sorted(R):
    L = [o for _, o in R[l]]
    print('  L%-2d MSE %.2e | mean %.3f var %.3f skew %.3f kurt %.3f | joint %.3f | coef %s' % (l, np.mean([o['mse'] for o in L]), *[np.nanmean([o[c] for o in L]) for c in ('mean', 'var', 'skew', 'kurt')],
          np.mean([o['joint'] for o in L]), np.round(np.mean([o['beta'] for o in L], 0), 2)))
