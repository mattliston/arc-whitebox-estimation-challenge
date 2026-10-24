#!/usr/bin/env python3
"""TAG fable-covhunter (CH-28): FINAL-LAYER STEP 0 fit. Is the per-unit final error e_i predictable from final-layer card features?
Target e_i / sigma_i (sigma = sd of z_15). Features are dimensionless per-unit quantities from final_dump/ (see final_dump.py) plus two
per-net context values. Models: ridge (standardized, alpha by grouped CV on TRAIN) and a small GBM (sklearn HistGradientBoosting).
Fitted on TRAIN nets, graded on held-out VALID nets: R2_0 = 1 - sum (e - ehat)^2 / sum e^2 (the implied final-MSE cut if ehat were
subtracted), pooled and per net; plus leave-one-net-out on TRAIN and the per-net bias share (mean(e)^2 / mean(e^2)).
usage: final_fit.py"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
inp = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
TRAIN, VALID = set(inp['train']), set(inp['valid'])
nets = {}
for f in sorted((HERE / 'final_dump').glob('final_rank*.npz')):
    nets[int(f.stem[-4:])] = dict(np.load(f))


def features(d):
    s = np.sqrt(d['vv']); a = d['mu'] / s; s3 = s ** 3
    g = d['aux_gate']; ls = np.log(s)
    X = dict(alpha=a, gate=g, pdf=d['aux_pdf'], edge=np.minimum(g, 1 - g), kink=(np.abs(a) < 1.5).astype(float),
             skew=(d['k3_bank'] + d['k3_star'] + d['k3_bterm']) / s3, skew_bank=d['k3_bank'] / s3, skew_star=d['k3_star'] / s3,
             skew_bterm=d['k3_bterm'] / s3, radial=np.full_like(a, float(d['radial'])), logs=ls - ls.mean(),
             host=d['host'] / s, gmean=d['aux_gaussian_mean'] / s, hostdev=(d['host'] - d['aux_gaussian_mean']) / s, delta=d['delta'] / s,
             postv=d['aux_post_v'] / d['vv'], postk3=d['aux_post_k3'] / (d['vv'] * s), postk4=d['aux_post_k4'] / d['vv'] ** 2,
             diag_share=d['diag_share'], wload16=d['wload16'], wload64=d['wload64'], vload16=d['vload16'], vload64=d['vload64'],
             w1=d['wsum1'] * 0.7071, w2=(d['wsum2'] - 1 / 512) * 11.3137, w3=d['wsum3'] * 93.478, w4=(d['wsum4'] - 1.1444e-5) * 836.09,
             m14=np.full_like(a, float(d['m14_mean'])), eig1=np.full_like(a, float(d['C14_eig'][0] / d['C14_eig'].sum())))
    return X, s


names = None; XS, YS, SS, ES, G = [], [], [], [], []
for r, d in nets.items():
    X, s = features(d)
    names = list(X) if names is None else names
    XS.append(np.stack([X[k] for k in names], 1)); SS.append(s); ES.append(d['e']); YS.append(d['e'] / s); G.append(np.full(len(s), r))
X = np.concatenate(XS); Y = np.concatenate(YS); S = np.concatenate(SS); E = np.concatenate(ES); G = np.concatenate(G)
tr = np.isin(G, list(TRAIN)); va = np.isin(G, list(VALID))
print(f'nets: {len(nets)} ({int(np.isin(list(nets), list(TRAIN)).sum())} TRAIN, {int(np.isin(list(nets), list(VALID)).sum())} VALID); features {len(names)}')
bias = [float(np.mean(nets[r]['e']) ** 2 / np.mean(nets[r]['e'] ** 2)) for r in nets]
print(f'per-net bias share mean(e)^2/mean(e^2): mean {np.mean(bias):.3f} max {np.max(bias):.3f}')


def grade(mask, ehat, label):
    e = E[mask]; eh = ehat
    pooled = 1 - np.sum((e - eh) ** 2) / np.sum(e ** 2)
    per = [1 - np.sum((E[mask & (G == r)] - eh[G[mask] == r]) ** 2) / np.sum(E[mask & (G == r)] ** 2) for r in np.unique(G[mask])]
    print(f'{label:34s} R2_0 pooled {pooled:+.4f} | per-net mean cut {np.mean(per):+.4f} median {np.median(per):+.4f} wins {sum(p > 0 for p in per)}/{len(per)}', flush=True)
    return pooled


sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
# ridge alpha by grouped CV on TRAIN
tr_nets = np.unique(G[tr]); best = None
for alpha in (1, 10, 100, 1000, 1e4, 1e5):
    sse = 0; sst = 0
    for r in tr_nets:
        m_fit = tr & (G != r); m_te = G == r
        mdl = Ridge(alpha=alpha).fit(Xs[m_fit], Y[m_fit]); eh = mdl.predict(Xs[m_te]) * S[m_te]
        sse += np.sum((E[m_te] - eh) ** 2); sst += np.sum(E[m_te] ** 2)
    r2 = 1 - sse / sst
    print(f'ridge alpha {alpha:>8g}: TRAIN leave-one-net-out R2_0 {r2:+.4f}')
    if best is None or r2 > best[1]: best = (alpha, r2)
rid = Ridge(alpha=best[0]).fit(Xs[tr], Y[tr])
grade(tr, rid.predict(Xs[tr]) * S[tr], f'ridge a={best[0]:g} TRAIN in-sample')
grade(va, rid.predict(Xs[va]) * S[va], f'ridge a={best[0]:g} VALID held-out')
coef = sorted(zip(names, rid.coef_), key=lambda x: -abs(x[1]))[:10]
print('ridge top coefficients (standardized):', ', '.join(f'{k} {c:+.3f}' for k, c in coef))
for depth, it in ((3, 200), (4, 400)):
    gbm = HistGradientBoostingRegressor(max_depth=depth, max_iter=it, learning_rate=0.05, min_samples_leaf=100, l2_regularization=1.0, random_state=0)
    gbm.fit(X[tr], Y[tr])
    grade(tr, gbm.predict(X[tr]) * S[tr], f'GBM d{depth} it{it} TRAIN in-sample')
    grade(va, gbm.predict(X[va]) * S[va], f'GBM d{depth} it{it} VALID held-out')
# raw-error target (not normalized) as a check
gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=100, l2_regularization=1.0, random_state=0)
gbm.fit(np.column_stack([X[tr], S[tr]]), E[tr])
grade(va, gbm.predict(np.column_stack([X[va], S[va]])), 'GBM raw-e target VALID held-out')
