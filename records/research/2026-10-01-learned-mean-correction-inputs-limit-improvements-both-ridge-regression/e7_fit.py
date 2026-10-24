#!/usr/bin/env python3
"""TAG opus-reexam E7: is the learned mean correction information-limited or under-fitted?
Per network and layer l (4..15) the torch card dump gives the learned net's inputs x_l (30 features), its new state hs_l (64), the
host mean, the learned correction delta and the output. With e_l = output - truth and the gate g_l = Phi(mu/sigma):
  carried_l = g_l * (W_l^T e_{l-1})          (previous-layer error pushed through the layer, first order)
  fh_l      = (host_l - truth_l) - carried_l (the host error freshly made at layer l)
The ideal correction is -fh_l; target y = -fh_l / sigma_l. Compared on HELD-OUT networks (the 24 validation networks):
  deployed : share of sum y^2 removed by the deployed delta/sigma
  ridge    : ridge regression on [x, hs] fitted on 24 training networks (linear readout of the same inputs)
  gbm      : gradient boosting on [x, hs] fitted on the 24 training networks (non-linear readout of the same inputs)
  gbm+     : the same with a few extra per-unit features the learned net does not see (skew split by carrier, masks)
If gbm is no better than deployed, the correction is information-limited for these inputs; if it is clearly better, the net is under-fit.
usage: e7_fit.py"""
import json, glob, sys
from math import erf
from pathlib import Path
import numpy as np
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
parts = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
VALID = set(parts['valid']); TRAIN = set(parts['train'])
Phi = np.vectorize(lambda t: 0.5 * (1 + erf(t / np.sqrt(2))))
rows = {}
for f in sorted(glob.glob(str(HERE / 'e7_rank*.npz'))):
    rank = int(Path(f).stem.split('rank')[1]); d = np.load(f)
    W = np.load(STORE / Path(MAN[rank]['path']).parent / 'weights.npy').astype(np.float64)
    e = d['pred'].astype(np.float64) - d['truth'].astype(np.float64)
    for l in range(4, 16):
        mu = d[f'mu_{l}']; v = d[f'v_{l}']; s = np.sqrt(v); g = Phi(mu / s)
        carried = g * (W[l].T @ e[l - 1])
        fh = (d[f'host_{l}'] - d['truth'][l]) - carried
        y = -fh / s; dep = d[f'delta_{l}'] / s
        k3s = [d[f'k3bank_{l}'] / s ** 3, d[f'k3star_{l}'] / s ** 3, d[f'k3b_{l}'] / s ** 3]
        a = mu / s
        extra = np.stack(k3s + [a, np.abs(a), np.exp(-a * a / 2), a * np.exp(-a * a / 2), (a * a - 1) * np.exp(-a * a / 2)], 1)
        rows.setdefault(l, []).append(dict(rank=rank, y=y, dep=dep, X=np.concatenate([d[f'x_{l}'], d[f'hs_{l}']], 1).astype(np.float32),
                                           extra=extra.astype(np.float32), s=s, fh=fh))
ranks = sorted({r['rank'] for r in rows[4]}); tr = [r for r in ranks if r in TRAIN]; va = [r for r in ranks if r in VALID]
print(f'networks: {len(tr)} training, {len(va)} validation (held out from the learned net and from every fit below)')
def cat(l, sel, key): return np.concatenate([r[key] for r in rows[l] if r['rank'] in sel])
def share(y, pred): return 1 - ((y - pred) ** 2).sum() / (y ** 2).sum()
print(' L | rms y (units of sigma) | fresh host MSE x1e9 | deployed (train nets) | deployed (valid) | ridge | gbm | gbm+extra | gbm on what deployed leaves (valid) | fresh MSE left x1e9: deployed -> gbm+')
tot = dict(dep=0.0, gbm=0.0, base=0.0)
for l in range(4, 16):
    Xtr, ytr, dtr = cat(l, tr, 'X'), cat(l, tr, 'y'), cat(l, tr, 'dep')
    Xva, yva, dva = cat(l, va, 'X'), cat(l, va, 'y'), cat(l, va, 'dep')
    Etr, Eva = cat(l, tr, 'extra'), cat(l, va, 'extra'); sva = cat(l, va, 's'); fhva = cat(l, va, 'fh')
    # ridge on standardized features
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - m) / sd; B = (Xva - m) / sd
    lam = 1e-2 * len(A)
    coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean())); ridge = B @ coef + ytr.mean()
    def gbm(Xa, ya, Xb):
        n = len(Xa); cut = int(n * 5 / 6)   # last sixth of the training networks = early-stopping set (whole networks)
        ds = lgb.Dataset(Xa[:cut], ya[:cut]); dv = lgb.Dataset(Xa[cut:], ya[cut:])
        mdl = lgb.train(dict(objective='regression', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.7,
                             lambda_l2=1.0, verbose=-1, num_threads=4), ds, num_boost_round=600, valid_sets=[dv],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
        return mdl.predict(Xb, num_iteration=mdl.best_iteration)
    p1 = gbm(Xtr, ytr, Xva)
    p2 = gbm(np.concatenate([Xtr, Etr], 1), ytr, np.concatenate([Xva, Eva], 1))
    p3 = gbm(np.concatenate([Xtr, Etr], 1), ytr - dtr, np.concatenate([Xva, Eva], 1))
    base = float(np.mean(fhva ** 2)) * 1e9; left_dep = float(np.mean((fhva + dva * sva) ** 2)) * 1e9; left_gbm = float(np.mean((fhva + p2 * sva) ** 2)) * 1e9
    tot['base'] += base; tot['dep'] += left_dep; tot['gbm'] += left_gbm
    print(f'{l:2d} | {np.sqrt(np.mean(yva ** 2)):.2e} | {base:5.2f} | {share(ytr, dtr):+.3f} | {share(yva, dva):+.3f} | {share(yva, ridge):+.3f} | '
          f'{share(yva, p1):+.3f} | {share(yva, p2):+.3f} | {share(yva - dva, p3):+.3f} | {left_dep:5.2f} -> {left_gbm:5.2f}', flush=True)
print(f"sum over layers 4-15 of fresh host MSE x1e9: {tot['base']:.2f}; left after deployed {tot['dep']:.2f}; left after gbm+ {tot['gbm']:.2f}")
