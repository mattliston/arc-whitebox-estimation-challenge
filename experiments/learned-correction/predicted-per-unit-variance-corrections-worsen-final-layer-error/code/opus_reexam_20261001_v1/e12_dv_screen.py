#!/usr/bin/env python3
"""TAG opus-reexam E12: is the card's per-unit VARIANCE error predictable from what the learned net sees?
The per-unit variance fed to the mean formula is the largest single lever (exact variance: -34% final error, analyst A6).
Target per unit and layer: y = (v_card - v_true) / v_true, v_true from Monte Carlo (network 35: 64M samples; 37, 39: 48M).
Features: the learned net's 30 inputs and 64-number state at that layer (e7 dumps of the current torch card) plus
a = mu/sigma, |a|, phi(a), a phi(a), the three parts of the card's skew.
Leave-one-network-out: fit ridge and gradient boosting on two networks, score on the third. R2 is noise-corrected with the
known sampling variance of v_true (2/N per unit, relative).
usage: e12_dv_screen.py"""
import json, numpy as np, lightgbm as lgb
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
G = P2 / 'public_g19_attribution_v1'
def truth_v(rank):
    mc = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); L = 16
    vB = np.array([np.diag(mc[f'cov_pre_{l}']) for l in range(L)]).astype(np.float64); NB = float(mc['P'])
    if (G / f'mps_truth_rank{rank:04d}_P32M.npz').exists():
        a = np.load(G / f'mps_truth_rank{rank:04d}_P32M.npz'); b = np.load(G / f'mps_truth_rank{rank:04d}_indep32M.npz')
        v = (a['mc_pre_v'].astype(np.float64) * float(a['P']) + b['mc_pre_v'].astype(np.float64) * float(b['P']) + vB * NB) / (float(a['P']) + float(b['P']) + NB)
        return v, float(a['P']) + float(b['P']) + NB
    d = np.load(HERE / f'analyst/e1_rank{rank:04d}_N16M_B16.npz'); N = float(d['N']); vA = []
    for l in range(L):
        tot = d[f'norm_{l}_cnt'].sum(); m1 = d[f'norm_{l}_0'].sum(0) / tot; m2 = d[f'norm_{l}_1'].sum(0) / tot; vA.append(m2 - m1 ** 2)
    vA = np.array(vA)
    return (vA * N + vB * NB) / (N + NB), N + NB
phi = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
ranks = [35, 37, 39]; data = {}
for r in ranks:
    d = np.load(HERE / f'e7/e7_rank{r:04d}.npz'); vt, N = truth_v(r); data[r] = {}
    for l in range(4, 16):
        v = d[f'v_{l}'].astype(np.float64); mu = d[f'mu_{l}'].astype(np.float64); s = np.sqrt(v); a = mu / s
        ok = vt[l] > 1e-8
        y = (v - vt[l]) / np.where(ok, vt[l], 1.0)
        extra = np.stack([a, np.abs(a), phi(a), a * phi(a), d[f'k3bank_{l}'] / s ** 3, d[f'k3star_{l}'] / s ** 3, d[f'k3b_{l}'] / s ** 3, np.log(np.maximum(v, 1e-12))], 1)
        X = np.concatenate([d[f'x_{l}'], d[f'hs_{l}'], extra], 1).astype(np.float32)
        data[r][l] = (X[ok], y[ok], 2.0 / N)
print('leave-one-network-out, per layer: rms relative variance error of the card (%), truth noise (%), noise-corrected held-out R2: ridge | boosting')
tot = dict(sig=0.0, ridge=0.0, gbm=0.0)
for l in range(4, 16):
    out = []
    for test in ranks:
        tr = [r for r in ranks if r != test]
        Xa = np.concatenate([data[r][l][0] for r in tr]); ya = np.concatenate([data[r][l][1] for r in tr]); Xb, yb, nz = data[test][l]
        m, sd = Xa.mean(0), Xa.std(0) + 1e-9; A = (Xa - m) / sd; B = (Xb - m) / sd
        coef = np.linalg.solve(A.T @ A + 1e-1 * len(A) * np.eye(A.shape[1]), A.T @ (ya - ya.mean())); pr = B @ coef + ya.mean()
        cut = int(len(Xa) * 0.8); idx = np.random.default_rng(0).permutation(len(Xa))
        mdl = lgb.train(dict(objective='regression', learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.7, lambda_l2=1.0, verbose=-1, num_threads=3),
                        lgb.Dataset(Xa[idx[:cut]], ya[idx[:cut]]), num_boost_round=400, valid_sets=[lgb.Dataset(Xa[idx[cut:]], ya[idx[cut:]])],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
        pg = mdl.predict(Xb, num_iteration=mdl.best_iteration)
        sig = (yb ** 2).mean() - nz                                  # noise-free energy of the target
        out.append((np.sqrt((yb ** 2).mean()), np.sqrt(nz), ((yb ** 2).mean() - ((yb - pr) ** 2).mean()) / sig, ((yb ** 2).mean() - ((yb - pg) ** 2).mean()) / sig))
    o = np.array(out).mean(0)
    print(f'{l:2d} | {o[0] * 100:.3f}% | {o[1] * 100:.3f}% | {o[2]:+.3f} | {o[3]:+.3f}   (per test network: ridge ' + ' '.join(f'{x[2]:+.2f}' for x in out) + ' ; boosting ' + ' '.join(f'{x[3]:+.2f}' for x in out) + ')')
