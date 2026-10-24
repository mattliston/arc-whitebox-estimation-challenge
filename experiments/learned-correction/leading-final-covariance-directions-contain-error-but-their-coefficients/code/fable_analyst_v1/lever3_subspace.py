#!/usr/bin/env python3
"""TAG fable-analyst (AN-55): lever-3 test. Can a learned correction in the card's own top covariance directions at L15 remove final error?
Per net (40 TRAIN640 nets spread over the split; torch s6bcard + B2 ep1, no refit): e = final prediction - truth688; U = top-32
eigenvectors of the card's linear-response post covariance at L15, G (W15^T C14 W15) G with G = diag(gate15) and C14 the card's carried
L14 covariance; each u_k signed so that u_k . pred >= 0. Coefficients c_k = u_k . e. Deployable features per (net, k): p_k = u_k . pred,
h_k = u_k . host (the scalar map), d_k = u_k . delta (the final hook), log eigenvalue. Models with leave-one-NET-out: (A) per-k shrinkage
c_k = beta_k p_k; (B) pooled ridge on [p, h, d, p log lam]. R^2 = 1 - SSE / sum c^2 (the share of the subspace error energy removed),
top-8 and top-32, plus the implied change of the final MSE. usage: lever3_subspace.py <threads> <out stem>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import s6bcard
torch.set_num_threads(int(sys.argv[1])); STEM = HERE / sys.argv[2]
pl = s6bcard.payload(torch, params=P2 / 'fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
train = json.loads((HERE / 'split_train640.json').read_text())['train']; nets = train[::16][:40]
K = 32; rows = []
for r in nets:
    w = s6bcard.weights(r); tp = np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    cd = s6bcard.card(torch, w, *pl); cap = {}
    def so(l, st, ctx):
        if l == 14: cap['C14'] = st[1].detach().double().clone()
        return st
    def dl(l, d, ctx):
        if l == 15: cap.update(host=ctx['host'].detach().double().numpy().copy(), delta=d.detach().double().numpy().copy(), mu=ctx['mu'].detach().double().numpy().copy(), vv=ctx['vv'].detach().double().numpy().copy())
        return d
    cd.hooks['state_out'] = so; cd.hooks['delta'] = dl
    t0 = time.time()
    with torch.no_grad(): pred = cd.run(False).numpy().astype(np.float64)[15]
    W = torch.as_tensor(w[15], dtype=torch.float64); Cp = (W.T @ cap['C14'] @ W).numpy()
    g = 0.5 * (1 + np.vectorize(__import__('math').erf)(cap['mu'] / np.sqrt(2 * cap['vv'])))
    val, vec = np.linalg.eigh(g[:, None] * Cp * g[None, :]); vec = vec[:, ::-1][:, :K]; val = val[::-1][:K]
    vec = vec * np.where(vec.T @ pred >= 0, 1.0, -1.0)[None, :]
    e = pred - tp
    rows.append(dict(rank=r, c=vec.T @ e, p=vec.T @ pred, h=vec.T @ cap['host'], d=vec.T @ cap['delta'], lam=val, E=float(e @ e)))
    print(json.dumps(dict(rank=r, mse=float(e @ e / 1024), frac_top8=float(np.sum((vec.T @ e)[:8] ** 2) / (e @ e)), frac_top32=float(np.sum((vec.T @ e) ** 2) / (e @ e)), wall=round(time.time() - t0, 1))), flush=True)
np.savez(STEM.with_suffix('.npz'), **{f'{k}_{x["rank"]}': x[k] for x in rows for k in ('c', 'p', 'h', 'd', 'lam')}, E=np.array([x['E'] for x in rows]))
n = len(rows); C = np.array([x['c'] for x in rows]); Pm = np.array([x['p'] for x in rows]); Hm = np.array([x['h'] for x in rows]); Dm = np.array([x['d'] for x in rows])
Lm = np.log(np.array([x['lam'] for x in rows])); E = np.array([x['E'] for x in rows])
def loo(pred_fn):
    Ch = np.zeros_like(C)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i]); Ch[i] = pred_fn(tr, i)
    return Ch
def shrink(tr, i):     # model A: per-k beta_k = sum c p / sum p^2 over training nets
    b = (C[tr] * Pm[tr]).sum(0) / (Pm[tr] ** 2).sum(0); return b * Pm[i]
def ridge(tr, i, alpha=1e-3):   # model B: pooled over k, features [p, h, d, p*loglam] (standardised by training scale)
    X = np.stack([Pm, Hm, Dm, Pm * Lm], -1)             # (n, K, 4)
    Xt, yt = X[tr].reshape(-1, 4), C[tr].reshape(-1); s = Xt.std(0) + 1e-30
    beta = np.linalg.solve((Xt / s).T @ (Xt / s) + alpha * len(yt) * np.eye(4), (Xt / s).T @ yt)
    return (X[i] / s) @ beta
out = dict(n_nets=n, nets=nets, K=K, energy_share_top8=float(np.mean([np.sum(x['c'][:8] ** 2) / x['E'] for x in rows])),
           energy_share_top32=float(np.mean([np.sum(x['c'] ** 2) / x['E'] for x in rows])), models={})
for name, fn in (('A_shrink', shrink), ('B_ridge', ridge)):
    Ch = loo(fn); rec = {}
    for k in (8, 32):
        sse = ((C[:, :k] - Ch[:, :k]) ** 2).sum(); ss = (C[:, :k] ** 2).sum()
        rec[f'R2_top{k}'] = float(1 - sse / ss)
        rec[f'final_mse_change_top{k}'] = float(np.mean([(((C[i, :k] - Ch[i, :k]) ** 2).sum() - (C[i, :k] ** 2).sum()) / E[i] for i in range(n)]))
    out['models'][name] = rec
(STEM.with_suffix('.json')).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'nets'}, indent=1))
