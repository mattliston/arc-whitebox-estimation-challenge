#!/usr/bin/env python3
"""TAG opus-reexam E6: when the collective gain (the norm of an earlier layer) fluctuates, do PAIR covariances scale with it the way
per-unit variances do? Rigid scaling (a scale mixture) predicts that the conditional covariance of every pair grows like norm^2
(elasticity 1 for diagonal AND off-diagonal entries). If only the variances follow, the gain mode is a common volatility of the
idiosyncratic parts and its third-order content is plane-supported rather than Sym(C x b).
Monte Carlo on CPU (torch float32): N inputs through the true net of <rank>; at target layers L in (6, 10, 14) the pre-activations of a
fixed random subset of 192 units are binned by the norm of the post layer L-1, L-3, L-6 and of the input; per bin: count, sum of r^2,
sum z, sum z z^T. Analysis: weighted least squares of each conditional covariance entry on (r^2/mean - 1); elasticity = slope / level.
usage: e6_gain_elasticity_mc.py <rank> <N_millions> <threads>"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
rank = int(sys.argv[1]); N = int(float(sys.argv[2]) * 1e6); torch.set_num_threads(int(sys.argv[3]))
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = torch.from_numpy(np.load((STORE / man['path']).parent / 'weights.npy').astype(np.float32))
LS = (6, 10, 14); LAGS = (1, 3, 6, 'in'); B = 16; m = 192; BATCH = 32768
gen = torch.Generator(); gen.manual_seed(777 + rank)
sub = {L: torch.randperm(1024, generator=gen)[:m] for L in LS}
def forward(x):
    norms = {-1: x.norm(dim=1)}; pre = {}
    y = x
    for l in range(15):
        z = y @ W[l]
        if l in LS: pre[l] = z[:, sub[l]].double()
        y = z.clamp_min(0.0); norms[l] = y.norm(dim=1)
    return pre, norms
def rkey(L, lag): return -1 if lag == 'in' else L - lag
x = torch.randn((BATCH, 1024), generator=gen); pre, norms = forward(x)
qs = torch.linspace(0, 1, B + 1)[1:-1]
edges = {(L, lag): torch.quantile(norms[rkey(L, lag)], qs) for L in LS for lag in LAGS}
ref = {L: pre[L].mean(0) for L in LS}
acc = {(L, lag): dict(cnt=torch.zeros(B, dtype=torch.float64), r2=torch.zeros(B, dtype=torch.float64),
                      s1=torch.zeros((B, m), dtype=torch.float64), s2=torch.zeros((B, m, m), dtype=torch.float64)) for L in LS for lag in LAGS}
t0 = time.time(); done = 0
while done < N:
    x = torch.randn((BATCH, 1024), generator=gen); pre, norms = forward(x)
    for L in LS:
        zc = pre[L] - ref[L]
        for lag in LAGS:
            r = norms[rkey(L, lag)]; ids = torch.bucketize(r, edges[(L, lag)]); a = acc[(L, lag)]
            a['cnt'].index_add_(0, ids, torch.ones(len(r), dtype=torch.float64)); a['r2'].index_add_(0, ids, (r * r).double())
            a['s1'].index_add_(0, ids, zc)
            for b in range(B):
                zb = zc[ids == b]; a['s2'][b] += zb.T @ zb
    done += BATCH
    if (done // BATCH) % 8 == 0: print(json.dumps(dict(samples=done, wall=round(time.time() - t0, 1))), flush=True)
out = dict(N=np.array(done))
for (L, lag), a in acc.items():
    for k, v in a.items(): out[f'{L}_{lag}_{k}'] = v.numpy()
    out[f'{L}_sub'] = sub[L].numpy(); out[f'{L}_ref'] = ref[L].numpy()
np.savez(HERE / f'e6_rank{rank:04d}_N{done // 1000000}M.npz', **out)
print(json.dumps(dict(done=done, wall=round(time.time() - t0, 1))), flush=True)
# ---- analysis
print(f'\nrank {rank}, N={done}: elasticity of conditional moments w.r.t. the squared norm (rigid scaling predicts 1 for all second moments, 0.5 for means)')
print(' L  norm of      | sd of log r2 | mean elast (0.5) | variance elast (diag) | off-diag covariance elast (energy-weighted) | same, pairs with |rho|>0.2 | n pairs')
for L in LS:
    for lag in LAGS:
        a = {k: out[f'{L}_{lag}_{k}'] for k in ('cnt', 'r2', 's1', 's2')}
        cnt = a['cnt']; p = cnt / cnt.sum(); r2 = a['r2'] / cnt; x_ = r2 / (p * r2).sum() - 1
        mu_b = a['s1'] / cnt[:, None]; M2 = a['s2'] / cnt[:, None, None]; C_b = M2 - mu_b[:, :, None] * mu_b[:, None, :]
        xm = (p * x_).sum(); xv = (p * (x_ - xm) ** 2).sum()
        slope = lambda Y: np.tensordot(p * (x_ - xm), Y - np.tensordot(p, Y, axes=(0, 0)), axes=(0, 0)) / xv
        level = lambda Y: np.tensordot(p, Y, axes=(0, 0))
        mu_tot = out[f'{L}_ref'] + level(mu_b)
        e_mean = (slope(mu_b) * mu_tot).sum() / (mu_tot ** 2).sum()
        A = level(C_b); S = slope(C_b)
        d = np.diag_indices(m); off = ~np.eye(m, dtype=bool)
        e_diag = (S[d] * A[d]).sum() / (A[d] ** 2).sum()
        e_off = (S[off] * A[off]).sum() / (A[off] ** 2).sum()
        sd = np.sqrt(np.diag(A)); rho = A / np.outer(sd, sd); big = off & (np.abs(rho) > 0.2)
        e_big = (S[big] * A[big]).sum() / (A[big] ** 2).sum() if big.any() else float('nan')
        print(f'{L:2d}  {"input" if lag == "in" else "post L-" + str(lag):11s} | {np.sqrt(xv):.4f} | {e_mean:.3f} | {e_diag:.3f} | {e_off:.3f} | {e_big:.3f} | {int(big.sum()) // 2}')
