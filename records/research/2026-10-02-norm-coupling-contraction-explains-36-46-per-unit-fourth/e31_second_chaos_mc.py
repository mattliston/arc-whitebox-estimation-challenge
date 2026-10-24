#!/usr/bin/env python3
"""TAG opus-reexam E31: is the unit-specific kurtosis a SECOND-CHAOS (matrix-level) object? Monte Carlo on the CPU, per-unit accumulators only.
Weights are iid Gaussian, so for unit i of a layer K4pre[i,i,i,i] = K4post(w_i^4) = constant + (12/n)(w_i' N w_i - (2/n) tr N) + fourth chaos,
N_ab = sum_c kappa4(x_a, x_b, x_c, x_c), and the same matrix gives Q_i = sum_k K4pre[i,i,k,k] = 2 w_i' N w_i + small. So the second-chaos part of
the per-unit fourth cumulant is (6/n) (Q_i - mean): measurable per unit as Q_i = E[z~_i^2 (S - tr C)] - 2 (C^2)_ii, S = |z~|^2 (centred squared norm).
Also Qm_i = E[z~_i^2 ((mhat . z~)^2 - mhat' C mhat)] - 2 (C mhat)_i^2 (contraction along the mean direction of the layer).
Means and covariances for centring: the 32M covariance truth. usage: e31_second_chaos_mc.py <rank> <threads> <samples per half> [layers=5,9,13,15]
output: e31_rank<rank>.npz (two independent halves)"""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2])); NH = int(sys.argv[3]); LAYERS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '5,9,13,15').split(',')]
torch.set_grad_enabled(False)
W = torch.from_numpy(np.ascontiguousarray(pc.weights(rank))).to(torch.float32)          # z_l = y_{l-1} @ W[l]
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
mu = {l: torch.from_numpy(MC[f'mean_pre_{l}'].astype(np.float32)) for l in LAYERS}
C = {l: MC[f'cov_pre_{l}'].astype(np.float64) for l in LAYERS}
trC = {l: float(np.trace(C[l])) for l in LAYERS}
mh = {l: (lambda m: m / np.linalg.norm(m))(MC[f'mean_pre_{l}'].astype(np.float64)) for l in LAYERS}
mCm = {l: float(mh[l] @ C[l] @ mh[l]) for l in LAYERS}; mht = {l: torch.from_numpy(mh[l].astype(np.float32)) for l in LAYERS}
B = 16384; nb = NH // B; t0 = time.time()
acc = {h: {l: dict(Q=torch.zeros(1024, dtype=torch.float64), Qm=torch.zeros(1024, dtype=torch.float64), v=torch.zeros(1024, dtype=torch.float64),
                   k=torch.zeros(1024, dtype=torch.float64), S1=0.0, S2=0.0) for l in LAYERS} for h in (0, 1)}
for h in (0, 1):
    gen = torch.Generator().manual_seed(31000 + h)
    for b in range(nb):
        x = torch.randn(B, 1024, generator=gen, dtype=torch.float32)
        for l in range(max(LAYERS) + 1):
            z = x @ W[l]
            if l in acc[h]:
                zt = z - mu[l]; z2 = zt * zt; S = z2.sum(1).double() - trC[l]; P = ((zt @ mht[l]).double()) ** 2 - mCm[l]
                a = acc[h][l]; z2d = z2.double()
                a['Q'] += (z2d * S[:, None]).sum(0); a['Qm'] += (z2d * P[:, None]).sum(0); a['v'] += z2d.sum(0); a['k'] += (z2d * z2d).sum(0)
                a['S1'] += float(S.sum()); a['S2'] += float((S * S).sum())
            x = torch.relu(z)
        if b % 8 == 0: print(json.dumps(dict(half=h, batch=b, of=nb, wall=round(time.time() - t0, 1))), flush=True)
N = nb * B; out = dict(N=N, layers=np.array(LAYERS))
for h in (0, 1):
    for l in LAYERS:
        a = acc[h][l]; C2 = (C[l] * C[l]).sum(1); Cm = C[l] @ mh[l]
        out[f'Q_{l}_{h}'] = (a['Q'] / N).numpy() - 2 * C2; out[f'Qm_{l}_{h}'] = (a['Qm'] / N).numpy() - 2 * Cm * Cm
        out[f'v_{l}_{h}'] = (a['v'] / N).numpy(); out[f'm4_{l}_{h}'] = (a['k'] / N).numpy(); out[f'varS_{l}_{h}'] = a['S2'] / N - (a['S1'] / N) ** 2
np.savez(HERE / f'e31_rank{rank:04d}.npz', **out)
print(json.dumps(dict(done=True, N_per_half=N, wall=round(time.time() - t0, 1))), flush=True)
