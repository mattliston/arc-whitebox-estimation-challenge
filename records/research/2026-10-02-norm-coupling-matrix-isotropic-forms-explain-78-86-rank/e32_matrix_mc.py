#!/usr/bin/env python3
"""TAG opus-reexam E32: the norm-coupling MATRIX of the fourth cumulant by Monte Carlo (CPU), for the transport-law and spectrum tests.
Ntilde_ij(l) = sum_k kappa4(z_i, z_j, z_k, z_k) = E[z~_i z~_j (S - tr C)] - 2 (C^2)_ij, S = |z~|^2, pre-activations of layer l, centred with the
32M truth means. Two independent halves. usage: e32_matrix_mc.py <rank> <threads> <samples per half> <layers, e.g. 9,12,13>
output: e32_rank<rank>.npz with M_<l>_<half> = E[z~ z~' (S - tr C)] (the 2 C^2 part is NOT subtracted here) and the sample count."""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2])); NH = int(sys.argv[3]); LAYERS = [int(x) for x in sys.argv[4].split(',')]
torch.set_grad_enabled(False)
W = torch.from_numpy(np.ascontiguousarray(pc.weights(rank))).to(torch.float32)
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
mu = {l: torch.from_numpy(MC[f'mean_pre_{l}'].astype(np.float32)) for l in LAYERS}
trC = {l: float(np.trace(MC[f'cov_pre_{l}'].astype(np.float64))) for l in LAYERS}
B = 16384; nb = NH // B; t0 = time.time()
acc = {h: {l: torch.zeros((1024, 1024), dtype=torch.float64) for l in LAYERS} for h in (0, 1)}
for h in (0, 1):
    gen = torch.Generator().manual_seed(32000 + h)
    for b in range(nb):
        x = torch.randn(B, 1024, generator=gen, dtype=torch.float32)
        for l in range(max(LAYERS) + 1):
            z = x @ W[l]
            if l in acc[h]:
                zt = z - mu[l]; S = (zt * zt).sum(1) - trC[l]
                acc[h][l] += ((zt * S[:, None]).T @ zt).double()
            x = torch.relu(z)
        if b % 16 == 0: print(json.dumps(dict(half=h, batch=b, of=nb, wall=round(time.time() - t0, 1))), flush=True)
N = nb * B; out = dict(N=N, layers=np.array(LAYERS))
for h in (0, 1):
    for l in LAYERS: out[f'M_{l}_{h}'] = (acc[h][l] / N).numpy()
np.savez(HERE / f'e32_rank{rank:04d}.npz', **out)
print(json.dumps(dict(done=True, N_per_half=N, wall=round(time.time() - t0, 1))), flush=True)
