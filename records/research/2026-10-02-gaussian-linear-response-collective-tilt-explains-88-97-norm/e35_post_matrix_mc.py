#!/usr/bin/env python3
"""TAG opus-reexam E35: where does the norm-coupling matrix change, at the ReLU or at the push? Monte Carlo (CPU), network 35, one layer l:
  pre space  : Mpre  = E[z~ z~' (S - E S)],   S = |z~|^2                       (layer l)
  post space : Mpost = E[x~ x~' (R - E R)],   R = |x~|^2, x = relu(z)          (layer l)   and Mmix = E[z~ z~' (R - E R)], tz = E[z~ (R - E R)]
  next layer : Mnext = E[z'~ z'~' (S' - E S')]                                  (layer l+1)
Centred with the 32M truth means. Two independent halves. usage: e35_post_matrix_mc.py <rank> <threads> <samples per half> <layer l>"""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2])); NH = int(sys.argv[3]); l = int(sys.argv[4])
torch.set_grad_enabled(False)
W = torch.from_numpy(np.ascontiguousarray(pc.weights(rank))).to(torch.float32)
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
f32 = lambda k: torch.from_numpy(MC[k].astype(np.float32))
mu, mun, mp = f32(f'mean_pre_{l}'), f32(f'mean_pre_{l + 1}'), f32(f'mean_post_{l}')
ES = float(np.trace(MC[f'cov_pre_{l}'].astype(np.float64))); ESn = float(np.trace(MC[f'cov_pre_{l + 1}'].astype(np.float64))); ER = float(np.trace(MC[f'cov_post_{l}'].astype(np.float64)))
B = 16384; nb = NH // B; t0 = time.time(); Z = lambda: torch.zeros((1024, 1024), dtype=torch.float64)
acc = {h: dict(Mpre=Z(), Mpost=Z(), Mmix=Z(), Mnext=Z(), tz=torch.zeros(1024, dtype=torch.float64)) for h in (0, 1)}
for h in (0, 1):
    gen = torch.Generator().manual_seed(35000 + h)
    for b in range(nb):
        x = torch.randn(B, 1024, generator=gen, dtype=torch.float32)
        for j in range(l + 2):
            z = x @ W[j]
            if j == l:
                zt = z - mu; S = (zt * zt).sum(1) - ES; xr = torch.relu(z); xt = xr - mp; R = (xt * xt).sum(1) - ER; a = acc[h]
                a['Mpre'] += ((zt * S[:, None]).T @ zt).double(); a['Mmix'] += ((zt * R[:, None]).T @ zt).double(); a['Mpost'] += ((xt * R[:, None]).T @ xt).double()
                a['tz'] += (zt * R[:, None]).sum(0).double()
            if j == l + 1:
                zt = z - mun; S = (zt * zt).sum(1) - ESn; acc[h]['Mnext'] += ((zt * S[:, None]).T @ zt).double()
            x = torch.relu(z)
        if b % 16 == 0: print(json.dumps(dict(half=h, batch=b, of=nb, wall=round(time.time() - t0, 1))), flush=True)
N = nb * B
np.savez(HERE / f'e35_rank{rank:04d}_L{l}.npz', N=N, layer=l, **{f'{k}_{h}': (v / N).numpy() for h in (0, 1) for k, v in acc[h].items()})
print(json.dumps(dict(done=True, N_per_half=N, wall=round(time.time() - t0, 1))), flush=True)
