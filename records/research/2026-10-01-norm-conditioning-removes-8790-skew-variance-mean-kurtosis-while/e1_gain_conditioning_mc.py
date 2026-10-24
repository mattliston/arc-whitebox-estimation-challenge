#!/usr/bin/env python3
"""TAG opus-reexam E1: does conditioning on the collective GAIN variable remove the per-unit (network-specific) skew/kurtosis?

For net <rank>, N Monte Carlo inputs x ~ N(0, I) through the true 1024x16 ReLU net (h = x @ W[l], y = relu(h)) on MPS.
At every layer l the pre-activation z_l = y_{l-1} @ W_l is binned by a scalar collective variable t and per-bin, per-unit
power sums of z (orders 1..4) are accumulated. Two collective variables:
  'norm' : t = |y_{l-1}| (norm of the previous post layer; layer 0 uses |x|)
  'pc1'  : t = z_l . u1, u1 = top eigenvector of the 32M MC covariance of z_l (fable_covhunter_v1/mcall file)
Also accumulated: plain power sums of the gain-normalised variable u = z_l / |y_{l-1}| under the plain measure and under the
gain-tilted measure (weights |y_{l-1}|).
Output npz: per layer power sums; the analysis (law of total cumulance: within-bin vs between-bin share of each unit's k3/k4) is
done by e1_analyze.py. Knobs on the command line: rank, N (millions), bins.
usage: e1_gain_conditioning_mc.py <rank> <N_millions> [bins=16]
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
rank = int(sys.argv[1]); N = int(float(sys.argv[2]) * 1e6); B = int(sys.argv[3]) if len(sys.argv) > 3 else 16
BATCH = 65_536; n = 1024; L = 16
dev = torch.device('mps')
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy')
wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(L)]
mc = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
u1 = []; ref = []
for l in range(L):
    C = mc[f'cov_pre_{l}']; w_, v_ = np.linalg.eigh(C)
    u1.append(torch.from_numpy(v_[:, -1].astype(np.float32)).to(dev))
    ref.append(torch.from_numpy(mc[f'mean_pre_{l}'].astype(np.float32)).to(dev))
gen = torch.Generator(device='cpu'); gen.manual_seed(20261001 + rank)
# pilot batch -> bin edges (quantiles) per layer and per collective variable
def forward(x):
    out = []
    y = x; nrm = x.norm(dim=1)
    for l in range(L):
        z = y @ wg[l]
        out.append((z, nrm))
        y = torch.clamp(z, min=0.0); nrm = y.norm(dim=1)
    return out
x = torch.randn((BATCH, n), generator=gen).to(dev)
edges = {}
qs = torch.linspace(0, 1, B + 1)[1:-1].to(dev)
acc_ref = {}
for l, (z, nrm) in enumerate(forward(x)):
    edges[('norm', l)] = torch.quantile(nrm, qs)
    edges[('pc1', l)] = torch.quantile((z - ref[l]) @ u1[l], qs)
    acc_ref[l] = (ref[l] / nrm.mean()).clone()   # fixed centring reference for u = z / |y_prev|
acc = {}
for l in range(L):
    for kind in ('norm', 'pc1'):
        acc[(kind, l)] = [torch.zeros((B, n), device=dev, dtype=torch.float32) for _ in range(4)]
        acc[(kind, l, 'cnt')] = torch.zeros(B, device=dev)
        acc[(kind, l, 't')] = [torch.zeros(B, device=dev) for _ in range(2)]
    acc[('u', l)] = [torch.zeros(n, device=dev) for _ in range(4)]       # plain sums of u^k, u = (z - 0)/nrm
    acc[('ut', l)] = [torch.zeros(n, device=dev) for _ in range(4)]      # tilted sums nrm * u^k
    acc[('usum', l)] = torch.zeros(2, device=dev)                        # count, sum nrm
cpu = {k: (np.zeros(tuple(v.shape)) if torch.is_tensor(v) else [np.zeros(tuple(a.shape)) for a in v]) for k, v in acc.items()}
def flush():
    for k, v in acc.items():
        if torch.is_tensor(v):
            cpu[k] += v.cpu().numpy().astype(np.float64); v.zero_()
        else:
            for i, a in enumerate(v):
                cpu[k][i] += a.cpu().numpy().astype(np.float64); a.zero_()
t0 = time.time(); done = 0; it = 0
while done < N:
    x = torch.randn((BATCH, n), generator=gen).to(dev)
    for l, (z, nrm) in enumerate(forward(x)):
        zc = z - ref[l]
        p2 = zc * zc; p3 = p2 * zc; p4 = p2 * p2
        for kind, t in (('norm', nrm), ('pc1', zc @ u1[l])):
            ids = torch.bucketize(t, edges[(kind, l)])
            a = acc[(kind, l)]
            a[0].index_add_(0, ids, zc); a[1].index_add_(0, ids, p2); a[2].index_add_(0, ids, p3); a[3].index_add_(0, ids, p4)
            acc[(kind, l, 'cnt')].index_add_(0, ids, torch.ones_like(t))
            acc[(kind, l, 't')][0].index_add_(0, ids, t); acc[(kind, l, 't')][1].index_add_(0, ids, t * t)
        u = z / nrm[:, None]
        uc = u - acc_ref[l][None, :]
        q2 = uc * uc; q3 = q2 * uc; q4 = q2 * q2
        a = acc[('u', l)]; a[0] += uc.sum(0); a[1] += q2.sum(0); a[2] += q3.sum(0); a[3] += q4.sum(0)
        w = nrm[:, None]
        a = acc[('ut', l)]; a[0] += (w * uc).sum(0); a[1] += (w * q2).sum(0); a[2] += (w * q3).sum(0); a[3] += (w * q4).sum(0)
        acc[('usum', l)] += torch.stack((torch.tensor(float(len(nrm)), device=dev), nrm.sum()))
    done += BATCH; it += 1
    if it % 4 == 0: flush()
    if it % 16 == 0: print(json.dumps(dict(samples=done, wall=round(time.time() - t0, 1))), flush=True)
flush()
out = dict(N=np.array(done), B=np.array(B))
for k, v in cpu.items():
    name = '_'.join(str(s) for s in k)
    if isinstance(v, list):
        for i, a in enumerate(v): out[f'{name}_{i}'] = a
    else: out[name] = v
for l in range(L):
    out[f'ref_{l}'] = ref[l].cpu().numpy(); out[f'uref_{l}'] = acc_ref[l].cpu().numpy()
dest = HERE / f'e1_rank{rank:04d}_N{done // 1000000}M_B{B}.npz'
np.savez(dest, **out); print(json.dumps(dict(done=done, wall=round(time.time() - t0, 1), out=str(dest))), flush=True)
