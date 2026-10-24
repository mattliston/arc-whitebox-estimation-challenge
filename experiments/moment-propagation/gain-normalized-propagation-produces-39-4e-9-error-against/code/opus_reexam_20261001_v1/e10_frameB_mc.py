#!/usr/bin/env python3
"""TAG opus-reexam E10: Monte Carlo truth of the GAIN-NORMALISED frame ("frame B").
A ReLU network without biases is homogeneous, so the overall size of the activation vector can be divided out at every layer.
With m_l = E[x_l] (reference: 32M-sample lab mean), l_l = m_l / |m_l|^2:
   G_l = l_l . x_l      (size of x_l along the mean direction; E[G_l] = 1)
   u_l = x_l / G_l      (the activation vector scaled to unit projection on the mean)
   frame-B expectation of f(u_l) = E[G_l f(u_l)]                (weights G_l)
Then E[x_{l+1}] = E_B[relu(W' u_l)] exactly, and the frame-B covariance is C^B_l = E[G_l (u_l - m_l)(u_l - m_l)'].
Accumulated per layer, in two independent halves (A/B) for noise:
   'B_l'  : sum G, sum G (u - m_l), sum G (u - m_l)(u - m_l)'         u = x_l / G_l            (frame-B post state of layer l)
   'Y_l'  : the same for y = x_l / G_{l-1} centred on m_l with weights G_{l-1}  (post state of layer l BEFORE re-normalisation), l >= 1
   'Z_l'  : per-unit power sums 1..4 of zeta = z_l / G_{l-1} (centred on W' m_{l-1}) with weights G_{l-1}, l >= 1
usage: e10_frameB_mc.py <rank> <N_millions>   -> e10_rank{rank}_N{N}M.npz"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
rank = int(sys.argv[1]); N = int(float(sys.argv[2]) * 1e6)
BATCH = 65_536; n = 1024; L = 16
dev = torch.device('mps')
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy')
wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(L)]
mc = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
m64 = [mc[f'mean_post_{l}'].astype(np.float64) for l in range(L)]
mref = [torch.from_numpy(m.astype(np.float32)).to(dev) for m in m64]
ell = [torch.from_numpy((m / (m @ m)).astype(np.float32)).to(dev) for m in m64]
zref = [None] + [torch.from_numpy((W[l].astype(np.float64).T @ m64[l - 1]).astype(np.float32)).to(dev) for l in range(1, L)]
gen = torch.Generator(device='cpu'); gen.manual_seed(31337 + rank)
keys = [('B', l) for l in range(L)] + [('Y', l) for l in range(1, L)]
half = {}
for h in (0, 1):
    half[h] = {k: dict(g=0.0, s1=np.zeros(n), M=np.zeros((n, n))) for k in keys}
    for l in range(1, L): half[h][('Z', l)] = dict(g=0.0, p=np.zeros((4, n)))
t0 = time.time(); done = 0; it = 0
while done < N:
    h = it % 2
    x = torch.randn((BATCH, n), generator=gen).to(dev)
    Gprev = None
    for l in range(L):
        z = x @ wg[l]
        x = torch.clamp(z, min=0.0)
        if l >= 1:
            zc = z / Gprev[:, None] - zref[l][None, :]           # zeta - reference, weights Gprev
            wv = Gprev[:, None]
            p2 = zc * zc
            acc = half[h][('Z', l)]
            acc['p'] += torch.stack(((wv * zc).sum(0), (wv * p2).sum(0), (wv * p2 * zc).sum(0), (wv * p2 * p2).sum(0))).cpu().numpy().astype(np.float64)
            acc['g'] += float(Gprev.sum())
            yc = x - Gprev[:, None] * mref[l][None, :]           # G_{l-1} (y - m_l)
            A = yc / Gprev.sqrt()[:, None]
            acc = half[h][('Y', l)]
            acc['g'] += float(Gprev.sum()); acc['s1'] += yc.sum(0).cpu().numpy().astype(np.float64)
            acc['M'] += (A.T @ A).cpu().numpy().astype(np.float64)
        G = x @ ell[l]
        xc = x - G[:, None] * mref[l][None, :]                    # G_l (u - m_l)
        A = xc / G.sqrt()[:, None]
        acc = half[h][('B', l)]
        acc['g'] += float(G.sum()); acc['s1'] += xc.sum(0).cpu().numpy().astype(np.float64)
        acc['M'] += (A.T @ A).cpu().numpy().astype(np.float64)
        Gprev = G
    done += BATCH; it += 1
    if it % 16 == 0: print(json.dumps(dict(samples=done, wall=round(time.time() - t0, 1))), flush=True)
out = dict(N=np.array(done), batch=np.array(BATCH))
for h in (0, 1):
    for k, acc in half[h].items():
        name = f'{k[0]}{k[1]}_h{h}'
        out[name + '_g'] = np.array(acc['g'])
        if 'M' in acc:
            out[name + '_s1'] = acc['s1']; out[name + '_M'] = acc['M'].astype(np.float32)
        else:
            out[name + '_p'] = acc['p']
dest = HERE / f'e10_rank{rank:04d}_N{done // 1000000}M.npz'
np.savez(dest, **out); print(json.dumps(dict(done=done, wall=round(time.time() - t0, 1), out=str(dest))), flush=True)
