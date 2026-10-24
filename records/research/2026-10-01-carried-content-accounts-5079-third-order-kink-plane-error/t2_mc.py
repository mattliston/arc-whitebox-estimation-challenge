#!/usr/bin/env python3
"""TAG opus-physicist T2-MC: Monte Carlo TRUTH of the mechanisms that make the post-activation third cumulant at layer l, as seen by the
next layer's (2,1) plane. Same input stream as public_g19 mc_planes_mps.py / covhunter mc_allayers_mps.py (seed = mlp_seed ^ SEED_XOR ^ extra),
consumed in two BLOCKS of 8,000,000 samples (122 batches of 65536 + one of 4608 each): block 0 = the samples of mps_planes_rank*.npz
(certificate: the K21 plane of z_{l+1} must reproduce that file), block 1 = an independent draw.
At each chosen layer l (z = pre-activations, x = relu z, centred with the 32M means of covhunter's mcall file; g = Phi(mu/sigma),
d = phi/sigma from the 32M mean and variance) the job accumulates, per block:
  third-order sums M21[u] = sum u_i^2 u_j, S1 = sum u, S2 = sum u^2 for u in
     z   (pre l)            x   (post l)                zn  (pre l+1 = x W_{l+1}, centred)
     yl  = (g o z) W_{l+1}  (gate-linear transport of the true pre-activation fluctuation)
     xG  = relu(zG) - E, zG ~ N(mu, C_l) (Gaussian surrogate with the true mean and covariance), yG = xG W_{l+1}
  second- and fourth-order sums of z: M11_z = sum z_i z_j, M22 = sum z_i^2 z_j^2, M31 = sum z_i^3 z_j
  fourth-cumulant feed (d_c/2) g_i g_j Q_ijcc pushed to the next plane:
     A1 = sum yl_a^2 z_k^2,  A2 = sum yl_a z_k,  A3 = sum yl_a q_a yl_b  with q = (d o z^2) W_{l+1} - its mean,  Sq = sum q
Stops at a batch boundary once TMAX seconds have elapsed (counts saved per block).
usage: t2_mc.py <rank> <layers e.g. 5,9,11,13> [TMAX seconds = 780] [seed_extra = 0]  -> t2mc_rank{rank}[_x{extra}].npz"""
import json, queue, sys, threading, time, math
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
import os
SEED_XOR = 0x47414D454F4E5F31; BATCH = int(os.environ.get('T2_BATCH', 65_536)); BLOCK = int(os.environ.get('T2_BLOCK', 8_000_000)); FLUSH = 4
TAGX = os.environ.get('T2_TAG', '')
rank = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(',')]
TMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 780.0; extra = int(sys.argv[4]) if len(sys.argv) > 4 else 0
dev = torch.device(sys.argv[5] if len(sys.argv) > 5 else 'mps'); n = 1024
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
rmp = STORE / man['path']; row = json.loads(rmp.read_text())
W = np.load(rmp.parent / 'weights.npy')
wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
f32 = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)
K = {}
for l in layers:
    mu = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C); s = np.sqrt(v); al = mu / s
    pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = torch.erfc(torch.from_numpy(-al / math.sqrt(2))).numpy() / 2
    d = pdf / s
    ev, V = np.linalg.eigh(C); Lc = V * np.sqrt(np.clip(ev, 0, None))[None, :]     # C = Lc Lc^T (C is rank-deficient: dead units upstream)
    assert np.abs(Lc @ Lc.T - C).max() < 1e-5 * np.abs(C).max(), 'surrogate factor'   # the 32M file has eigenvalues down to -5e-6 (float32 sums)
    K[l] = dict(mu=f32(mu), g=f32(g), d=f32(d), LcT=f32(Lc.T), m=f32(MC[f'mean_post_{l}']), mG=f32(s * pdf + mu * g),
                mun=f32(MC[f'mean_pre_{l + 1}']), q0=f32((v * d) @ W[l + 1].astype(np.float64)))
rng = np.random.Generator(np.random.PCG64DXSM((int(row['mlp_seed']) ^ SEED_XOR ^ extra) & ((1 << 64) - 1)))
sizes = []
for blk in (0, 1):
    left = BLOCK
    while left: b = min(BATCH, left); sizes.append((blk, b)); left -= b
q = queue.Queue(maxsize=2)
def producer():
    for _, b in sizes: q.put(rng.standard_normal(size=(b, n), dtype=np.float32))
threading.Thread(target=producer, daemon=True).start()
torch.manual_seed(20261001 + extra)
PLANES = ('z', 'x', 'zn', 'yl', 'xG', 'yG'); MATS = [f'M21_{u}' for u in PLANES] + ['M11_z', 'M22', 'M31', 'A1', 'A2', 'A3']
VECS = [f'S1_{u}' for u in PLANES] + [f'S2_{u}' for u in PLANES] + ['S3_z', 'Sq']
gm = {(l, k): torch.zeros((n, n), device=dev) for l in layers for k in MATS}
gv = {(l, k): torch.zeros(n, device=dev) for l in layers for k in VECS}
cm = {(l, blk, k): np.zeros((n, n)) for l in layers for blk in (0, 1) for k in MATS}
cv = {(l, blk, k): np.zeros(n) for l in layers for blk in (0, 1) for k in VECS}
cnt = [0, 0]
def flush(blk):
    for (l, k), t in gm.items(): cm[(l, blk, k)] += t.cpu().numpy().astype(np.float64); t.zero_()
    for (l, k), t in gv.items(): cv[(l, blk, k)] += t.cpu().numpy().astype(np.float64); t.zero_()
t0 = time.time(); pending = 0; cur = 0; stopped = False
with torch.no_grad():
    for it, (blk, b) in enumerate(sizes):
        if blk != cur:
            flush(cur); pending = 0; cur = blk
        if time.time() - t0 > TMAX: stopped = True; break
        x = torch.from_numpy(q.get()).to(dev)
        h = x @ wg[0]
        for l in range(max(layers) + 1):
            xp = torch.clamp(h, min=0.0)
            hn = xp @ wg[l + 1]
            if l in layers:
                k = K[l]; zt = h - k['mu']; z2 = zt * zt
                yl = (zt * k['g']) @ wg[l + 1]; qt = (z2 * k['d']) @ wg[l + 1] - k['q0']
                xG = torch.clamp(torch.randn((b, n), device=dev) @ k['LcT'] + k['mu'], min=0.0) - k['mG']; yG = xG @ wg[l + 1]
                for name, u in (('z', zt), ('x', xp - k['m']), ('zn', hn - k['mun']), ('yl', yl), ('xG', xG), ('yG', yG)):
                    u2 = u * u
                    gv[(l, f'S1_{name}')] += u.sum(0); gv[(l, f'S2_{name}')] += u2.sum(0); gm[(l, f'M21_{name}')].addmm_(u2.T, u)
                gm[(l, 'M11_z')].addmm_(zt.T, zt); gm[(l, 'M22')].addmm_(z2.T, z2); z3 = z2 * zt; gm[(l, 'M31')].addmm_(z3.T, zt); gv[(l, 'S3_z')] += z3.sum(0)
                y2 = yl * yl
                gm[(l, 'A1')].addmm_(y2.T, z2); gm[(l, 'A2')].addmm_(yl.T, zt); gm[(l, 'A3')].addmm_((yl * qt).T, yl); gv[(l, 'Sq')] += qt.sum(0)
            h = hn
        cnt[blk] += b; pending += 1
        if pending == FLUSH: flush(blk); pending = 0
        if (it + 1) % 16 == 0: print(json.dumps(dict(samples=sum(cnt), wall=round(time.time() - t0, 1))), flush=True)
    flush(cur)
out = dict(layers=np.array(layers), counts=np.array(cnt), stopped=np.array(stopped))
for (l, blk, k), a in cm.items():
    if cnt[blk]: out[f'{k}_L{l}_b{blk}'] = (a / cnt[blk]).astype(np.float32)
for (l, blk, k), a in cv.items():
    if cnt[blk]: out[f'{k}_L{l}_b{blk}'] = a / cnt[blk]
dest = HERE / f't2mc_rank{rank:04d}{"" if extra == 0 else f"_x{extra}"}{TAGX}.npz'
np.savez(dest, **out)
print(json.dumps(dict(done=True, counts=cnt, stopped=stopped, wall=round(time.time() - t0, 1), out=str(dest))), flush=True)
