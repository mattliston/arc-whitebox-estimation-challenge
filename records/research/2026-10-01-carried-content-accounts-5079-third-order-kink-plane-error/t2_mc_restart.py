#!/usr/bin/env python3
"""TAG opus-physicist T2c-MC: HOW OLD is the non-isotropic fourth-order content that feeds births (and K31)? Gaussian restarts.
For each target layer l and each depth k, the pre-activation state at layer l-k is replaced by a Gaussian with the TRUE mean and
covariance (32M, covhunter mcall), the TRUE network runs k layers forward to the pre-activations of layer l, and the same sums as
t2_mc.py are accumulated at l (centred with the true 32M mean of layer l; g, d from the true mean and variance):
   M11_z, M21_z, M22, M31, S1_z, S2_z, S3_z   (pre planes K21, K22, K31 of the restart)
   A1 = sum yl_a^2 z_k^2, A2 = sum yl_a z_k, A3 = sum yl_a q_a yl_b, Sq, S1_yl, S2_yl   (fourth-cumulant feed kernels)
Everything made before layer l-k is absent: the restart at depth k contains exactly the non-Gaussian content created by the last k
ReLU layers. Two independent blocks of BLOCK samples each (surrogate normals from torch, seeded).
usage: t2_mc_restart.py <rank> <layers e.g. 9,13> <depths e.g. 1,2,4> [BLOCK millions = 4] [TMAX s = 600]  -> t2mcR_rank{rank}.npz"""
import json, sys, time, math, os
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
rank = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(',')]; depths = [int(x) for x in sys.argv[3].split(',')]
BLOCK = int(float(sys.argv[4]) * 1e6) if len(sys.argv) > 4 else 4_000_000; TMAX = float(sys.argv[5]) if len(sys.argv) > 5 else 600.0
dev = torch.device(os.environ.get('T2_DEV', 'mps')); BATCH = int(os.environ.get('T2_BATCH', 65_536)); TAGX = os.environ.get('T2_TAG', ''); n = 1024; FLUSH = 4
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy')
wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
f32 = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)
K = {}; S = {}
for l in layers:
    mu = MC[f'mean_pre_{l}'].astype(np.float64); v = np.diag(MC[f'cov_pre_{l}']).astype(np.float64); s = np.sqrt(v); al = mu / s
    pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = torch.erfc(torch.from_numpy(-al / math.sqrt(2))).numpy() / 2; d = pdf / s
    K[l] = dict(mu=f32(mu), g=f32(g), d=f32(d), q0=f32((v * d) @ W[l + 1].astype(np.float64)))
    for k in depths:
        l0 = l - k
        if l0 in S: continue
        C0 = MC[f'cov_pre_{l0}'].astype(np.float64); ev, V = np.linalg.eigh(C0); Lc = V * np.sqrt(np.clip(ev, 0, None))[None, :]
        S[l0] = dict(LcT=f32(Lc.T), mu=f32(MC[f'mean_pre_{l0}']))
MATS = ('M11_z', 'M21_z', 'M22', 'M31', 'A1', 'A2', 'A3'); VECS = ('S1_z', 'S2_z', 'S3_z', 'S1_yl', 'S2_yl', 'Sq')
keys = [(l, k) for l in layers for k in depths]
gm = {(lk, m): torch.zeros((n, n), device=dev) for lk in keys for m in MATS}; gv = {(lk, m): torch.zeros(n, device=dev) for lk in keys for m in VECS}
cm = {(lk, b, m): np.zeros((n, n)) for lk in keys for b in (0, 1) for m in MATS}; cv = {(lk, b, m): np.zeros(n) for lk in keys for b in (0, 1) for m in VECS}
cnt = [0, 0]
def flush(b):
    for (lk, m), t in gm.items(): cm[(lk, b, m)] += t.cpu().numpy().astype(np.float64); t.zero_()
    for (lk, m), t in gv.items(): cv[(lk, b, m)] += t.cpu().numpy().astype(np.float64); t.zero_()
sizes = []
for b in (0, 1):
    left = BLOCK
    while left: x = min(BATCH, left); sizes.append((b, x)); left -= x
torch.manual_seed(20261002); t0 = time.time(); pending = 0; cur = 0; stopped = False
with torch.no_grad():
    for it, (b, nb) in enumerate(sizes):
        if b != cur: flush(cur); pending = 0; cur = b
        if time.time() - t0 > TMAX: stopped = True; break
        for (l, k) in keys:
            l0 = l - k
            h = torch.randn((nb, n), device=dev) @ S[l0]['LcT'] + S[l0]['mu']
            for j in range(l0, l): h = torch.clamp(h, min=0.0) @ wg[j + 1]
            kk = K[l]; zt = h - kk['mu']; z2 = zt * zt; z3 = z2 * zt
            yl = (zt * kk['g']) @ wg[l + 1]; qt = (z2 * kk['d']) @ wg[l + 1] - kk['q0']
            key = (l, k)
            gm[(key, 'M11_z')].addmm_(zt.T, zt); gm[(key, 'M21_z')].addmm_(z2.T, zt); gm[(key, 'M22')].addmm_(z2.T, z2); gm[(key, 'M31')].addmm_(z3.T, zt)
            gm[(key, 'A1')].addmm_((yl * yl).T, z2); gm[(key, 'A2')].addmm_(yl.T, zt); gm[(key, 'A3')].addmm_((yl * qt).T, yl)
            gv[(key, 'S1_z')] += zt.sum(0); gv[(key, 'S2_z')] += z2.sum(0); gv[(key, 'S3_z')] += z3.sum(0)
            gv[(key, 'S1_yl')] += yl.sum(0); gv[(key, 'S2_yl')] += (yl * yl).sum(0); gv[(key, 'Sq')] += qt.sum(0)
        cnt[b] += nb; pending += 1
        if pending == FLUSH: flush(b); pending = 0
        if (it + 1) % 16 == 0: print(json.dumps(dict(samples=sum(cnt), wall=round(time.time() - t0, 1))), flush=True)
    flush(cur)
out = dict(layers=np.array(layers), depths=np.array(depths), counts=np.array(cnt), stopped=np.array(stopped))
for ((l, k), b, m), a in cm.items():
    if cnt[b]: out[f'{m}_L{l}_k{k}_b{b}'] = (a / cnt[b]).astype(np.float32)
for ((l, k), b, m), a in cv.items():
    if cnt[b]: out[f'{m}_L{l}_k{k}_b{b}'] = a / cnt[b]
dest = HERE / f't2mcR_rank{rank:04d}{TAGX}.npz'
np.savez(dest, **out)
print(json.dumps(dict(done=True, counts=cnt, stopped=stopped, wall=round(time.time() - t0, 1), out=str(dest))), flush=True)
