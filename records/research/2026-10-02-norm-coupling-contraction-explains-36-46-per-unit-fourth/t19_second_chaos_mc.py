#!/usr/bin/env python3
"""TAG opus-physicist T19-MC (lead's request of 08:25Z): Monte Carlo of the SECOND-CHAOS matrix of the fourth cumulant.
For the pre-activations z of layer l (centred with the 32M mean, zt), S = |zt|^2 the squared norm of the layer and p = mhat . zt the
projection on the unit mean direction mhat = mu / |mu|:
   Ntilde_ij  = sum_k kappa4(z_i, z_j, z_k, z_k) = Cov(zt_i zt_j, S) - 2 (C^2)_ij
   Nmean_ij   = kappa4(z_i, z_j, p, p)           = Cov(zt_i zt_j, p^2) - 2 (C mhat)_i (C mhat)_j
(diagonals = the lead's Q_i and Qm_i). Same input stream and blocks as t2_mc.py (seed = mlp_seed ^ SEED_XOR ^ extra; two blocks of
8,000,000 samples, batches of 65,536), so block 0 holds the samples of the stored 8M planes and of t2mc block 0, block 1 is independent.
Accumulated per layer and block (weights centred with the 32M values to keep float32 sums well conditioned):
   M11 = sum zt zt',  T = sum (S - trC0) zt zt',  Tm = sum (p^2 - p2_0) zt zt',  S1 = sum zt,  and the scalars sum S, S^2, p^2, p^4, S p^2.
usage: t19_second_chaos_mc.py <rank> <layers e.g. 5,9,11,13,15> [device = mps] [TMAX seconds = 900]  -> t19mc_rank{rank}.npz"""
import json, queue, sys, threading, time, os
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
SEED_XOR = 0x47414D454F4E5F31; BATCH = int(os.environ.get('T2_BATCH', 65_536)); BLOCK = int(os.environ.get('T2_BLOCK', 8_000_000)); FLUSH = 4
rank = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(',')]
dev = torch.device(sys.argv[3] if len(sys.argv) > 3 else 'mps'); TMAX = float(sys.argv[4]) if len(sys.argv) > 4 else 900.0; n = 1024
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
rmp = STORE / man['path']; row = json.loads(rmp.read_text())
W = np.load(rmp.parent / 'weights.npy')
wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
f32 = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)
K = {}
for l in layers:
    mu = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); mh = mu / np.linalg.norm(mu)
    K[l] = dict(mu=f32(mu), mh=f32(mh), trC=float(np.trace(C)), p2=float(mh @ C @ mh))
rng = np.random.Generator(np.random.PCG64DXSM((int(row['mlp_seed']) ^ SEED_XOR) & ((1 << 64) - 1)))
sizes = []
for blk in (0, 1):
    left = BLOCK
    while left: b = min(BATCH, left); sizes.append((blk, b)); left -= b
q = queue.Queue(maxsize=2)
def producer():
    for _, b in sizes: q.put(rng.standard_normal(size=(b, n), dtype=np.float32))
threading.Thread(target=producer, daemon=True).start()
MATS = ('M11', 'T', 'Tm'); SCAL = ('S', 'SS', 'P2', 'P4', 'SP2')
gm = {(l, k): torch.zeros((n, n), device=dev) for l in layers for k in MATS}
gv = {l: torch.zeros(n, device=dev) for l in layers}
cm = {(l, blk, k): np.zeros((n, n)) for l in layers for blk in (0, 1) for k in MATS}
cv = {(l, blk): np.zeros(n) for l in layers for blk in (0, 1)}
cs = {(l, blk, k): 0.0 for l in layers for blk in (0, 1) for k in SCAL}
cnt = [0, 0]
def flush(blk):
    for (l, k), t in gm.items(): cm[(l, blk, k)] += t.cpu().numpy().astype(np.float64); t.zero_()
    for l, t in gv.items(): cv[(l, blk)] += t.cpu().numpy().astype(np.float64); t.zero_()
t0 = time.time(); pending = 0; cur = 0; stopped = False
with torch.no_grad():
    for it, (blk, b) in enumerate(sizes):
        if blk != cur:
            flush(cur); pending = 0; cur = blk
        if time.time() - t0 > TMAX: stopped = True; break
        x = torch.from_numpy(q.get()).to(dev)
        h = x @ wg[0]
        for l in range(max(layers) + 1):
            if l in layers:
                k = K[l]; zt = h - k['mu']; S = (zt * zt).sum(1) - k['trC']; p = zt @ k['mh']; p2 = p * p - k['p2']
                gm[(l, 'M11')].addmm_(zt.T, zt); gm[(l, 'T')].addmm_(zt.T, zt * S[:, None]); gm[(l, 'Tm')].addmm_(zt.T, zt * p2[:, None]); gv[l] += zt.sum(0)
                Sd, pd = S.cpu().double(), p2.cpu().double()        # centred weights; the scalars go to float64 on the CPU
                for name, val in (('S', Sd.sum()), ('SS', (Sd * Sd).sum()), ('P2', pd.sum()), ('P4', (pd * pd).sum()), ('SP2', (Sd * pd).sum())): cs[(l, blk, name)] += float(val)
            if l < 15: h = torch.clamp(h, min=0.0) @ wg[l + 1]
        cnt[blk] += b; pending += 1
        if pending == FLUSH: flush(blk); pending = 0
        if (it + 1) % 16 == 0: print(json.dumps(dict(samples=sum(cnt), wall=round(time.time() - t0, 1))), flush=True)
    flush(cur)
out = dict(layers=np.array(layers), counts=np.array(cnt), stopped=np.array(stopped))
for (l, blk, k), a in cm.items():
    if cnt[blk]: out[f'{k}_L{l}_b{blk}'] = (a / cnt[blk]).astype(np.float64 if k == 'M11' else np.float32)
for (l, blk), a in cv.items():
    if cnt[blk]: out[f'S1_L{l}_b{blk}'] = a / cnt[blk]
for (l, blk, k), a in cs.items():
    if cnt[blk]: out[f'{k}_L{l}_b{blk}'] = np.array(a / cnt[blk])
for l in layers: out[f'trC0_L{l}'] = np.array(K[l]['trC']); out[f'p20_L{l}'] = np.array(K[l]['p2'])
dest = HERE / f't19mc_rank{rank:04d}.npz'
np.savez(dest, **out)
print(json.dumps(dict(done=True, counts=cnt, stopped=stopped, wall=round(time.time() - t0, 1), out=str(dest))), flush=True)
