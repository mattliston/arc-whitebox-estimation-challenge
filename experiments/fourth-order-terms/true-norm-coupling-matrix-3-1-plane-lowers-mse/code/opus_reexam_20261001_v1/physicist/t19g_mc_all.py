#!/usr/bin/env python3
"""TAG opus-physicist T19g-MC (lead's order of 09:01Z, closed-loop ceiling): the second-chaos matrix Ntilde at ALL layers 1-15 of one network.
   Ntilde_ij = Cov(zt_i zt_j, S) - 2 (C^2)_ij,  zt = z - mu,  S = |zt|^2.
Centring with the EXACT pre-activation means mu_l = (organizers' 1e9-sample post means of layer l-1) @ W_l, so no 32M file is needed
(network 651 has none). Per layer and block: M11 = sum zt zt', T = sum (S - S0) zt zt', S1 = sum zt, VS = sum (S - S0) zt, sum (S - S0)
(S0 = the mean of S over the first batch: a constant that keeps the float32 sums well conditioned).
Input stream as t2_mc.py (seed = mlp_seed ^ SEED_XOR), two blocks of <block> samples.
usage: t19g_mc_all.py <rank> [block = 4000000] [device = mps]  -> t19g_rank{rank}.npz"""
import json, queue, sys, threading, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
SEED_XOR = 0x47414D454F4E5F31; BATCH = 65_536; FLUSH = 4
rank = int(sys.argv[1]); BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000; dev = torch.device(sys.argv[3] if len(sys.argv) > 3 else 'mps'); n = 1024; layers = list(range(1, 16))
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
rmp = STORE / man['path']; row = json.loads(rmp.read_text()); W = np.load(rmp.parent / 'weights.npy')
tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
MU = {l: tp[l - 1] @ W[l].astype(np.float64) for l in layers}; mu_g = {l: torch.from_numpy(MU[l].astype(np.float32)).to(dev) for l in layers}
rng = np.random.Generator(np.random.PCG64DXSM((int(row['mlp_seed']) ^ SEED_XOR) & ((1 << 64) - 1)))
sizes = []
for blk in (0, 1):
    left = BLOCK
    while left: b = min(BATCH, left); sizes.append((blk, b)); left -= b
q = queue.Queue(maxsize=2)
def producer():
    for _, b in sizes: q.put(rng.standard_normal(size=(b, n), dtype=np.float32))
threading.Thread(target=producer, daemon=True).start()
gm = {(l, k): torch.zeros((n, n), device=dev) for l in layers for k in ('M11', 'T')}
gv = {(l, k): torch.zeros(n, device=dev) for l in layers for k in ('S1', 'VS')}
cm = {(l, blk, k): np.zeros((n, n)) for l in layers for blk in (0, 1) for k in ('M11', 'T')}
cv = {(l, blk, k): np.zeros(n) for l in layers for blk in (0, 1) for k in ('S1', 'VS')}
cs = {(l, blk): 0.0 for l in layers for blk in (0, 1)}; S0 = {}; cnt = [0, 0]
def flush(blk):
    for (l, k), t in gm.items(): cm[(l, blk, k)] += t.cpu().numpy().astype(np.float64); t.zero_()
    for (l, k), t in gv.items(): cv[(l, blk, k)] += t.cpu().numpy().astype(np.float64); t.zero_()
t0 = time.time(); pending = 0; cur = 0
with torch.no_grad():
    for it, (blk, b) in enumerate(sizes):
        if blk != cur:
            flush(cur); pending = 0; cur = blk
        h = torch.from_numpy(q.get()).to(dev) @ wg[0]
        for l in range(1, 16):
            h = torch.clamp(h, min=0.0) @ wg[l]
            zt = h - mu_g[l]; S = (zt * zt).sum(1)
            if l not in S0: S0[l] = float(S.mean())
            S = S - S0[l]; zs = zt * S[:, None]
            gm[(l, 'M11')].addmm_(zt.T, zt); gm[(l, 'T')].addmm_(zt.T, zs); gv[(l, 'S1')] += zt.sum(0); gv[(l, 'VS')] += zs.sum(0); cs[(l, blk)] += float(S.cpu().double().sum())
        cnt[blk] += b; pending += 1
        if pending == FLUSH: flush(blk); pending = 0
        if (it + 1) % 16 == 0: print(json.dumps(dict(samples=sum(cnt), wall=round(time.time() - t0, 1))), flush=True)
    flush(cur)
out = dict(counts=np.array(cnt))
for (l, blk, k), a in cm.items(): out[f'{k}_L{l}_b{blk}'] = (a / cnt[blk]).astype(np.float32)
for (l, blk, k), a in cv.items(): out[f'{k}_L{l}_b{blk}'] = a / cnt[blk]
for (l, blk), a in cs.items(): out[f'S_L{l}_b{blk}'] = np.array(a / cnt[blk])
for l in layers: out[f'S0_L{l}'] = np.array(S0[l]); out[f'mu_L{l}'] = MU[l]
dest = HERE / f't19g_rank{rank:04d}.npz'; np.savez(dest, **out)
print(json.dumps(dict(done=True, counts=cnt, wall=round(time.time() - t0, 1), out=str(dest))), flush=True)
