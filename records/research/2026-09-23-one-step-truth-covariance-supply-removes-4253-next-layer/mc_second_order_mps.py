#!/usr/bin/env python3
"""TAG fable-mc (MC-1): Monte Carlo TRUTH of second-order statistics (mean, full covariance) of the PRE- and POST-activations at chosen
layers, on MPS, in two independent halves (A = even batches, B = odd batches) so the MC noise is measurable (A-B). Same input stream
convention as public_g19_attribution_v1/mc_planes_mps.py (seed from the row's mlp_seed ^ SEED_XOR ^ extra).
usage: mc_second_order_mps.py <rank> <P_millions> <layers e.g. 4,7,10,13> [seed_extra]  -> mc2_rank{r}_P{P}M.npz
keys: mean_pre_l, cov_pre_l, mean_post_l, cov_post_l (full sample) and *_A_l, *_B_l halves (cov only), P."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
SEED_XOR = 0x47414D454F4E5F31; BATCH = 65_536
rank = int(sys.argv[1]); P = int(float(sys.argv[2]) * 1e6); layers = [int(x) for x in sys.argv[3].split(',')]
extra = int(sys.argv[4]) if len(sys.argv) > 4 else 0
dev = torch.device('mps')
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
rmp = STORE / man['path']; row = json.loads(rmp.read_text())
W = np.load(rmp.parent / 'weights.npy', mmap_mode='r')
wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
rng = np.random.Generator(np.random.PCG64DXSM((int(row['mlp_seed']) ^ SEED_XOR ^ extra) & ((1 << 64) - 1)))
n = 1024
acc = {(l, kind, half): [np.zeros(n), np.zeros((n, n)), 0] for l in layers for kind in ('pre', 'post') for half in (0, 1)}
done, it, t0 = 0, 0, time.time()
ref = {}
while done < P:
    b = min(BATCH, P - done); half = it % 2
    x = torch.from_numpy(rng.standard_normal(size=(b, n), dtype=np.float32)).to(dev)
    for l in range(16):
        h = x @ wg[l]; y = torch.clamp(h, min=0.0)
        if l in layers:
            for kind, v in (('pre', h), ('post', y)):
                key = (l, kind)
                if key not in ref: ref[key] = v.mean(0).detach().clone()
                vc = v - ref[key]
                a = acc[(l, kind, half)]
                a[0] += vc.sum(0).cpu().numpy().astype(np.float64); a[1] += (vc.T @ vc).cpu().numpy().astype(np.float64); a[2] += b
        x = y
    done += b; it += 1
    if it % 32 == 0: print(json.dumps(dict(samples=done, wall=round(time.time() - t0, 1))), flush=True)
out = dict(P=np.array(P))
for l in layers:
    for kind in ('pre', 'post'):
        s = [acc[(l, kind, h)] for h in (0, 1)]
        r0 = ref[(l, kind)].cpu().numpy().astype(np.float64)
        N = s[0][2] + s[1][2]; mc = (s[0][0] + s[1][0]) / N; M = (s[0][1] + s[1][1]) / N
        out[f'mean_{kind}_{l}'] = (mc + r0).astype(np.float64); out[f'cov_{kind}_{l}'] = (M - np.outer(mc, mc)).astype(np.float64)
        for h, tag in ((0, 'A'), (1, 'B')):
            mh = s[h][0] / s[h][2]; Mh = s[h][1] / s[h][2]
            out[f'cov_{kind}_{tag}_{l}'] = (Mh - np.outer(mh, mh)).astype(np.float32)
dest = HERE / f'mc2_rank{rank:04d}_P{int(P/1e6)}M.npz'
np.savez(dest, **out); print(json.dumps(dict(done=P, wall=round(time.time() - t0, 1), out=str(dest))), flush=True)
