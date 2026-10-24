#!/usr/bin/env python3
"""TAG fable-covhunter (CH-1): Monte Carlo TRUTH of the mean and full covariance of the PRE- and POST-activations at ALL 16 layers, on
MPS, in two independent halves (A = even batches, B = odd batches) for the noise floor. Same input stream as
fable_mc_v1/mc_second_order_mps.py (seed = row mlp_seed ^ SEED_XOR ^ extra, batch 65536), so layers 4/7/10/13 reproduce its files.
Sums are accumulated on the GPU in float32 over CHUNK batches, then flushed into float64 CPU sums (the original copied 8 MB per layer
per batch). Input normals are drawn on a background thread (numpy fills release the GIL).
usage: mc_allayers_mps.py <rank> <P_millions> [seed_extra] [layers, default all]  -> mcall_rank{r}_P{P}M.npz
keys per layer l: mean_pre_l, cov_pre_l, mean_post_l, cov_post_l (float64, full sample); cov_{pre,post}_{A,B}_l (float32 halves),
mean_{pre,post}_{A,B}_l (float64 halves); P."""
import json, queue, sys, threading, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
SEED_XOR = 0x47414D454F4E5F31; BATCH = 65_536; CHUNK = 16
rank = int(sys.argv[1]); P = int(float(sys.argv[2]) * 1e6)
extra = int(sys.argv[3]) if len(sys.argv) > 3 else 0
layers = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else list(range(16))
dev = torch.device('mps')
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
rmp = STORE / man['path']; row = json.loads(rmp.read_text())
W = np.load(rmp.parent / 'weights.npy')
wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
rng = np.random.Generator(np.random.PCG64DXSM((int(row['mlp_seed']) ^ SEED_XOR ^ extra) & ((1 << 64) - 1)))
n = 1024
sizes = []
while sum(sizes) < P: sizes.append(min(BATCH, P - sum(sizes)))
q = queue.Queue(maxsize=2)
def producer():
    for b in sizes: q.put(rng.standard_normal(size=(b, n), dtype=np.float32))
threading.Thread(target=producer, daemon=True).start()
keys = [(l, kind) for l in layers for kind in ('pre', 'post')]
g1 = {(k, h): torch.zeros(n, device=dev) for k in keys for h in (0, 1)}
g2 = {(k, h): torch.zeros((n, n), device=dev) for k in keys for h in (0, 1)}
c1 = {(k, h): np.zeros(n) for k in keys for h in (0, 1)}
c2 = {(k, h): np.zeros((n, n)) for k in keys for h in (0, 1)}
cnt = [0, 0]
ref = {}
def flush():
    for kh in g1:
        c1[kh] += g1[kh].cpu().numpy().astype(np.float64); c2[kh] += g2[kh].cpu().numpy().astype(np.float64)
        g1[kh].zero_(); g2[kh].zero_()
t0 = time.time()
for it, b in enumerate(sizes):
    half = it % 2
    x = torch.from_numpy(q.get()).to(dev)
    for l in range(max(layers) + 1):
        h = x @ wg[l]; y = torch.clamp(h, min=0.0)
        if l in layers:
            for kind, v in (('pre', h), ('post', y)):
                k = (l, kind)
                if k not in ref: ref[k] = v.mean(0).detach().clone()
                vc = v - ref[k]
                g1[(k, half)] += vc.sum(0); g2[(k, half)].addmm_(vc.T, vc)
        x = y
    cnt[half] += b
    if (it + 1) % CHUNK == 0 or it + 1 == len(sizes): flush()
    if (it + 1) % 32 == 0: print(json.dumps(dict(samples=sum(cnt), wall=round(time.time() - t0, 1))), flush=True)
out = dict(P=np.array(P))
for (l, kind) in keys:
    r0 = ref[(l, kind)].cpu().numpy().astype(np.float64)
    s = [(c1[((l, kind), h)], c2[((l, kind), h)], cnt[h]) for h in (0, 1)]
    N = s[0][2] + s[1][2]; mc = (s[0][0] + s[1][0]) / N; M = (s[0][1] + s[1][1]) / N
    out[f'mean_{kind}_{l}'] = mc + r0; out[f'cov_{kind}_{l}'] = M - np.outer(mc, mc)
    for h, tag in ((0, 'A'), (1, 'B')):
        mh = s[h][0] / s[h][2]; Mh = s[h][1] / s[h][2]
        out[f'mean_{kind}_{tag}_{l}'] = mh + r0; out[f'cov_{kind}_{tag}_{l}'] = (Mh - np.outer(mh, mh)).astype(np.float32)
tag = f'{P / 1e6:g}'.replace('.', 'p')
dest = HERE / f'mcall_rank{rank:04d}_P{tag}M{"" if extra == 0 else f"_x{extra}"}.npz'
assert not dest.exists(), dest
np.savez(dest, **out); print(json.dumps(dict(done=P, wall=round(time.time() - t0, 1), out=str(dest))), flush=True)
