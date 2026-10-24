#!/usr/bin/env python3
"""TAG analyst mc_truth profile v1 (lead 2026-10-12 13:3xZ; contingency for a faster truth producer): where the time of one batch of mc_truth.py
(32768 samples x 16 layers on MPS, float32) goes, and how fast the candidate changes are. Runs BETWEEN truth blocks (the chain's GPU slot), about a
minute of GPU. Network: generated seed 7100136 (weights only; nothing is written but this script's own JSON lines).
Measured per batch (seconds, K batches after a warm-up, synchronised at the ends):
  base      = mc_truth.batch_sums as it is (randn + mask, 16 x [matmul, relu, d = y - ref, d.sum, (d*d).sum], stack, .cpu(), float64 add on the CPU)
  mm        = randn + 16 x [matmul, relu] only;  randn = randn + mask only;  stats = 16 x [d = y - ref, d.sum, (d*d).sum] on stored activations
  nosync    = base's arithmetic, the 32 x 1024 sums accumulated on the GPU in float32 over 8 batches, one transfer per 8 batches
  inplace   = nosync + no temporaries: layer l's activations are shifted and squared IN PLACE after layer l + 1's matmul has read them
  Each candidate's sums against base's on the SAME inputs (same generator state): largest relative difference (float32 rounding level expected).
usage: profile_mc_v1.py [K=12]   -> JSON lines on stdout"""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[_k] = '1'
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import loader
dev = torch.device('mps'); B, L, n = 32768, 16, 1024; K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
w = loader.weights(7100136); wg = [torch.from_numpy(np.ascontiguousarray(w[l])).to(dev) for l in range(L)]
gen = torch.Generator(device=dev)
sync = torch.mps.synchronize
say = lambda **r: print(json.dumps(r), flush=True)


def draw(flip):
    x = torch.randn((B, n), generator=gen, device=dev, dtype=torch.float32); x.masked_fill_(x < -7.0, 0.0)
    return x.neg_() if flip else x


gen.manual_seed(3); y = draw(False); ref = []
for l in range(L):
    y = torch.relu_(y @ wg[l]); ref.append(y.mean(0))


def base(i):
    x = draw(bool(i & 1)); s, q = [], []; y = x
    for l in range(L):
        y = torch.relu_(y @ wg[l]); d = y - ref[l]; s.append(d.sum(0)); q.append((d * d).sum(0))
    return torch.stack(s + q).cpu().numpy().astype(np.float64)


def sums_plain(x):
    parts = [None] * (2 * L); y = x
    for l in range(L):
        y = torch.relu_(y @ wg[l]); d = y - ref[l]; parts[l] = d.sum(0); parts[L + l] = (d * d).sum(0)
    return torch.stack(parts)


def sums_inplace(x):
    parts = [None] * (2 * L); prev = x
    for l in range(L):
        y = torch.relu_(prev @ wg[l])
        if l > 0:
            d = prev.sub_(ref[l - 1]); parts[l - 1] = d.sum(0); parts[L + l - 1] = d.square_().sum(0)
        prev = y
    d = prev.sub_(ref[L - 1]); parts[L - 1] = d.sum(0); parts[2 * L - 1] = d.square_().sum(0)
    return torch.stack(parts)


def grouped(fn):
    def run(i, st={}):
        if i % 8 == 0: st['acc'] = torch.zeros((2 * L, n), device=dev)
        st['acc'] += fn(draw(bool(i & 1)))
        if i % 8 == 7: st.setdefault('host', np.zeros((2 * L, n)))[...] += st['acc'].cpu().numpy().astype(np.float64)
    return run


def mm(i):
    y = draw(bool(i & 1))
    for l in range(L): y = torch.relu_(y @ wg[l])
    return y


def randn_only(i): return draw(bool(i & 1))


acts = []
gen.manual_seed(5); y = draw(False)
for l in range(L):
    y = torch.relu_(y @ wg[l]); acts.append(y.clone())


def stats(i):
    out = []
    for l in range(L):
        d = acts[l] - ref[l]; out.append(d.sum(0)); out.append((d * d).sum(0))
    return out


def timed(name, fn, k=K):
    gen.manual_seed(11); fn(0); sync()
    t0 = time.perf_counter()
    for i in range(k): fn(i)
    sync(); sec = (time.perf_counter() - t0) / k
    say(part=name, s_per_batch=round(sec, 4), samples_per_s=round(B / sec), batches=k); return sec


acc = np.zeros((2 * L, n))
t = dict(base=timed('base', lambda i: acc.__iadd__(base(i))), mm=timed('mm', mm), randn=timed('randn', randn_only), stats=timed('stats', stats),
         nosync=timed('nosync', grouped(sums_plain), 16), inplace=timed('inplace', grouped(sums_inplace), 16))
for name, fn in (('nosync', sums_plain), ('inplace', sums_inplace)):            # the same inputs through base's arithmetic and the candidate's
    gen.manual_seed(21); x = draw(False); ref_sums = sums_plain(x.clone()).cpu().numpy().astype(np.float64)
    cand = fn(x.clone()).cpu().numpy().astype(np.float64)
    say(check=name, max_rel_diff_sum=float(np.max(np.abs(cand[:L] - ref_sums[:L]) / (np.abs(ref_sums[:L]) + 1e-3 * B))),
        max_rel_diff_sq=float(np.max(np.abs(cand[L:] - ref_sums[L:]) / np.abs(ref_sums[L:]))))
say(summary=True, shares_of_base=dict(mm_incl_randn=round(t['mm'] / t['base'], 3), randn=round(t['randn'] / t['base'], 3), stats=round(t['stats'] / t['base'], 3)),
    speedup_vs_base=dict(nosync=round(t['base'] / t['nosync'], 3), inplace=round(t['base'] / t['inplace'], 3)),
    matmul_TFLOPS_in_mm=round(2 * B * n * n * L / (t['mm'] - t['randn']) / 1e12, 2), base_TFLOPS=round(2 * B * n * n * L / t['base'] / 1e12, 2))
