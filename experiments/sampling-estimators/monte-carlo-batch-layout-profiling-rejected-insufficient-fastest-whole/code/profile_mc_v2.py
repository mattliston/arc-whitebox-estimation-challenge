#!/usr/bin/env python3
"""TAG analyst mc_truth profile v2 (lead 2026-10-12 13:3xZ, second probe): v1 found the batch matmul-bound (randn + 16 x [matmul, relu] = 79% of a batch,
the matmul at 4.9 TFLOPS on MPS; statistics 18%; one host sync per batch 5%). The only lever left is the matmul itself: throughput of the 16-layer
chain y = relu(y @ W) per sample at batch sizes 8192 ... 65536 and with the weights in the other layout (y @ Wt.t(), torch.nn.functional.linear),
float32 only (no lower precision). Runs BETWEEN truth blocks (the chain's GPU slot), well under a minute of GPU; network: generated seed 7100136.
usage: profile_mc_v2.py   -> JSON lines on stdout"""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[_k] = '1'
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import loader
dev = torch.device('mps'); L, n = 16, 1024
w = loader.weights(7100136); wg = [torch.from_numpy(np.ascontiguousarray(w[l])).to(dev) for l in range(L)]
wt = [x.t().contiguous() for x in wg]
gen = torch.Generator(device=dev); gen.manual_seed(7); sync = torch.mps.synchronize
F = torch.nn.functional


def chain(B, how):
    y = torch.randn((B, n), generator=gen, device=dev, dtype=torch.float32)
    for l in range(L):
        y = torch.relu_(y @ wg[l]) if how == 'y@W' else torch.relu_(y @ wt[l].t()) if how == 'y@Wt.t()' else torch.relu_(F.linear(y, wt[l]))
    return y


for how in ('y@W', 'y@Wt.t()', 'linear'):
    for B in (8192, 16384, 32768, 49152, 65536):
        k = max(2, int(6 * 32768 / B)); chain(B, how); sync(); t0 = time.perf_counter()
        for _ in range(k): chain(B, how)
        sync(); sec = (time.perf_counter() - t0) / k
        print(json.dumps(dict(how=how, batch=B, s_per_batch=round(sec, 4), samples_per_s=round(B / sec), matmul_TFLOPS=round(2 * B * n * n * L / sec / 1e12, 2))), flush=True)
