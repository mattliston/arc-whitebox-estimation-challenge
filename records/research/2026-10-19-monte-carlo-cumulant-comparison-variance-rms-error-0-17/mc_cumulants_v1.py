"""TAG fable-lead 2026-10-19: Monte Carlo per-unit cumulants of a fresh network's PRE-activations at every layer (truth for the card's
per-unit (mu, v, k3, k4)): x ~ N(0, I), z_l = y_{l-1} @ W[l], y_l = relu(z_l) (loader convention, no biases). Streams batches, accumulates
raw power sums in float64 per unit, converts to central moments / cumulants. Writes <out>/mc_cum_seed<seed>.npz with mu, var, k3, k4
(16, 1024) for the pre-activations and post_mean, post_var (16, 1024).
usage: mc_cumulants_v1.py <seed> <n samples> <batch> <out dir>   (env OMP/OPENBLAS threads as you like)"""
import sys
import time
from pathlib import Path

import numpy as np

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
seed, n, batch, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
out.mkdir(exist_ok=True)
W = loader.weights(seed).astype(np.float32)          # (16, 1024, 1024), z = y @ W[l]
S = np.zeros((4, 16, 1024)); P = np.zeros((2, 16, 1024))
rng = np.random.default_rng(12345 + seed)
t0 = time.monotonic(); done = 0
while done < n:
    b = min(batch, n - done)
    y = rng.standard_normal((b, 1024), dtype=np.float32)
    for l in range(16):
        z = y @ W[l]
        zd = z.astype(np.float64)
        S[0, l] += zd.sum(0); z2 = zd * zd; S[1, l] += z2.sum(0); S[2, l] += (z2 * zd).sum(0); S[3, l] += (z2 * z2).sum(0)
        y = np.maximum(z, 0)
        yd = y.astype(np.float64); P[0, l] += yd.sum(0); P[1, l] += (yd * yd).sum(0)
    done += b
m1 = S[0] / n; m2 = S[1] / n; m3 = S[2] / n; m4 = S[3] / n
mu = m1; var = m2 - m1 ** 2
c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
k4 = c4 - 3 * var ** 2
pm = P[0] / n; pv = P[1] / n - pm ** 2
np.savez(out / ('mc_cum_seed%d.npz' % seed), mu=mu, var=var, k3=c3, k4=k4, post_mean=pm, post_var=pv, n=n, seed=seed)
print('seed %d: %d samples in %.0f s; layer 8 mean excess kurtosis k4/var^2 %.4f (unit sd %.4f), layer 12 %.4f' % (seed, n, time.monotonic() - t0, np.mean(k4[8] / var[8] ** 2), np.std(k4[8] / var[8] ** 2), np.mean(k4[12] / var[12] ** 2)))
