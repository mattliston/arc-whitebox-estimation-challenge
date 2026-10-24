"""TAG opus-trainer (2026-10-10 13:20Z): the zero-dial gradient table of every per-layer dial family (otr_pl / otr_m NEW, incl. the low-rank covariance
corrections cvdd ... cvmm at 0) over a set of gradient runs: per family, layers 1-14: layers beyond 2.5 standard errors, mean t^2, split-half cosine of the
14-vector, and t of the band sums (layers 1-9, 10-14). With no signal: mean t^2 about 1, 0.2 of 14 layers beyond 2.5.
usage (from O/trainer): python -B cert/cv/dial_table.py <gradient label>"""
import sys
from pathlib import Path

import numpy as np
import torch

rows = [torch.load(f, map_location='cpu', weights_only=False) for f in sorted(Path('run_M/results').glob('a_%s_r*.pt' % sys.argv[1]))]
N = len(rows); out = []
for fam in sorted(rows[0]['pl_grads']):
    G = np.stack([r['pl_grads'][fam].double().numpy()[1:15] * (1e8 / r['scale']) for r in rows])
    if not np.abs(G).sum():
        continue
    m, se = G.mean(0), G.std(0, ddof=1) / N ** .5
    t = np.where(se > 0, m / np.maximum(se, 1e-300), 0.0)
    h1, h2 = G[: N // 2].mean(0), G[N // 2:].mean(0)
    cos = float(h1 @ h2 / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-300))
    band = []
    for lo, hi in ((0, 9), (9, 14)):
        B = G[:, lo:hi].sum(1); band.append(float(B.mean() / (B.std(ddof=1) / N ** .5 + 1e-300)))
    out.append((float((t * t).mean()), fam, int((np.abs(t) > 2.5).sum()), cos, band))
print('%d gradient runs (%s); per family over layers 1-14: layers beyond 2.5 se / mean t^2 / split-half cosine / t of band sums 1-9 and 10-14' % (N, sys.argv[1]))
for tt, fam, k, cos, band in sorted(out, reverse=True):
    print('%-6s %2d / %6.2f / %+.2f / %+6.2f %+6.2f' % (fam, k, tt, cos, band[0], band[1]))
