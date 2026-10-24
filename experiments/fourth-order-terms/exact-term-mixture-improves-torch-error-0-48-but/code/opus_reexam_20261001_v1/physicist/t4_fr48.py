#!/usr/bin/env python3
"""TAG opus-physicist: the confirmation table on held-out ranks NOT used for the mixture fit (k4l_pred_rank*_<tag>.npz with arms base, FK, MBK ...).
Final-layer MSE against the organizers' truth; pooled change against base and between arms, paired bootstrap over networks.
usage: t4_fr48.py <tag> [<arm> <reference arm> ...]"""
import sys, glob
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; rng = np.random.default_rng(0); tag = sys.argv[1]
D = {int(Path(f).name.split('_')[2][4:]): np.load(f) for f in glob.glob(str(HERE / f'k4l_pred_rank*_{tag}.npz'))}; R = sorted(D)
m = lambda r, k, L=15: float(((D[r][k][L] - D[r]['truth'][L]) ** 2).mean()); ma = lambda r, k: float(((D[r][k] - D[r]['truth']) ** 2).mean())
def cmp(a, b):
    x = np.array([m(r, a) for r in R]); y = np.array([m(r, b) for r in R]); per = x / y - 1
    idx = rng.integers(0, len(R), size=(4000, len(R))); bs = x[idx].sum(1) / y[idx].sum(1) - 1
    print(f"  {a:8s} against {b:8s}: pooled {100 * (x.sum() / y.sum() - 1):+.2f}% [95% {100 * np.percentile(bs, 2.5):+.2f}, {100 * np.percentile(bs, 97.5):+.2f}]  better on {int((per < 0).sum())}/{len(R)}"
          f"  per-network mean {100 * per.mean():+.2f}% (se {100 * per.std(ddof=1) / len(R) ** 0.5:.2f})  worst {100 * per.max():+.2f}%  all-layer {100 * (sum(ma(r, a) for r in R) / sum(ma(r, b) for r in R) - 1):+.2f}%")
arms = [k for k in D[R[0]].files if k not in ('truth', 'base')]
print(f"{len(R)} networks {R}; base pooled final-layer MSE {np.mean([m(r, 'base') for r in R]):.4e}")
for a in arms: cmp(a, 'base')
pairs = sys.argv[2:]
for a, b in zip(pairs[::2], pairs[1::2]): cmp(a, b)
