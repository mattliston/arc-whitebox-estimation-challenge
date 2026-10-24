#!/usr/bin/env python3
"""TAG analyst A13b certificate: the new per-unit truths of a network (fresh/pu/mps_truth_rank<rank>_A32M / _B32M) against an existing independent pair
(public_g19_attribution_v1/mps_truth_rank<rank>_P32M / _indep32M, CPU generator, another code). For each quantity and layer: the mean square of the
difference of the two pairs' means, divided by what the noise of both pairs predicts ((A - B)^2 / 4 + (P - I)^2 / 4, averaged over units): about 1 when the
two productions agree to their noise; and the layer mean of the difference in units of its standard error (a bias test). numpy, seconds.
usage: pu_cert.py <rank>"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent.parent; rank = int(sys.argv[1])
A, B = (np.load(HERE / 'pu' / f'mps_truth_rank{rank:04d}_{t}.npz') for t in ('A32M', 'B32M'))
P, I = (np.load(P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{rank:04d}_{t}.npz') for t in ('P32M', 'indep32M'))
print(f"network {rank}: ratio of the squared difference of the two productions to their predicted noise (1 = agreement), and the layer-mean bias in standard errors")
for k in ('mc_pre_m', 'mc_pre_v', 'mc_pre_k3', 'mc_pre_k4', 'mc_post'):
    cells = []
    for l in (1, 3, 5, 7, 9, 11, 12, 13, 14, 15):
        d = (A[k][l] + B[k][l]) / 2 - (P[k][l] + I[k][l]) / 2
        noise = (A[k][l] - B[k][l]) ** 2 / 4 + (P[k][l] - I[k][l]) ** 2 / 4
        ratio = float((d ** 2).mean() / noise.mean()); bias = float(d.mean() / np.sqrt(noise.mean() / d.size))
        cells.append(f"L{l} {ratio:.2f} ({bias:+.1f})")
    print(f"  {k[3:]:7s}: " + '  '.join(cells))
