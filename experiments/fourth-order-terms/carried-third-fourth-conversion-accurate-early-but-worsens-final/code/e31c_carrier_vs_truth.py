#!/usr/bin/env python3
"""TAG opus-reexam E31c: the carried conversion term q_i = (12/n) sum_r c_r pp_ri^2 (E30 dumps, four retention values) against the Monte Carlo
second-chaos content (6/n) Q_i (E31), network 35, per layer: weighted correlation of the unit-specific parts (v projected out) and the slope."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; rank = int(sys.argv[1]); n = 1024
D = np.load(HERE / f'e31_rank{rank:04d}.npz'); PU = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz')
for l in [int(x) for x in D['layers']]:
    va = PU['mc_pre_v'][l].astype(np.float64); mu = PU['mc_pre_m'][l].astype(np.float64); al = mu / np.sqrt(va)
    w = ((al * al - 1) * np.exp(-al * al / 2)) ** 2 + 1e-12; sw = np.sqrt(w)
    base = np.stack([va], 1) * sw[:, None]; res = lambda y: y * sw - base @ np.linalg.lstsq(base, y * sw, rcond=None)[0]
    Q = (6.0 / n) * (D[f'Q_{l}_0'] + D[f'Q_{l}_1']) / 2; rq = res(Q); out = dict(layer=l, Q_unit_specific_rms=float('%.3g' % np.sqrt((rq ** 2).mean())))
    for name in ('t0', 't05', 't10', 't13'):
        q = np.load(HERE / f'e30_dump_rank{rank:04d}_{name}.npz')[f'q_{l}']; rc = res(q)
        out[name] = dict(corr=round(float(np.corrcoef(rc, rq)[0, 1]), 3), slope=round(float((rc @ rq) / (rc @ rc)), 3), size_ratio=round(float(np.sqrt((rc ** 2).mean() / (rq ** 2).mean())), 3))
    print(json.dumps(out))
