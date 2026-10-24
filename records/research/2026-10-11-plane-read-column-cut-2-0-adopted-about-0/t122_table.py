#!/usr/bin/env python3
"""TAG opus-physicist T122 table: read column cut threshold scan on the S48 card (torch, the trainer's worker v11 pinned), each arm paired per network with
rdc 2.0 (S48's setting): final-layer MSE pooled change, mean log (se), networks lower, all-layer; per layer 5-15 pooled; and every arm against no cut.
usage: t122_table.py"""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
R = {}
for f in (HERE / 't122_rdc_scan' / 'results').glob('*.json'):
    d = json.loads(f.read_text()); _, s, arm = d['id'].split('_', 2); R.setdefault(int(s), {})[arm] = np.array(d['mse_layers'], float)
for ref in ('r20', 'none'):
    for arm in ('none', 'r24', 'r22', 'r20', 'r18'):
        if arm == ref:
            continue
        nets = sorted(s for s in R if ref in R[s] and arm in R[s])
        if len(nets) < 2:
            continue
        a = np.array([R[s][arm] for s in nets]); b = np.array([R[s][ref] for s in nets]); lr = np.log(a[:, -1] / b[:, -1])
        print('%-4s vs %-4s on %2d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %2d; all-layer %+.3f%% | L5-15: %s' % (
            arm, ref, len(nets), 100 * (a[:, -1].mean() / b[:, -1].mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)),
            int((a[:, -1] < b[:, -1]).sum()), 100 * (a.mean(1).mean() / b.mean(1).mean() - 1),
            ' '.join('%+.2f' % (100 * (a[:, l].mean() / b[:, l].mean() - 1)) for l in range(5, 16))))
