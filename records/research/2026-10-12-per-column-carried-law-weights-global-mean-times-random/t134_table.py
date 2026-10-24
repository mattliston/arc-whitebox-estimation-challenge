#!/usr/bin/env python3
"""TAG opus-physicist T134 table: learned-OFF screen of so2m / so2t on the S51 + gk card (t134_make.py). Final-layer MSE of each arm vs base_off on the
same network (pooled %, mean of per-network log changes with its se, lower count), all-layer %, layers 11-15; the positive control (learned ON vs OFF
on 7100584); the lead's rule: >= 1% learned-off -> jmix refit proposal, both < 0.5% -> closed (so2t's native price: about 1,100 requests, +0.6% bill).
usage: t134_table.py"""
import json
import math
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent / 't134_refit_screen' / 'results'
R = {}
for f in RUN.glob('t134_*.json'):
    d = json.loads(f.read_text())
    R[d['id']] = np.array(d['mse_layers'], dtype=np.float64)
seeds = sorted({int(k.split('_')[2]) for k in R if k.startswith('t134_b_')})
for arm in ('so2m', 'so2t'):
    nets = [s for s in seeds if 't134_b_%d_base' % s in R and 't134_b_%d_%s' % (s, arm) in R]
    if len(nets) < 2:
        print('%s: %d networks so far' % (arm, len(nets)))
        continue
    a = np.array([R['t134_b_%d_%s' % (s, arm)] for s in nets]); b = np.array([R['t134_b_%d_base' % s] for s in nets])
    lr = np.log(a[:, -1] / b[:, -1])
    print('%s (learned OFF) vs base_off on %d (%d-%d): final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%; L11-15 %s; '
          'per network %s' % (arm, len(nets), nets[0], nets[-1], 100 * (a[:, -1].mean() / b[:, -1].mean() - 1), 100 * lr.mean(),
                              100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a[:, -1] < b[:, -1]).sum()), len(nets),
                              100 * (a.mean(1).mean() / b.mean(1).mean() - 1), ' '.join('%+.2f' % (100 * (a[:, l].mean() / b[:, l].mean() - 1)) for l in range(11, 16)),
                              ' '.join('%+.2f' % (100 * x) for x in lr)))
on = [s for s in seeds if 't134_a_ctl_on_%d' % s in R and 't134_b_%d_base' % s in R]      # learned ON on all 8 (lead 00:0xZ: the per-network spread of the learned gain)
if on:
    g = np.array([100 * (R['t134_a_ctl_on_%d' % s][-1] / R['t134_b_%d_base' % s][-1] - 1) for s in on])
    ga = np.array([100 * (R['t134_a_ctl_on_%d' % s][:15].mean() / R['t134_b_%d_base' % s][:15].mean() - 1) for s in on])
    print('learned ON vs OFF (control; the learned mean correction\'s value) on %d: final mean %+.2f%%, sd %.2f%%, range %+.2f..%+.2f%% | layers 0-14 mean %+.2f%%, sd %.2f%%; '
          'per network final %s' % (len(on), g.mean(), g.std(ddof=1) if len(g) > 1 else 0.0, g.min(), g.max(), ga.mean(), ga.std(ddof=1) if len(ga) > 1 else 0.0,
                                   ' '.join('%d:%+.2f' % (s, x) for s, x in zip(on, g))))
