#!/usr/bin/env python3
"""TAG opus-reexam E25: is the learned mean correction calibrated on fresh networks, layer by layer?
On the S37 card (F4 0.6 + K4L, learned on) at every layer l the card computes host_l (the mean formula) and adds the learned
correction delta_l. With the organizers' all-layer truth m_l the residual the correction should remove is r_l = m_l - host_l.
Per layer and network: slope_l = <delta, r> / <delta, delta> (1 = calibrated, < 1 = over-confident), cos(delta, r), |delta| / |r|.
(The truth noise is uncorrelated with delta, so the slope is unbiased; layers 0-2 are at the truth noise floor.)
usage: e25_delta_slope.py <threads> <rank> [<rank> ...]  -> one json line per network + a pooled table"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
import f4 as F4
pc.set_threads(sys.argv[1]); ranks = [int(x) for x in sys.argv[2:]]
num = np.zeros(16); den = np.zeros(16); rr = np.zeros(16)
for rank in ranks:
    cd = pc.build(rank); F4.install(cd, torch, 0.6, age=2, anchor='cheap'); tp = pc.truth(rank); rec = {}
    def dh(l, d, ctx):
        rec[l] = (d.detach().to(torch.float64).numpy().copy(), ctx['host'].detach().to(torch.float64).numpy().copy()); return d
    pc.wrap(cd, 'delta', dh)
    pred = pc.run(cd)                     # (K4L left out: it changes host by a small per-unit term and is not needed for the calibration question)
    out = dict(rank=rank, final_x1e9=round(float(pc.mse(pred, tp)[15]) * 1e9, 3), slope={}, cos={}, size={})
    for l in range(1, 16):
        d, h = rec[l]; r = tp[l] - np.maximum(h, 0)
        num[l] += d @ r; den[l] += d @ d; rr[l] += r @ r
        out['slope'][l] = round(float(d @ r / (d @ d)), 3); out['cos'][l] = round(float(d @ r / np.sqrt((d @ d) * (r @ r))), 3); out['size'][l] = round(float(np.sqrt((d @ d) / (r @ r))), 3)
    print(json.dumps(out), flush=True)
print(json.dumps(dict(pooled_slope={l: round(float(num[l] / den[l]), 3) for l in range(1, 16)}, pooled_cos={l: round(float(num[l] / np.sqrt(den[l] * rr[l])), 3) for l in range(1, 16)},
                      pooled_size={l: round(float(np.sqrt(den[l] / rr[l])), 3) for l in range(1, 16)})))
