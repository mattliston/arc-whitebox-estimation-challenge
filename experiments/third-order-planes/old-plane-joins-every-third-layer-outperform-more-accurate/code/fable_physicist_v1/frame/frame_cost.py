#!/usr/bin/env python3
"""TAG fable-physicist (FJ-2): native per-join cost (dense MACs; 1 unit = 2^30 MACs = 2^31 FLOPs) of each frame method, and the total
frame-work cost of each join schedule, from the r35 bank dump sizes (fable_oldtier_economics_v1/bank_r35: cohort widths, alive rows).
Frame methods (incremental over nothing; the bank = sketch): sketch, power1, over448, warm (= over448 with a 448-wide warm block), eig.
Common per-join work (all methods): Fj = Qn^T J, joined core, overlap, rotation overlap@F (r^2 Nprev), core update (2 r^3)."""
import json
import numpy as np
from pathlib import Path
D = Path(__file__).resolve().parents[2] / 'fable_oldtier_economics_v1' / 'bank_r35'
coh = np.load(D / 'layer_14.npz')['cohorts']
act_at = {l: int(np.load(D / f'layer_{l - 1:02d}.npz')['alive'].sum()) for l in range(5, 15)}
r = 384; U = 2 ** 30
SCHED = {'every': {l: [l - 5] for l in range(5, 15)},
         'even': {6: [0, 1], 8: [2, 3], 10: [4, 5], 12: [6, 7], 14: [8, 9]},
         'every3': {7: [0, 1, 2], 10: [3, 4, 5], 13: [6, 7, 8]},
         'every3b': {8: [0, 1, 2], 11: [3, 4, 5], 14: [6, 7, 8]}}
def join_cost(act, width, nprev, frame):
    G1 = 2 * act * width * r + (2 * act * r * r + r ** 3 if nprev else 0)      # one G application on r columns
    qr = act * r * r
    common = r * act * width + r * width * r + (r * act * r + r * r * nprev + 2 * r ** 3 if nprev else 0)
    if frame == 'sketch': fr = G1 + qr
    elif frame == 'power1': fr = 2 * (G1 + qr)
    elif frame in ('over448', 'warm'):
        k = 448; Gk = G1 * k / r
        fr = 2 * Gk + act * k * k + k * act * k + 4.5 * k ** 3 + act * k * r
    elif frame == 'eig':
        fr = act * act * width + (act * r * r + act * act * r if nprev else 0) + 4.5 * act ** 3
    return (fr + common) / U, fr / U
out = {}
for s, joins in SCHED.items():
    for frame in ('sketch', 'power1', 'over448', 'warm', 'eig'):
        tot = frm = 0; nprev = 0
        for L in sorted(joins):
            width = int(sum(coh[cix][1] for cix in joins[L]))
            t_, f_ = join_cost(act_at[L], width, nprev, frame); tot += t_; frm += f_; nprev += width
        out[(s, frame)] = (tot, frm)
base = out[('every', 'sketch')][0]
print('schedule  frame    joins  frame-work units (all joins)   total join units   vs bank join units')
for (s, frame), (tot, frm) in out.items():
    print(f'{s:8s} {frame:8s} {len(SCHED[s]):3d}    {frm:8.2f}                      {tot:8.2f}          {tot - base:+8.2f}')
