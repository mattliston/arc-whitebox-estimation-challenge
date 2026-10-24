#!/usr/bin/env python3
"""TAG opus-reexam E15 capture (copy of physicist/t4_capture_cols.py writing into O/): the card's pushed bank columns pp (pre-activation,
all columns incl. the old tier as U F) and weights at chosen layers, closed loop, learned ON.
usage: e15_capture_cols.py <rank> <threads> <layers>  -> e15cols_rank{rank}_L{l}.npz"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2]); layers = [int(x) for x in sys.argv[3].split(',')]
cd = pc.build(rank); push0 = cd.shared['push']; got = {}
def push(t, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
    r = push0(t, layer, w, p, lam, previous, Q, F, core, stops, rank=rank)
    if layer in layers: got[layer] = dict(pp=r[0].numpy().copy(), lam=lam.numpy().copy(), old=np.array(int(r[4])), stops=np.array(stops))
    return r
cd.shared['push'] = push
def pin(l, v, ctx):
    if l in got: got[l].update(alive=ctx['alive'].numpy().copy(), resp=ctx['response_mask'].numpy().copy())
    return v
pc.wrap(cd, 'pair_in', pin)
pred = pc.run(cd)
for l, g in got.items():
    np.savez(HERE / f'e15cols_rank{rank:04d}_L{l}.npz', **g)
    print(json.dumps(dict(layer=l, pp=list(g['pp'].shape), old_columns=int(g['old']), n_stops=len(g['stops']))))
print(json.dumps(dict(final_x1e9=round(float(pc.mse(pred, pc.truth(rank))[15]) * 1e9, 3))))
