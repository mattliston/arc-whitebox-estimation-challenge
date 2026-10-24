#!/usr/bin/env python3
"""TAG fable-covhunter (CH-39): closed-loop oracle of the carried b vector on the torch S6b card (fable_analyst_v1/s6bcard.py, learned ON),
at every layer L1-14, strength eps (b -> b + eps (target - b)), final-MSE ratio vs S6b:
  c3m   : target = c3 m, the exact input-radial value (c3 = 1/E[r]^2 = 9.7704e-4, m = post mean)
  split : the b source term uses the exact input-radial 6 c3 instead of the card's radial: b -> b - eps (radial - 6 c3)/6 d v
  rad6m : target = (new radial / 6) m, the scale of the least-squares Sym content of the TRUE K21 (homog.py: b_fit ~ radial/6 mu)
usage: homog_oracle.py <ranks> <cfgs e.g. c3m:0.5,c3m:1,split:0.5,split:1,rad6m:0.5,rad6m:1> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
for _p in (str(HERE), str(P2 / 'fable_analyst_v1')):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np, torch
import s6bcard as s6
ranks = [int(x) for x in sys.argv[1].split(',')]; cfgs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3])); out = Path(sys.argv[4])
C3 = 9.770395e-04
pl = s6.payload(torch)


def hook(kind, eps):
    def h(l, b, ctx):
        if not 1 <= l <= 14: return b
        if kind == 'c3m': return b + eps * (C3 * ctx['m'] - b)
        if kind == 'split': return b - eps * (ctx['radial'] - 6 * C3) / 6 * ctx['d'] * ctx['vv']
        if kind == 'rad6m': return b + eps * (ctx['new_radial'] / 6 * ctx['m'] - b)
        raise ValueError(kind)
    return h


for rk in ranks:
    w = s6.weights(rk); truth = s6.truthpost(rk); base = None
    for cfg in ['base'] + cfgs:
        t0 = time.time(); cd = s6.card(torch, w, *pl)
        if cfg != 'base':
            kind, eps = cfg.split(':'); cd.hooks['b_post'] = hook(kind, float(eps))
        with torch.no_grad(): pred = cd.run(False)[15].to(torch.float64).numpy()
        mse = float(((pred - truth[15]) ** 2).mean()); base = mse if cfg == 'base' else base
        rec = dict(rank=rk, card='S6b', cfg=cfg, mse=mse, ratio=mse / base, wall=round(time.time() - t0, 1))
        print(json.dumps(rec), flush=True)
        with out.open('a') as f: f.write(json.dumps(rec) + '\n')
