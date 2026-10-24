#!/usr/bin/env python3
"""TAG opus-reexam E28: how often do the clamps of the birth pair geometry bind? (S37 physics, torch card, one forward)
The pair of hub h matches three numbers: birth coefficient d (1 + radial/3), the (2,1) coefficient beta, and the hub's own third
cumulant ('innovation'). u = beta / birth clamped to +-8 v; disc = 4 innovation / birth - 3 u^2 floored at 0.01 v^2; gap = sqrt(disc)
capped at 16 v. When the floor binds the pair no longer matches the innovation and the difference goes to the star (plane only).
usage: e28_pair_clamps.py <rank> <threads>"""
import sys, inspect, textwrap, types
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
import f4 as F4
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2])
cd = pc.build(rank, learned=True); REC = {}
owner = next(k for k in type(cd).__mro__ if 'step' in k.__dict__)
src = textwrap.dedent(inspect.getsource(owner.step))
line = "gap = t.minimum(t.maximum(disc, .01 * vv * vv).sqrt(), 16 * vv)"
assert src.count(line) == 1
src = src.replace(line, line + "\n    REC[layer] = dict(disc=(disc / (vv * vv)).detach(), u=(beta / safe_birth / vv).detach(), sel=selected.detach(), g=g.detach(), innov=innovation.detach(), birth=birth.detach(), vv=vv.detach())")
g = dict(sys.modules[owner.__module__].__dict__); g['REC'] = REC; exec(src, g); cd.step = types.MethodType(g['step'], cd)
F4.install(cd, torch, 0.6, age=2, anchor='cheap')
pred = pc.run(cd); tp = pc.truth(rank); print('final MSE x1e9', round(float(pc.mse(pred, tp)[15]) * 1e9, 4))
print('layer hubs bound | sum|mismatch| / sum|innovation| (all hubs) | energy ratio | gate of bound hubs: quartiles | median |u|/sqrt(v) bound vs free | bound with innovation and birth of opposite sign')
for l in sorted(REC):
    r = REC[l]; s = r['sel']; d = r['disc'][s]; u = r['u'][s]; a = r['innov'][s]; c = r['birth'][s]; vv = r['vv'][s]; gg = r['g'][s]
    fb = d < 0.01
    if int(fb.sum()) == 0: print(l, len(s), 0); continue
    pair_e3 = c * (3 * (u * vv) ** 2 + 0.01 * vv * vv) / 4          # what the floored pair puts on e^3
    mis = (pair_e3 - a)[fb]
    print(f"{l:5d} {len(s):5d} {int(fb.sum()):5d} | {100*float(mis.abs().sum() / a.abs().sum()):6.1f}% | {100*float((mis ** 2).sum() / (a ** 2).sum()):6.1f}% | "
          f"[{float(gg[fb].quantile(0.25)):.2f}, {float(gg[fb].median()):.2f}, {float(gg[fb].quantile(0.75)):.2f}] | "
          f"{float(u[fb].abs().median()):.3f} vs {float(u[~fb].abs().median()):.3f} | {int(((a * c) < 0)[fb].sum())}")
