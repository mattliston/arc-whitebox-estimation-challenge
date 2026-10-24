#!/usr/bin/env python3
"""TAG fable-analyst (AN-68): compare two native VALID40 row files net by net (the porter's paired_valid40_*.json, 'cand:<rank>' rows):
pooled raw = sum MSE_X / sum MSE_Y - 1, util = mean util ratio of means, adjusted = mean adj_X / mean adj_Y - 1 (the board metric is the mean
of MSE x util over MLPs), each with a 95% bootstrap interval over nets (4000 draws), plus per-net wins and min / median / max.
usage: attrib_pair.py <X json> <Y json> [label]"""
import sys, json
from pathlib import Path
import numpy as np
X, Y = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); label = sys.argv[3] if len(sys.argv) > 3 else 'X vs Y'
assert X['ranks'] == Y['ranks']; nets = X['ranks']
g = lambda D, k: np.array([D['results'][f'cand:{r}'][k] for r in nets])
mX, uX, aX, mY, uY, aY = g(X, 'mse'), g(X, 'util'), g(X, 'adjusted'), g(Y, 'mse'), g(Y, 'util'), g(Y, 'adjusted')
rng = np.random.default_rng(0); idx = rng.integers(0, len(nets), (4000, len(nets)))
def stat(num, den, pooled_sum):
    p = (num.sum() / den.sum() if pooled_sum else num.mean() / den.mean()) - 1
    b = (num[idx].sum(1) / den[idx].sum(1)) - 1
    return p, np.quantile(b, .025), np.quantile(b, .975)
out = {}
for name, num, den in (('raw', mX, mY), ('util', uX, uY), ('adjusted', aX, aY)):
    p, lo, hi = stat(num, den, True); per = num / den - 1
    out[name] = dict(pooled=float(p), ci95=[float(lo), float(hi)], wins=int((per < 0).sum()), per_net=[float(per.min()), float(np.median(per)), float(per.max())])
    print(f"{label}: {name:8s} {100 * p:+.3f}% [{100 * lo:+.3f}, {100 * hi:+.3f}]  better {out[name]['wins']}/40  per-net {100 * per.min():+.2f} / {100 * np.median(per):+.2f} / {100 * per.max():+.2f}%")
Path(__file__).resolve().parent.joinpath(f"attrib_{label.replace(' ', '_')}.json").write_text(json.dumps(out, indent=1))
