#!/usr/bin/env python3
"""TAG opus-physicist T101 table: module-audit arms against the fired S47e / Nd point (arm A of t93_tailscan, same request and worker arithmetic).
Ids are '<prefix>_<seed>_<arm>' (the arm may contain underscores). Per arm: final-layer change pooled, mean log (se), networks lower, all-layer change.
usage: t101_table.py <run dir> [<run dir> ...]"""
import json, math, sys
from pathlib import Path
import numpy as np
R = {}
for d_ in [Path(__file__).resolve().parent / 't93_tailscan'] + [Path(x) for x in sys.argv[1:]]:
    for f in sorted((d_ / 'results').glob('*.json')):
        d = json.loads(f.read_text()); _, s, a = d['id'].split('_', 2)
        if d_.name == 't93_tailscan' and a != 'A': continue
        R.setdefault(int(s), {})[a] = (d['mse'], float(np.mean(d['mse_layers'])))
rows = []
for arm in sorted({a for r in R.values() for a in r if a != 'A'}):
    nets = sorted(s for s in R if 'A' in R[s] and arm in R[s])
    if len(nets) < 2: continue
    a = np.array([R[s][arm][0] for s in nets]); b = np.array([R[s]['A'][0] for s in nets]); lr = np.log(a / b)
    al = np.array([R[s][arm][1] for s in nets]); bl = np.array([R[s]['A'][1] for s in nets])
    rows.append((100 * (a.mean() / b.mean() - 1), arm, len(nets), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a < b).sum()), 100 * (al.mean() / bl.mean() - 1)))
for p, arm, n, ml, se, low, alr in sorted(rows):
    print('%-16s n %2d: final %+8.3f%% pooled, mean log %+8.3f%% (se %.3f%%), lower on %2d of %2d; all-layer %+8.3f%%' % (arm, n, p, ml, se, low, n, alr))
