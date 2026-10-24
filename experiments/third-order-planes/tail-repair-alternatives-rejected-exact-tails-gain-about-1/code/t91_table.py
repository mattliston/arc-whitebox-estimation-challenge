#!/usr/bin/env python3
"""TAG opus-physicist T91 table: the common tail-repair scale on the fit seeds. Arms: A (fired, from t90_repair_ceiling), x1.25 (E1 of
t91_scale16, or the global orep = yrep = 1.25 arm D of t90 where E1 was not rerun), x1.5 (E2). Per arm: final-layer change vs A pooled,
mean log (se), networks lower, all-layer change; the parabola in the scale through x1 / x1.25 / x1.5 of the mean log change -> the best
common scale and its in-sample change. Certificate: E1 = D exactly where both exist (the per-layer worker = the global option).
usage: t91_table.py [t91 run dir]"""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 't91_scale16'
R = {}
for d_ in (HERE / 't90_repair_ceiling', RUN):
    for f in sorted((d_ / 'results').glob('*.json')):
        d = json.loads(f.read_text()); _, s, a = d['id'].rsplit('_', 2); R.setdefault(int(s), {})[a] = (d['mse'], float(np.mean(d['mse_layers'])))
cert = [(s, R[s]['E1'][0] / R[s]['D'][0] - 1) for s in R if 'E1' in R[s] and 'D' in R[s]]
print('certificate E1 (per-layer worker, uniform x1.25) vs D (global option x1.25):', ['%d %.2e' % c for c in cert] or 'none')
for s in R:
    if 'E1' not in R[s] and 'D' in R[s]: R[s]['E1'] = R[s]['D']
for arm, name in (('E1', 'x1.25'), ('E2', 'x1.5')):
    nets = sorted(s for s in R if 'A' in R[s] and arm in R[s])
    a = np.array([R[s][arm][0] for s in nets]); b = np.array([R[s]['A'][0] for s in nets]); lr = np.log(a / b)
    al = np.array([R[s][arm][1] for s in nets]); bl = np.array([R[s]['A'][1] for s in nets])
    print('%s vs fired on %d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%' % (
        name, len(nets), 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a < b).sum()), len(nets), 100 * (al.mean() / bl.mean() - 1)))
nets = sorted(s for s in R if all(k in R[s] for k in ('A', 'E1', 'E2')))
Y = np.array([[0.0, math.log(R[s]['E1'][0] / R[s]['A'][0]), math.log(R[s]['E2'][0] / R[s]['A'][0])] for s in nets])
c = np.polyfit([1.0, 1.25, 1.5], Y.mean(0), 2)
xo = -c[1] / (2 * c[0]) if c[0] > 0 else float('nan')
print('mean-log parabola on %d networks: best common scale %.3f, change there %+.3f%% (in-sample); per network best scale: %s' % (
    len(nets), xo, 100 * np.polyval(c, xo) if c[0] > 0 else float('nan'),
    ' '.join('%.2f' % (-q[1] / (2 * q[0]) if q[0] > 0 else float('nan')) for q in (np.polyfit([1.0, 1.25, 1.5], y, 2) for y in Y))))
