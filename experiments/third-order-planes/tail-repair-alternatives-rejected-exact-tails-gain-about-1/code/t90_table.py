#!/usr/bin/env python3
"""TAG opus-physicist T90 table: paired arms of a worker run directory (results/<prefix>_<seed>_<arm>.json) against arm A.
Per arm: final-layer MSE change pooled (ratio of the mean MSEs), mean log ratio over networks with its standard error, networks lower,
and the all-layer change (ratio of the mean over networks of the 16-layer mean MSE).
usage: t90_table.py <run dir> [base arm, default A]"""
import json, math, sys
from pathlib import Path
import numpy as np
RUN = Path(sys.argv[1]); BASE = sys.argv[2] if len(sys.argv) > 2 else 'A'
R = {}
for f in sorted((RUN / 'results').glob('*.json')):
    d = json.loads(f.read_text()); _, seed, arm = d['id'].rsplit('_', 2)
    R.setdefault(int(seed), {})[arm] = (d['mse'], float(np.mean(d['mse_layers'])))
errors = sorted(p.name for p in (RUN / 'results').glob('*.error'))
arms = sorted({a for r in R.values() for a in r if a != BASE})
print('T90 %s: %d networks with arm %s; errors: %s' % (RUN.name, sum(BASE in r for r in R.values()), BASE, errors or 'none'))
for arm in arms:
    nets = sorted(s for s in R if BASE in R[s] and arm in R[s])
    if len(nets) < 2: continue
    a = np.array([R[s][arm][0] for s in nets]); b = np.array([R[s][BASE][0] for s in nets]); lr = np.log(a / b)
    al = np.array([R[s][arm][1] for s in nets]); bl = np.array([R[s][BASE][1] for s in nets])
    print('  %s vs %s on %d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%' % (
        arm, BASE, len(nets), 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a < b).sum()), len(nets),
        100 * (al.mean() / bl.mean() - 1)))
