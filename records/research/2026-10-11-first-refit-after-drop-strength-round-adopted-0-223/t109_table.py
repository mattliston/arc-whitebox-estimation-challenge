#!/usr/bin/env python3
"""TAG opus-physicist T109 table: sealed 7100200-231 look, S48T (MS point + one-direction star) vs A (S47e), paired. Pooled final-layer change, SE of the
per-network % changes, mean log (se), networks lower, all-layer change, and the per-network list. Reads only when all 64 results are in.
usage: t109_table.py [run dir]"""
import json, math, sys
from pathlib import Path
import numpy as np
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 't109_sealed200_s48t'
R = {}
for f in (RUN / 'results').glob('*.json'):
    d = json.loads(f.read_text()); _, s, a = d['id'].split('_', 2); R.setdefault(int(s), {})[a] = (d['mse'], float(np.mean(d['mse_layers'])))
assert len(R) == 32 and all(len(v) == 2 for v in R.values()), 'sealed batch incomplete'
v = np.array([[R[s]['A'][0], R[s]['S48T'][0], R[s]['A'][1], R[s]['S48T'][1]] for s in sorted(R)]); b, c, bl, cl = v.T
pct = 100 * (c / b - 1); lr = np.log(c / b); pooled = 100 * (c.mean() / b.mean() - 1); se = pct.std(ddof=1) / math.sqrt(len(pct))
print('S48T vs S47e on sealed 7100200-231 (n %d): final %+.3f%% pooled (SE of per-network %% changes %.3f%%, %+.2f SE), mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%' % (
    len(v), pooled, se, pooled / se, 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((c < b).sum()), len(v), 100 * (cl.mean() / bl.mean() - 1)))
print('per network:', ' '.join('%d:%+.2f' % (s, p) for s, p in zip(sorted(R), pct)))
