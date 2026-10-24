#!/usr/bin/env python3
"""TAG opus-physicist T113 table: arms vs BASE (S47e + one-direction star) in physicist/t113_floors_l15, paired per seed: final-layer change pooled,
mean log (se), networks lower, all-layer; and for the births-floor arms the mean births per layer (0-14) against BASE from sel_counts.jsonl.
usage: t113_table.py"""
import json, math
from pathlib import Path
import numpy as np
RUN = Path(__file__).resolve().parent / 't113_floors_l15'
R = {}
for f in (RUN / 'results').glob('*.json'):
    d = json.loads(f.read_text()); _, s, a = d['id'].split('_', 2); R.setdefault(int(s), {})[a] = (d['mse'], float(np.mean(d['mse_layers'])))
C = {}
if (RUN / 'sel_counts.jsonl').exists():
    for line in (RUN / 'sel_counts.jsonl').read_text().splitlines():
        d = json.loads(line); arm, s = d['tag'].rsplit('_', 1); C.setdefault(arm, {})[int(s)] = d['counts']
def births(arm):
    rows = [[C[arm][s].get(str(l), {}).get('births', 0) for l in range(15)] for s in C.get(arm, {})]
    return np.mean(rows, 0) if rows else None
b0 = births('BASE')
for arm in ('B005ALL', 'B003ALL', 'B005L', 'B003L', 'L15_30', 'L15_25', 'L15_20'):
    nets = sorted(s for s in R if 'BASE' in R[s] and arm in R[s])
    if len(nets) < 2: continue
    a = np.array([R[s][arm][0] for s in nets]); b = np.array([R[s]['BASE'][0] for s in nets]); lr = np.log(a / b)
    al = np.array([R[s][arm][1] for s in nets]); bl = np.array([R[s]['BASE'][1] for s in nets])
    extra = ''
    bb = births(arm)
    if bb is not None and b0 is not None:
        extra = ' | births per layer 0-14: ' + ' '.join('%+.0f' % x for x in (bb - b0)) + ' (base ' + ' '.join('%.0f' % x for x in b0) + ')'
    print('%-8s n %2d: final %+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %2d of %2d; all-layer %+.3f%%%s' % (
        arm, len(nets), 100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a < b).sum()), len(nets), 100 * (al.mean() / bl.mean() - 1), extra))
