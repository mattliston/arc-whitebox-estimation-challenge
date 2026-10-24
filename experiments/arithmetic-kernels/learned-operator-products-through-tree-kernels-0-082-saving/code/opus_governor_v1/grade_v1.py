"""[opus-governor] forced-trigger accuracy vs S2tr (46e9315d), same native diag judge: plain and noise-free (cross-half) final-layer
MSE log-ratio %, flops %, max |prediction change|. usage: grade_v1.py <name> <seed>...  (diag/<name>_<seed>.json vs diag/base_<seed>.json;
seed 7103248's base is opus_s2tr_v1/diag_S2tr_7103248.json, the same judge)"""
import json
import math
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
O = D.parents[2]
name, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]


def load(p, s):
    row = json.loads(p.read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z:
        return row, z[str(s)].astype(np.float64)


rows = []
for s in seeds:
    bp = D.parent/'opus_s2tr_v1/diag_S2tr_7103248.json' if s == 7103248 else D/f'diag/base_{s}.json'
    (b, yb), (v, yv) = load(bp, s), load(D/f'diag/{name}_{s}.json', s)
    t = np.load(O/f'analyst/fresh/truth_seed{s}.npz')
    TA, TB = t['mc_post_A'][-1], t['mc_post_B'][-1]
    ub, uv = np.mean((yb[-1] - TA)*(yb[-1] - TB)), np.mean((yv[-1] - TA)*(yv[-1] - TB))
    rows.append(dict(seed=s, MSE=100*math.log(v['mse_final']/b['mse_final']), xMSE=100*math.log(uv/ub),
                     flops=100*(v['flops']/b['flops'] - 1), ops=v['ops'] - b['ops'], finite=bool(np.isfinite(yv).all()),
                     max_dy_final=float(np.abs(yv[-1] - yb[-1]).max()), rms_y_final=float(np.sqrt(np.mean(yb[-1]**2)))))
for r in rows:
    print('%d MSE %+.4f%% xMSE %+.4f%% flops %+.3f%% ops %+d finite %s max|dy| %.2e (rms y %.2e)' % (
        r['seed'], r['MSE'], r['xMSE'], r['flops'], r['ops'], r['finite'], r['max_dy_final'], r['rms_y_final']))
m = {k: sum(r[k] for r in rows)/len(rows) for k in ('MSE', 'xMSE', 'flops')}
print(name, 'mean over %d seeds: MSE %+.4f%% xMSE %+.4f%% flops %+.3f%%' % (len(rows), m['MSE'], m['xMSE'], m['flops']))
(D/f'GRADE_{name}_v1.json').open('x').write(json.dumps(dict(name=name, rows=rows, mean=m), indent=1) + '\n')
