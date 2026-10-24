#!/usr/bin/env python3
"""TAG opus-physicist T92 table: Nd vs S46, stage 2 (the pinned judge's sealed seeds 7100168-199) and the pool with stage 1 (the trainer's
check on 7100232-263, trainer/run_Nd/results a_0mNdC_base / a_0mNdC_pt), as the lead's 06:06Z rule reads them: pooled final-layer change
(ratio of mean MSEs), SE of the per-network % changes, mean log ratio (se), networks lower, all-layer change (ratio of the mean 16-layer MSE).
Reads only after the whole judge batch is complete (every lane has COMPLETE.json).
usage: t92_table.py <judge run dir>"""
import json, math, sys
from pathlib import Path
import numpy as np
RUN = Path(sys.argv[1]); O = Path(__file__).resolve().parent.parent
lanes = sorted(RUN.glob('seed*'))
assert lanes and all((l / 'COMPLETE.json').is_file() for l in lanes), 'judge batch incomplete'


def rows_judge():
    out = {}
    for lane in lanes:
        rec = json.loads((lane / 'COMPLETE.json').read_text())['records']
        b, c = (json.loads(Path(r['path']).read_text()) for r in rec)
        out[int(lane.name[4:])] = (b['mse'], c['mse'], float(np.mean(b['mse_layers'])), float(np.mean(c['mse_layers'])))
    return out


def rows_stage1():
    out, res = {}, O / 'trainer/run_Nd/results'
    for f in sorted(res.glob('a_0mNdC_pt_r*.json')):
        s = int(f.stem.split('_r')[-1]); g = res / f'a_0mNdC_base_r{s}.json'
        if g.exists():
            c, b = json.loads(f.read_text()), json.loads(g.read_text())
            out[s] = (b['mse'], c['mse'], float(np.mean(b['mse_layers'])), float(np.mean(c['mse_layers'])))
    return out


def line(name, R):
    v = np.array(list(R.values())); b, c, bl, cl = v.T; pct = 100 * (c / b - 1); lr = np.log(c / b)
    pooled = 100 * (c.mean() / b.mean() - 1); se = pct.std(ddof=1) / math.sqrt(len(pct))
    print('%-28s n %2d: final %+.3f%% pooled (SE of per-network %% changes %.3f%%, pooled/SE %+.2f), mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%' % (
        name, len(v), pooled, se, pooled / se, 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((c < b).sum()), len(v), 100 * (cl.mean() / bl.mean() - 1)))


J, S1 = rows_judge(), rows_stage1()
line('stage 2 (sealed 168-199)', J)
if S1:
    line('stage 1 (232-263, trainer)', S1)
    line('pooled stages 1 + 2', {**{('a', k): v for k, v in S1.items()}, **{('b', k): v for k, v in J.items()}})
print('per network (seed: change %):', ' '.join('%d:%+.2f' % (s, 100 * (v[1] / v[0] - 1)) for s, v in sorted(J.items())))
