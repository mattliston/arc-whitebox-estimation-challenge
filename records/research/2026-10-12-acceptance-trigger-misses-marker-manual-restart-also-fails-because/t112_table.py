#!/usr/bin/env python3
"""TAG opus-physicist T112 table (= t92_table.py generalised): a candidate vs its base, stage 2 (the pinned judge's sealed seeds) and the pool with
stage 1 (the trainer's check results <stage-1 results dir>/<base prefix>_r<seed>.json and <point prefix>_r<seed>.json), as the lead's 06:06Z rule reads them: pooled final-layer change
(ratio of mean MSEs), SE of the per-network % changes, mean log ratio (se), networks lower, all-layer change (ratio of the mean 16-layer MSE).
Reads only after the whole judge batch is complete (every lane has COMPLETE.json).
usage: t112_table.py <judge run dir> [<stage-1 results dir> <base prefix> <point prefix>]"""
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


S1 = sys.argv[2:5] if len(sys.argv) >= 5 else None


def rows_stage1():
    out = {}
    if not S1: return out
    res, bpre, ppre = Path(S1[0]), S1[1], S1[2]
    for f in sorted(res.glob(ppre + '_r*.json')):
        s = int(f.stem.split('_r')[-1]); g = res / f'{bpre}_r{s}.json'
        if g.exists():
            c, b = json.loads(f.read_text()), json.loads(g.read_text())
            out[s] = (b['mse'], c['mse'], float(np.mean(b['mse_layers'])), float(np.mean(c['mse_layers'])))
    return out


def line(name, R):
    v = np.array(list(R.values())); b, c, bl, cl = v.T; pct = 100 * (c / b - 1); lr = np.log(c / b)
    pooled = 100 * (c.mean() / b.mean() - 1); se = pct.std(ddof=1) / math.sqrt(len(pct))
    print('%-28s n %2d: final %+.3f%% pooled (SE of per-network %% changes %.3f%%, pooled/SE %+.2f), mean log %+.3f%% (se %.3f%%), lower on %d of %d; all-layer %+.3f%%' % (
        name, len(v), pooled, se, pooled / se, 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((c < b).sum()), len(v), 100 * (cl.mean() / bl.mean() - 1)))


J, S1_ = rows_judge(), rows_stage1()
line('stage 2 (sealed)', J)
if S1_:
    line('stage 1 (trainer check)', S1_)
    line('pooled stages 1 + 2', {**{('a', k): v for k, v in S1_.items()}, **{('b', k): v for k, v in J.items()}})
print('per network (seed: change %):', ' '.join('%d:%+.2f' % (s, 100 * (v[1] / v[0] - 1)) for s, v in sorted(J.items())))
