"""[opus-retune] paired noise-free grade vs a shared control dir (same math as opus_lead_screen_v1/grade_vs_2b108_opus_v2.py):
per seed U = mean((y-A)(y-B)) on L15 with truth halves mc_post_A/B; xMSE = 100 log(Uv/Ub), bill = 100 log(Fv/Fb), xnet = xMSE + bill.
Reports per-seed means (se, count lower) AND the score-aggregate 100 log(sum Uv / sum Ub) + 100 log(sum Fv / sum Fb).
usage: grade_retune_v1.py <variant dir> <control dir>"""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path(__file__).resolve().parents[3]
V, C = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for p in sorted(V.glob('native_*.json')):
    a = json.loads(p.read_text())['rows'][0]; s = a['seed']; c = C / f'native_{s}.json'
    if not c.exists(): continue
    b = json.loads(c.read_text())['rows'][0]
    ya = np.load(str(p).replace('.json', '_pred.npz'))[str(s)].astype(np.float64)[-1]
    yb = np.load(str(c).replace('.json', '_pred.npz'))[str(s)].astype(np.float64)[-1]
    t = np.load(O / f'analyst/fresh/truth_seed{s}.npz'); A, B = t['mc_post_A'][-1], t['mc_post_B'][-1]
    ua, ub = float(np.mean((ya - A) * (ya - B))), float(np.mean((yb - A) * (yb - B)))
    rows.append(dict(seed=s, mse=100 * math.log(a['mse_final'] / b['mse_final']), xmse=100 * math.log(ua / ub), bill=100 * math.log(a['flops'] / b['flops']),
                     ua=ua, ub=ub, fa=a['flops'], fb=b['flops']))
    r = rows[-1]; r['xnet'] = r['xmse'] + r['bill']
    print('%d  MSE %+7.3f%%  xMSE %+7.3f%%  bill %+7.3f%%  xnet %+7.3f%%' % (s, r['mse'], r['xmse'], r['bill'], r['xnet']))
if len(rows) > 1:
    for k in ('mse', 'xmse', 'bill', 'xnet'):
        v = [r[k] for r in rows]
        print('mean %-4s %+7.3f%%  se %.3f  lower %d/%d' % (k, st.mean(v), st.stdev(v) / len(v) ** .5, sum(x < 0 for x in v), len(v)))
    ax = 100 * math.log(sum(r['ua'] for r in rows) / sum(r['ub'] for r in rows)); ab = 100 * math.log(sum(r['fa'] for r in rows) / sum(r['fb'] for r in rows))
    print('AGGREGATE xMSE %+7.3f%%  bill %+7.3f%%  score %+7.3f%%  (n=%d)' % (ax, ab, ax + ab, len(rows)))
