"""[opus-releaseidea] swap v1 (var_*) vs final (ctl_*) on pairs_more (same process layout: first + warm network after a captured setup),
plus the earlier 4 pairs (COMPARE_swap_v1.json) pooled: bill %, ops, noise-free cross-half xMSE % at the final layer."""
import json, math, statistics as st
from pathlib import Path
import numpy as np
X = Path(__file__).resolve().parent; O = X.parents[2]; F = O/'analyst/fresh'; D = X/'pairs_more'
rows = []
for j1 in sorted(D.glob('var_*.json')):
    j0 = D/j1.name.replace('var_', 'ctl_')
    if not j0.exists():
        continue
    a, b = json.loads(j0.read_text()), json.loads(j1.read_text())
    p0, p1 = np.load(str(j0).replace('.json', '_pred.npz')), np.load(str(j1).replace('.json', '_pred.npz'))
    for pos, (r0, r1) in zip(('first', 'warm'), zip(a['rows'], b['rows'])):
        s = r0['seed']; assert s == r1['seed']
        y0, y1 = p0[str(s)].astype(np.float64), p1[str(s)].astype(np.float64)
        t = np.load(F/f'truth_seed{s}.npz'); A, B = t['mc_post_A'][-1].astype(np.float64), t['mc_post_B'][-1].astype(np.float64)
        U = lambda y: float(np.mean((y[-1] - A) * (y[-1] - B)))
        rows.append(dict(seed=s, position=pos, bill_pct=100 * (r1['flops'] / r0['flops'] - 1), ops=r1['ops'] - r0['ops'],
                         xmse_pct=100 * math.log(U(y1) / U(y0)), bytes_equal=p0[str(s)].tobytes() == p1[str(s)].tobytes()))
for r in rows:
    print(f"{r['seed']} {r['position']:5s} bill {r['bill_pct']:+.3f}%  ops {r['ops']:+5d}  xMSE {r['xmse_pct']:+.3f}%  bytes_equal {r['bytes_equal']}")
allrows = rows + json.loads((X/'COMPARE_swap_v1.json').read_text())
for name, rr in (('new pairs', rows), ('all pairs', allrows)):
    for pos in ('first', 'warm'):
        v = [r['bill_pct'] for r in rr if r['position'] == pos]
        if len(v) > 1:
            print(f"{name} {pos}: bill mean {st.mean(v):+.3f}% (SE {st.stdev(v)/math.sqrt(len(v)):.3f}, n {len(v)}, changed {sum(abs(x) > 1e-6 for x in v)})")
    x = [r['xmse_pct'] for r in rr]
    if len(x) > 1:
        print(f"{name} xMSE mean {st.mean(x):+.3f}% (SE {st.stdev(x)/math.sqrt(len(x)):.3f}, n {len(x)})")
json.dump(rows, open(X/'COMPARE_more_v1.json', 'w'), indent=1)
