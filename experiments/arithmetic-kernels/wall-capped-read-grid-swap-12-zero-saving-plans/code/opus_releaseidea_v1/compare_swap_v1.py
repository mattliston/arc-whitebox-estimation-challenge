"""[opus-releaseidea] swap v1 vs final on the final's fresh first+warm pairs (opus_stack_v1/pairs_final var_* = final parity copy,
pairs_swap var_* = swap parity copy, each with its own captured setup): per network bill %, ops, noise-free cross-half xMSE %
(U = mean((y-A)(y-B)) at the final layer, analyst/fresh truths), bytes equality and max |dy| at the final layer."""
import json, math, statistics as st
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
ST = O/'lead/runs/opus_stack_v1'; F = O/'analyst/fresh'
rows = []
for a, b in ((7103088, 7103089), (7103090, 7103091), (7103092, 7103093), (7103094, 7103095)):
    f0 = ST/f'pairs_final/var_{a}_{b}'; f1 = ST/f'pairs_swap/var_{a}_{b}'
    if not (f1.parent/f'{f1.name}.json').exists():
        continue
    j0, j1 = json.loads((f0.parent/f'{f0.name}.json').read_text()), json.loads((f1.parent/f'{f1.name}.json').read_text())
    p0, p1 = np.load(f0.parent/f'{f0.name}_pred.npz'), np.load(f1.parent/f'{f1.name}_pred.npz')
    for pos, (r0, r1) in zip(('first', 'warm'), zip(j0['rows'], j1['rows'])):
        s = r0['seed']; assert s == r1['seed']
        y0, y1 = p0[str(s)].astype(np.float64), p1[str(s)].astype(np.float64)
        t = np.load(F/f'truth_seed{s}.npz'); A, B = t['mc_post_A'][-1].astype(np.float64), t['mc_post_B'][-1].astype(np.float64)
        U = lambda y: float(np.mean((y[-1] - A) * (y[-1] - B)))
        rows.append(dict(seed=s, position=pos, bill_pct=100 * (r1['flops'] / r0['flops'] - 1), ops=r1['ops'] - r0['ops'],
                         xmse_pct=100 * math.log(U(y1) / U(y0)), bytes_equal=p0[str(s)].tobytes() == p1[str(s)].tobytes(),
                         max_dy_final=float(np.max(np.abs(y1[-1] - y0[-1])))))
for r in rows:
    print(f"{r['seed']} {r['position']:5s} bill {r['bill_pct']:+.3f}%  ops {r['ops']:+5d}  xMSE {r['xmse_pct']:+.3f}%  bytes_equal {r['bytes_equal']}  max|dy| {r['max_dy_final']:.1e}")
for pos in ('first', 'warm'):
    v = [r['bill_pct'] for r in rows if r['position'] == pos]
    if len(v) > 1:
        print(f'{pos}: bill mean {st.mean(v):+.3f}% (SE {st.stdev(v)/math.sqrt(len(v)):.3f}, n {len(v)})')
x = [r['xmse_pct'] for r in rows]
if len(x) > 1:
    print(f'xMSE mean {st.mean(x):+.3f}% (SE {st.stdev(x)/math.sqrt(len(x)):.3f}, n {len(x)})')
json.dump(rows, open(Path(__file__).with_name('COMPARE_swap_v1.json'), 'w'), indent=1)
