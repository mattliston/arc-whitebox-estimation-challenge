"""[opus-tank] pooled B-vs-JORD effect over all seeds with both runs: S8 (screen_b10 vs opus_grow_v1/screen_jord), A80 (screen_b10 vs
opus_retest_v1/screenA80_jord), X24 7103096-119 (screenX24_b10 vs screenX24_jord). Noise-free cross-half final U, + bill."""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = Path(__file__).resolve().parent; F = O/'analyst/fresh'
J8, J80 = O/'lead/runs/opus_grow_v1/screen_jord', O/'lead/runs/opus_retest_v1/screenA80_jord'


def screen(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z[str(d['seed'])].astype(np.float64)
    return d['flops'], y


util = lambda f: max(f / 2**41, .1)
rows = []
for s in list(range(7103248, 7103256)) + list(range(7103080, 7103096)) + list(range(7103096, 7103120)):
    a, j = ((D/'screen_b10', J8) if s >= 7103248 else (D/'screen_b10', J80) if s < 7103096 else (D/'screenX24_b10', D/'screenX24_jord'))
    try: (fa, ya), (fj, yj) = screen(a/f'native_{s}.json'), screen(j/f'native_{s}.json')
    except FileNotFoundError: continue
    with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'].astype(np.float64), z['mc_post_B'].astype(np.float64)
    U = lambda y: float(np.mean((y[-1] - A[-1]) * (y[-1] - B[-1])))
    v = 100 * math.log(U(ya) / U(yj)); b = 100 * math.log(util(fa) / util(fj)); rows.append((s, v + b, v, b))


def summ(rr, name):
    n = len(rr); m = st.mean(r[1] for r in rr); se = st.stdev(r[1] for r in rr) / math.sqrt(n)
    print(f"{name}: n{n} B vs JORD xnet {m:+.3f} (se {se:.3f}) xMSE {st.mean(r[2] for r in rr):+.3f} lower {sum(r[2] < 0 for r in rr)}/{n}")


summ(rows, 'POOLED')
summ([r for r in rows if 7103096 <= r[0] < 7103120], 'X24 new')
summ([r for r in rows if r[0] < 7103096 or r[0] >= 7103248], 'first 24')
