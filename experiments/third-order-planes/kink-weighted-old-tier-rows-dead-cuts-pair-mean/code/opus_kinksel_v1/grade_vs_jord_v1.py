"""[opus-kinksel] 8-seed screen (7103248-55, screen_judge_opus_v1) of arms vs the JORD-only screen (opus_grow_v1/screen_jord):
noise-free cross-half U log + bill log, paired per seed. usage: grade_vs_jord_v1.py tag [tag ...] (screen dirs screen_<tag> here)"""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
F = O/'analyst/fresh'; G = Path(__file__).resolve().parent; J = O/'lead/runs/opus_grow_v1/screen_jord'; util = lambda f: max(f / 2**41, .1)
def screen(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z[str(d['seed'])].astype(np.float64)
    return d['flops'], y
for tag in sys.argv[1:]:
    rows = []
    for s in range(7103248, 7103256):
        try: (fc, yc), (fb, yb) = screen(G/f'screen_{tag}/native_{s}.json'), screen(J/f'native_{s}.json')
        except FileNotFoundError: continue
        with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
        U = lambda y: float(np.mean((y[-1] - A) * (y[-1] - B)))
        ul, bl = 100 * math.log(U(yc) / U(yb)), 100 * math.log(util(fc) / util(fb)); rows.append((s, ul, bl, ul + bl, yc[-1].tobytes() == yb[-1].tobytes()))
    if not rows: print(tag, 'no rows'); continue
    n = len(rows); m = lambda i: st.mean(r[i] for r in rows); se = lambda i: (st.stdev(r[i] for r in rows) / math.sqrt(n)) if n > 1 else float('nan')
    print(f"{tag:10s} n{n} vs JORD: xMSE {m(1):+7.3f} (se {se(1):.3f})  bill {m(2):+6.3f}  xnet {m(3):+7.3f} (se {se(3):.3f})  lower {sum(r[3] < 0 for r in rows)}/{n}  identical {sum(r[4] for r in rows)}  per-seed {' '.join('%+.2f' % r[1] for r in rows)}")
