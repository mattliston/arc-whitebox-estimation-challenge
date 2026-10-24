"""[opus-hunteridea] oracle ladder grade: per oracle, noise-free cross-half xMSE (per-seed log ratio vs ctl_s2tr), bill, per-seed values."""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
X = Path(__file__).resolve().parent; O = X.parents[2]; CT = O/'lead/runs/opus_grow_v1/ctl_s2tr'
def scr(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: return d['flops'], z[str(d['seed'])].astype(np.float64)
for t in sys.argv[2:]:
    xs, bs = [], []
    for s in [int(v) for v in sys.argv[1].split(',')]:
        try: (fc, yc), (fb, yb) = scr(X/f'screen_{t}/native_{s}.json'), scr(CT/f'native_{s}.json')
        except FileNotFoundError: continue
        with np.load(O/f'analyst/fresh/truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
        U = lambda y: float(np.mean((y[-1] - A) * (y[-1] - B))); xs.append(100 * math.log(U(yc) / U(yb))); bs.append(100 * math.log(fc / fb))
    if xs: print(f"{t:10s} n{len(xs)} xMSE {st.mean(xs):+7.3f}% (se {st.stdev(xs)/math.sqrt(len(xs)) if len(xs) > 1 else float('nan'):.3f}) bill {st.mean(bs):+6.2f}%  per-seed " + ' '.join(f'{x:+.2f}' for x in xs))
