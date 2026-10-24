"""[opus-grow] 8-seed screen (7103248-55, screen_judge_opus_v1) of grow variants vs the shared S2tr control arm (ctl_s2tr):
noise-free cross-half U log + bill log. Bar: xnet <= -0.15% -> 24 seeds. rate = -dxMSE/dbill. usage: grade_grow_v1.py tag [tag ...]"""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
F = O/'analyst/fresh'; G = O/'lead/runs/opus_grow_v1'; util = lambda f: max(f / 2**41, .1)
def screen(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z[str(d['seed'])].astype(np.float64)
    return d['flops'], y
out = {}
for tag in sys.argv[1:]:
    rows = []
    for s in range(7103248, 7103256):
        try: (fc, yc), (fb, yb) = screen(G/f'screen_{tag}/native_{s}.json'), screen(G/f'ctl_s2tr/native_{s}.json')
        except FileNotFoundError: continue
        with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
        U = lambda y: float(np.mean((y[-1] - A) * (y[-1] - B)))
        ident = yc[-1].astype(np.float32).tobytes() == yb[-1].astype(np.float32).tobytes()
        ul, bl = 100 * math.log(U(yc) / U(yb)), 100 * math.log(util(fc) / util(fb)); rows.append((s, ul, bl, ul + bl, ident))
    if not rows: continue
    n = len(rows); m = lambda i: st.mean(r[i] for r in rows); se = lambda i: (st.stdev(r[i] for r in rows) / math.sqrt(n)) if n > 1 else float('nan')
    rate = (-m(1) / m(2)) if abs(m(2)) > 1e-9 else float('inf')
    out[tag] = dict(n=n, xMSE=round(m(1), 3), xMSE_se=round(se(1), 3), bill=round(m(2), 3), xnet=round(m(3), 3), xnet_se=round(se(3), 3),
                    lower=sum(r[3] < 0 for r in rows), rate=round(rate, 3), identical_final=sum(r[4] for r in rows),
                    to24=(n == 8 and m(3) <= -0.15))
    print(f"{tag:8s} n{n} xMSE {m(1):+7.3f} (se {se(1):.3f})  bill {m(2):+6.3f}  xnet {m(3):+7.3f} (se {se(3):.3f})  lower {out[tag]['lower']}/{n}  rate {rate:+.2f}  ident {out[tag]['identical_final']}  to24 {out[tag]['to24']}")
if out:
    p = G/'GROW_GRADE_latest.json'; p.write_text(json.dumps(out, indent=1))
