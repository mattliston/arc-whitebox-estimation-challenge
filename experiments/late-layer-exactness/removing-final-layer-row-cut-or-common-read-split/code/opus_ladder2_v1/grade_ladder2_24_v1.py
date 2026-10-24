"""[opus-ladder2] 8-24 seed grade (7103248-55 vs ctl_s2tr, 7103080-095 vs ctl_s2tr_a80, screen_judge_opus_v1) of ladder-2 arms vs the shared S2tr control (opus_grow_v1/ctl_s2tr):
noise-free cross-half U log + bill log (= grade_grow_v1 metric), plus per-layer secondary xMSE at L8/L11/L14. usage: grade_nt_v1.py dir [dir ...]"""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
F = O/'analyst/fresh'; CTL = O/'lead/runs/opus_grow_v1/ctl_s2tr'; CTL80 = O/'lead/runs/opus_retune_v1/ctl_s2tr_a80'; D = Path(__file__).resolve().parent
util = lambda f: max(f / 2**41, .1)


def screen(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z[str(d['seed'])].astype(np.float64)
    return d['flops'], y


out = {}
for tag in sys.argv[1:]:
    rows = []
    for s in list(range(7103248, 7103256)) + list(range(7103080, 7103096)):
        try: (fc, yc), (fb, yb) = screen(D/tag/f'native_{s}.json'), screen((CTL if s >= 7103248 else CTL80)/f'native_{s}.json')
        except FileNotFoundError: continue
        with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'].astype(np.float64), z['mc_post_B'].astype(np.float64)
        U = lambda y, l=-1: float(np.mean((y[l] - A[l]) * (y[l] - B[l])))
        ul, bl = 100 * math.log(U(yc) / U(yb)), 100 * math.log(util(fc) / util(fb))
        lay = [100 * math.log(U(yc, l) / U(yb, l)) for l in (7, 10, 13)]
        rows.append((s, ul, bl, ul + bl, *lay))
        print(f'  {tag} {s} xMSE {ul:+.3f} bill {bl:+.3f} xnet {ul + bl:+.3f}  L8 {lay[0]:+.2f} L11 {lay[1]:+.2f} L14 {lay[2]:+.2f}')
    if not rows: continue
    n = len(rows); m = lambda i: st.mean(r[i] for r in rows); se = lambda i: (st.stdev(r[i] for r in rows) / math.sqrt(n)) if n > 1 else float('nan')
    out[tag] = dict(n=n, xMSE=round(m(1), 3), xMSE_se=round(se(1), 3), bill=round(m(2), 3), xnet=round(m(3), 3), xnet_se=round(se(3), 3),
                    lower=sum(r[1] < 0 for r in rows), L8=round(m(4), 3), L11=round(m(5), 3), L14=round(m(6), 3))
    print(f"{tag:12s} n{n} xMSE {m(1):+7.3f} (se {se(1):.3f})  bill {m(2):+6.3f}  xnet {m(3):+7.3f} (se {se(3):.3f})  lower {out[tag]['lower']}/{n}  L8 {m(4):+.2f} L11 {m(5):+.2f} L14 {m(6):+.2f}")
if out:
    (D/'LADDER2_GRADE24_latest.json').write_text(json.dumps(out, indent=1))
