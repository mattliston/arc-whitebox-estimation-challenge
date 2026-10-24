"""[opus-late] grade late-only variants vs shared S2tr controls, noise-free cross-half U. S8 7103248-55: screen_<tag> vs opus_grow_v1/ctl_s2tr;
A80 7103080-095: screenA80_<tag> vs opus_retune_v1/ctl_s2tr_a80. xnet = mean per-seed (U log + bill log); aggregate = log(sum U ratio) + log(sum
FLOPs ratio) (public-score form). rate = -dxMSE/dbill. Bar: 24-seed xnet <= -0.3% and aggregate < 0. usage: grade_late_v1.py tag [tag...]"""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
F = O/'analyst/fresh'; L = O/'lead/runs/opus_late_v1'
CT = {'S8': (range(7103248, 7103256), 'screen_{t}', O/'lead/runs/opus_grow_v1/ctl_s2tr'), 'A80': (range(7103080, 7103096), 'screenA80_{t}', O/'lead/runs/opus_retune_v1/ctl_s2tr_a80')}
util = lambda f: max(f / 2**41, .1)
def screen(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z[str(d['seed'])].astype(np.float64)
    return d['flops'], y
out = {}
for tag in sys.argv[1:]:
    rows = []
    for setname, (seeds, cdir, bdir) in CT.items():
        for s in seeds:
            try: (fc, yc), (fb, yb) = screen(L/cdir.format(t=tag)/f'native_{s}.json'), screen(bdir/f'native_{s}.json')
            except FileNotFoundError: continue
            with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
            Uc, Ub = (float(np.mean((y[-1] - A) * (y[-1] - B))) for y in (yc, yb))
            rows.append(dict(set=setname, seed=s, Uc=Uc, Ub=Ub, fc=fc, fb=fb, x=100 * math.log(Uc / Ub), b=100 * math.log(util(fc) / util(fb))))
    if not rows: continue
    def summ(rr):
        n = len(rr); x = [r['x'] for r in rr]; b = [r['b'] for r in rr]; net = [r['x'] + r['b'] for r in rr]
        agg = 100 * math.log(sum(r['Uc'] for r in rr) / sum(r['Ub'] for r in rr)) + 100 * math.log(sum(r['fc'] for r in rr) / sum(r['fb'] for r in rr))
        return dict(n=n, xMSE=round(st.mean(x), 3), xMSE_se=round(st.stdev(x) / math.sqrt(n), 3) if n > 1 else None, bill=round(st.mean(b), 3),
                    xnet=round(st.mean(net), 3), xnet_se=round(st.stdev(net) / math.sqrt(n), 3) if n > 1 else None, aggregate=round(agg, 3),
                    lower=sum(v < 0 for v in net), rate=round(-st.mean(x) / st.mean(b), 2) if abs(st.mean(b)) > 1e-6 else None)
    res = {k: summ([r for r in rows if r['set'] == k]) for k in CT if any(r['set'] == k for r in rows)}
    if len(res) == 2: res['ALL'] = summ(rows)
    out[tag] = res
    for k, v in res.items(): print(f"{tag:8s} {k:4s} n{v['n']:<3d} xMSE {v['xMSE']:+7.3f} (se {v['xMSE_se']})  bill {v['bill']:+6.3f}  xnet {v['xnet']:+7.3f} (se {v['xnet_se']})  agg {v['aggregate']:+7.3f}  lower {v['lower']}/{v['n']}  rate {v['rate']}")
(L/'LATE_GRADE_latest.json').write_text(json.dumps(out, indent=1))
