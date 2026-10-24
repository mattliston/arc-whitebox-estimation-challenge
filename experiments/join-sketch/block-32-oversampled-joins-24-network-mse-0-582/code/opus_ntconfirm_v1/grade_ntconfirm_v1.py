"""[opus-ntconfirm] 24-seed noise-free grade: S8 7103248-55 (candidate dir vs opus_grow_v1/ctl_s2tr) + A80 7103080-095 (candidate dir vs
opus_retune_v1/ctl_s2tr_a80). xMSE = per-seed cross-half U log ratio, bill = FLOPs log ratio, xnet = sum; aggregate = log(sum U) + log(sum F).
usage: grade_ntconfirm_v1.py <S8 candidate screen dir> <A80 candidate screen dir> [out.json]"""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
S8, A80 = Path(sys.argv[1]), Path(sys.argv[2])
SETS = (('S8', range(7103248, 7103256), S8, O/'lead/runs/opus_grow_v1/ctl_s2tr'), ('A80', range(7103080, 7103096), A80, O/'lead/runs/opus_retune_v1/ctl_s2tr_a80'))
def scr(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: return d['flops'], z[str(d['seed'])].astype(np.float64)
rows = []
for name, seeds, cd, bd in SETS:
    for s in seeds:
        try: (fc, yc), (fb, yb) = scr(cd/f'native_{s}.json'), scr(bd/f'native_{s}.json')
        except FileNotFoundError: continue
        with np.load(O/f'analyst/fresh/truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
        Uc, Ub = (float(np.mean((y[-1] - A) * (y[-1] - B))) for y in (yc, yb))
        rows.append(dict(set=name, seed=s, x=100 * math.log(Uc / Ub), b=100 * math.log(fc / fb), Uc=Uc, Ub=Ub, fc=fc, fb=fb))
def summ(rr):
    n = len(rr); x = [r['x'] for r in rr]; b = [r['b'] for r in rr]; t = [a + c for a, c in zip(x, b)]
    agg = 100 * math.log(sum(r['Uc'] for r in rr) / sum(r['Ub'] for r in rr)) + 100 * math.log(sum(r['fc'] for r in rr) / sum(r['fb'] for r in rr))
    se = lambda v: st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float('nan')
    return dict(n=n, xMSE=round(st.mean(x), 3), xMSE_se=round(se(x), 3), bill=round(st.mean(b), 3), xnet=round(st.mean(t), 3), xnet_se=round(se(t), 3),
                aggregate=round(agg, 3), lower=sum(v < 0 for v in t))
for r in rows: print(f"{r['set']:3s} {r['seed']}  xMSE {r['x']:+7.3f}  bill {r['b']:+6.3f}  xnet {r['x'] + r['b']:+7.3f}")
res = {k: summ([r for r in rows if r['set'] == k]) for k in ('S8', 'A80') if any(r['set'] == k for r in rows)}
if len(rows) > 1: res['ALL'] = summ(rows)
for k, v in res.items(): print(k, v)
if len(sys.argv) > 3: Path(sys.argv[3]).write_text(json.dumps(dict(rows=rows, summary=res), indent=1))
