"""[opus-storeval] store-rank grade vs S2tr (res_ctl_<r>.json): per arm xMSE = 100 log(U/U_ctl), bill = 100 log(F/F_ctl) (cold), xnet; plus pairs."""
import json, math, statistics as st, sys
from pathlib import Path
K = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_kinkoracle_v1')
ld = lambda p: json.loads(p.read_text())
ctl = {int(p.stem.split('_')[-1]): ld(p) for p in K.glob('res_ctl_*.json')}
def arm(tag): return {int(p.stem.split('_')[-1]): ld(p) for p in K.glob(f'res_d_{tag}_*.json')}
def summ(name, a, b):
    rs = sorted(set(a) & set(b))
    if len(rs) < 2: return
    x = [100 * math.log(a[r]['U'] / b[r]['U']) for r in rs]; f = [100 * math.log(a[r]['flops'] / b[r]['flops']) for r in rs]; t = [p + q for p, q in zip(x, f)]
    se = lambda v: st.stdev(v) / math.sqrt(len(v))
    print(f'{name:22s} n {len(rs):2d} xMSE {st.mean(x):+.3f} (se {se(x):.3f}) bill {st.mean(f):+.3f} xnet {st.mean(t):+.3f} (se {se(t):.3f}) lower {sum(v < 0 for v in t)}/{len(t)}')
for tag in (sys.argv[1:] or ('jord', 'tankb10', 'dc35', 'stkdcv2rc30', 'stk4')):
    summ(f'{tag} vs S2tr', arm(tag), ctl)
if arm('tankb10') and arm('jord'): summ('tankb10 vs jord', arm('tankb10'), arm('jord'))
