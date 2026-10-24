"""[opus-u64late] candidate (2b108+GB128r+late U64) vs GB128r controls, 24 seeds, noise-free cross-half + bill; pre-registered rule."""
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
F = O/'analyst/fresh'; L8 = O/'lead/runs/opus_u64late_v1'
util = lambda f: max(f / 2**41, .1)
def dial(p):
    d = json.loads(p.read_text())
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z['y'].astype(np.float64)
    return d['flops'], y
def screen(p):
    d = json.loads(p.read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z[str(d['seed'])].astype(np.float64)
    return d['flops'], y
rows = []
for s in list(range(7103080, 7103096)) + list(range(7103248, 7103256)):
    try:
        if s < 7103200: (fc, yc), (fb, yb) = dial(L8/f'res/u64l_{s}.json'), dial(O/f'lead/runs/opus_r256vgb_v1/res/gb128r_{s}.json')
        else: (fc, yc), (fb, yb) = screen(L8/f'screen_S8/native_{s}.json'), screen(O/f'lead/runs/opus_joinx_v1/screen_FRAG/native_{s}.json')
    except FileNotFoundError: continue
    with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
    U = lambda y: float(np.mean((y[-1] - A) * (y[-1] - B)))
    same = next((i for i in range(16) if yc[i].astype(np.float32).tobytes() != yb[i].astype(np.float32).tobytes()), 16)
    ul = 100 * math.log(U(yc) / U(yb)); bl = 100 * math.log(util(fc) / util(fb))
    rows.append(dict(seed=s, xMSE=round(ul, 3), bill=round(bl, 3), net=round(ul + bl, 3), first_diff_layer=same))
for r in rows: print('%d  xMSE %+7.3f%%  bill %+.3f%%  net %+7.3f%%  first_diff_layer %d' % (r['seed'], r['xMSE'], r['bill'], r['net'], r['first_diff_layer']))
def summ(rr):
    n = len(rr); v = lambda k: [r[k] for r in rr]
    return dict(n=n, xMSE=round(st.mean(v('xMSE')), 3), bill=round(st.mean(v('bill')), 3), net=round(st.mean(v('net')), 3),
                net_se=round(st.stdev(v('net')) / math.sqrt(n), 3) if n > 1 else None, lower=sum(x < 0 for x in v('net')))
all24 = summ(rows); last8 = summ([r for r in rows if r['seed'] >= 7103248]); a80 = summ([r for r in rows if r['seed'] < 7103200])
verdict = 'PASS' if (len(rows) == 24 and all24['net'] < 0 and last8['net'] < 0) else ('FAIL' if len(rows) == 24 else 'INCOMPLETE')
print('ALL24', all24); print('A80', a80); print('LAST8', last8); print('VERDICT', verdict)
if len(sys.argv) > 1:
    with open(sys.argv[1], 'x') as f: json.dump(dict(rows=rows, all24=all24, a80=a80, last8=last8, verdict=verdict), f, indent=1)
