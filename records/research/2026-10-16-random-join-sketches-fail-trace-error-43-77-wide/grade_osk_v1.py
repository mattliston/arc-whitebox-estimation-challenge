"""[opus-sketchgb] candidate arm (tag) vs GB128r controls (primary) and vs 2b108 controls (secondary), 24 seeds, noise-free cross-half U
+ bill. usage: grade_osk_v1.py <tag> [out.json]. A80 7103080-95: res/<tag>_<s>.json (dial_judge_v2); S8 7103248-55: screen_<tag>/native_<s>.json."""
import json, math, glob, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
F = O/'analyst/fresh'; SK = O/'lead/runs/opus_sketchgb_v1'; tag = sys.argv[1]
util = lambda f: max(f / 2**41, .1)
def dial(p):
    d = json.loads(Path(p).read_text())
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z['y'].astype(np.float64)
    return d['flops'], y
def screen(p):
    d = json.loads(Path(p).read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z[str(d['seed'])].astype(np.float64)
    return d['flops'], y
rows = []
for s in list(range(7103080, 7103096)) + list(range(7103248, 7103256)):
    try:
        if s < 7103200:
            c, g, b = dial(SK/f'res/{tag}_{s}.json'), dial(O/f'lead/runs/opus_r256vgb_v1/res/gb128r_{s}.json'), dial(O/f'lead/runs/opus_acc_dials_v1/res/x2b108_{s}.json')
        else:
            c, g = screen(SK/f'screen_{tag}/native_{s}.json'), screen(O/f'lead/runs/opus_joinx_v1/screen_FRAG/native_{s}.json')
            b = screen(next(iter(glob.glob(str(O/f'lead/runs/astra_smirnov_final_native_ten_v1/run_final_parity_s*_v1/native_{s}_ON.json')))))
    except (FileNotFoundError, StopIteration): continue
    with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
    U = lambda y: float(np.mean((y[-1] - A) * (y[-1] - B)))
    vs = lambda x, y: (100 * math.log(U(x[1]) / U(y[1])), 100 * math.log(util(x[0]) / util(y[0])))
    (ug, bg), (ub, bb) = vs(c, g), vs(c, b)
    rows.append(dict(seed=s, xMSE_g=round(ug, 3), bill_g=round(bg, 3), net_g=round(ug + bg, 3), xMSE_b=round(ub, 3), bill_b=round(bb, 3), net_b=round(ub + bb, 3)))
for r in rows: print('%d  vsGB128r xMSE %+7.3f bill %+.3f net %+7.3f | vs2b108 xMSE %+7.3f bill %+.3f net %+7.3f' % tuple(r[k] for k in ('seed', 'xMSE_g', 'bill_g', 'net_g', 'xMSE_b', 'bill_b', 'net_b')))
def summ(rr):
    n = len(rr); out = dict(n=n)
    for k in ('xMSE_g', 'bill_g', 'net_g', 'xMSE_b', 'bill_b', 'net_b'):
        v = [r[k] for r in rr]; out[k] = round(st.mean(v), 3); out[k + '_se'] = round(st.stdev(v) / math.sqrt(n), 3) if n > 1 else None
    out['lower_g'] = sum(r['net_g'] < 0 for r in rr); return out
all24 = summ(rows) if rows else {}; a80 = summ([r for r in rows if r['seed'] < 7103200]) if rows else {}; last8 = summ([r for r in rows if r['seed'] >= 7103248]) if any(r['seed'] >= 7103248 for r in rows) else {}
verdict = ('PASS' if all24['net_g'] <= 0 else 'FAIL') if len(rows) == 24 else 'INCOMPLETE'
print('ALL24', all24); print('A80', a80); print('LAST8', last8); print('VERDICT (pre-registered: ALL24 mean net vs GB128r <= 0)', verdict)
if len(sys.argv) > 2:
    with open(sys.argv[2], 'x') as f: json.dump(dict(tag=tag, rows=rows, all24=all24, a80=a80, last8=last8, verdict=verdict), f, indent=1)
