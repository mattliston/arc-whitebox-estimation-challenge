"""[opus-r256vgb] noise-free cross-half (U = mean((y-A)(y-B)), final layer) + bill (util floor .1) for R256, GB128r and 2b108 on
16 A80 seeds (dial_judge_v2 runs; 2b108 = acc lane x2b108) and the earlier 8 seeds 7103248-55 (screen runs; 2b108 = native ten)."""
import glob, json, math, statistics as st, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
F = O/'analyst/fresh'; RV = O/'lead/runs/opus_r256vgb_v1/res'; ACC = O/'lead/runs/opus_acc_dials_v1/res'
util = lambda f: max(f / 2**41, .1)
def dial(p):
    d = json.loads(p.read_text())
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z['y'][-1].astype(np.float64)
    return d['flops'], y
def screen(p):
    d = json.loads(p.read_text())['rows'][0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y = z[str(d['seed'])][-1].astype(np.float64)
    return d['flops'], y
def U(y, s):
    with np.load(F/f'truth_seed{s}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
    return float(np.mean((y - A) * (y - B)))
rows = []
for s in range(7103080, 7103096):
    try: b, r, g = dial(ACC/f'x2b108_{s}.json'), dial(RV/f'r256_{s}.json'), dial(RV/f'gb128r_{s}.json')
    except FileNotFoundError: continue
    rows.append(dict(set='A80', seed=s, b=b, r=r, g=g))
SC = {'r': O/'lead/runs/opus_gbfix_v1/screen_R256', 'g': O/'lead/runs/opus_joinx_v1/screen_FRAG'}
CT = O/'lead/runs/astra_smirnov_final_native_ten_v1'
for s in range(7103248, 7103256):
    c = next(Path(x) for x in glob.glob(str(CT/f'run_final_parity_s*_v1/native_{s}_ON.json')))
    rows.append(dict(set='S8', seed=s, b=screen(c), r=screen(SC['r']/f'native_{s}.json'), g=screen(SC['g']/f'native_{s}.json')))
def pair(x, y, s):
    (fx, yx), (fy, yy) = x, y
    ul = 100 * math.log(U(yx, s) / U(yy, s)); bl = 100 * math.log(util(fx) / util(fy)); return ul, bl, ul + bl
out = {}
for name, (k1, k2) in {'R256_vs_2b108': ('r', 'b'), 'GB128r_vs_2b108': ('g', 'b'), 'GB128r_vs_R256': ('g', 'r')}.items():
    for subset in ('A80', 'S8', 'ALL24'):
        rr = [r for r in rows if subset == 'ALL24' or r['set'] == subset]
        vals = [pair(r[k1], r[k2], r['seed']) for r in rr]
        if not vals: continue
        n = len(vals); col = lambda i: [v[i] for v in vals]
        m = lambda i: st.mean(col(i)); se = lambda i: (st.stdev(col(i)) / math.sqrt(n)) if n > 1 else float('nan')
        out[f'{name}_{subset}'] = dict(n=n, xMSE=round(m(0), 3), xMSE_se=round(se(0), 3), bill=round(m(1), 3), xnet=round(m(2), 3), xnet_se=round(se(2), 3), xnet_lower=sum(v[2] < 0 for v in vals))
        print(f'{name:18s} {subset:5s} n={n:2d}  xMSE {m(0):+.3f}% (se {se(0):.3f})  bill {m(1):+.3f}%  xnet {m(2):+.3f}% (se {se(2):.3f}, lower {sum(v[2] < 0 for v in vals)}/{n})')
if len(sys.argv) > 1:
    with open(sys.argv[1], 'x') as f: json.dump(dict(rows=[dict(set=r['set'], seed=r['seed']) for r in rows], summary=out), f, indent=1)
