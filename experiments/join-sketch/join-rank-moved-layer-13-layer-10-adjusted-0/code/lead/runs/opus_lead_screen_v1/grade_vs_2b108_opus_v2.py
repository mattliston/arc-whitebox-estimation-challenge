"""v2: paired screen grade against the exact 2b108 native-ten controls (7103248-57) or any control dir with native_<seed>.json,
reporting the plain MSE change AND the truth-noise-free cross-half change U = mean((y-A)(y-B)) on the final layer.
usage: grade_vs_2b108_opus_v2.py <screen outdir> [control outdir]"""
import glob, json, math, statistics as st, sys
from pathlib import Path
import numpy as np

O = Path(__file__).resolve().parents[3]
CTRL = O / 'lead/runs/astra_smirnov_final_native_ten_v1'
bill = lambda f: max(f / 2**41, .1)


def control(s):
    if len(sys.argv) > 2:
        return Path(sys.argv[2]) / f'native_{s}.json'
    return next(Path(x) for x in glob.glob(str(CTRL / f'run_final_parity_s*_v1/native_{s}_ON.json')))


rows = []
for p in sorted(Path(sys.argv[1]).glob('native_*.json')):
    a = json.loads(p.read_text())['rows'][0]; s = a['seed']
    c = control(s)
    if not c.exists():
        continue
    b = json.loads(c.read_text())['rows'][0]
    ya = np.load(str(p).replace('.json', '_pred.npz'))[str(s)].astype(np.float64)
    yb = np.load(str(c).replace('.json', '_pred.npz'))[str(s)].astype(np.float64)
    t = np.load(O / f'analyst/fresh/truth_seed{s}.npz')
    A, B = t['mc_post_A'][-1], t['mc_post_B'][-1]
    ua, ub = np.mean((ya[-1] - A) * (ya[-1] - B)), np.mean((yb[-1] - A) * (yb[-1] - B))
    same = next((l for l in range(16) if ya[l].astype(np.float32).tobytes() != yb[l].astype(np.float32).tobytes()), 16)
    m, u, f = 100 * math.log(a['mse_final'] / b['mse_final']), 100 * math.log(ua / ub), 100 * math.log(bill(a['flops']) / bill(b['flops']))
    rows.append((s, m, u, f, u + f, same, a.get('r320_resp', {}).get('10') if isinstance(a.get('r320_resp'), dict) else None))
    print('%d  MSE %+7.3f%%  xhalfMSE %+7.3f%%  bill %+7.3f%%  xnet %+7.3f%%  same_layers %2d  resp10 %s' % rows[-1])
if len(rows) > 1:
    for i, name in ((1, 'MSE'), (2, 'xMSE'), (3, 'bill'), (4, 'xnet')):
        v = [r[i] for r in rows]
        print('mean %-5s %+7.3f%%  se %.3f  lower %d/%d' % (name, st.mean(v), st.stdev(v) / len(v) ** .5, sum(x < 0 for x in v), len(v)))
