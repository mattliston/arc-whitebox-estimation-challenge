"""Paired screen grade against the exact 2b108 native-ten controls (seeds 7103248-57).
usage: grade_vs_2b108_opus_v1.py <screen outdir>"""
import glob, json, math, statistics as st, sys
from pathlib import Path
import numpy as np

O = Path(__file__).resolve().parents[3]
CTRL = O / 'lead/runs/astra_smirnov_final_native_ten_v1'
bill = lambda f: max(f / 2**41, .1)
rows = []
for p in sorted(Path(sys.argv[1]).glob('native_*.json')):
    a = json.loads(p.read_text())['rows'][0]; s = a['seed']
    c = next(Path(x) for x in glob.glob(str(CTRL / f'run_final_parity_s*_v1/native_{s}_ON.json')))
    b = json.loads(c.read_text()); assert b['sha256'].startswith('2b10843a'); b = b['rows'][0]
    ya = np.load(str(p).replace('.json', '_pred.npz'))[str(s)]; yb = np.load(str(c).replace('.json', '_pred.npz'))[str(s)]
    same = next((l for l in range(16) if ya[l].tobytes() != yb[l].tobytes()), 16)
    m, f = 100 * math.log(a['mse_final'] / b['mse_final']), 100 * math.log(bill(a['flops']) / bill(b['flops']))
    rows.append((s, m, f, m + f, same, a['ops'] - b['ops'] if 'ops' in b else None))
    print('%d  MSE %+7.3f%%  bill %+7.3f%%  net %+7.3f%%  identical_layers %2d  ops_delta %s' % rows[-1])
if len(rows) > 1:
    for i, name in ((1, 'MSE'), (2, 'bill'), (3, 'net')):
        v = [r[i] for r in rows]
        print('mean %-4s %+7.3f%%  se %.3f  lower %d/%d' % (name, st.mean(v), st.stdev(v) / len(v) ** .5, sum(x < 0 for x in v), len(v)))
