"""[opus-lead] grade a candidate boundary dir against a control boundary dir on every common rank that has a 2 x 32M store truth
(analyst/fresh/pu/mps_truth_rank<r>_{A,B}32M.npz, key mc_post): U = mean((p15-A15)*(p15-B15)) (noise-free cross-half, final layer),
raw = U ratio, bill = flops ratio, adjusted = product; also the per-layer U ratio (secondary; decisions are final-layer only).
usage: grade_store_v1.py <candidate boundary dir> <control boundary dir> [--layers]"""
import json, sys, math, statistics as st
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
PU = O / 'analyst/fresh/pu'
cand, ctrl = Path(sys.argv[1]), Path(sys.argv[2])
layers = '--layers' in sys.argv
rows = lambda p: {r['rank']: r for r in map(json.loads, (p / 'rows.jsonl').read_text().splitlines()) if 'rank' in r}
a, b = rows(cand), rows(ctrl)
out = []
for rank in sorted(set(a) & set(b)):
    ta, tb = PU / f'mps_truth_rank{rank:04d}_A32M.npz', PU / f'mps_truth_rank{rank:04d}_B32M.npz'
    if not (ta.exists() and tb.exists() and (cand / f'pred_{rank}.bin').exists() and (ctrl / f'pred_{rank}.bin').exists()):
        continue
    p1 = np.fromfile(cand / f'pred_{rank}.bin', dtype='<f4').reshape(16, 1024).astype('d')
    p0 = np.fromfile(ctrl / f'pred_{rank}.bin', dtype='<f4').reshape(16, 1024).astype('d')
    A, B = np.load(ta)['mc_post'].astype('d'), np.load(tb)['mc_post'].astype('d')
    u1 = np.mean((p1 - A) * (p1 - B), axis=1); u0 = np.mean((p0 - A) * (p0 - B), axis=1)
    fr = a[rank]['flops'] / b[rank]['flops']
    rec = dict(rank=rank, raw_pct=100 * (u1[15] / u0[15] - 1), bill_pct=100 * (fr - 1), adjusted_pct=100 * (u1[15] / u0[15] * fr - 1),
               u1=float(u1[15]), u0=float(u0[15]), requests=a[rank]['requests'], control_requests=b[rank]['requests'],
               max_abs_dy=float(np.max(np.abs(p1 - p0))), layer_raw_pct=[100 * (float(x) / float(y) - 1) for x, y in zip(u1, u0)])
    out.append(rec)
    line = 'rank %4d: raw %+.3f%% bill %+.3f%% adjusted %+.3f%% | U %.4e -> %.4e | req %d -> %d | max|dy| %.2e' % (
        rank, rec['raw_pct'], rec['bill_pct'], rec['adjusted_pct'], rec['u0'], rec['u1'], rec['control_requests'], rec['requests'], rec['max_abs_dy'])
    if layers:
        line += ' | L8-14 ' + ' '.join('%+.1f' % x for x in rec['layer_raw_pct'][8:15])
    print(line)
if out:
    n = len(out)
    raw = [r['raw_pct'] for r in out]; adj = [r['adjusted_pct'] for r in out]; bill = [r['bill_pct'] for r in out]
    pooled = 100 * (sum(r['u1'] for r in out) / sum(r['u0'] for r in out) - 1)
    print('MEAN of %d: raw %+.3f%% (se %.3f, lower on %d/%d, pooled %+.3f%%) bill %+.3f%% adjusted %+.3f%% (se %.3f)' % (
        n, st.mean(raw), (st.stdev(raw) / math.sqrt(n)) if n > 1 else 0.0, sum(x < 0 for x in raw), n, pooled, st.mean(bill), st.mean(adj), (st.stdev(adj) / math.sqrt(n)) if n > 1 else 0.0))
json.dump(out, open(cand / 'GRADE_STORE_v1.json', 'w'), indent=1)
