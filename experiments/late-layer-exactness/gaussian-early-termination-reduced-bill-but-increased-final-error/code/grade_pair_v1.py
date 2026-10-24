"""[opus-lead] grade a candidate boundary dir against a control pred dir on the fixed ranks, Astra's formula: U = mean((p15-A15)*(p15-B15)) over
the 1024 units with the two 32M truth halves; raw = U ratio, bill = flops ratio, adjusted = product. Truth paths from PILOT_CONTROLS_v1 (35/100)
and PREREG_EXTENSION6_v1 (150..287). usage: grade_pair_v1.py <candidate boundary dir> <control pred dir (rows.jsonl + pred_*.bin)>"""
import json, sys, math, statistics as st
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
cand, ctrl = Path(sys.argv[1]), Path(sys.argv[2])
truths = {}
for m in (O/'lead/runs/astra_crunch_compute_v1/global_radius_checks_v1/PILOT_CONTROLS_v1.json', O/'lead/runs/astra_crunch_compute_v1/small_stack_v25a_v1/PREREG_EXTENSION6_v1.json'):
    for r in json.loads(m.read_text())['rows']:
        t = r['truths']; t = eval(t) if isinstance(t, str) else t
        truths[r['rank']] = [(q['path'], q['key']) for q in t]
rows = lambda p: {r['rank']: r for r in map(json.loads, (p/'rows.jsonl').read_text().splitlines())}
a, b = rows(cand), rows(ctrl)
out = []
for rank in sorted(set(a) & set(b) & set(truths)):
    p1 = np.fromfile(cand/f'pred_{rank}.bin', dtype='<f4').reshape(16, 1024).astype('d'); p0 = np.fromfile(ctrl/f'pred_{rank}.bin', dtype='<f4').reshape(16, 1024).astype('d')
    A, B = [np.load(path)[key].astype('d') for path, key in truths[rank]]
    u1 = float(np.mean((p1[15] - A[15]) * (p1[15] - B[15]))); u0 = float(np.mean((p0[15] - A[15]) * (p0[15] - B[15])))
    fr = a[rank]['flops'] / b[rank]['flops']; dy = float(np.max(np.abs(p1 - p0)))
    out.append(dict(rank=rank, raw_pct=100 * (u1 / u0 - 1), bill_pct=100 * (fr - 1), adjusted_pct=100 * (u1 / u0 * fr - 1), requests=a[rank]['requests'], control_requests=b[rank]['requests'], max_abs_dy=dy, residual=a[rank]['residual_s'], control_residual=b[rank]['residual_s']))
    print('rank %4d: raw %+.3f%% bill %+.3f%% adjusted %+.3f%% | requests %d -> %d (%+d) | max|dy| %.2e | residual %.4f vs %.4f' % (rank, out[-1]['raw_pct'], out[-1]['bill_pct'], out[-1]['adjusted_pct'], b[rank]['requests'], a[rank]['requests'], a[rank]['requests'] - b[rank]['requests'], dy, a[rank]['residual_s'], b[rank]['residual_s']))
if out:
    print('MEAN of %d: raw %+.3f%% bill %+.3f%% adjusted %+.3f%% (se %.3f) | requests %+.0f (%+.1f%%)' % (len(out), st.mean(r['raw_pct'] for r in out), st.mean(r['bill_pct'] for r in out), st.mean(r['adjusted_pct'] for r in out), (st.stdev(r['adjusted_pct'] for r in out) / math.sqrt(len(out))) if len(out) > 1 else 0, st.mean(r['requests'] - r['control_requests'] for r in out), 100 * st.mean(r['requests'] / r['control_requests'] - 1 for r in out)))
json.dump(out, open(cand/'GRADE_PAIR_v1.json', 'w'), indent=1)
