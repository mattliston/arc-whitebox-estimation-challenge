"""TAG opus_engineer a53 (2026-10-10): native against torch at the LEVEL, network by network, at several points of the S42 -> S45 line.
Final-layer MSE against the 688-sample truth (the same truth the trainer's 'mse' uses); native from the saved 16 x 1024 predictions of
fresh-process sessions on list a, torch from the trainer's judgements on the same networks (field 'mse').
For each point: level gap = native / torch - 1; for each later point: change gap = native change - torch change (both from S42).
usage: level_table.py   (points and files below)"""
import json
from pathlib import Path
import numpy as np
E = Path(__file__).resolve().parents[1]
O, P2 = E.parent, E.parents[1]
R = O/'trainer/run_M/results'
RANKS = (651, 378, 424, 328, 287, 641, 548, 557, 316, 390)
POINTS = [('S42', ('hc_s2_d6_r2_fresh_a', 's42_deepdraw_fresh_a'), 'a_0hS41_u2'), ('S43', ('s43mix_fresh_a',), 'a_0s43c_j'),
          ('S44', ('s44nq_fresh_a',), 'a_0mCJq'), ('S45', ('s45h_fresh_a',), 'a_0mHJ'), ('S46', ('s46s_fresh_a',), 'a_0mIJv')]
truth = {r: np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64) for r in RANKS}
nat, tor = {}, {}
for name, sessions, tag in POINTS:
    for r in RANKS:
        m = [float(np.mean((np.load(E/f'logs/quick_{s}_pred_{r}.npy').astype(np.float64)[15] - truth[r]) ** 2)) for s in sessions]
        nat[name, r] = float(np.mean(m))
        tor[name, r] = float(json.load(open(R/f'{tag}_r{r:04d}.json'))['mse'])
print('network  ' + '  '.join('%s level' % p[0] for p in POINTS) + '   ' + '  '.join('%s change gap' % p[0] for p in POINTS[1:]))
gaps = {p[0]: [] for p in POINTS}; cg = {p[0]: [] for p in POINTS[1:]}
for r in RANKS:
    line = '%7d ' % r
    for name, _s, _t in POINTS:
        g = 100 * (nat[name, r] / tor[name, r] - 1); gaps[name].append(g); line += '  %+8.2f%%' % g
    line += '   '
    for name, _s, _t in POINTS[1:]:
        c = 100 * (nat[name, r] / nat['S42', r] - tor[name, r] / tor['S42', r]); cg[name].append(c); line += '  %+8.2f%%' % c
    print(line)
se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
print('   mean ' + ''.join('  %+8.2f%%' % np.mean(gaps[p[0]]) for p in POINTS) + '   ' + ''.join('  %+8.2f%%' % np.mean(cg[p[0]]) for p in POINTS[1:]))
print('     se ' + ''.join('  %8.2f%% ' % se(gaps[p[0]]) for p in POINTS) + '   ' + ''.join('  %8.2f%% ' % se(cg[p[0]]) for p in POINTS[1:]))
pooled = lambda d, name: 100 * (sum(d[name, r] for r in RANKS) / sum(tor[name, r] for r in RANKS) - 1)
print(' pooled ' + ''.join('  %+8.2f%%' % pooled(nat, p[0]) for p in POINTS))
