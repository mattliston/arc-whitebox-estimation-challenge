"""radlab errbudget_v1 (10-12, lead 21:5xZ): ERROR BUDGET AFTER gk, from per-layer MSEs only (read-only; numpy, single-threaded).
Inputs: the trainer's a_RS51_base evals on A41 (S50 point) = 'base'; scout_card run_k1 / run_k4 / run_k5 arms (gk, gk15, gk05, gkpB, gk+a4k31s6,
gk+gb75, gk+bsk10, gk+xh); truth noise per layer from analyst/fresh (sum_units mc_post_var / N / 1024 is subtracted from every layer MSE).
BORN vs CARRIED, per layer l: M_l = c_l M_{l-1} + B_l (noise-free MSE; carried = the previous layer's mean error pushed through the TRUE mean
map, born = everything new at layer l, including errors that reach the mean through the covariance). c_l from radlab/carry_v1.py (true
networks: c_rand = isotropic error, ~0.85 at L10-15; a mean-aligned error carries ~1.0): two bounds, c = c_rand and c = 1. The final-layer
share of what layer l bears = B_l x prod_{k>l} c_k / M_15 (the cross term between born and carried error is ignored).
An arm's change by layer: d_l = M_l(arm) - M_l(ref) on the same networks; born part of the change dB_l = d_l - c_l d_{l-1}, its share of the
final change dB_l x prod c / M_15(ref) (sums to d_15 / M_15(ref)).
usage: python -B radlab/errbudget_v1.py"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

O = Path(__file__).resolve().parents[1]
BASE = O / 'trainer' / 'run_RAD_S51_gen' / 'results'
RUNS = [O / 'scout_card' / r / 'results' for r in ('run_k1', 'run_k4', 'run_k5')]
CR = np.load(O / 'radlab' / 'cache' / 'carry_c_rand_v1.npy')
arms = defaultdict(dict)
for rd in RUNS:
    for f in sorted(rd.glob('sc_*.json')):
        d = json.loads(f.read_text()); seed, arm = d['id'].split('_r')[1].split('_', 1)
        arms[arm].setdefault(int(seed), np.array(d['mse_layers'], float))
noise = {}


def nz(s):
    if s not in noise:
        with np.load(O / 'analyst' / 'fresh' / ('truth_seed%d.npz' % s)) as z:
            noise[s] = z['mc_post_var'].sum(1) / float(z['N']) / 1024.0
    return noise[s]


for s in range(7100584, 7100648):
    p = BASE / ('a_RS51_base_r%d.json' % s)
    if p.exists():
        arms['base'][s] = np.array(json.loads(p.read_text())['mse_layers'], float)
M = {a: {s: v - nz(s) for s, v in d.items()} for a, d in arms.items()}      # noise-free per-layer MSE
C1 = np.ones(16)


def tail(c, l):
    return float(np.prod(c[l + 1:16]))


def budget(ref, nets, label):
    X = np.mean([M[ref][s] for s in nets], 0)
    print('\n%s: %d networks; noise-free MSE by layer (1e-9) and what each layer bears of the final error' % (label, len(nets)))
    print(' L   MSE    growth | c_rand  born  final-share | c=1 born  final-share')
    for l in range(1, 16):
        g = X[l] - X[l - 1]
        b1 = X[l] - CR[l] * X[l - 1]; s1 = b1 * tail(CR, l) / X[15]
        print('%2d %7.3f %+7.3f | %.3f %+7.3f  %5.1f%% | %+7.3f  %5.1f%%' % (l, 1e9 * X[l], 1e9 * g, CR[l], 1e9 * b1, 100 * s1, 1e9 * g, 100 * g / X[15]))
    for name, c in (('c_rand', CR), ('c=1', C1)):
        sh = [((X[l] - c[l] * X[l - 1]) * tail(c, l)) / X[15] for l in range(1, 16)] + [X[0] * tail(c, 0) / X[15]]
        late, mid, early = sum(sh[11:15]), sum(sh[7:11]), sum(sh[0:7]) + sh[15]
        print('  %-6s: born at L12-15 %.0f%% (L15 alone %.0f%%), L8-11 %.0f%%, L0-7 %.0f%%' % (name, 100 * late, 100 * sh[14], 100 * mid, 100 * early))
    return X


def change(arm, ref, label):
    nets = sorted(set(M[arm]) & set(M[ref]))
    if len(nets) < 2:
        return
    D = np.array([M[arm][s] - M[ref][s] for s in nets]); R = np.array([M[ref][s] for s in nets])
    rel = D / R; d, r = D.mean(0), R.mean(0)
    se = rel.std(0, ddof=1) / np.sqrt(len(nets))
    print('\n%s (%d networks): pooled change by layer, %% (se); born part of the change and its share of the final change (c_rand / c=1)' % (label, len(nets)))
    print('   L: ' + ' '.join('%6d' % l for l in range(5, 16)))
    print('   %%: ' + ' '.join('%+6.2f' % (100 * d[l] / r[l]) for l in range(5, 16)))
    print('  se: ' + ' '.join('%6.2f' % (100 * se[l]) for l in range(5, 16)))
    for name, c in (('c_rand', CR), ('c=1', C1)):
        sh = [(d[l] - c[l] * d[l - 1]) * tail(c, l) / r[15] for l in range(1, 16)]
        print('  born %-6s ' % name + ' '.join('%+6.3f' % (100 * x) for x in sh[4:15]) + '   (sum %+.3f%% = final %+.3f%%; L12-15 %+.3f, L8-11 %+.3f, L1-7 %+.3f)' % (
            100 * sum(sh), 100 * d[15] / r[15], 100 * sum(sh[11:15]), 100 * sum(sh[7:11]), 100 * sum(sh[0:7])))


gk_nets = sorted(set(M['gk']) & set(M['base']))
budget('base', gk_nets, 'BASE (S50 point) on the gk networks')
budget('gk', gk_nets, 'AFTER gk')
change('gk', 'base', 'gk vs base')
for a in ('gk15', 'gk05', 'gk+a4k31s6', 'gk+gb75', 'gk+bsk10', 'gkpB', 'gk+xh'):
    if a in M:
        change(a, 'gk', '%s vs gk' % a)
