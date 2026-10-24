#!/usr/bin/env python3
"""TAG opus-physicist T148 table: pe3 (the pair map's local skew term e3 x (1 + pe3)) at L8-11 on top of C vs T141's C, paired per network: (p) -0.2,
(q) -0.1; the control (C under worker_generated_v16) vs T141's C. Pre-declared 17:14:11Z (TEAM_LOG): every arm with an 8-network (7100584-591) mean log
vs C <= -0.15% extends to 7100592-607; pass at 24 = all-24 mean log vs C <= -0.15% and the out-of-selection 16 agree in sign; between passing arms the
lower all-24 mean log goes to s56k1's 'extra' line.
usage: t148_table.py               the table
       t148_table.py decide        writes t148_extend.txt = the arms to extend (letters, empty = none)"""
import json
import math
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; RUN = HERE / 't148_pe3'
SCREEN, OOS = list(range(7100584, 7100592)), list(range(7100592, 7100608))
NAME = dict(p='pe3 -0.2 at L8-11 (pair-map local skew x0.8)', q='pe3 -0.1 at L8-11 (pair-map local skew x0.9)')
load = lambda f: json.loads(f.read_text())['mse_layers']
C = {int(f.stem.split('_r')[1].split('_')[0]): load(f) for f in (HERE / 't141_k4l_ext' / 'results').glob('ph141_r*_c.json')}
X = {f.stem: load(f) for f in (RUN / 'results').glob('ph148*.json')}
arm = lambda a: {int(k.split('_r')[1]): v for k, v in X.items() if k.startswith(('ph148%s_r' % a, 'ph148x%s_r' % a))}


def stats(a, seeds):
    A = arm(a); s = [x for x in seeds if x in A]
    if len(s) < 2:
        return None
    x = np.array([A[q][-1] for q in s]); c = np.array([C[q][-1] for q in s]); lr = np.log(x / c)
    lx = np.array([A[q] for q in s]); lc = np.array([C[q] for q in s])
    return dict(s=s, mean=lr.mean(), text=(
        '(%s) %s vs C on %d (%d-%03d): final %+.3f%% pooled, mean log %+.3f%% (se %.3f), lower on %d of %d [noise-free x1.36 %+.3f%%]; L8-15 %s; per network %s'
        % (a, NAME[a], len(s), s[0], s[-1] % 1000, 100 * (x.mean() / c.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(s)),
           int((x < c).sum()), len(s), 136 * lr.mean(), ' '.join('%+.2f' % (100 * (lx[:, l].mean() / lc[:, l].mean() - 1)) for l in range(8, 16)),
           ' '.join('%+.2f' % (100 * v) for v in lr))))


z = X.get('ph148_ctl_r7100584')
if z:
    print('control (C under worker_generated_v16) vs T141 C on 7100584: %s (worst rel %.2e)' % ('BIT FOR BIT' if z == C[7100584] else 'DIFFERS',
                                                                                     max(abs(a / b - 1) for a, b in zip(z, C[7100584]))))
if len(sys.argv) > 1 and sys.argv[1] == 'decide':
    go = ''
    for a in 'pq':
        st = stats(a, SCREEN)
        print(st['text'] + (' -> extends to 7100592-607' if st['mean'] <= -0.0015 else ' -> no extension (bar -0.15%)'))
        go += a if st['mean'] <= -0.0015 else ''
    (HERE / 't148_extend.txt').write_text(go)
    sys.exit(0)
for a in 'pq':
    for seeds in (SCREEN, OOS, SCREEN + OOS):
        st = stats(a, seeds)
        if st and (seeds is SCREEN or len(st['s']) > len(SCREEN)):
            print(st['text'])
    full, oos = stats(a, SCREEN + OOS), stats(a, OOS)
    if full and oos and len(full['s']) == 24:
        print('(%s) pass at 24: %s (all-24 mean log %+.3f%%, out-of-selection 16 %+.3f%%)%s'
              % (a, 'YES' if full['mean'] <= -0.0015 and oos['mean'] < 0 else 'no', 100 * full['mean'], 100 * oos['mean'], ''))
P, Q = arm('p'), arm('q')
both = [s for s in SCREEN + OOS if s in P and s in Q]
if len(both) >= 4:
    d = np.array([math.log(P[s][-1] / Q[s][-1]) for s in both])
    print('step: (p) -0.2 vs (q) -0.1 on %d paired: mean log %+.3f%% (se %.3f), (p) lower on %d' % (len(both), 100 * d.mean(), 100 * d.std(ddof=1) / math.sqrt(len(d)), int((d < 0).sum())))
