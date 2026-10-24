#!/usr/bin/env python3
"""TAG opus-physicist T147 table: the lead's bigger bets (h) K31L4 age 4 at 0.6, (i) age 4 at 0.6 with age 3 lowered by 0.6, (j) law dfit 4 (C^2 four-form),
(k) law dfit 3, each on top of C vs T141's C, paired per network; the control (age-4 instance at 0) vs T141's C.
Pre-declared (TEAM_LOG): every arm with an 8-network (7100584-591) mean log vs C <= -0.15% extends to 7100592-607; pass at 24 = all-24 mean log vs C
<= -0.15% and the out-of-selection 16 agree in sign; (j) is port-worthy (lead) only at an all-24 mean log <= -1.0%.
usage: t147_table.py               the table
       t147_table.py decide        writes t147_extend.txt = the arms to extend (letters, empty = none)"""
import json
import math
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; RUN = HERE / 't147_bets'
SCREEN, OOS = list(range(7100584, 7100592)), list(range(7100592, 7100608))
NAME = dict(h='K31L4 age 4 at 0.6 at read layers 1-14 (ages 1-3 unchanged)', i='K31L4 age 4 at 0.6, age 3 0.7277 -> 0.1277 (redistribution)',
            j='pair law dfit 4 (constant + C^2 four-form, exact C^2)', k='pair law dfit 3 (constant regressor only)')
load = lambda f: json.loads(f.read_text())['mse_layers']
C = {int(f.stem.split('_r')[1].split('_')[0]): load(f) for f in (HERE / 't141_k4l_ext' / 'results').glob('ph141_r*_c.json')}
X = {f.stem: load(f) for f in (RUN / 'results').glob('ph147*.json')}
arm = lambda a: {int(k.split('_r')[1]): v for k, v in X.items() if k.startswith(('ph147%s_r' % a, 'ph147x%s_r' % a))}


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


z = X.get('ph147_z_r7100584')
if z:
    print('control (age-4 instance at 0) vs T141 C on 7100584: %s (worst rel %.2e)' % ('BIT FOR BIT' if z == C[7100584] else 'DIFFERS',
                                                                                     max(abs(a / b - 1) for a, b in zip(z, C[7100584]))))
if len(sys.argv) > 1 and sys.argv[1] == 'decide':
    go = ''
    for a in 'hijk':
        st = stats(a, SCREEN)
        if st is None:      # 18:4xZ fix: an arm whose requests all errored (j / k on worker_sc8) crashed this step and blocked (h)'s extension
            print('(%s) %s: no results (errored) -> no extension' % (a, NAME[a]))
            continue
        print(st['text'] + (' -> extends to 7100592-607' if st['mean'] <= -0.0015 else ' -> no extension (bar -0.15%)'))
        go += a if st['mean'] <= -0.0015 else ''
    (HERE / 't147_extend.txt').write_text(go)
    sys.exit(0)
for a in 'hijk':
    for seeds in (SCREEN, OOS, SCREEN + OOS):
        st = stats(a, seeds)
        if st and (seeds is SCREEN or len(st['s']) > len(SCREEN)):
            print(st['text'])
    full, oos = stats(a, SCREEN + OOS), stats(a, OOS)
    if full and oos and len(full['s']) == 24:
        print('(%s) pass at 24: %s (all-24 mean log %+.3f%%, out-of-selection 16 %+.3f%%)%s'
              % (a, 'YES' if full['mean'] <= -0.0015 and oos['mean'] < 0 else 'no', 100 * full['mean'], 100 * oos['mean'],
                 ('; port-worthy (<= -1.0%%): %s' % ('YES' if full['mean'] <= -0.01 else 'no')) if a == 'j' else ''))
J, K = arm('j'), arm('k')
both = [s for s in SCREEN + OOS if s in J and s in K]
if len(both) >= 4:
    d = np.array([math.log(J[s][-1] / K[s][-1]) for s in both])
    print('C^2 part: (j) dfit 4 vs (k) dfit 3 on %d paired: mean log %+.3f%% (se %.3f), (j) lower on %d' % (len(both), 100 * d.mean(), 100 * d.std(ddof=1) / math.sqrt(len(d)), int((d < 0).sum())))
