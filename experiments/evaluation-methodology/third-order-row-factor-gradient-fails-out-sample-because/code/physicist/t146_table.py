#!/usr/bin/env python3
"""TAG opus-physicist T146 table: the K31 row factor c0 += 0.05 / 0.10 at L8-14 on top of C vs T141's C, paired per network (results in
physicist/t145_coadapt). Pre-declared 15:54:32Z (TEAM_LOG): the arm with the lower 8-network mean log vs C extends to 7100592-607 if that mean log is
<= -0.15%; pass at 24 = all-24 mean log vs C <= -0.15% and the out-of-selection 16 agree in sign. Lead 16:0xZ: when the better arm extends and (f)
beats (e) on 8, (g) c0 += 0.15 at L8-14 runs on all 24 as well (the bowl); between passing arms the lower all-24 mean log goes to scout-native.
usage: t146_table.py               the table
       t146_table.py decide        writes t146_extend.txt = the arm to extend (empty = none) and t146_g.txt = 'g' or empty"""
import json
import math
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; RUN = HERE / 't145_coadapt'
SCREEN, OOS = list(range(7100584, 7100592)), list(range(7100592, 7100608))
NAME = dict(e='K31 row factor c0 +0.05 at L8-14', f='K31 row factor c0 +0.10 at L8-14', g='K31 row factor c0 +0.15 at L8-14')
load = lambda f: json.loads(f.read_text())['mse_layers']
C = {int(f.stem.split('_r')[1].split('_')[0]): load(f) for f in (HERE / 't141_k4l_ext' / 'results').glob('ph141_r*_c.json')}
X = {f.stem: load(f) for f in (RUN / 'results').glob('ph146*.json')}
arm = lambda a: {int(k.split('_r')[1]): v for k, v in X.items() if k.startswith(('ph146%s_r' % a, 'ph146x%s_r' % a))}


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


if len(sys.argv) > 1 and sys.argv[1] == 'decide':
    st = {a: stats(a, SCREEN) for a in 'ef'}
    for a in 'ef':
        print(st[a]['text'])
    w = min('ef', key=lambda a: st[a]['mean'])
    go = st[w]['mean'] <= -0.0015
    g = go and st['f']['mean'] < st['e']['mean']
    print('T146 better arm (%s) %+.3f%% vs C on 8 -> %s%s' % (w, 100 * st[w]['mean'], 'extends to 7100592-607' if go else 'no extension (bar -0.15%)',
                                                           '; (f) beats (e), so (g) c0 +0.15 runs on all 24' if g else ''))
    (HERE / 't146_extend.txt').write_text(w if go else '')
    (HERE / 't146_g.txt').write_text('g' if g else '')
    sys.exit(0)
for a in 'efg':
    for seeds in (SCREEN, OOS, SCREEN + OOS):
        st = stats(a, seeds)
        if st and (seeds is SCREEN or len(st['s']) > len(SCREEN)):
            print(st['text'])
    full, oos = stats(a, SCREEN + OOS), stats(a, OOS)
    if full and oos and len(full['s']) == 24:
        print('(%s) pass at 24: %s (all-24 mean log %+.3f%%, out-of-selection 16 %+.3f%%)'
              % (a, 'YES' if full['mean'] <= -0.0015 and oos['mean'] < 0 else 'no', 100 * full['mean'], 100 * oos['mean']))
A24 = {a: arm(a) for a in 'efg'}
both = [s for s in SCREEN + OOS if s in A24['f'] and s in A24['g']]
if len(both) >= 8:
    d = np.array([math.log(A24['g'][s][-1] / A24['f'][s][-1]) for s in both])
    print('bowl: (g) c0 +0.15 vs (f) c0 +0.10 on %d paired: mean log %+.3f%% (se %.3f), (g) lower on %d' % (len(both), 100 * d.mean(), 100 * d.std(ddof=1) / math.sqrt(len(d)), int((d < 0).sum())))
