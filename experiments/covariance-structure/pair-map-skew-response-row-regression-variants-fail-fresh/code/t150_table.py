#!/usr/bin/env python3
"""TAG opus-physicist T150 table: the pair law's dfit 4 (j) / dfit 3 (k) vs their matched control (c) = dfit 2, all three with the cutsim option
law_response_fit = false, on top of C; also (c) vs T141's C (the cost of dropping the native response-row fit). Pre-declared (TEAM_LOG): an arm with an
8-network (7100584-591) mean log vs (c) <= -0.15% runs on 7100592-607 together with (c); pass at 24 = all-24 mean log vs (c) <= -0.15% and the
out-of-selection 16 agree in sign; (j) is port-worthy (lead) only at <= -1.0% on 24.
usage: t150_table.py | t150_table.py decide (writes t150_extend.txt = the arms to extend, 'c' included when any arm extends)"""
import json
import math
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; RUN = HERE / 't150_dfit'
SCREEN, OOS = list(range(7100584, 7100592)), list(range(7100592, 7100608))
NAME = dict(c='law dfit 2 (control), all-row fit', j='law dfit 4 (C^2 four-form, exact C^2), all-row fit', k='law dfit 3 (constant regressor), all-row fit')
load = lambda f: json.loads(f.read_text())['mse_layers']
T141 = {int(f.stem.split('_r')[1].split('_')[0]): load(f) for f in (HERE / 't141_k4l_ext' / 'results').glob('ph141_r*_c.json')}
X = {f.stem: load(f) for f in (RUN / 'results').glob('ph150*.json')}
arm = lambda a: {int(k.split('_r')[1]): v for k, v in X.items() if k.startswith(('ph150%s_r' % a, 'ph150x%s_r' % a))}


def stats(a, ref, rname, seeds):
    A = arm(a); s = [x for x in seeds if x in A and x in ref]
    if len(s) < 2:
        return None
    x = np.array([A[q][-1] for q in s]); c = np.array([ref[q][-1] for q in s]); lr = np.log(x / c)
    lx = np.array([A[q] for q in s]); lc = np.array([ref[q] for q in s])
    return dict(s=s, mean=lr.mean(), text=(
        '(%s) %s vs %s on %d (%d-%03d): final %+.3f%% pooled, mean log %+.3f%% (se %.3f), lower on %d of %d; L8-15 %s; per network %s'
        % (a, NAME[a], rname, len(s), s[0], s[-1] % 1000, 100 * (x.mean() / c.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(s)),
           int((x < c).sum()), len(s), ' '.join('%+.2f' % (100 * (lx[:, l].mean() / lc[:, l].mean() - 1)) for l in range(8, 16)),
           ' '.join('%+.2f' % (100 * v) for v in lr))))


CC = arm('c')
st = stats('c', T141, "T141's C", SCREEN + OOS)
if st:
    print(st['text'])
if len(sys.argv) > 1 and sys.argv[1] == 'decide':
    go = ''
    for a in 'jk':
        st = stats(a, CC, '(c)', SCREEN)
        if st is None:
            print('(%s) %s: no results -> no extension' % (a, NAME[a])); continue
        print(st['text'] + (' -> extends to 7100592-607' if st['mean'] <= -0.0015 else ' -> no extension (bar -0.15%)'))
        go += a if st['mean'] <= -0.0015 else ''
    (HERE / 't150_extend.txt').write_text(('c' + go) if go else '')
    sys.exit(0)
for a in 'jk':
    for seeds in (SCREEN, OOS, SCREEN + OOS):
        st = stats(a, CC, '(c)', seeds)
        if st and (seeds is SCREEN or len(st['s']) > len(SCREEN)):
            print(st['text'])
    full, oos = stats(a, CC, '(c)', SCREEN + OOS), stats(a, CC, '(c)', OOS)
    if full and oos and len(full['s']) == 24:
        print('(%s) pass at 24: %s (all-24 mean log vs (c) %+.3f%%, out-of-selection 16 %+.3f%%)%s'
              % (a, 'YES' if full['mean'] <= -0.0015 and oos['mean'] < 0 else 'no', 100 * full['mean'], 100 * oos['mean'],
                 ('; port-worthy (<= -1.0%%): %s' % ('YES' if full['mean'] <= -0.01 else 'no')) if a == 'j' else ''))
J, K = arm('j'), arm('k')
both = [s for s in SCREEN + OOS if s in J and s in K]
if len(both) >= 4:
    d = np.array([math.log(J[s][-1] / K[s][-1]) for s in both])
    print('C^2 part: (j) dfit 4 vs (k) dfit 3 on %d paired: mean log %+.3f%% (se %.3f), (j) lower on %d' % (len(both), 100 * d.mean(), 100 * d.std(ddof=1) / math.sqrt(len(d)), int((d < 0).sum())))
