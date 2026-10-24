#!/usr/bin/env python3
"""TAG opus-physicist T151 table: the carrier-dial scan on top of C. Screen: each arm vs C's reference on the FRESH 7100608-623 (both in this run), paired.
Confirmation: a passing arm on 7100592-607 vs T141's C there. Pre-declared (TEAM_LOG): per dial (nfs: sp / sm; nfd: dp / dm; K31L4: kp) the arm with the
lower 16-network screen mean log is confirmed if that mean log is <= -0.15%; PASS = the all-32 mean log (screen 16 + confirmation 16) <= -0.15% and the
confirmation 16 agree in sign.
usage: t151_table.py | t151_table.py decide (writes t151_confirm.txt = the arms to confirm, space separated)"""
import json
import math
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; RUN = HERE / 't151_carrier'
FRESH, CONF = list(range(7100608, 7100624)), list(range(7100592, 7100608))
NAME = dict(sp='law plane strength nfs +0.15 at L7-14', sm='law plane strength nfs -0.15 at L7-14', dp='law dscale nfd +0.15 at L1-14',
            dm='law dscale nfd -0.15 at L1-14', kp='K31L4 plane x1.2 at L7-14 (all ages)')
DIALS = (('sp', 'sm'), ('dp', 'dm'), ('kp',))
load = lambda f: json.loads(f.read_text())['mse_layers']
X = {f.stem: load(f) for f in (RUN / 'results').glob('ph151*.json')}
REF = {int(k.rsplit('_r', 1)[1]): v for k, v in X.items() if k.startswith('ph151_ref_r')}
REF.update({int(f.stem.split('_r')[1].split('_')[0]): load(f) for f in (HERE / 't141_k4l_ext' / 'results').glob('ph141_r*_c.json')})
arm = lambda a: {int(k.rsplit('_r', 1)[1]): v for k, v in X.items() if k.startswith(('ph151%s_r' % a, 'ph151x%s_r' % a))}


def stats(a, seeds):
    A = arm(a); s = [x for x in seeds if x in A and x in REF]
    if len(s) < 2:
        return None
    x = np.array([A[q][-1] for q in s]); c = np.array([REF[q][-1] for q in s]); lr = np.log(x / c)
    lx = np.array([A[q] for q in s]); lc = np.array([REF[q] for q in s])
    return dict(s=s, mean=lr.mean(), text=(
        '(%s) %s vs C on %d (%d-%03d): final %+.3f%% pooled, mean log %+.3f%% (se %.3f), lower on %d of %d; L8-15 %s; per network %s'
        % (a, NAME[a], len(s), min(s), max(s) % 1000, 100 * (x.mean() / c.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(s)),
           int((x < c).sum()), len(s), ' '.join('%+.2f' % (100 * (lx[:, l].mean() / lc[:, l].mean() - 1)) for l in range(8, 16)),
           ' '.join('%+.2f' % (100 * v) for v in lr))))


if len(sys.argv) > 1 and sys.argv[1] == 'decide':
    go = []
    for dial in DIALS:
        st = {a: stats(a, FRESH) for a in dial}
        for a in dial:
            print(st[a]['text'] if st[a] else '(%s) %s: no results' % (a, NAME[a]))
        ok = [a for a in dial if st[a]]
        if ok:
            w = min(ok, key=lambda a: st[a]['mean'])
            print('dial %s: better arm (%s) %+.3f%% -> %s' % ('/'.join(dial), w, 100 * st[w]['mean'],
                                                             'confirm on 7100592-607' if st[w]['mean'] <= -0.0015 else 'no confirmation (bar -0.15%)'))
            if st[w]['mean'] <= -0.0015:
                go.append(w)
    (HERE / 't151_confirm.txt').write_text(' '.join(go))
    sys.exit(0)
for dial in DIALS:
    for a in dial:
        for seeds in (FRESH, CONF, FRESH + CONF):
            st = stats(a, seeds)
            if st and (seeds is FRESH or seeds is CONF or len(st['s']) > len(FRESH)):
                print(st['text'])
        full, oos = stats(a, FRESH + CONF), stats(a, CONF)
        if full and oos and len(full['s']) == 32:
            print('(%s) PASS: %s (all-32 mean log %+.3f%%, confirmation 16 %+.3f%%)' % (a, 'YES' if full['mean'] <= -0.0015 and oos['mean'] < 0 else 'no',
                                                                                       100 * full['mean'], 100 * oos['mean']))
