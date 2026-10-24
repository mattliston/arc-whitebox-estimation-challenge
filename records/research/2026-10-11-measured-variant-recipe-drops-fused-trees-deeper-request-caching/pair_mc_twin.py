#!/usr/bin/env python3
"""TAG opus-release pair MC for a hypothetical TWIN (2026-10-11 20:30 UTC; a copy of O/analyst/a16_pair_mc.py, its place model and rival pairs
unchanged): our two designated files are given as (graded base page, score factor, P(loss)) so a file that is not graded yet (the S48 safe, an
S49 twin) can be priced from a graded page of the same structure scaled by a constant (approximation: every network alike).
usage: pair_mc_twin.py <base id> f1 L1 f2 L2 [label]   (file 1 = ranked first when clean, file 2 = the fallback)"""
import os
for k in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]/'analyst'; B = 2.0 ** 41; NT = 12; NS = 50000
base = int(sys.argv[1]); f1, l1, f2, l2 = (float(a) for a in sys.argv[2:6]); label = sys.argv[6] if len(sys.argv) > 6 else ''
rows = {}
for l in open(HERE / 'scan_merged.jsonl'):
    r = json.loads(l)
    if r.get('http') == 200: rows[r['id']] = r
team = lambda r: r.get('team') or r['who']
ok = lambda r: (len(r.get('flops') or []) == 100 and r.get('resid') and r.get('wall') and r['status'] == 'GRADED' and r.get('score') is not None
                and not (r.get('agg') or {}).get('n_failed_mlps'))
best = {}
for r in rows.values():
    if ok(r) and (team(r) not in best or r['score'] < best[team(r)]['score']): best[team(r)] = r
R = json.loads((HERE / 'a16_loss_risk.json').read_text()); one = R['lost_network']['score_one_of_50']; ER = R['entries_risk']


def per_net(r):
    s = np.array([p[1] for p in sorted(r['pub'])]) * np.maximum(0.1, np.array(r['flops'][:50]) / B)
    assert abs(s.mean() / r['score'] - 1) < 2e-3, (r['id'], s.mean(), r['score']); return s


ours = 'mliston'; rivals = [team(r) for r in sorted(best.values(), key=lambda r: r['score']) if team(r) != ours][:NT - 1]
pairs = []                                                     # per rival: (first file, second file) as (entry, P(loss)); a team without risk entries: its best twice at 0
for t in rivals:
    e = ER.get(t) or [dict(id=best[t]['id'], p_any=0.0)]
    pairs.append([(rows[x['id']], x['p_any']) for x in (e[:2] if len(e) > 1 else e * 2)])
rng = np.random.default_rng(29); idx = rng.integers(0, 50, (NS, 50))
S1 = np.array([per_net(a[0])[idx].mean(1) for a, b in pairs]); S2 = np.array([per_net(b[0])[idx].mean(1) for a, b in pairs])
p1 = np.array([a[1] for a, b in pairs])[:, None]; p2 = np.array([b[1] for a, b in pairs])[:, None]
simB = per_net(rows[base])[idx].mean(1)
u1, u2 = rng.random(S1.shape), rng.random(S1.shape); v1, v2 = rng.random(NS), rng.random(NS)      # common random numbers for every factor and case
j = rivals.index('J2W')
res = {}
for case in ('independent', 'J2W together'):
    w2 = u2.copy()
    if case == 'J2W together': w2[j] = u1[j]
    sc = np.where(u1 >= p1, S1, np.where(w2 >= p2, S2, one))
    us = np.where(v1 >= l1, simB*f1, np.where(v2 >= l2, simB*f2, np.inf))
    place = 1 + (sc < np.where(np.isinf(us), one*1.0001, us)).sum(0)
    res[case] = ((place == 1).mean(), (place <= 3).mean(), place.mean())
print(f"{label:52s} file1 x{f1:.5f} L {100*l1:.1f}% | file2 x{f2:.5f} L {100*l2:.1f}% | P(podium) {100*res['independent'][1]:.2f}% (J2W together {100*res['J2W together'][1]:.2f}%) "
      f"| P(1st) {100*res['independent'][0]:.2f}% | expected place {res['independent'][2]:.3f}")
