#!/usr/bin/env python3
"""TAG opus-physicist T82 (numpy): table of T80 arms with a GENERAL bill proxy (any join schedule): per layer push ~ RP_l x live_prev_l x young_l, young read ~
RR_l x resp_l x young_l x alive_l (young = columns pushed minus the old tier's), the old tier's push ~ B x live_prev_l x basis_l (basis = the old tier's rank in force,
recorded), each join ~ J x its rank; RP / RR from the engineer's S40 census of 651 over T76's base counts (as T77), B and J from the census fit (T77 notes). ROUGH
(about 2x low on the old tier's rank against the engineer's native price). Per arm against base on the same networks: final-layer MSE change, bill proxy change, and
their product (the score proxy).
usage: t82_joins_table.py <tag>"""
import json, sys, math, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
tag = sys.argv[1]
d = json.load(open(O / 'engineer/census/s40_651.json')); TOT = d['nets'][0]['flops']
push = collections.Counter(); read = collections.Counter()
for p in d['products']:
    if '_deadcut_push' in p['chain']: push[p['layer']] += p['flops']
    if '_cp_read_young_sq' in p['chain']: read[p['layer']] += p['flops']
c5 = {int(k): v for k, v in json.loads(open(HERE / 't76_parity.jsonl').readline())['counts'].items()}
B_RANK, J_RANK = 8101.4, 5.645e9 / 363.0
RANK0 = {l: (0 if l < 7 else 384 if l < 10 else 320 if l < 13 else 384) for l in range(1, 16)}
RP = {l: max(push[l] - B_RANK * c5[l]['live_prev'] * RANK0[l] - (J_RANK * RANK0[l] if l in (7, 10, 13) else 0.0), 0.0) / (c5[l]['live_prev'] * max(c5[l]['cols'] - c5[l]['old'], 1)) for l in range(1, 15)}
RR = {l: read[l] / (c5[l]['resp'] * (c5[l]['cols'] - c5[l]['old']) * c5[l]['alive']) for l in range(1, 15)}


def bill(counts, joins, sched):
    ranks = dict(zip(sorted(int(k) for k in joins), (int(x) for x in sched.split('-'))))
    b = 0.0
    for l in range(1, 16):
        c = counts.get(str(l))
        if not c or 'cols' not in c: continue
        young = c['cols'] - c['old']
        b += RP.get(l, RP[14]) * c['live_prev'] * young + RR.get(l, 0.0) * c.get('resp', 0) * young * c.get('alive', 0)
        b += B_RANK * c['live_prev'] * c.get('basis', 0) + (J_RANK * ranks[l] if l in ranks else 0.0)
    return b


R = collections.defaultdict(dict)
for line in open(HERE / f't80_{tag}.jsonl'):
    x = json.loads(line); R[x['arm']][x['net']] = x
base = R.get('base', {})
print('T82 %s: arms against base on the same networks; bill proxy in %% of the whole bill (ROUGH); score proxy = error ratio x bill ratio' % tag)
for arm, rows in R.items():
    if arm == 'base': continue
    nets = [n for n in rows if n in base and 'counts' in rows[n] and 'counts' in base[n]]
    if not nets: continue
    a = np.array([rows[n]['mse'] for n in nets]); b0 = np.array([base[n]['mse'] for n in nets]); lr = np.log(a / b0)
    db = np.mean([(bill(rows[n]['counts'], rows[n]['joins'], rows[n]['sched']) - bill(base[n]['counts'], base[n]['joins'], base[n]['sched'])) / TOT for n in nets])
    se = np.std(lr, ddof=1) / math.sqrt(len(lr)) if len(lr) > 1 else float('nan'); er = a.sum() / b0.sum()
    print('  %-70s n %2d | error %+.2f%% (se %.2f%%, lower %d) | bill %+.1f%% | score %+.2f%%' % (arm[:70], len(nets), 100 * (er - 1), 100 * se, int((a < b0).sum()), 100 * db, 100 * (er * (1 + db) - 1)))
