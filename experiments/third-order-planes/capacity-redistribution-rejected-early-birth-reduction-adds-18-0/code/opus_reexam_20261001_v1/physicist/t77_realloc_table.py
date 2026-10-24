#!/usr/bin/env python3
"""TAG opus-physicist T77 (numpy): table of the T76 reallocation arms: per arm against 'base' on the same networks, pooled final-layer MSE change, the mean of the
per-network log ratios and its se, networks lower; and the BILL PROXY: per layer push ~ rate_push_l x live_prev_l x cols_l and young read ~ rate_read_l x resp_l x
young_l x alive_l (young = columns pushed minus old-tier columns), rates = the engineer's S40 census of network 651 (O/engineer/census/s40_651.json, push 43%,
young read 36% of its bill) divided by the base counts of T76's parity network 5; the old tier: push 8,101 flops per (live_prev x rank unit) per layer and the join
5.6e9 flops at the base ranks, prorated by rank (least squares on the census, 25% residuals: a ROUGH proxy); the change against base in % of the whole bill.
usage: t77_realloc_table.py <tag>"""
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
RANK0 = {l: (0 if l < 7 else 384 if l < 10 else 320 if l < 13 else 384) for l in range(1, 16)}
RP = {l: max(push[l] - 8101.4 * c5[l]['live_prev'] * RANK0[l] - (5.645e9 / 363.0 * RANK0[l] if l in (7, 10, 13) else 0.0), 0.0) / (c5[l]['live_prev'] * max(c5[l]['cols'] - c5[l]['old'], 1)) for l in range(1, 15)}
RR = {l: read[l] / (c5[l]['resp'] * (c5[l]['cols'] - c5[l]['old']) * c5[l]['alive']) for l in range(1, 15)}


B_RANK, J_RANK = 8101.4, 5.645e9 / 363.0      # census fit (T77 notes): push of the old tier per (live_prev x rank unit) per layer; join cost per rank unit at a join


def bill(counts, sched='384-320-384', joins=(7, 10, 13)):
    r7, r10, r13 = (int(x) for x in sched.split('-')); j1, j2, j3 = joins
    rank = {l: (0 if l < j1 else r7 if l < j2 else r10 if l < j3 else r13) for l in range(1, 16)}
    b = 0.0
    for l in range(1, 16):
        c = counts.get(str(l))
        if not c or 'cols' not in c: continue
        b += RP.get(l, 0.0) * c['live_prev'] * (c['cols'] - c['old']) + RR.get(l, 0.0) * c.get('resp', 0) * (c['cols'] - c['old']) * c.get('alive', 0)
        b += B_RANK * c['live_prev'] * rank[l] + (J_RANK * rank[l] if l in joins else 0.0)
    return b


R = collections.defaultdict(dict)
for line in open(HERE / f't76_{tag}.jsonl'):
    x = json.loads(line); R[x['arm']][x['rank']] = x
base = R.get('base', {})
print('T77 reallocation arms (%s), against base on the same networks; bill proxy in %% of the whole bill' % tag)
for arm, rows in R.items():
    if arm == 'base': continue
    nets = [r for r in rows if r in base]
    if not nets: continue
    a = np.array([rows[r]['mse'] for r in nets]); b0 = np.array([base[r]['mse'] for r in nets]); lr = np.log(a / b0)
    jl = next((tuple(int(x) for x in part.split(':')[1].split('-')) for part in arm.split(';') if part.startswith('joins:')), (7, 10, 13))
    db = np.mean([(bill(rows[r]['counts'], rows[r].get('sched', '384-320-384'), jl) - bill(base[r]['counts'], base[r].get('sched', '384-320-384'))) / TOT for r in nets])
    se = np.std(lr, ddof=1) / math.sqrt(len(lr)) if len(lr) > 1 else float('nan')
    print('  %-42s n %2d | pooled %+.3f%% | mean log %+.3f%% (se %.3f%%) | lower %d of %d | bill proxy %+.2f%%' % (
        arm, len(nets), 100 * (a.sum() / b0.sum() - 1), 100 * lr.mean(), 100 * se, int((a < b0).sum()), len(nets), 100 * db))
