"""TAG opus_engineer a62 v2 (lead 19:3xZ: price join schedules, incl. 7/10/- = no third join) = price_joins_v1.py plus:
(a) the final layer priced from the census's own layer-15 parts (its push serves k3 only, on the rows |mu/sigma| <= 3.5): young and old
    cost per column there, not layer 14's;
(b) any schedule SYNTHESIZED from the physicist's base records (no torch run needed for the bill): cohort b's width = cols(b+1) - cols(b)
    of the base record, and a schedule 'J1:b0-b1,J2:b0-b1,...' puts the cohorts b0..b1 into the old tier at layer J (so the old columns
    at layer l = the widths of every cohort joined at or before l; the young = cols - old; the total columns per layer are the base's).
Model per network: change = sum_l [y_l (Ky_var - Ky_base) + o_l r_var(l)/r_fired(l) (Ko_var - Ko_base)] + joins_var - joins_base, a join's
cost = its joined columns x its rank x the fired joins' cost per column and rank; in % of S47e's mean bill on 651 / 378.
usage: price_joins_v2.py <base push_counts.jsonl> <schedule: 'synth:7:0-2,10:3-5' or a variant push_counts.jsonl> <ranks, e.g. 384,320>"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = open(HERE/'table_v1.py').read().split("\ntot = {n['rank']")[0]
argv, sys.argv = sys.argv, ['x', str(HERE/'census_s47e_651_378.json')]
ns = {}
exec(compile(src, 'table_v1.py', 'exec'), ns)
sys.argv = argv
d = json.load(open(HERE/'census_s47e_651_378.json'))
tot = {n['rank']: n['flops'] for n in d['nets']}
young = collections.Counter(); old = collections.Counter(); join = collections.Counter()
for rank, layer, line, chain, op, fl, n in d['rows']:
    lines = [c[1] for c in chain if c[0] == '_shared_old384_push']
    is_join = any(x in (8432, 8434, 8436, 8457, 8460, 8461) for x in lines)
    for m, frac in ns['classify'](layer, line, chain):
        if m in ('BASE bank push: young columns', 'BASE bank read: young square'):
            young[(rank, layer)] += frac * fl
        elif m in ('BASE bank push: old tier', 'BASE bank read: old tier'):
            (join if is_join else old)[(rank, layer)] += frac * fl
K = {(r['rank'], r['layer']): (r['young'], r['old']) for r in json.load(open(HERE/'rows2_s47e_651_378.json'))['records']}
ranks = (651, 378)
for r in ranks:   # layer 15: young = layer 14's young + its newborn cohort (about layer 14's young minus layer 13's), old = layer 14's
    K[(r, 15)] = (K[(r, 14)][0] + (K[(r, 14)][0] - K[(r, 13)][0]), K[(r, 14)][1])
y = {l: sum(young[(r, l)] / K[(r, l)][0] for r in ranks) / 2 for l in range(1, 16)}
o = {l: sum(old[(r, l)] / K[(r, l)][1] for r in ranks) / 2 for l in range(7, 16)}
for l in range(1, 7):
    o[l] = o[7]
fired_rank = {l: (384 if l < 10 else 320 if l < 13 else 384) for l in range(1, 16)}
joined = {7: 6144, 10: 11776 - 6144, 13: 16576 - 11776}
per_join_col = sum(join[(r, j)] / joined[j] / fired_rank[j] for r in ranks for j in joined) / (2 * len(joined))
mean_bill = sum(tot.values()) / len(tot)
print('per-column costs (Mflop per column, mean of 651 / 378): young ' + ' '.join('L%d %.2f' % (l, y[l] / 1e6) for l in range(1, 16))
      + ' | old ' + ' '.join('L%d %.2f' % (l, o[l] / 1e6) for l in range(7, 16)) + ' | join %.3f per column and rank unit' % (per_join_col / 1e6))
base = {json.loads(l)['tag'].split('_')[-1]: json.loads(l) for l in open(sys.argv[1])}
vranks = [int(x) for x in sys.argv[3].split(',')]


def synth(rec, spec):
    cols = {int(l): v['cols'] for l, v in rec['counts'].items()}
    width = {b: cols[b + 1] - cols[b] for b in range(0, 15)}
    joins = {}
    for part in spec.split(','):
        j, rng = part.split(':')
        b0, b1 = (int(x) for x in rng.split('-'))
        joins[int(j)] = list(range(b0, b1 + 1))
    counts = {}
    for l in range(0, 16):
        o_l = sum(width[b] for j, bs in joins.items() if j <= l for b in bs)
        counts[str(l)] = dict(cols=cols[l], old=o_l)
    return dict(tag=rec['tag'], joins={str(j): bs for j, bs in joins.items()}, counts=counts)


if sys.argv[2].startswith('synth:'):
    var = {k: synth(v, sys.argv[2][len('synth:'):]) for k, v in base.items()}
else:
    var = {json.loads(l)['tag'].split('_')[-1]: json.loads(l) for l in open(sys.argv[2])}


def schedule(rec, rank_list):
    joins = sorted(int(j) for j in rec['joins'])
    rank_at = {}
    for i, j in enumerate(joins):
        for l in range(j, 16):
            rank_at[l] = rank_list[i]
    counts = {int(l): (v['cols'] - v['old'], v['old']) for l, v in rec['counts'].items()}
    cost = 0.0
    for l in range(1, 16):
        ky, ko = counts[l]
        cost += y[l] * ky
        if ko:
            cost += o[l] * ko * rank_at[l] / fired_rank[l]
    for j in joins:
        cost += per_join_col * rank_at[j] * (counts[j][1] - counts[j - 1][1])
    return cost


changes = []
for seed in sorted(set(base) & set(var)):
    b = schedule(base[seed], [384, 320, 384])
    v = schedule(var[seed], vranks)
    changes.append(100 * (v - b) / mean_bill)
print('%s vs 7/10/13: mean bill change %+.2f%% (range %+.2f..%+.2f) over %d networks; young column-layers %+.1f%%'
      % (sys.argv[2], sum(changes) / len(changes), min(changes), max(changes), len(changes),
         100 * (sum(sum(v['cols'] - v['old'] for v in var[s]['counts'].values()) for s in var)
                / sum(sum(v['cols'] - v['old'] for v in base[s]['counts'].values()) for s in var) - 1)))
