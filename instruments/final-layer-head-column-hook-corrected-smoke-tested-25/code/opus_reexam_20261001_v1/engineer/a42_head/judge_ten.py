"""TAG opus_engineer a42: a candidate's fresh-network session against the stored sessions of its base (same rank list, same order, one
process each): per network and pooled final-layer MSE, bill, requests; the lead's fire rule of 2026-10-02T19:13Z for S42
(final-layer error at least 1.3% lower pooled with at least 8 of 10 networks lower, bill within +0.1%, requests inside the caps
43,800 first network / 42,700 others). The base's runs differ only by their rounding draw, so the candidate is judged against their
MEAN per network (and shown against each).
usage: judge_ten.py <candidate session> <base session> [<base session> ...]"""
import json
import sys
from pathlib import Path
E = Path(__file__).resolve().parents[1]
load = lambda name: {r['rank']: r for r in json.load(open(E/f'logs/quick_{name}.json'))['rows']}
cand = load(sys.argv[1]); bases = [load(x) for x in sys.argv[2:]]
order = [r['rank'] for r in json.load(open(E/f'logs/quick_{sys.argv[1]}.json'))['rows']]
ranks = [r for r in order if all(r in b for b in bases)]
assert ranks == [r for r in order] and all([x['rank'] for x in json.load(open(E/f'logs/quick_{name}.json'))['rows']][:len(ranks)] == ranks for name in sys.argv[2:]), 'rank lists differ in content or order'
requests = lambda x: x['ops'] + (x['views']['total'] if 'views' in x else 0)   # charged ops (+ index requests when the session counted them)
print(f'{sys.argv[1]} against {", ".join(sys.argv[2:])}; networks in run order: {ranks}')
print(' rank   candidate MSE e-9   ' + '   '.join(f'vs base {i + 1}' for i in range(len(bases))) + '   vs their mean    bill      requests')
changes, lower = [], 0
for r in ranks:
    c = cand[r]; mean = sum(b[r]['mse'] for b in bases) / len(bases)
    ch = 100 * (c['mse'] / mean - 1); changes.append(ch); lower += c['mse'] < mean
    print(f'{r:5d}   {1e9 * c["mse"]:15.4f}   ' + '   '.join(f'{100 * (c["mse"] / b[r]["mse"] - 1):+8.2f}%' for b in bases)
          + f'   {ch:+12.2f}%   {100 * (c["flops"] / bases[0][r]["flops"] - 1):+7.3f}%   {requests(c)}' + (f' ({requests(c) - requests(bases[0][r]):+d})' if ('views' in c) == ('views' in bases[0][r]) else ''))
n = len(ranks)
pool = lambda rows: sum(rows[r]['mse'] for r in ranks)
base_pool = sum(pool(b) for b in bases) / len(bases)
pooled = 100 * (pool(cand) / base_pool - 1)
mean = sum(changes) / n
se = (sum((x - mean) ** 2 for x in changes) / (n - 1) / n) ** 0.5
bill = 100 * (sum(cand[r]['flops'] for r in ranks) / sum(bases[0][r]['flops'] for r in ranks) - 1)
print(f'POOLED final-layer MSE: {1e9 * pool(cand) / n:.4f} e-9 against {1e9 * base_pool / n:.4f} e-9 = {pooled:+.2f}%   (each base: '
      + ', '.join(f'{100 * (pool(cand) / pool(b) - 1):+.2f}%' for b in bases) + f'); mean of the per-network changes {mean:+.2f}% (se {se:.2f}); lower on {lower} of {n}')
first, later = requests(cand[ranks[0]]), [requests(cand[r]) for r in ranks[1:]]
print(f'bill {bill:+.3f}% against base 1; requests by this count: first network {first}, others {min(later) if later and later[0] else None}-{max(later) if later and later[0] else None}'
      f' (the release lane counts at the platform boundary; its caps are 43,800 / 42,700)')
rule = pooled <= -1.3 and lower >= 0.8 * n and bill <= 0.1
print('FIRE RULE (pooled <= -1.3%, at least 8 of 10 lower, bill within +0.1%): ' + ('MET' if rule else 'NOT MET') + '; the request caps are the release lane\'s to confirm at the boundary')
