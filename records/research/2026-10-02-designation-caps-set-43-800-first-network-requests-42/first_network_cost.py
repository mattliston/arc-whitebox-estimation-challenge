"""TAG opus-release: what being the FIRST network after a setup costs in requests, function by function, without the difference
between two networks: two --sites sessions of one file with the two networks in opposite order (A: 35 then 100; B: 100 then
35). Each network is then seen once as the first network of a process and once as the second; first - second, per function.
(The second network of a process still binds keys the first one did not use; later networks bind fewer. So this is the cost
of being first against being second, a lower bound of first against a late network.)
usage: first_network_cost.py <session dir A (35 first)> <session dir B (100 first)> [min requests to list]"""
import collections
import json
from pathlib import Path
import sys

VIEW = {'__getitem__', 'unstack', 'reshape', 'transpose', 'asarray', 'empty', 'zeros', 'swapaxes', 'moveaxis', 'ravel', 'diag',
        'stack', 'astype', 'arange', 'full', 'ones', 'expand_dims', 'squeeze', 'broadcast_to', 'T', 'copy'}


def load(d):
    out = {}
    for pos, x in enumerate((Path(d)/'rows.jsonl').read_text().splitlines()):
        r = json.loads(x); f = collections.Counter(); v = collections.Counter()
        for key, n in r['by_site'].items():
            line, func, op = key.split(' ', 2)
            f[func] += n
            if op in VIEW:
                v[func] += n
        out[r['rank']] = dict(pos=pos, requests=r['requests'], f=f, v=v, flops=r['flops'])
    return out


a, b = load(sys.argv[1]), load(sys.argv[2]); low = int(sys.argv[3]) if len(sys.argv) > 3 else 8
total = collections.Counter(); views = collections.Counter()
for rank in sorted(set(a) & set(b)):
    first, second = (a[rank], b[rank]) if a[rank]['pos'] == 0 else (b[rank], a[rank])
    assert first['pos'] == 0 and second['pos'] == 1, rank
    print(f"network {rank}: first after the setup {first['requests']} requests, second in the process {second['requests']}: "
          f"{first['requests'] - second['requests']:+d}; bill {first['flops']/second['flops'] - 1:+.3%}")
    for k in set(first['f']) | set(second['f']):
        total[(rank, k)] = first['f'][k] - second['f'][k]; views[(rank, k)] = first['v'][k] - second['v'][k]
ranks = sorted(set(a) & set(b)); names = sorted({k for _r, k in total}, key=lambda k: -sum(total[(r, k)] for r in ranks))
print(f"{'function':30s} " + ' '.join(f'net {r:>4d}' for r in ranks) + '   of them without arithmetic')
for k in names:
    if max(abs(total[(r, k)]) for r in ranks) >= low:
        print(f"  {k:28s} " + ' '.join(f'{total[(r, k)]:+8d}' for r in ranks) + '   ' + ' / '.join(f'{views[(r, k)]:+d}' for r in ranks))
rest = [k for k in names if max(abs(total[(r, k)]) for r in ranks) < low]
print(f"  {'(' + str(len(rest)) + ' smaller)':28s} " + ' '.join(f'{sum(total[(r, k)] for k in rest):+8d}' for r in ranks))
