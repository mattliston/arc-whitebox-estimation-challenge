"""TAG opus_engineer a56 (2026-10-11): request census of a --sites boundary session (a56_req/run_sites.sh): per function and per site
(line, function, operation), for each network of the session (first = cold after the setup), with the no-arithmetic share (views,
reshapes, allocations, conversions: release/census_by_function.py's VIEW set).
usage: sites_table.py <session dir> [top functions] [top sites]"""
import collections
import json
import sys
from pathlib import Path

VIEW = {'__getitem__', 'unstack', 'reshape', 'transpose', 'asarray', 'empty', 'zeros', 'swapaxes', 'moveaxis', 'ravel', 'diag',
        'stack', 'astype', 'arange', 'full', 'ones', 'expand_dims', 'squeeze', 'broadcast_to', 'T', 'copy'}
d = Path(sys.argv[1]); topf = int(sys.argv[2]) if len(sys.argv) > 2 else 45; tops = int(sys.argv[3]) if len(sys.argv) > 3 else 60
rows = [json.loads(x) for x in (d/'rows.jsonl').read_text().splitlines()]
for r in rows:
    f, v, s = collections.Counter(), collections.Counter(), collections.Counter()
    ops = collections.Counter()
    for key, n in r['by_site'].items():
        line, func, op = key.split(' ', 2)
        f[func] += n; s[key] += n; ops[op] += n
        if op in VIEW:
            v[func] += n
    tot, noar = r['requests'], sum(n for k, n in r['by_site'].items() if k.split(' ', 2)[2] in VIEW)
    print(f"== network {r['rank']}: {tot} requests, {noar} without arithmetic ({100 * noar / tot:.1f}%); bill {r['flops']:,}")
    print('   by operation:', ', '.join(f'{k} {n}' for k, n in ops.most_common(25)))
    print(f"   {'function':36s} {'requests':>8s} {'no arith':>8s}")
    for k, n in f.most_common(topf):
        print(f"   {k:36s} {n:8d} {v[k]:8d}")
    print(f"   top sites (line function operation):")
    for k, n in s.most_common(tops):
        print(f"   {n:6d}  {k}")
