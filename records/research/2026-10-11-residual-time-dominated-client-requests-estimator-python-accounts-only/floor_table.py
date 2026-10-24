"""TAG opus_engineer a58 (2026-10-11): per-request residual floor from a boundary session of floor_candidate.py (ranks 35 add, 36 view,
45 reshape, 100 none) against a real candidate's session: residual instructions and cycles per request of each kind (the empty
network subtracted), and what that floor leaves of the candidate's own residual per request.
usage: floor_table.py <floor session dir> [<candidate session dir> ...]"""
import json
import sys
from pathlib import Path

rows = {r['rank']: r for r in map(json.loads, (Path(sys.argv[1])/'rows.jsonl').read_text().splitlines())}
base = rows[100]
print(f"empty network: requests {base['requests']}, residual {base['residual_s']*1e3:.2f} ms, {base['residual_instructions']/1e6:.2f} M instructions")
floor = {}
for rank, kind in ((35, 'in-place add'), (36, 'basic-slice view'), (45, 'reshape view')):
    r = rows[rank]
    n = r['requests'] - base['requests']
    di = (r['residual_instructions'] - base['residual_instructions']) / n
    dc = (r['residual_cycles'] - base['residual_cycles']) / n
    ds = (r['residual_s'] - base['residual_s']) / n
    floor[kind] = di
    print(f"{kind:18s} requests {n:6d}: {di/1e3:6.1f} k instructions, {dc/1e3:6.1f} k cycles, {ds*1e6:5.2f} us residual per request")
for d in sys.argv[2:]:
    for r in map(json.loads, (Path(d)/'rows.jsonl').read_text().splitlines()):
        per = r['residual_instructions'] / r['requests']
        lo, hi = min(floor.values()), max(floor.values())
        print(f"{Path(d).name} network {r['rank']}: {r['requests']} requests, {per/1e3:.1f} k instructions per request; "
              f"beyond the floor ({lo/1e3:.1f}..{hi/1e3:.1f} k): {(r['residual_instructions'] - hi * r['requests'])/1e6:.0f}.."
              f"{(r['residual_instructions'] - lo * r['requests'])/1e6:.0f} M instructions "
              f"= {100*(1 - hi * r['requests'] / r['residual_instructions']):.0f}..{100*(1 - lo * r['requests'] / r['residual_instructions']):.0f}% of its residual")
