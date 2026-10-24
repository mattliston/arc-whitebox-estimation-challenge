"""TAG opus-release: one line per stage C run (one worker session each: setup, network 35 first, network 100 warm) against a
reference session: requests of the first and of the warm network, the setup (binding phase, keys bound, deferred fusions,
pace), bill, kernel work (instructions without page-fault work), the projection of stage C, and whether the stored prediction
bytes equal the reference session's (pred_<rank>.bin). The first-network residual is predicted at the bank's platform prices:
warm requests x (median residual of the bank's other networks / the bank's warm requests) + (first - warm) x FIRST_PRICE.
usage: variant_table.py <reference run dir> <run dir> [<run dir> ...]"""
import hashlib
import json
from pathlib import Path
import sys

import project

R = Path(__file__).resolve().parent
WARM_PRICE = 0.3067/42678          # S40 #333559: median residual of the 93 networks that are no first network / requests of network 100
FIRST_PRICE = 12.7e-6              # extra residual of a first network per extra request (S36-S40: 0.0236 s over about 1,880 requests)


def session(run):
    leg = sorted(p for p in Path(run).iterdir() if p.is_dir() and (p/'rows.jsonl').exists() and p.name.startswith('C'))[0]
    rows = [json.loads(x) for x in (leg/'rows.jsonl').read_text().splitlines()]
    man = json.loads((leg/'manifest.json').read_text())
    pred = {r['rank']: hashlib.sha256((leg/f"pred_{r['rank']}.bin").read_bytes()).hexdigest() for r in rows}
    stage = json.loads((Path(run)/'stage_c.json').read_text()) if (Path(run)/'stage_c.json').exists() else None
    return dict(rows=rows, setup=man['setup'], pred=pred, stage=stage, sha=man['sha256'][:8])


ref = session(sys.argv[1])
print(f"reference {sys.argv[1]} (sha {ref['sha']}): requests {[r['requests'] for r in ref['rows']]}")
print(f"{'run':30s} {'sha':8s} {'first':>6s} {'warm':>6s} {'f-w':>5s} {'vs ref first/warm':>17s} {'bind req':>8s} {'keys':>4s} {'fused':>5s} {'pace':>5s} "
      f"{'bill first/warm %':>17s} {'kernel':>6s} {'wall mean/worst':>15s} {'first net s':>11s} {'warm s':>6s} bytes")
for run in sys.argv[1:]:
    s = session(run); rows = s['rows']; f, w = rows[0], rows[1]
    log = {x[0]: x[1] for x in s['setup']['pb_log']}
    same = all(s['pred'][r['rank']] == ref['pred'].get(r['rank']) for r in rows)
    ff = [sum(project.fault_free(r).values()) for r in rows]; rf = [sum(project.fault_free(r).values()) for r in ref['rows']]
    p = (s['stage'] or {}).get('projection') or {}
    print(f"{Path(run).name:30s} {s['sha']:8s} {f['requests']:6d} {w['requests']:6d} {f['requests'] - w['requests']:5d} "
          f"{f['requests'] - ref['rows'][0]['requests']:+8d}/{w['requests'] - ref['rows'][1]['requests']:+6d}   "
          f"{s['setup'].get('binding_requests', 0):8d} {log.get('wall guard', s['setup'].get('keys_total', 0)):4} {log.get('lvfuse', 0):5} "
          f"{s['setup'].get('slowdown') or 0:5.3f} {(f['flops']/ref['rows'][0]['flops'] - 1)*100:+8.3f}/{(w['flops']/ref['rows'][1]['flops'] - 1)*100:+7.3f}  "
          f"{sum(ff)/sum(rf):6.4f} {p.get('mean_wall_s', 0):7.1f}/{p.get('worst_wall_s', 0):5.1f}   "
          f"{w['requests']*WARM_PRICE + (f['requests'] - w['requests'])*FIRST_PRICE:11.4f} {w['requests']*WARM_PRICE:6.4f} {'identical' if same else 'DIFFER'}")
