"""v3 (trainer, 10-12 17:3xZ, engineer 17:3xZ: v2 matched a verdict string QUOTED inside an announcement and applied x2a1 by mistake): a verdict counts only when the line BEGINS '[engineer] <stamp> ROUTE PARITY <key>: PASSED'.
v2 (trainer, 10-12 17:2xZ, engineer 17:22:55Z: the x2a1 per-layer route, chain_x2a_v1.sh): = v1 + 'ROUTE PARITY x2a1_7: PASSED' -> x2a1 bands [7,7], [1,6], [8,10], [11,14], all.
TAG opus-trainer RAD (10-12 17:0xZ; engineer 16:5xZ: route parity verdicts 'ROUTE PARITY <fam>_<layer>: PASSED / FAILED'): one pass. For each
engineer line with 'ROUTE PARITY bscale_0: PASSED', 'X1_14: PASSED' or 'k22_14: PASSED' not yet applied, add that single layer as its own rule-(b)
band to cert/RAD/ROUTES.json (bscale -> bands [[0,0],[0,15]], X1 -> [[14,14],[0,15]]: both leave all_only; k22 -> [[14,14]]); the old table is
archived as ROUTES.json.archived_<stamp>; one TEAM_LOG line per change. FAILED, not comparable and anything else change nothing.
usage (from O/trainer): python -B cert/RAD/routes_update_v3.py   (cert/RAD/routes_update_v3.sh calls it every 2 min)"""
import json
import re
import time
from pathlib import Path

P = Path('cert/RAD/ROUTES.json')
r = json.loads(P.read_text())
eng = [l for l in Path('../TEAM_LOG.md').read_text().splitlines() if l.startswith('[engineer] ')]
ADD = {'bscale_0': ('bscale', [[0, 0], [0, 15]]), 'X1_14': ('X1', [[14, 14], [0, 15]]), 'k22_14': ('k22', [[14, 14]]), 'x2a1_7': ('x2a1', [[7, 7], [1, 6], [8, 10], [11, 14], [0, 15]])}
done = set(r.get('applied_parity', []))
changed = []
for key, (fam, bands) in ADD.items():
    if key in done or not any(re.match(r'\[engineer\] \S+ ROUTE PARITY %s: PASSED' % re.escape(key), l) for l in eng):      # v3: anchored
        continue
    r.setdefault('bands', {})[fam] = bands
    r['all_only'] = [f for f in r['all_only'] if f != fam]
    r['per_layer'] = [f for f in r['per_layer'] if f != fam]
    done.add(key); changed.append('%s -> bands %s' % (fam, bands))
if changed:
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    r['applied_parity'] = sorted(done); r['utc'] = stamp
    P.rename(P.with_name('ROUTES.json.archived_%s' % stamp.replace(':', '').replace('-', '')))
    P.write_text(json.dumps(r, indent=1))
    open('../TEAM_LOG.md', 'a').write('[trainer] %s ROUTES.json UPDATED from the engineer\'s ROUTE PARITY PASSED lines (cert/RAD/routes_update_v3.py): %s; rounds pick it up at their next arm stage.\n' % (stamp, '; '.join(changed)))
    print('updated:', changed)
