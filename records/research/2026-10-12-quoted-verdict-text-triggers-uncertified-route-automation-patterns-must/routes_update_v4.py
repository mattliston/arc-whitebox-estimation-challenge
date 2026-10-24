"""v4 (trainer, written 2026-10-12T22:5xZ; lead 22:50:09Z RULE (b')): = v3 (a verdict counts only when the line BEGINS '[engineer] <stamp> ROUTE
PARITY <key>: PASSED') plus the DELIVERY criterion: from the same line, every parity network's 'corr <c>, native/torch size <s>' gives
D = 2 s c - s^2 (the share of a torch-sized dial's gain the native route delivers; s^2 = added error); the route enters only if at least two
networks are listed and D >= 0.5 on every one. Keys in ROUTES.json 'vetoed' (bscale_0, lead 22:50:09Z) are never applied. A PASSED key that
fails (b') is recorded once under 'rejected_b_prime' with its D values (the table versioned, one TEAM_LOG line), so it is not re-judged.
v3 / v2 / v1 docstrings: routes_update_v3.py. Table: bscale_0 -> bscale bands [[0,0],[0,15]]; X1_14 -> [[14,14],[0,15]]; k22_14 -> [[14,14]];
x2a1_7 -> x2a1 [[7,7],[1,6],[8,10],[11,14],[0,15]] (the per-layer x2a route, engineer 17:22Z). The old table is archived as
ROUTES.json.archived_<stamp>. usage (from O/trainer): python -B cert/RAD/routes_update_v4.py   (cert/RAD/routes_update_v4.sh: every 2 min)"""
import json
import re
import time
from pathlib import Path

P = Path('cert/RAD/ROUTES.json')
r = json.loads(P.read_text())
eng = [l for l in Path('../TEAM_LOG.md').read_text().splitlines() if l.startswith('[engineer] ')]
ADD = {'bscale_0': ('bscale', [[0, 0], [0, 15]]), 'X1_14': ('X1', [[14, 14], [0, 15]]), 'k22_14': ('k22', [[14, 14]]),
       'x2a1_7': ('x2a1', [[7, 7], [1, 6], [8, 10], [11, 14], [0, 15]])}
done, vetoed, rejected = set(r.get('applied_parity', [])), r.get('vetoed', {}), r.get('rejected_b_prime', {})
changed, lines = [], []
for key, (fam, bands) in ADD.items():
    if key in done or key in vetoed or key in rejected:
        continue
    hits = [l for l in eng if re.match(r'\[engineer\] \S+ ROUTE PARITY %s: PASSED' % re.escape(key), l)]
    if not hits:
        continue
    nets = re.findall(r'(\d{7}): corr ([+-]?[0-9.]+), native/torch size ([0-9.]+)', hits[-1])
    D = {n: round(2 * float(s) * float(c) - float(s) ** 2, 3) for n, c, s in nets}
    if len(D) >= 2 and all(v >= 0.5 for v in D.values()):
        r.setdefault('bands', {})[fam] = bands
        r['all_only'] = [f for f in r['all_only'] if f != fam]
        r['per_layer'] = [f for f in r['per_layer'] if f != fam]
        done.add(key); changed.append('%s -> bands %s' % (fam, bands)); lines.append('%s ADDED (D %s)' % (key, D))
    else:
        rejected[key] = dict(D=D, line=hits[-1][:60], why='rule (b\'): D >= 0.5 on at least two networks required')
        changed.append('%s rejected' % key); lines.append('%s NOT ADDED by rule (b\') (D %s)' % (key, D))
if changed:
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    r['applied_parity'] = sorted(done); r['rejected_b_prime'] = rejected; r['utc'] = stamp
    P.rename(P.with_name('ROUTES.json.archived_%s' % stamp.replace(':', '').replace('-', '')))
    P.write_text(json.dumps(r, indent=1))
    open('../TEAM_LOG.md', 'a').write('[trainer] %s ROUTES.json UPDATED by rule (b\') (cert/RAD/routes_update_v4.py; D = 2 s c - s^2 per parity network, '
                                      '>= 0.5 required on both): %s; rounds pick it up at their next arm stage.\n' % (stamp, '; '.join(lines)))
    print('updated:', lines)
