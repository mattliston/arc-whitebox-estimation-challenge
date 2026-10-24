"""Decision table: for each candidate name, pool paired evidence (Mac32 compare JSON, VM store scorer JSON, fresh12 scorer JSONs)
and project hosted score and wall margin against #337506 (hosted 1.4092527407367254e-9; slowest public wall 114.9 s; residual max 0.364 s).
usage: decision_table_v1.py NAME:MAC.json:STORE.json:FRESH_o8.json:FRESH_e4.json[:PAIRED_WALL%[:PAIRED_RESID%]] [...]   (use - for a missing file; paired wall/resid override the store JSON's load-confounded wall)
"""
import json, sys, statistics, math
HOSTED = 1.4092527407367254e-9; MAXWALL = 114.9; RESID = 0.364; MACRATIO = 1.0024
def load(p): return None if p == '-' else json.load(open(p))
print('%-16s %8s %8s %8s %8s | %9s %7s %7s | %10s %7s %7s' % ('candidate', 'mac32', 'vm32', 'fresh12', 'pooled', 'proj_host', 'wall%', 'resid%', 'proj_max_s', 'burst12', 'burst15'))
for spec in sys.argv[1:]:
    parts = spec.split(':'); name, mac, store, fo8, fe4 = parts[:5]; pw = float(parts[5]) if len(parts) > 5 else None; pr = float(parts[6]) if len(parts) > 6 else None
    mac, store, fo8, fe4 = map(load, (mac, store, fo8, fe4))
    vals = []; wts = []
    m = mac['all32']['adjusted_pct'] if mac else None
    v = store['all']['mean'] if store else None
    fr = None
    if fo8 and fe4:
        rows = fo8['rows'] + fe4['rows']; fr = statistics.mean(r['adjusted_pct'] for r in rows)
    for x, w in ((m, 32), (v, 32), (fr, 12)):
        if x is not None: vals.append(x * w); wts.append(w)
    pooled = sum(vals) / sum(wts) if wts else float('nan')
    wall = pw if pw is not None else (store['wall']['mean'] if store else 0.0); resid = pr if pr is not None else (store['residual']['mean'] if store else 0.0)
    proj = HOSTED * (1 + pooled / 100); maxs = MAXWALL * (1 + wall / 100)
    f = lambda x: 'n/a' if x is None else '%+.3f' % x
    print('%-16s %8s %8s %8s %+8.3f | %9.4e %+7.2f %+7.2f | %10.1f %7s %7s' % (name, f(m), f(v), f(fr), pooled, proj, wall, resid, maxs, 'PASS' if maxs * 1.12 < 120 else 'FAIL', 'PASS' if maxs * 1.15 < 120 else 'FAIL'))
