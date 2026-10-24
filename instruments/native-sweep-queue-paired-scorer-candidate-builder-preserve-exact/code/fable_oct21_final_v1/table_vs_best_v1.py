"""Candidate table vs our CURRENT BEST (#337565 = extgate_s15_cap3, hosted 1.4036387e-9). Local paired numbers are measured
vs the #337506 reference; this subtracts the fired file's own local numbers per set and projects hosted from the held25-pooled
delta (hosted/held ratio calibrated on #337565: -0.40% hosted for -0.36% held25-pooled).
usage: table_vs_best_v1.py NAME [NAME ...]   (reads MAC32_NAME_vs_49bd_v1.json locally, STORE/FRESH json fetched into vmjson/)
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; VJ = HERE / 'vmjson'
BEST = 1.4036387231743654e-9; REF506 = 1.4092527407367254e-9
FIRED = dict(mac=(-0.104, -0.393), vm=(-0.212, -0.330), fresh=-0.272)   # extgate_s15_cap3 local numbers vs #337506
def mac(n):
    p = HERE / ('MAC32_%s_vs_49bd_v1.json' % n)
    if not p.exists(): return None
    d = json.load(open(p)); return (d['all32']['adjusted_pct'], d['heldout25']['adjusted_pct'])
def vm(n):
    p = VJ / ('STORE_%s_vs_49bd_v1.json' % n)
    if not p.exists(): return None
    d = json.load(open(p)); return (d['all']['mean'], d['held25']['mean'])
def fresh(n):
    rows = []
    for c in ('original8', 'extension4'):
        p = VJ / ('FRESH_%s_%s_vs_49bd_v1.json' % (c, n))
        if p.exists(): rows += json.load(open(p))['rows']
    return (sum(r['adjusted_pct'] for r in rows) / len(rows), len(rows)) if rows else None
print('%-24s %16s %16s %12s | %10s %10s | %12s %9s' % ('candidate', 'mac32 all/held', 'vm32 all/held', 'fresh12', 'd_all', 'd_held', 'proj_hosted', 'vs_best'))
for n in sys.argv[1:]:
    m, v, f = mac(n), vm(n), fresh(n)
    da, dh, wa, wh = 0.0, 0.0, 0, 0
    if m: da += (m[0] - FIRED['mac'][0]) * 32; dh += (m[1] - FIRED['mac'][1]) * 25; wa += 32; wh += 25
    if v: da += (v[0] - FIRED['vm'][0]) * 32; dh += (v[1] - FIRED['vm'][1]) * 25; wa += 32; wh += 25
    if f: da += (f[0] - FIRED['fresh']) * f[1]; dh += (f[0] - FIRED['fresh']) * f[1]; wa += f[1]; wh += f[1]
    da = da / wa if wa else float('nan'); dh = dh / wh if wh else float('nan')
    proj = BEST * (1 + dh / 100 * (0.40 / 0.36)) if wh else float('nan')
    fm = '%+.2f/%+.2f' % m if m else '-'; fv = '%+.2f/%+.2f' % v if v else '-'; ff = ('%+.2f (n%d)' % f) if f else '-'
    print('%-24s %16s %16s %12s | %+9.2f%% %+9.2f%% | %12.4e %+8.2f%%' % (n, fm, fv, ff, da, dh, proj, (proj / BEST - 1) * 100))
