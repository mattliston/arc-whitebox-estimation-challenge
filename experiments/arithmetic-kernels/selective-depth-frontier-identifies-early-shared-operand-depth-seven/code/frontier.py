#!/usr/bin/env python3
"""TAG fable-engineer2 (lead 2026-09-25: selective depth-7 frontier + depth-5 padding lever): prices every benched class move from
bench_frontier.py rows and the S27 shape census (calls per net on nets 35 and 100, leaf_products.py), converts added backend to platform
wall by WALL v3 (engineer2's project_wall_v3 law: platform kernel % = r x local % on the worst MLP's kernel Kw, plus op cost x added ops;
r 0.60/0.78/0.95, op 0.30/0.41/0.50 ms), ranks by bill saved per platform second, and prints the cumulative frontier against the
projected platform wall max (S27 #332295: actual max 77.5 s; fitted expected worst 73.7 s; Kw 66.62 s) up to 90 s.
usage: frontier.py <bench.jsonl> [lever filter: d7|p6|all] [base net for the backend denominator, default 100]"""
import json, sys
from pathlib import Path
D = Path(__file__).resolve().parent; B27 = D.parent / 'bill27'
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
lever = sys.argv[2] if len(sys.argv) > 2 else 'all'; bnet = int(sys.argv[3]) if len(sys.argv) > 3 else 100
base = {r['base_net']: r for r in rows if 'base_net' in r}
cls = [r for r in rows if 'kind' in r]
KW, MAX27, EW27, CAP = 66.62, 77.5, 73.7, 90.0
R = (0.60, 0.78, 0.95); OPC = (0.30e-3, 0.41e-3, 0.50e-3)
calls = {}
for net in (35, 100):
    for p in json.load(open(B27 / f'leaf_products_S27_n{net}.json'))['products']:
        if p['P'] != 1 or p['fn'] != '_ipt_leaf_unwind_ab': continue
        c = calls.setdefault((p['m'], p['k'], p['n'], p['L']), {35: 0, 100: 0, 'comp': set()})
        c[net] += p['calls']; c['comp'].add(p['component'])
bill = {n: base[n]['flops'] for n in base}; bk = base[bnet]['backend_s']
print(f"base backend per net: " + ", ".join(f"net {n} {base[n]['backend_s']:.2f} s ({base[n]['ops']} ops, {base[n]['flops']/2**31:.2f} U)" for n in base) + f"; denominator net {bnet}")
items = []
for r in cls:
    if 'skip' in r or (lever != 'all' and r['kind'] != lever): continue
    m, k, n = r['dims']; Ln = 6 if r['kind'] == 'd7' else 5
    c = calls.get((m, k, n, Ln))
    if c is None: continue
    per = (c[35] + c[100]) / 2
    d = r['delta']
    dbill = d['bill'] * per; dback = d['backend_s'] * per; dops = d['ops'] * per
    plat = [rr * KW * dback / bk + oc * dops for rr, oc in ((R[2], OPC[2]), (R[1], OPC[1]), (R[0], OPC[0]))]   # pess, central, opt
    items.append(dict(kind=r['kind'], dims=(m, k, n), comp='+'.join(sorted(c['comp'])), calls=per, dbill=dbill, dbill_pct=100 * dbill / bill[bnet],
                      dback=dback, dops=dops, plat=plat, err=d['err_ratio'], bill_rel=d['bill_rel'], back_rel=d['backend_rel'],
                      rate=(-100 * dbill / bill[bnet]) / plat[1] if plat[1] > 0 else float('inf')))
items.sort(key=lambda x: -x['rate'])
print(f"\n{'lever':4s} {'m':>5s} {'k':>5s} {'n':>5s} {'calls':>5s} {'bill/call':>9s} {'backend/call':>12s} {'bill %':>7s} {'wall s (pess/cent/opt)':>24s} {'%/s':>6s} {'err x':>5s}  component")
cum = [0.0, 0.0, 0.0]; cb = 0.0; rowsout = []
for x in items:
    if x['dbill'] >= 0: continue
    cum = [a + b for a, b in zip(cum, x['plat'])]; cb += x['dbill_pct']
    x['cum_bill_pct'] = cb; x['cum_plat'] = list(cum); rowsout.append(x)
    print(f"{x['kind']:4s} {x['dims'][0]:5d} {x['dims'][1]:5d} {x['dims'][2]:5d} {x['calls']:5.1f} {100*x['bill_rel']:+8.2f}% {100*x['back_rel']:+11.0f}% {x['dbill_pct']:+7.3f} "
          f"{x['plat'][0]:+7.2f}/{x['plat'][1]:+6.2f}/{x['plat'][2]:+6.2f} {x['rate']:6.3f} {x['err']:5.2f}  {x['comp']}   CUM {cb:+.3f}% {cum[1]:+.2f} s -> max {MAX27+cum[1]:.1f} s")
print("\nFRONTIER (central wall): cumulative bill vs projected platform max (S27 max 77.5 s; fitted expected worst 73.7 s)")
for lim in (1, 2, 3, 5, 7.5, 10, 12.5):
    sel = [x for x in rowsout if x['cum_plat'][1] <= lim]
    if sel:
        x = sel[-1]; print(f"  <= +{lim:4.1f} s: {len(sel):2d} classes, bill {x['cum_bill_pct']:+.3f}%, wall +{x['cum_plat'][0]:.2f}/+{x['cum_plat'][1]:.2f}/+{x['cum_plat'][2]:.2f} s -> max {MAX27+x['cum_plat'][1]:.1f} s")
neg = [x for x in items if x['dbill'] >= 0]
if neg: print("\nclasses that LOSE bill: " + ", ".join(f"{x['kind']} {x['dims']} {100*x['bill_rel']:+.2f}%" for x in neg))
