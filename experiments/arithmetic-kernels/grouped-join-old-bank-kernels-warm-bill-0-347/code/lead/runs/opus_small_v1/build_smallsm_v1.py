"""[opus-small] build the test fragment (own environment, template v1b) + rule from the exact grids: route (m,k,n) to its form's family
where the Smirnov tariff cost <= (1 - MARGIN) x the incumbent _shared_old384_mm tariff cost. Validates decisions on the logged shapes."""
import json, glob, sys, hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'); S = O/'lead/runs/opus_small_v1'
MARGIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
FAMS = {('signed_executable_1', 5): ('osm_e1_L5', 5, 10, 11, 38, 4), ('signed_executable_0', 5): ('osm_e0_L5', 5, 10, 11, 11, 38),
        ('signed_executable_0', 4): ('osm_e0_L4', 4, 40, 22, 8, 4)}
FORMS = {'xxR': ('signed_executable_1', 5, 10), 'Rxx': ('signed_executable_0', 5, 10), 'xRx': ('signed_executable_0', 5, 10),
         'xRR': ('signed_executable_0', 4, 40), 'RRR': ('signed_executable_0', 4, 40)}
inc = json.load(open(S/'GRID_inc_shard0.json')); rule = {}; per = {}
for form, (name, L, P) in FORMS.items():
    f = S/f'GRID_sm_{form}_{name}_L{L}_P{P}.json'
    if not f.exists(): print('missing grid', f.name); continue
    sm = json.load(open(f)); n = 0
    for key, c in sm.items():
        if c is None or key not in inc: continue
        if c <= (1 - MARGIN) * inc[key]:
            rule[tuple(map(int, key.split(',')))] = (name, L); n += 1
    per[form] = (n, len(sm))
print('routed shapes per form', per, 'total', len(rule))
# validation on logged shapes (live incumbent cost) -> per-net saving of the rule as built
price = json.load(open(S/'PRICE_small_v1.json'))['best']; tot = {}; sav = {}
for g in sorted(glob.glob(str(S/'shapes/shapes_*.json'))):
    d = json.load(open(g)); tot[d['seed']] = d['total']; sv = 0
    for r in d['rows']:
        a, b = r['a'], r['b']
        if len(a) != 2 or len(b) != 2: continue
        s = (a[0], a[1], b[1]) if a[1] == b[0] else (a[0], a[1], b[0])
        if r['entry'] == '_shared_old384_mm' and s in rule:
            form = ''.join('R' if x == 384 else 'x' for x in s); name, L, P = FORMS[form]
            gs = json.load(open(S/f'GRID_sm_{form}_{name}_L{L}_P{P}.json'))['%d,%d,%d' % s]
            sv += r['flops'] - gs
    sav[d['seed']] = sv
print('validation on the 8 logged nets: saving % of warm bill', {k: round(100 * v / tot[k], 3) for k, v in sav.items()}, 'mean', round(sum(100 * v / tot[k] for k, v in sav.items()) / len(sav), 3))
lit = (O/'lead/runs/opus_pushsm_v1/record_sx0_literal.txt').read_text().strip()
t = (S/'fragment_smallsm_template_v1b.py.txt').read_text()
t = t.replace('__RECORDS__', 'dict(_ASTRA_SM40_RECORDS, signed_executable_0=' + lit + ')')
t = t.replace('__FAMILIES__', repr(FAMS)).replace('__RULE__', repr(rule))
out = S/'fragment_smallsm_test_v1.py.txt'; out.write_text(t); print('fragment', out.name, hashlib.sha256(t.encode()).hexdigest()[:12], len(t), 'bytes')
