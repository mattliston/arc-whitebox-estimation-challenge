"""[opus-small] build fragment smallsm_v2 against the shared SMX API of OPUS-PUSHSM pushsm_v2: register three tight families, route
join / old-bank _shared_old384_mm products by an exact (m,k,n) rule from the flopscope-tariff grids (Smirnov family cost <= (1-MARGIN)
x incumbent cost, padding counted), validate on the 8 logged networks, and assemble v4g + pushsm_v2 + smallsm_v2 (+ parity copy)."""
import json, glob, sys, hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'); S = O/'lead/runs/opus_small_v1'
MARGIN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
FAMS = {'osm_e1_L5': ('osm_e1_L5', 5, 10, 11, 38, 4), 'osm_e0_L5': ('osm_e0_L5', 5, 10, 11, 11, 38), 'osm_e0_L4': ('osm_e0_L4', 4, 40, 22, 8, 4)}
FORMS = {'xxR': ('signed_executable_1', 'osm_e1_L5', 5, 10), 'Rxx': ('signed_executable_0', 'osm_e0_L5', 5, 10),
         'xRR': ('signed_executable_0', 'osm_e0_L4', 4, 40),
         'RRR': ('signed_executable_0', 'osm_e0_L4', 4, 40)}
inc = json.load(open(S/'GRID_inc_shard0.json')); rule, per, smc = {}, {}, {}
for form, (rec, fam, L, P) in FORMS.items():
    f = S/f'GRID_sm_{form}_{rec}_L{L}_P{P}.json'
    if not f.exists(): print('missing grid', f.name); continue
    sm = json.load(open(f)); n = 0
    for key, c in sm.items():
        if c is None or key not in inc: continue
        s = tuple(map(int, key.split(','))); smc[s] = c
        if c <= (1 - MARGIN) * inc[key]: rule[s] = (rec, fam); n += 1
    per[form] = (n, len(sm))
print('MARGIN', MARGIN, 'routed shapes per form', per, 'total', len(rule))
tot, sav = {}, {}
for g in sorted(glob.glob(str(S/'shapes/shapes_*.json'))):
    d = json.load(open(g)); tot[d['seed']] = d['total']; sv = 0
    for r in d['rows']:
        a, b = r['a'], r['b']
        if r['entry'] != '_shared_old384_mm' or len(a) != 2 or len(b) != 2: continue
        s = (a[0], a[1], b[1]) if a[1] == b[0] else (a[0], a[1], b[0])
        if s in rule: sv += r['flops'] - smc[s]
    sav[d['seed']] = sv
print('rule on the 8 logged nets (live incumbent): saving % of warm bill', {k: round(100 * v / tot[k], 3) for k, v in sav.items()},
      'mean', round(sum(100 * v / tot[k] for k, v in sav.items()) / len(sav), 3))
frag = f'''

# BEGIN OPUS-SMALL FRAGMENT smallsm_v2 (2026-10-16; append LAST, after OPUS-PUSHSM pushsm_v2, on GOV v4g 805c3598)
# JOIN and OLD-BANK products through the shared Smirnov dispatch (_SMX_*, OPUS-PUSHSM pushsm_v2): _shared_old384_mm calls made outside
# any UBX / cluster context are routed to _smx_call exactly where _SMALLSM_RULE lists their (m, k, n). The rule is an exact table priced
# with flopscope's own tariff (padding counted) over every join / old-bank form a network can produce (rows 480..1024, widths
# 3072..7168, step 32, and the rank-384 squares): a shape is routed when its family's cost is <= {1 - MARGIN:.2f} x the incumbent's.
# Families (capacities in units of 2^L): {', '.join(f'{v}' for v in FAMS.values())}. Same real arithmetic, rounding-level change.
# _SMX_ON (governor EXACT set) is checked at call time; anything not routed runs the parent unchanged.
assert callable(_smx_call) and callable(_smx_fits) and callable(_smx_register) and '_SMX_ON' in globals()
_SMALLSM_FAMILIES = {repr(FAMS)}
for _osm_f in _SMALLSM_FAMILIES.values():
    _smx_register(_osm_f)
del _osm_f
_SMALLSM_RULE = {repr(rule)}
_SMALLSM_STATS = dict(routed=0)
_SMALLSM_PARENT = _shared_old384_mm


def _shared_old384_mm(left, right):
    if (_SMX_ON and _AB_ON and _WINOGRAD and _SCRATCH is not None and _UBX_CTX[0] is None and _ASTRA_CLUSTER_CTX[0] is None
            and len(left.shape) == len(right.shape) == 2 and str(left.dtype) == str(right.dtype) == 'float32'):
        m, k, n = int(left.shape[0]), int(left.shape[1]), int(right.shape[1])
        hit = _SMALLSM_RULE.get((m, k, n))
        if hit is not None and _smx_fits(hit[0], hit[1], m, k, n):
            _SMALLSM_STATS['routed'] += 1
            return _smx_call(hit[0], hit[1], left, right)
    return _SMALLSM_PARENT(left, right)
# END OPUS-SMALL FRAGMENT smallsm_v2
'''
fp = S/'fragment_smallsm_v2.py.txt'; fp.write_text(frag); print('fragment', fp.name, hashlib.sha256(frag.encode()).hexdigest()[:12], len(frag), 'bytes')
v4g = (O/'lead/runs/opus_governor_v1/candidate_GOV_v4g.py').read_text(); assert hashlib.sha256(v4g.encode()).hexdigest().startswith('805c3598')
par = (O/'lead/runs/opus_governor_v1/test_parity_v4g.py').read_text(); assert hashlib.sha256(par.encode()).hexdigest().startswith('99713daa')
psm = (O/'lead/runs/opus_pushsm_v1/fragment_pushsm_v2.py.txt').read_text()
for name, base in (('cand_v4g_psm2', v4g), ('parity_v4g_psm2', par)):
    for tag, text in ((name, base + psm), (name + '_osm2', base + psm + frag)):
        (S/f'{tag}.py').write_text(text); print(tag, hashlib.sha256(text.encode()).hexdigest()[:12])
print('pushsm_v2 sha', hashlib.sha256(psm.encode()).hexdigest()[:12])
