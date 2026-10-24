"""TAG opus_engineer a62 v3 (2026-10-12 00:1xZ) = v2 with 'k4' routed (fragment_k4_v1.py + k4_from_pt_v1.py: band shifts constant inside
1-7 / 8-11 / 12-14, layer 15 by the terminal wrapper). v2 header: (2026-10-11 23:4xZ, S50's arms) = v1 + every family must have a KNOWN ROUTE at all (not only the per-layer
ones): a family outside ROUTED with any non-zero value is flagged (S50's 'k4' has no exporter form; trainer/export_dials.py would
assert, and a silent drop here would lose it). v1 header: (2026-10-11 21:0xZ, before RAD-S48's port): a trainer checkpoint's per-layer families and dials, read WITHOUT
torch (a54_jmiv/read_pt.py's loader), against a reference checkpoint. Flags every family the native port does not carry by a known
route: per-layer variation over layers 1-14 is carried only for iso / bscale / X1 / s_carry (DLY@ from the .pt), k4la1 (DLY@ from
physics_dials.json), kc (KC@), nfs (the law's per-layer strengths) and ssb (fragment_star_beta); any other family must be constant over
layers 1-14 (the exporter's one shift). Families absent from the reference are listed as NEW.
usage: pt_families_v1.py <params.pt> [<reference params.pt>]"""
import sys
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parents[1]
src = (E/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
PER_LAYER_OK = {'iso', 'bscale', 'X1', 's_carry', 'k4la1', 'kc', 'nfs', 'ssb', 'edx', 'nkl1', 'nkl2', 'nkl3'}
ROUTED = PER_LAYER_OK | {'X0', 'X2', 's_new', 'c0', 'c1', 'rc', 'edge', 'bsrc', 'k21n', 'k31', 'ds', 'f4a1', 'f4a2', 'k4la2', 'x1a1', 'x1a2',
                         'x2a1', 'x2a2', 'nfd', 'n6b', 'n6r', 'n6d', 'k4'}   # the exporter's forms (export_dials.py) + this lane's fragments (k4: fragment_k4_v1); NOT e2   # edx: fragment_early_edge_v1 + edx_from_pt_v1; nkl*: fragment_k3l_bylayer_v1 + k3l_from_pt_v1
a = ns['load'](sys.argv[1])
r = ns['load'](sys.argv[2]) if len(sys.argv) > 2 else None
pl = {k: np.asarray(v, dtype=float).ravel() for k, v in a.get('pl', {}).items()}
rl = {k: np.asarray(v, dtype=float).ravel() for k, v in r.get('pl', {}).items()} if r else {}
bad = []
for k in sorted(pl):
    v = pl[k]
    varies = v.size >= 15 and float(np.ptp(v[1:15])) > 0
    flag = 'NEW ' if r and k not in rl else ''
    if k not in ROUTED and np.any(v != 0):
        flag += 'NO NATIVE ROUTE (non-zero) '
        bad.append(k)
    elif varies and k not in PER_LAYER_OK:
        flag += 'PER-LAYER, NO NATIVE ROUTE '
        bad.append(k)
    moved = '' if k not in rl else ' max |change| %.3g' % float(np.max(np.abs(v - rl[k])))
    print('%-8s layers %s%s %s%s' % (k, 'vary' if varies else 'flat', moved, np.round(v[:3], 4), (' <- ' + flag) if flag else ''))
for k in sorted(set(rl) - set(pl)):
    print('%-8s MISSING here (in the reference)' % k)
d = {k: float(v) for k, v in a['dials'].items()}
rd = {k: float(v) for k, v in r['dials'].items()} if r else {}
print('dials', ' '.join('%s %.4f%s' % (k, v, '' if k not in rd else ' (%+.4f)' % (v - rd[k])) for k, v in sorted(d.items())))
print('array families', len(a.get('arrays', {})), '' if not r else '(reference %d)' % len(r.get('arrays', {})))
print('VERDICT', 'FLAG ' + ', '.join(bad) if bad else 'every per-layer family has a native route')
