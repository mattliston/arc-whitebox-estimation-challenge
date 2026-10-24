"""TAG opus_engineer a62 (2026-10-11 21:0xZ, before RAD-S48's port): a trainer checkpoint's per-layer families and dials, read WITHOUT
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
PER_LAYER_OK = {'iso', 'bscale', 'X1', 's_carry', 'k4la1', 'kc', 'nfs', 'ssb', 'edx', 'nkl1', 'nkl2', 'nkl3'}   # edx: fragment_early_edge_v1 + edx_from_pt_v1; nkl*: fragment_k3l_bylayer_v1 + k3l_from_pt_v1
a = ns['load'](sys.argv[1])
r = ns['load'](sys.argv[2]) if len(sys.argv) > 2 else None
pl = {k: np.asarray(v, dtype=float).ravel() for k, v in a.get('pl', {}).items()}
rl = {k: np.asarray(v, dtype=float).ravel() for k, v in r.get('pl', {}).items()} if r else {}
bad = []
for k in sorted(pl):
    v = pl[k]
    varies = v.size >= 15 and float(np.ptp(v[1:15])) > 0
    flag = 'NEW ' if r and k not in rl else ''
    if varies and k not in PER_LAYER_OK:
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
