"""TAG opus_engineer a62 (2026-10-11 21:3xZ, for RAD-S48's port): the young pairs' fourth-chaos strengths by cohort age AND read layer from a
trainer parameter file, read WITHOUT torch (a54_jmiv/read_pt.py's loader), by the exporter's rule taken at every layer l = 1..14 instead of
layer 1 only: (age 1, age 2, age 3) = (nkl1 + 1 + nkl2 + nkl3, 1 + nkl2 + nkl3, nkl3) at l (law_literals_from_pt_v1.py's rule). Layer 1's
tuple is what the law converter writes as _NP_K3L; this writes fragment_k3l_<name>.py = `_NP_K3L_BY_LAYER = {l: tuple}` for the layers
whose tuple differs from layer 1's (appended AFTER fragment_k3l_bylayer_v1.py). Trailing ages that are 0 at EVERY layer are dropped as
the law converter drops them; the count must then match.
usage: k3l_from_pt_v1.py <params.pt> <name>"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
pt, name = sys.argv[1], sys.argv[2]
out = HERE/f'fragment_k3l_{name}.py'
assert not out.exists(), out
a = ns['load'](pt)
pl = a.get('pl', {})
get = lambda f: [float(x) for x in np.asarray(pl.get(f, np.zeros(16)), dtype=np.float64).ravel()]
k1, k2, k3 = get('nkl1'), get('nkl2'), get('nkl3')
table = {l: [k1[l] + 1.0 + k2[l] + k3[l], 1.0 + k2[l] + k3[l], k3[l]] for l in range(1, 15)}
count = 3
while count and all(not table[l][count - 1] for l in table):
    count -= 1
table = {l: tuple(v[:count]) for l, v in table.items()}
base = table[1]
diff = {l: v for l, v in table.items() if v != base}
for l in range(1, 15):
    print('layer %2d  ages %s%s' % (l, ' '.join('%.6f' % x for x in table[l]), '' if l not in diff else '  <- differs from layer 1'))
out.write_text('# opus-engineer a62 k3l_from_pt_v1.py: K3L strengths by read layer of %s (layer 1 = _NP_K3L %r)\n_NP_K3L_BY_LAYER = %r\n'
               % (pt, base, diff))
print('written', out, '(%d layers differ)' % len(diff))
