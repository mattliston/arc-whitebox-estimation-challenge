"""TAG opus_engineer a62 (file written 2026-10-12 04:22Z): the 'rso' / 'rsy' families of a trainer parameter file -> fragment_rsv_<name>.py
(_RS = {layer: (1 + rso, 1 + rsy)} for the layers where either shift is non-zero), read WITHOUT torch (a54_jmiv/read_pt.py's loader), for
fragment_rs_v1.py. Writes '_RS = {}' when both families are absent or zero.
usage: rs_from_pt_v1.py <params.pt> <name>"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
pt, name = sys.argv[1], sys.argv[2]
out = HERE/f'fragment_rsv_{name}.py'
assert not out.exists(), out
a = ns['load'](pt)
pl = a.get('pl', {})
rso = np.asarray(pl.get('rso', np.zeros(16)), dtype=np.float64).ravel()
rsy = np.asarray(pl.get('rsy', np.zeros(16)), dtype=np.float64).ravel()
assert rso.size == rsy.size == 16, (rso.shape, rsy.shape)
rs = {l: (1.0 + float(rso[l]), 1.0 + float(rsy[l])) for l in range(16) if rso[l] != 0 or rsy[l] != 0}
print('rso by layer:', np.round(rso, 6).tolist())
print('rsy by layer:', np.round(rsy, 6).tolist())
out.write_text('# opus-engineer a62 rs_from_pt_v1.py: the rso / rsy families of %s ((1 + rso, 1 + rsy) by layer)\n_RS = %r\n' % (pt, rs))
print('_RS =', rs)
print('written', out)
