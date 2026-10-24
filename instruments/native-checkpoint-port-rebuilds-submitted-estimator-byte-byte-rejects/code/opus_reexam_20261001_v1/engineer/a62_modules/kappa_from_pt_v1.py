"""TAG opus_engineer a62 (file written 2026-10-12 03:53Z; S50 arm fkappa_1_14): the 'kappa' family of a trainer parameter file ->
fragment_kappa_<name>.py, read WITHOUT torch (a54_jmiv/read_pt.py's loader). Torch: kappa_l = 1 + eps['kappa'][l] in the step's birth
weight at every layer whose step forms births (0-14; the last layer forms none). Native: _KAPPA = {l: 1 + eps} for the layers with a
non-zero shift (make_kappa_v1.py's edited birth line). Writes `_KAPPA = {}` when the file has no non-zero 'kappa'.
usage: kappa_from_pt_v1.py <params.pt> <name>"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
pt, name = sys.argv[1], sys.argv[2]
out = HERE/f'fragment_kappa_{name}.py'
assert not out.exists(), out
a = ns['load'](pt)
eps = np.asarray(a.get('pl', {}).get('kappa', np.zeros(16)), dtype=np.float64).ravel()
assert eps.size == 16, eps.shape
assert eps[15] == 0, 'kappa shift at layer 15 (no births there): %r' % eps[15]
kappa = {l: 1.0 + float(eps[l]) for l in range(15) if eps[l] != 0}
print('kappa shift by layer:', np.round(eps, 6).tolist())
out.write_text('# opus-engineer a62 kappa_from_pt_v1.py: the kappa family of %s (1 + shift, layers with a non-zero shift)\n_KAPPA = %r\n' % (pt, kappa))
print('_KAPPA =', kappa)
print('written', out)
