"""TAG opus_engineer a62 (file written 2026-10-12 07:33Z): the 'k4la2' family of a trainer parameter file -> fragment_k4la2v_<name>.py for
fragment_k4la2_v1.py, read WITHOUT torch (a54_jmiv/read_pt.py's loader). Torch: the age-2 strength at the layer the term is applied is
K4L_C (1.0) + eps['k4la2'][l]; age-2 terms exist at layers 2-15. Writes _K4LA2_BY_LAYER = {l: 1 + eps[l], l = 2..15} when the shift VARIES over
2-15 (a flat shift is the builder lever's one age-2 literal, as before); '_K4LA2_BY_LAYER = {}' otherwise.
usage: k4la2_from_pt_v1.py <params.pt> <name>"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
pt, name = sys.argv[1], sys.argv[2]
a = ns['load'](pt)
eps = np.asarray(a.get('pl', {}).get('k4la2', np.zeros(16)), dtype=np.float64).ravel()
assert eps.size == 16, eps.shape
print('k4la2 shift by layer:', np.round(eps, 6).tolist())
table = {l: 1.0 + float(eps[l]) for l in range(2, 16)} if float(np.ptp(eps[2:16])) > 0 else {}
out = HERE/f'fragment_k4la2v_{name}.py'
assert not out.exists(), out
out.write_text('# opus-engineer a62 k4la2_from_pt_v1.py: the k4la2 family of %s (1 + shift at the layer the age-2 term is applied, 2-15)\n_K4LA2_BY_LAYER = %r\n' % (pt, table))
print('_K4LA2_BY_LAYER =', table)
print('written', out)
