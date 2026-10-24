"""TAG opus_engineer a62 (file written 2026-10-12 16:49Z): the 'k22' family of a trainer parameter file -> fragment_k22v_<name>.py for make_k22_v1.py
(_K22_BY_LAYER = {l: 1 + eps['k22'][l]} for the layers 1-14 with a non-zero shift; '{}' when none), read WITHOUT torch (read_pt.py).
usage: k22_from_pt_v1.py <params.pt> <name>"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
pt, name = sys.argv[1], sys.argv[2]
out = HERE/f'fragment_k22v_{name}.py'
assert not out.exists(), out
eps = np.asarray(ns['load'](pt).get('pl', {}).get('k22', np.zeros(16)), dtype=np.float64).ravel()
table = {l: 1.0 + float(eps[l]) for l in range(1, 15) if eps[l] != 0}
print('k22 shift by layer:', np.round(eps, 6).tolist())
out.write_text('# opus-engineer a62 k22_from_pt_v1.py: the k22 family of %s (1 + shift, layers 1-14)\n_K22_BY_LAYER = %r\n' % (pt, table))
print('_K22_BY_LAYER =', table)
print('written', out)
