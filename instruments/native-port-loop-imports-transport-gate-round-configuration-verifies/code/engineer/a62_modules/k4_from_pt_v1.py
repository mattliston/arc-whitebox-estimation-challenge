"""TAG opus_engineer a62 (2026-10-12 00:1xZ; S50 arm fk4): the 'k4' family of a trainer parameter file -> fragment_k4_<name>.py, read WITHOUT
torch (a54_jmiv/read_pt.py's loader). Torch (otr_pl.py install): layers 1-14 scale the radial k4 by band(l) + eps['k4'][l], band = the dials
k4a (1-7) / k4b (8-11) / k4c (12-14); layer 15 by 1 + eps['k4'][15] (the 'k4_15' hook). Native: _S19_K4BAND = (k4a + shift, k4b + shift,
k4c + shift) with the shift required constant inside each band, and _K4_15 = eps['k4'][15] for fragment_k4_v1.py's terminal wrapper.
Writes `_K4_FAMILY = False` (nothing to carry) when the file has no non-zero 'k4'.
usage: k4_from_pt_v1.py <params.pt> <name>"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
pt, name = sys.argv[1], sys.argv[2]
out = HERE/f'fragment_k4_{name}.py'
assert not out.exists(), out
a = ns['load'](pt)
k4 = np.asarray(a.get('pl', {}).get('k4', np.zeros(16)), dtype=np.float64).ravel()
dials = {k: float(v) for k, v in a['dials'].items()}
if not np.any(k4[1:] != 0):
    out.write_text('# opus-engineer a62 k4_from_pt_v1.py: no k4 shift in %s\n_K4_FAMILY = False\n' % pt)
    print('no k4 shift; written', out)
    sys.exit(0)
bands = ((1, 7), (8, 11), (12, 14))
for lo, hi in bands:
    assert np.all(k4[lo:hi + 1] == k4[lo]), 'k4 shift not constant inside band %d-%d: %s' % (lo, hi, k4[lo:hi + 1])
band = tuple(dials[d] + float(k4[lo]) for d, (lo, hi) in zip(('k4a', 'k4b', 'k4c'), bands))
print('k4 shift by band (1-7, 8-11, 12-14):', [float(k4[lo]) for lo, hi in bands], 'layer 15:', float(k4[15]), '| layer 0 (unused):', float(k4[0]))
print('_S19_K4BAND =', band)
out.write_text('# opus-engineer a62 k4_from_pt_v1.py: the k4 family of %s (dials k4a / k4b / k4c + the band shifts; layer 15 = _K4_15)\n'
               '_K4_FAMILY = True\n_S19_K4BAND = %r\n_K4_15 = %r\n' % (pt, band, float(k4[15])))
print('written', out)
