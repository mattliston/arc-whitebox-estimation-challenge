"""TAG opus_engineer a62 (lead 19:2xZ: RAD-S48's star strength 'ssb'): writes fragment_star_beta_<name>.py from a trainer parameter file,
read WITHOUT torch (a54_jmiv/read_pt.py's loader): beta at the star's layer l (1-15) = 1 + pl['ssb'][l] (otr_m_v4.py comment STAR STRENGTH),
native index l - 1 (_SS_BETA[i] = the star moved into layer i + 1, the index _ss_factors reads from _PHYS_LAYER). The fragment is
fragment_star_beta_v1.py with its _SS_BETA literal replaced; with no 'ssb' family in the file every beta is 1 (the shipped form).
usage: star_beta_from_pt_v1.py <params.pt> <name>   -> a62_modules/fragment_star_beta_<name>.py (refuses to overwrite)"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE.parent
pt, name = sys.argv[1], sys.argv[2]
out = HERE/f'fragment_star_beta_{name}.py'
assert not out.exists()
src = open(E/'a54_jmiv/read_pt.py').read().split('\na = load(sys.argv[1])')[0]
ns = {'__name__': 'read_pt_loader'}
exec(compile(src, 'read_pt.py', 'exec'), ns)
a = ns['load'](pt)
ssb = np.asarray(a.get('pl', {}).get('ssb', np.zeros(16)), dtype=np.float64)
beta = tuple(float(1.0 + ssb[l]) for l in range(1, 16))
print('beta by the star\'s layer 1..15:', ' '.join('%d:%.6g' % (l, b) for l, b in zip(range(1, 16), beta)), '| layer 0 entry (unused):', float(ssb[0]))
text = (HERE/'fragment_star_beta_v1.py').read_text()
old = '_SS_BETA = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)\n'
assert text.count(old) == 1
text = text.replace(old, '_SS_BETA = %r   # from %s (pl ssb: beta = 1 + ssb[l], native index l - 1)\n' % (beta, pt))
compile(text, str(out), 'exec')
out.write_text(text)
print('written', out)
