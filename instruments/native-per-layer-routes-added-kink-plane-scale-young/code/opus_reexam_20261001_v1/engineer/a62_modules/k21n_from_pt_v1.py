"""TAG opus_engineer a62 (file written 2026-10-12 07:11Z): the 'k21n' family of a trainer parameter file -> fragment_k21nv_<name>.py for fragment_k21n_v1.py,
read WITHOUT torch (a54_jmiv/read_pt.py's loader). Torch: k21n(l) = V3['k21n'] + eps['k21n'][l] at layers 2-14 (cov_pm acts there only).
Writes _K21N_BY_LAYER = {l: base + eps[l], l = 2..14} when the shift VARIES over layers 2-14 (a flat shift is the exporter's literal);
'_K21N_BY_LAYER = {}' otherwise. --base <V3 k21n> (default -0.028, export_dials.py BASE); --check-literal <candidate.py>: with a flat
shift, base + shift must equal the file's _GN_KNOBS['k21n'] (the self-test on a point the exporter wrote, e.g. S50's -0.048).
usage: k21n_from_pt_v1.py <params.pt> <name> [--base B] [--check-literal <candidate.py>]"""
import re
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
args = sys.argv[1:]
pt, name = args[0], args[1]
base = float(args[args.index('--base') + 1]) if '--base' in args else -0.028
lit = args[args.index('--check-literal') + 1] if '--check-literal' in args else None
a = ns['load'](pt)
eps = np.asarray(a.get('pl', {}).get('k21n', np.zeros(16)), dtype=np.float64).ravel()
assert eps.size == 16, eps.shape
print('k21n shift by layer:', np.round(eps, 6).tolist(), '| base', base)
varies = float(np.ptp(eps[2:15])) > 0
if lit is not None and not varies:
    m = re.findall(r"_GN_KNOBS = \{[^}]*'k21n': ([-0-9.e]+)", Path(lit).read_text())
    assert m, 'no _GN_KNOBS literal in %s' % lit
    got, want = float(m[-1]), base + float(eps[2])
    print('literal check: file k21n %r vs base + shift %r -> %s' % (got, want, 'SAME' if abs(got - want) < 1e-12 else 'DIFFERENT'))
    assert abs(got - want) < 1e-12
table = {l: base + float(eps[l]) for l in range(2, 15)} if varies else {}
out = HERE/f'fragment_k21nv_{name}.py'
assert not out.exists(), out
out.write_text('# opus-engineer a62 k21n_from_pt_v1.py: the k21n family of %s (V3 base %r + shift, layers 2-14)\n_K21N_BY_LAYER = %r\n' % (pt, base, table))
print('_K21N_BY_LAYER =', table)
print('written', out)
