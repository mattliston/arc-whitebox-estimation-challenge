"""TAG opus_engineer a62 (file written 2026-10-12 17:14Z): the x1a1 / x1a2 / x2a1 / x2a2 families of a trainer parameter file ->
fragment_x2av_<name>.py for make_x2a_v1.py, read WITHOUT torch (read_pt.py). The exporter writes the native _F4C_C from the LAYER-1 values
(export_dials_s46_v*: exact_S1 / exact_S2 per age = pl[family][1]); every birth j = 0..13 whose torch strengths (age a read at layer
j + a <= 14) differ from those gets its own entry ((S1, S2) age 1, (S1, S2) age 2), so the file matches torch at every birth whatever layer
the flat value came from; '{}' when all are flat. Values with %.12g, as the builder writes the literal.
usage: x2a_from_pt_v1.py <params.pt> <name>"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
pt, name = sys.argv[1], sys.argv[2]
out = HERE/f'fragment_x2av_{name}.py'
assert not out.exists(), out
pl = ns['load'](pt).get('pl', {})
eps = {f: np.asarray(pl.get(f, np.zeros(16)), dtype=np.float64).ravel() for f in ('x1a1', 'x1a2', 'x2a1', 'x2a2')}
g = lambda x: float('%.12g' % x)
literal = tuple((g(eps['x1a%d' % a][1]), g(eps['x2a%d' % a][1])) for a in (1, 2))
table = {}
for j in range(14):
    st = tuple((g(eps['x1a%d' % a][j + a]), g(eps['x2a%d' % a][j + a])) if j + a <= 14 else literal[a - 1] for a in (1, 2))
    if st != literal:
        table[j] = st
for f in sorted(eps):
    print('%s by layer:' % f, np.round(eps[f], 6).tolist())
out.write_text('# opus-engineer a62 x2a_from_pt_v1.py: F4c exact-term strengths by birth of %s (literal = layer 1: %r)\n_F4C_C_BY_BIRTH = %r\n' % (pt, literal, table))
print('literal (layer 1):', literal)
print('_F4C_C_BY_BIRTH =', table)
print('written', out)
