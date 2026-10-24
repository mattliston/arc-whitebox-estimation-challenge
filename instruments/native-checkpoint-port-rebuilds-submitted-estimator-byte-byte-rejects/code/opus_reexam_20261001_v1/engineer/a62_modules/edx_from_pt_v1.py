"""TAG opus_engineer a62 (2026-10-11 21:2xZ, for RAD-S48's port): the early edge's per-layer scales from a trainer parameter file, read
WITHOUT torch (a54_jmiv/read_pt.py's loader): scale at layer l = EDGEXS[l] + pl['edx'][l] (otr_m_v5.py comment EARLY EDGE; EDGEXS is the
zero leaf of the gradient runs, taken from the file when it carries one under 'exs'), applied natively only at layers 1-7 (the request's
edgex [1, 7]; the torch card skips 8-14 and layer 0 for it). Writes fragment_edx_<name>.py = `_EDX = {l: scale}` (non-zero layers only),
appended AFTER fragment_early_edge_v1.py. Refuses a non-zero scale outside 1-7 and an existing output.
usage: edx_from_pt_v1.py <params.pt> <name>"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt.py', 'exec'), ns)
pt, name = sys.argv[1], sys.argv[2]
out = HERE/f'fragment_edx_{name}.py'
assert not out.exists(), out
a = ns['load'](pt)
pl = a.get('pl', {})
scale = np.zeros(16)
if 'edx' in pl:
    scale += np.asarray(pl['edx'], dtype=float).ravel()
if 'exs' in a:
    scale += np.asarray(a['exs'], dtype=float).ravel()
bad = [l for l in range(16) if scale[l] != 0 and not 1 <= l <= 7]
assert not bad, 'non-zero early-edge scale outside layers 1-7: %s' % bad
values = {l: float(scale[l]) for l in range(1, 8) if scale[l] != 0}
print('early edge scales by layer:', ' '.join('%d:%.6g' % (l, scale[l]) for l in range(1, 8)) or 'none',
      '| sources:', ', '.join(k for k in ('edx', 'exs') if k in pl or k in a) or 'none (all zero)')
out.write_text('# opus-engineer a62 edx_from_pt_v1.py: early-edge scales of %s (EDGEXS + pl edx, layers 1-7)\n_EDX = %r\n' % (pt, values))
print('written', out)
