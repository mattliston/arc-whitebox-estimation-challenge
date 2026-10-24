"""TAG opus_engineer a62 (lead 17:1xZ: the trainer's MS round -> native literals): the pair law's native literals from a trainer parameter
file, read WITHOUT torch (a54_jmiv/read_pt.py's loader), by the same rules as trainer/export_dials.py (pair_law_dials) followed by
a38_pairlaw/dials_from_trainer.py:
  _NP_STRENGTH = (0, 1 + nfs[1..14], 0)    _NP_DSCALE = 1 + nfd (one value when layers 1-14 agree)
  _NP_BIRTHS / _NP_ROWS / _NP_DIAG = 1 + n6b[1] / n6r[1] / n6d[1]   (one value; the exporter asserts layers 1-14 agree and takes layer 1)
  _NP_K3L = (age 1, age 2, age 3) = (nkl1 + 1 + nkl2 + nkl3, 1 + nkl2 + nkl3, nkl3) at layer 1, trailing zeros dropped
With --check <native.py> it compares the result with that file's literals (the conversion of the point the file was built from must
reproduce them exactly). With --fragment <out.py> it writes the literals as an appended fragment that also rebuilds _NP_TABLES.
usage: law_literals_from_pt_v1.py <params.pt> [--check <native.py>] [--fragment <out.py> <note>]"""
import re
import sys
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parents[1]
sys.argv, args = sys.argv[:2], sys.argv[2:]
src = open(E/'a54_jmiv/read_pt.py').read().split('\na = load(sys.argv[1])')[0]
ns = {'__name__': 'read_pt_loader'}
exec(compile(src, 'read_pt.py', 'exec'), ns)
a = ns['load'](sys.argv[1])
pl = {k: np.asarray(v, dtype=np.float64) for k, v in a.get('pl', {}).items()}
get = lambda f: [float(x) for x in pl.get(f, np.zeros(16))]
for f in ('n6b', 'n6r', 'n6d', 'nkl1', 'nkl2', 'nkl3', 'nfd'):
    assert all(x == get(f)[1] for x in get(f)[1:15]), f
strength = [1.0 + x for x in get('nfs')]
factor = [1.0 + x for x in get('nfd')]
k3l = [get('nkl1')[1] + 1.0 + get('nkl2')[1] + get('nkl3')[1], 1.0 + get('nkl2')[1] + get('nkl3')[1], get('nkl3')[1]]
while k3l and not k3l[-1]:
    k3l.pop()
lit = dict(_NP_STRENGTH=tuple([0.0] + strength[1:15] + [0.0]),
           _NP_DSCALE=(factor[1] if len(set(factor[1:15])) == 1 else tuple(factor)),
           _NP_BIRTHS=1.0 + get('n6b')[1], _NP_ROWS=1.0 + get('n6r')[1], _NP_DIAG=1.0 + get('n6d')[1],
           _NP_K3L=tuple(k3l))
for k, v in lit.items():
    print(k, '=', repr(v))
side = {'n6b': get('n6b')[0], 'n6r': get('n6r')[0], 'n6d': get('n6d')[0], 'nfd': get('nfd')[0]}
print('layer 0 values the scalars do not carry (the exporter takes layer 1):', side)
if '--check' in args:
    text = Path(args[args.index('--check') + 1]).read_text()
    bad = 0
    for k, v in lit.items():
        found = re.findall(r'^%s = (.*?)(?:\s+#.*)?$' % k, text, flags=re.M)
        native = eval(found[-1])
        native = tuple(native) if isinstance(native, (list, tuple)) else native
        same = native == v
        bad += not same
        print('check', k, 'SAME' if same else 'DIFFERS: native %r' % (native,))
    print('CHECK', 'PASS' if not bad else 'FAIL')
if '--fragment' in args:
    out = Path(args[args.index('--fragment') + 1])
    note = args[args.index('--fragment') + 2]
    assert not out.exists()
    lines = ['', '# BEGIN OPUS-ENGINEER a62 PAIR-LAW LITERALS (%s; a62_modules/law_literals_from_pt_v1.py from %s, by the rules of' % (note, sys.argv[1]),
             '# trainer/export_dials.py + a38_pairlaw/dials_from_trainer.py). Appended after the release fragments: rebinds the literals and rebuilds',
             '# the law\'s tables, which the file builds at import from them (_np_const would otherwise rebuild them in the first prediction).']
    lines += ['%s = %r' % (k, v) for k, v in lit.items()]
    lines += ['_NP_TABLES = _np_tables(1024)', '# END OPUS-ENGINEER a62 PAIR-LAW LITERALS', '']
    out.write_text('\n'.join(lines))
    print('written', out)
