"""TAG opus-release: self-test of literal_diff.py on small synthetic sources (each case: what must be found). Run: python -B selftest_literal_diff.py"""
import base64
from pathlib import Path
import tempfile

import literal_diff

BLOB = [base64.b64encode(bytes(range(256))*k).decode() for k in (1, 2)]
BASE = '''"""header v1"""
import math
_K = {'k21n': -0.028, 'iso': 0.5}
_SCALE = 1.0
_AGES = 2
_ON = True
_MODE = 'ij,ij->i'
_PACKED = {'w': ((4, 4), %r)}
def f(x, y=3):
    """doc"""
    for i in range(_AGES):
        x = x * _SCALE + _K['k21n'] / 6
    return x if _ON else -x
''' % BLOB[0]
CASES = [   # (name, edit of BASE, literal-only?, kinds that must be listed, number of code differences)
    ('same', lambda s: s, True, {}, 0),
    ('comment and layout', lambda s: s.replace('_SCALE = 1.0', '_SCALE   =   1.0   # tuned'), True, {}, 0),
    ('float dial', lambda s: s.replace('_SCALE = 1.0', '_SCALE = 0.8'), True, {'number': 1}, 0),
    ('sign change', lambda s: s.replace('-0.028', '0.048'), True, {'number': 1}, 0),
    ('int to float', lambda s: s.replace("'iso': 0.5", "'iso': 1").replace('/ 6', '/ 6.2'), True, {'number': 2}, 0),
    ('packed array', lambda s: s.replace(BLOB[0], BLOB[1]), True, {'data': 1}, 0),
    ('docstring', lambda s: s.replace('header v1', 'header v2'), True, {'text': 1}, 0),
    ('named dial', lambda s: s.replace('_ON = True', '_ON = True\n_DIV = 6.2').replace('/ 6', '/ _DIV'), True, {'number': 1}, 0),
    ('integer', lambda s: s.replace('_AGES = 2', '_AGES = 3'), False, {'integer': 1}, 0),
    ('default argument', lambda s: s.replace('y=3', 'y=4'), False, {'integer': 1}, 0),
    ('shape', lambda s: s.replace('(4, 4)', '(4, 8)'), False, {'integer': 1}, 0),
    ('switch', lambda s: s.replace('_ON = True', '_ON = False'), False, {'switch': 1}, 0),
    ('signature string', lambda s: s.replace('ij,ij->i', 'ij,ji->i'), False, {'string': 1}, 0),
    ('kind of constant', lambda s: s.replace('_SCALE = 1.0', "_SCALE = '1.0'"), False, {'type': 1}, 0),
    ('operator', lambda s: s.replace('x * _SCALE', 'x + _SCALE'), False, {}, 1),
    ('name', lambda s: s.replace('range(_AGES)', 'range(y)'), False, {}, 1),
    ('added statement', lambda s: s.replace('    return x', '    x = x + 1\n    return x'), False, {}, 1),
    ('added function', lambda s: s + 'def g():\n    return 1\n', False, {}, 1),
    ('named but rebound', lambda s: s.replace('_ON = True', '_ON = True\n_DIV = 6.2\n_DIV = 6.3').replace('/ 6', '/ _DIV'), False, {}, 3),
    ('named but shadowed by an argument', lambda s: s.replace('_ON = True', '_ON = True\ny = 6.2').replace('/ 6', '/ y'), False, {}, 2),
]
with tempfile.TemporaryDirectory() as d:
    a = Path(d)/'a.py'
    a.write_text(BASE)
    compile(BASE, 'a.py', 'exec')
    bad = 0
    for name, edit, literal_only, kinds, code in CASES:
        b = Path(d)/'b.py'
        b.write_text(edit(BASE))
        compile(b.read_text(), 'b.py', 'exec')
        assert (edit(BASE) != BASE) or name == 'same', name
        r = literal_diff.compare(a, b)
        ok = r['literal_only'] == literal_only and r['counts'] == kinds and len(r['code']) == code
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: literal-only {r['literal_only']}, constants {r['counts']}, code {len(r['code'])}"
              + ('' if ok else f"   expected {literal_only}, {kinds}, {code}: {literal_diff.lines(r)}"))
    print('ALL OK' if not bad else f'{bad} CASES FAILED')
    raise SystemExit(bool(bad))
