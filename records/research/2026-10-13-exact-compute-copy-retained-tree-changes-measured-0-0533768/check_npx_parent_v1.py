"""TAG scout-native check_npx_parent_v1 (2026-10-12): may fragment_npx_v1.py be appended to <parent.py>? Pass = the parent's FIRST
_np_left (the pair law v9, the one _K3LB_PARENT_LEFT captures) equals the fragment's _npx_left except for the one NPX change, and the
parent defines _K3LB_PARENT_LEFT (the K3L-by-layer wrapper) after it. usage: check_npx_parent_v1.py <parent.py> [...]"""
import ast
import sys
from pathlib import Path

OLD = "            cross = hold(fnp.einsum('ik,ik->k', full, hold(fnp.concatenate([pieces[i ^ 1] for i in range(len(pieces))], axis=1)), optimize=False))"
frag = Path(__file__).resolve().parent / 'fragment_npx_v1.py'


def first_def(text, name):
    lines = text.split('\n')
    node = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name][0]
    return '\n'.join(lines[node.lineno - 1:node.end_lineno])


ftext = frag.read_text()
mine = first_def(ftext, '_npx_left')
NEW = '\n'.join(l for l in mine.split('\n') if 'scout-native NPX' in l or 'the same number for the pair' in l)
ok_all = True
for p in sys.argv[1:]:
    text = Path(p).read_text()
    parent = first_def(text, '_np_left')
    ok = parent.count(OLD) == 1 and parent.replace(OLD, NEW).replace('def _np_left(', 'def _npx_left(', 1) == mine and '_K3LB_PARENT_LEFT = _np_left' in text
    ok_all &= ok
    print(Path(p).parent.name, 'append NPX' if ok else 'DIFFERENT: do not append')
sys.exit(0 if ok_all else 1)
