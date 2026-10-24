"""Append-only source build with exact reversal of three local body edits."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'hill_climb_star_projection32_v1/candidate.py'
SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
EDITS = [
    ('        _t31_carry = {}\n', '        _t31_carry = {}\n        _gt13_source = None\n'),
    ("            if l in _T31_LAYERS:\n",
     "            if l == 13:\n"
     "                if _gt13_source is None:\n"
     "                    raise RuntimeError('GT13 missing post12 source')\n"
     "                _gt13_delta4, _gt13_extra = _gt13_faces(_gt13_source, w32, w, Cpre, mu, K21)\n"
     "                _gt13_source = None\n"
     "                k4 = k4 + _gt13_delta4\n"
     "                m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric, extra31=_gt13_extra)\n"
     "                del _gt13_delta4, _gt13_extra\n"
     "            elif l in _T31_LAYERS:\n"),
    ("            if _t31_commit is not None:\n                _t31_commit(l, _e2_H(locals()))\n",
     "            if _t31_commit is not None:\n                _t31_commit(l, _e2_H(locals()))\n"
     "            if l == 12:\n"
     "                _gt13_source = _gt13_prepare(Cpre, aux, C)\n"),
]


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def assemble():
    parent = PARENT.read_bytes()
    assert hashlib.sha256(parent).hexdigest() == SHA and len(parent) == 930166
    source = parent.decode()
    parent_tree = ast.parse(source)
    bindings = {}
    for node in parent_tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('_T31_LAYERS', '_t31_commit', '_S19_GN_TORCH'):
                    bindings[target.id] = ast.literal_eval(node.value)
    assert bindings == dict(_T31_LAYERS=(), _t31_commit=None, _S19_GN_TORCH=False)
    fn = [x for x in parent_tree.body if isinstance(x, ast.FunctionDef) and x.name == '_predict_binary'][-1]
    original = ''.join(source.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno])
    changed = original
    for before, after in EDITS:
        assert changed.count(before) == 1
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(EDITS):
        assert inverse.count(after) == 1
        inverse = inverse.replace(after, before)
    assert inverse == original and ast.dump(ast.parse(inverse)) == ast.dump(ast.parse(original))
    fragment = (HERE / 'fragment.py').read_bytes()
    tree = ast.parse(fragment)
    assert not any(isinstance(x, (ast.Import, ast.ImportFrom)) for x in ast.walk(tree))
    assert all(isinstance(x.target, ast.Subscript) and isinstance(x.target.value, ast.Name)
               and x.target.value.id == '_GT13_STATS' for x in ast.walk(tree) if isinstance(x, ast.AugAssign))
    candidate = parent + b'\n\n' + fragment + b'\n\n' + changed.encode()
    ast.parse(candidate)
    assert candidate[:len(parent)] == parent
    return candidate, changed.encode()


def main():
    candidate, override = assemble()
    target = HERE / 'candidate.py'
    expected = dict(path=str(target), sha256=hashlib.sha256(candidate).hexdigest(), bytes=len(candidate))
    protocol = json.loads((HERE / 'protocol.json').read_text())
    assert protocol['parent'] == pin(PARENT) and protocol['fragment'] == pin(HERE / 'fragment.py') and protocol['candidate'] == expected
    assert not target.exists() and not (HERE / 'build.json').exists()
    for path, data in ((target, candidate), (HERE / 'predict_override.py', override)):
        with path.open('xb') as stream: stream.write(data)
    result = dict(passed=True, parent=pin(PARENT), candidate=pin(target), builder=pin(Path(__file__).resolve()),
                  protocol=pin(HERE / 'protocol.json'), fragment=pin(HERE / 'fragment.py'), override=pin(HERE / 'predict_override.py'),
                  exact_parent_prefix=True, exact_inverse=True, ast_inverse=True, edits=[dict(before=a, after=b) for a,b in EDITS],
                  numerical_execution=False, original_Q22_inputs_unchanged=True, source_scope='post12 birth -> pre13 only')
    with (HERE / 'build.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
