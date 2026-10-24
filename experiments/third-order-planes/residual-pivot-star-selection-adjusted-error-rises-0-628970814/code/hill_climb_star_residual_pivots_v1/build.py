"""Exclusive source assembly and inverse/AST checks; no candidate execution."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'hill_climb_star_projection32_v1/candidate.py'
SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    parent, fragment = PARENT.read_bytes(), (HERE / 'fragment.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == SHA and len(parent) == 930166
    old = [n for n in ast.parse(parent).body if isinstance(n, ast.FunctionDef) and n.name == '_lr_factor'][-1]
    tree = ast.parse(fragment)
    assert [n.name for n in tree.body if isinstance(n, ast.FunctionDef)] == ['_srp_columns', '_lr_factor']
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    new = tree.body[-1]
    assert ast.dump(ast.Module(body=old.body[:2] + old.body[3:], type_ignores=[])) == ast.dump(ast.Module(body=new.body[:2] + new.body[3:], type_ignores=[]))
    assert ast.unparse(old.body[2]) == "cols = fnp.sort(fnp.argsort(-cn, kind='stable')[:_LR_K])"
    assert ast.unparse(new.body[2]) == 'cols = _srp_columns(A32, A64, cn, _LR_K)'
    candidate = parent + b'\n\n' + fragment
    ast.parse(candidate)
    assert candidate[:-len(fragment)-2] == parent
    target = HERE / 'candidate.py'
    expected = dict(path=str(target), sha256=hashlib.sha256(candidate).hexdigest(), bytes=len(candidate))
    protocol = json.loads((HERE / 'protocol.json').read_text())
    assert protocol['parent'] == pin(PARENT) and protocol['fragment'] == pin(HERE / 'fragment.py') and protocol['candidate'] == expected
    assert not target.exists() and not (HERE / 'build.json').exists()
    with target.open('xb') as stream:
        stream.write(candidate)
    result = dict(passed=True, parent=pin(PARENT), candidate=pin(target), fragment=pin(HERE / 'fragment.py'),
                  protocol=pin(HERE / 'protocol.json'), builder=pin(Path(__file__).resolve()),
                  exact_parent_prefix=True, exact_inverse=True, original_factor_arithmetic_except_columns=True,
                  original_rank_power_and_route_unchanged=True, numerical_execution=False)
    with (HERE / 'build.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
