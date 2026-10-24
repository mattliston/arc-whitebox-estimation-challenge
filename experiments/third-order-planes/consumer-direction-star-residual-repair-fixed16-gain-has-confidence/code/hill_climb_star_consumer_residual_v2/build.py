"""Source-only, exclusive S32-plus-fragment build. Does not import the candidate."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'hill_climb_star_projection32_v1/candidate.py'
PARENT_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    assert not (HERE / 'candidate.py').exists()
    assert not (HERE / 'build.json').exists()
    parent = PARENT.read_bytes()
    fragment = (HERE / 'fragment.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT_SHA
    assert len(parent) == 930166
    original = ast.parse(parent)
    extra = ast.parse(fragment)
    names = [node.name for node in extra.body if isinstance(node, ast.FunctionDef)]
    assert names == ['_csr_vectors', '_flms_cov_star', '_flms_star_diagonal']
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(extra))
    combined = parent + b'\n\n' + fragment
    tree = ast.parse(combined)
    assert ast.dump(ast.Module(body=tree.body[:len(original.body)], type_ignores=[])) == ast.dump(original)
    assert combined[:-len(fragment)-2] == parent
    protocol = json.loads((HERE / 'protocol.json').read_text())
    expected = dict(path=str(HERE / 'candidate.py'), sha256=hashlib.sha256(combined).hexdigest(), bytes=len(combined))
    assert protocol['parent'] == pin(PARENT)
    assert protocol['fragment'] == pin(HERE / 'fragment.py')
    assert protocol['candidate'] == expected
    with (HERE / 'candidate.py').open('xb') as output:
        output.write(combined)
    result = dict(passed=True, parent=pin(PARENT), fragment=pin(HERE/'fragment.py'),
                  candidate=pin(HERE/'candidate.py'), builder=pin(Path(__file__).resolve()),
                  protocol=pin(HERE/'protocol.json'), exact_parent_prefix=True,
                  exact_inverse=True, ast_parent_prefix_unchanged=True,
                  numerical_execution=False, appended_functions=names)
    with (HERE / 'build.json').open('x') as output:
        json.dump(result, output, indent=2); output.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
