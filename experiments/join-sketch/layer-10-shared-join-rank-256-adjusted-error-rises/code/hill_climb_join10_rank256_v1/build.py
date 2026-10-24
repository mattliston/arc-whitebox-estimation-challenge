"""Mechanical source assembly only; never import or execute the estimator."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked(record):
    path = Path(record['path'])
    assert pin(path) == record
    return path.read_bytes()


def main():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    parent, fragment = checked(protocol['parent']), checked(protocol['fragment'])
    assert protocol['parent']['sha256'] == 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
    source = parent + fragment
    tree = ast.parse(source)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '_R320_RULE' for t in n.targets)]
    assert len(assignments) == 2
    assert ast.literal_eval(assignments[0].value) == {7: 384, 10: 320, 13: 384}
    assert ast.literal_eval(assignments[-1].value) == {7: 384, 10: 256, 13: 384}
    assert source[:-len(fragment)] == parent
    candidate = HERE / 'candidate.py'
    assert not candidate.exists() and not (HERE / 'build.json').exists()
    with candidate.open('xb') as stream:
        stream.write(source)
    receipt = dict(passed=True, estimator_executed=False, parent=protocol['parent'],
        fragment=protocol['fragment'], candidate=pin(candidate), builder=pin(Path(__file__)),
        protocol=pin(HERE / 'protocol.json'), cost_forecast=pin(HERE / 'cost_forecast.json'),
        exact_parent_prefix=True, exact_inverse=True, appended_bytes=len(fragment),
        effective_rule={7: 384, 10: 256, 13: 384}, extra_overrides=0)
    with (HERE / 'build.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
