"""Source-only, exclusive construction of the single admitted direct32 Q2 arm."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
PARENT = D.parent / 'hill_climb_exact_bundle_v1/candidate.py'
PARENT_SHA = '3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7'
SUFFIX = b'\n_LR_Q = 2\n'


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def inspect_source(data):
    tree = ast.parse(data)
    values, locations = {}, {}
    names = ('_LR_K', '_LR_R', '_LR_Q', '_LRN_POWER')
    factors = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
                    locations[target.id] = node.lineno
        elif isinstance(node, ast.FunctionDef) and node.name == '_lr_factor':
            factors.append(node)
    factor = factors[-1]
    expected = ast.parse('range(_LR_Q if _LRN_POWER else 0)', mode='eval').body
    loops = [n for n in ast.walk(factor) if isinstance(n, ast.For)]
    assert len(loops) == 1
    assert ast.dump(loops[0].iter, include_attributes=False) == ast.dump(expected, include_attributes=False)
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_tsr_mm'
               for n in ast.walk(loops[0]))
    factor_digest = hashlib.sha256(ast.dump(factor, include_attributes=False).encode()).hexdigest()
    return dict(settings=values, assignment_lines=locations, last_lr_factor_line=factor.lineno,
                power_loop_line=loops[0].lineno, factor_ast_sha256=factor_digest,
                loop='range(_LR_Q if _LRN_POWER else 0)')


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    assert (D / 'PROTOCOL.md').is_file()
    parent = PARENT.read_bytes()
    before = inspect_source(parent)
    assert before['settings'] == {'_LR_K': 32, '_LR_R': None, '_LR_Q': 1, '_LRN_POWER': True}
    candidate = parent + SUFFIX
    after = inspect_source(candidate)
    assert after['settings'] == {'_LR_K': 32, '_LR_R': None, '_LR_Q': 2, '_LRN_POWER': True}
    assert after['factor_ast_sha256'] == before['factor_ast_sha256']
    assert candidate[:-len(SUFFIX)] == parent
    out = D / 'candidate.py'
    with out.open('xb') as f:
        f.write(candidate)
    result = dict(parent=pin(PARENT), candidate=pin(out), builder=pin(Path(__file__)),
                  protocol=pin(D / 'PROTOCOL.md'), suffix=SUFFIX.decode(),
                  prefix_byte_exact=True, suffix_inverse_exact=True,
                  before=before, after=after, source_only=True, numerical_execution=False,
                  qualified=False, submitted=False)
    with (D / 'build.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['candidate']))


if __name__ == '__main__':
    main()
