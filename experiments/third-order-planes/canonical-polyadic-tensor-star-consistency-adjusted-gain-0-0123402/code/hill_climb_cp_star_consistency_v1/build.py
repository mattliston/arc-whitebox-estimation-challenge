"""One source-only CP/star state-consistency counterfactual on the exact bundle."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
PARENT = D.parent / 'hill_climb_exact_bundle_v1/candidate.py'
PARENT_SHA = '3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7'
BEFORE = '            cp_post = _e2_H(inherited + born) * live'
AFTER = '            cp_post = _e2_H(_e2_H(_S19_LAM[0] * inherited) + _e2_H(_S19_LAM[1] * born)) * live'


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    parent = PARENT.read_bytes()
    source = parent.decode()
    tree = ast.parse(source)
    scales = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_S19_LAM' for t in node.targets):
            scales = ast.literal_eval(node.value)
    assert scales == (1.0032287195131746, 0.9932375877606482)
    assert source.count(BEFORE) == 1 and AFTER not in source
    owners = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and BEFORE in ast.get_source_segment(source, node)]
    assert len(owners) == 1 and owners[0].name == '_predict_binary'
    body = ast.get_source_segment(source, owners[0])
    assert body.index('born = born + _bd_plane_delta') < body.index(BEFORE.strip())
    assert body.index(BEFORE.strip()) < body.index('lam = fnp.concatenate([lam * _S19_LAM[0], _s19_new, -_s19_new])')
    candidate = source.replace(BEFORE, AFTER)
    ast.parse(candidate)
    assert candidate.replace(AFTER, BEFORE).encode() == parent
    out = D / 'candidate.py'
    with out.open('xb') as f:
        f.write(candidate.encode())
    result = dict(parent=pin(PARENT), candidate=pin(out), builder=pin(Path(__file__)),
                  protocol=pin(D / 'PROTOCOL.md'), before=BEFORE, after=AFTER,
                  one_callsite_edit=True, source_inverse_exact=True,
                  owner=dict(function=owners[0].name, line=owners[0].lineno,
                             cp_post_line=source[:source.index(BEFORE)].count('\n')+1),
                  unchanged_scales=list(scales), born_includes_bd_plane_delta=True,
                  complete_tensor_exactness_claimed=False,
                  source_only=True, numerical_execution=False, qualified=False, submitted=False)
    with (D / 'build.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['candidate']))


if __name__ == '__main__':
    main()
