"""Four exact rank-literal changes; all other source bytes are inherited."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'lazy_rotation_objective_v1/candidate.py'
EXPECTED = 'b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert sha(PARENT) == EXPECTED
    source = PARENT.read_text()
    tree = ast.parse(source)
    push = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_shared_old384_push'][-1]
    bound = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_slr_virtual_rotation_envelope'][-1]
    assignment = [n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(x, ast.Name) and x.id == '_SLR_RANK' for x in n.targets)]
    assert len(assignment) == 1 and isinstance(assignment[0].value, ast.Constant)
    push_nodes = [n for n in ast.walk(push) if isinstance(n, ast.Constant) and type(n.value) is int and n.value == 384]
    bound_nodes = [n for n in ast.walk(bound) if isinstance(n, ast.Constant) and type(n.value) is float and n.value == 384.]
    assert len(push_nodes) == 2 and len(bound_nodes) == 1 and assignment[0].value.value == 384
    nodes = push_nodes+bound_nodes+[assignment[0].value]
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1]+len(line))
    spans = []
    for node in nodes:
        assert node.lineno == node.end_lineno
        start, stop = starts[node.lineno-1]+node.col_offset, starts[node.end_lineno-1]+node.end_col_offset
        before = source[start:stop]
        after = '256.0' if type(node.value) is float else '256'
        assert before == ('384.0' if type(node.value) is float else '384') and len(before) == len(after)
        spans.append(dict(start=start, stop=stop, before=before, after=after, line=node.lineno))
    candidate = source
    for edit in sorted(spans, key=lambda e:e['start'], reverse=True):
        candidate = candidate[:edit['start']]+edit['after']+candidate[edit['stop']:]
    inverse = candidate
    for edit in sorted(spans, key=lambda e:e['start'], reverse=True):
        assert inverse[edit['start']:edit['stop']] == edit['after']
        inverse = inverse[:edit['start']]+edit['before']+inverse[edit['stop']:]
    assert inverse == source
    ast.parse(candidate)
    target = HERE/'candidate.py'
    with target.open('x') as stream:
        stream.write(candidate)
    proof = dict(parent=str(PARENT), parent_sha256=EXPECTED, candidate_sha256=sha(target),
                 edits=spans, whole_source_inverse_byte_exact=True,
                 unchanged_other_source_bytes=True, unchanged_metric=True, unchanged_age=True,
                 unchanged_learned_literals=True, unchanged_kernels=True,
                 scope='rank256 with matching Lazy dimension/overflow metadata', source_sha256=sha(__file__))
    with (HERE/'build.json').open('x') as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
