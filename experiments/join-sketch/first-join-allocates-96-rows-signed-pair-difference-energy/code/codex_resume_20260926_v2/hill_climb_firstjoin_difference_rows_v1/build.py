"""Stdlib-only one-site source build; never imports the estimator."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
PARENT = D.parent/'hill_climb_precision_g3_symdiag128_v1/candidate.py'
PARENT_SHA = '88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    original = PARENT.read_text(); tree = ast.parse(original)
    capture = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(x, ast.Name) and x.id == '_OTT_ORIGINAL_PUSH' for x in n.targets))
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == '_shared_old384_push' and n.lineno < capture.lineno][-1]
    before = ast.get_source_segment(original, node)
    old = "            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])"
    new = "            _jn_rows = _fjdr_rows(layer, _jn_d, J, weight, coefficient, state, start, stop, rank)"
    assert before.count(old) == 1 and "Nystrom approximate Gram core" in before
    after = before.replace(old, new, 1)
    assert original.count(before) == 1
    prefix = original.replace(before, after, 1)
    suffix = '\n'+(D/'fragment.py').read_text()
    candidate = prefix+suffix
    assert candidate[:-len(suffix)].replace(after, before, 1) == original
    other = ast.parse(prefix)
    assert len(tree.body) == len(other.body)
    changed = [(a.name, a.lineno) for a,b in zip(tree.body,other.body) if ast.dump(a) != ast.dump(b)]
    assert changed == [('_shared_old384_push', node.lineno)]
    assert ast.dump(ast.parse(after).body[0].body[-1]) == ast.dump(ast.parse(before).body[0].body[-1])
    ast.parse(candidate)
    assert '_JOIN_SCHEDULE = {7: (0, 2), 10: (3, 5), 13: (6, 8)}' in original
    assert '_R320_RULE = {7: 384, 10: 320, 13: 384}' in original
    with (D/'candidate.py').open('x') as f: f.write(candidate)
    receipt = dict(parent=pin(PARENT), parent_build=pin(PARENT.parent/'build.json'),
        candidate=pin(D/'candidate.py'), builder=pin(__file__), fragment=pin(D/'fragment.py'),
        primitive=pin(D/'probe.py'), protocol=pin(D/'PROTOCOL.md'),
        policy=pin(D.parent/'hill_climb_policy_20260928_v2.json'),
        edits=[dict(label='first_join_nystrom_row_allocation',before=before,after=after)],
        suffix_bytes=len(suffix.encode()), source_inverse_exact=True,
        all_unlisted_source_bytes_unchanged=True, changed_function_asts=changed,
        scope=dict(layer=7,rank=384,incumbent_energy_rows=288,contrast_rows=96,
                   joining_cohorts=[0,1,2], setup_bound=682,setup_lvfuse=649,setup_psdb_fused=9),
        no_new_cache_or_arena=True, source_only=True, numerical_execution=False,
        execution_authorized=False)
    with (D/'build.json').open('x') as f: json.dump(receipt,f,indent=2); f.write('\n')
    print(json.dumps(receipt['candidate']))


if __name__ == '__main__': main()
