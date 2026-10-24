"""One frozen precision parent plus the unchanged nominal core-carry wrapper."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
PARENT = H2 / 'hill_climb_precision_symmetry_bundle_v1/candidate.py'
FRAGMENT = H2 / 'hill_climb_core_carry_consistency_v1/fragment.py'
PARENT_SHA = '1571846b21141a16c2677ca371c4175a774d424d375ee656eb49a0b4e65ac5f3'
FRAGMENT_SHA = 'a2defc728c49800f8bf078581db046ca75c40e8225d8db25b70e462c0baa6f4d'


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    assert pin(FRAGMENT)['sha256'] == FRAGMENT_SHA
    parent = PARENT.read_bytes()
    source = parent.decode()
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    body = lambda n: ''.join(lines[n.lineno-1:n.end_lineno])
    values = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in ('_S19_LAM', '_JOIN_SCHEDULE', '_JOIN_NYSTROM', '_t31_commit'):
                    values[t.id] = ast.literal_eval(n.value)
    assert values['_S19_LAM'] == (1.0032287195131746, 0.9932375877606482)
    assert values['_JOIN_SCHEDULE'] == {7: (0, 2), 10: (3, 5), 13: (6, 8)}
    assert values['_JOIN_NYSTROM'] is True and values['_t31_commit'] is None
    commits = [n for n in functions if n.name == '_shared_old384_commit']
    assert len(commits) == 1 and "state['core']" not in body(commits[0])
    live = [n for n in functions if '_shared_old384_commit(l, g, alive, selected, shared_old)' in body(n)
            and 'lam = fnp.concatenate([lam * _S19_LAM[0], _s19_new, -_s19_new])' in body(n)]
    assert len(live) == 1 and live[0].name == '_predict_binary'
    between = body(live[0]).split('_shared_old384_commit(l, g, alive, selected, shared_old)', 1)[1].split(
        'lam = fnp.concatenate([lam * _S19_LAM[0], _s19_new, -_s19_new])', 1)[0]
    assert 'core' not in between and 'shared_old' not in between
    pushes = [n for n in functions if n.name == '_shared_old384_push' and 'Nystrom approximate Gram core' in body(n)]
    assert len(pushes) == 1
    assert '_need_core = layer < max(_JOIN_SCHEDULE)' in body(pushes[0])
    assert "state['core'] = _shared_old384_finite(core, 'Nystrom approximate Gram core') if _need_core else None" in body(pushes[0])
    assert '_OTT_ORIGINAL_PUSH = _shared_old384_push' in source
    parent_result = H2 / 'hill_climb_precision_symmetry_bundle_v1/native8_extension_assessment.json'
    assert pin(parent_result)['sha256'] == 'f06e9cf9fe34da8f90c206f5a871914d53e7cded255d6a02d44f660693cf773a'
    previous = json.loads(parent_result.read_text())
    assert previous['passed_integrity'] is True and previous['admitted'] is False
    assert previous['gates']['adjusted_gain'] is False and previous['adjusted_wins'] == 8
    assert previous['candidate'] == pin(PARENT)
    suffix = b'\n' + FRAGMENT.read_bytes()
    candidate = parent + suffix
    ast.parse(candidate.decode())
    assert candidate[:-len(suffix)] == parent
    with (HERE / 'candidate.py').open('xb') as f:
        f.write(candidate)
    result = dict(parent=pin(PARENT), candidate=pin(HERE / 'candidate.py'),
                  builder=pin(Path(__file__)), protocol=pin(HERE / 'PROTOCOL.md'),
                  fragment=pin(FRAGMENT), append_only=True, source_inverse_exact=True,
                  suffix_bytes=len(suffix), parent_result=pin(parent_result),
                  exact_parent_core_build=pin(H2 / 'hill_climb_core_carry_consistency_v1/build.json'),
                  parent_original_bank_gate_passed=False, parent_failure_preserved=True,
                  unchanged_scales=list(values['_S19_LAM']),
                  commit_line=commits[0].lineno, active_body_line=live[0].lineno,
                  core_constructor_line=pushes[0].lineno, no_intervening_core_consumer=True,
                  nominal_core_update_layers=[7, 8, 9, 10, 11, 12],
                  first_affected_core_consumer_layers=[10, 13],
                  required_parent_prefix_layers=list(range(10)),
                  source_only=True, numerical_execution=False, qualified=False, submitted=False)
    with (HERE / 'build.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['candidate']))


if __name__ == '__main__':
    main()
