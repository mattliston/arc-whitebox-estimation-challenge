"""Fixed R4-only source edit on retained precision; stdlib preparation only."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
PARENT = H2 / 'hill_climb_precision_symmetry_bundle_v1/candidate.py'
PARENT_SHA = '1571846b21141a16c2677ca371c4175a774d424d375ee656eb49a0b4e65ac5f3'
EDITS = [
    dict(label='gaussian_and_power_views_R4',
         before='              Fr16=Fr[1:7], Gr16=Gr[1:7], P16=PB[1:7], P6=PB[1:7][:, None], RR=PB[0:3][:, None], P12=PB[1:3][:, None],',
         after='              Fr16=Fr[1:5], Gr16=Gr[1:5], P16=PB[1:5], P6=PB[1:5][:, None], RR=PB[0:3][:, None], P12=PB[1:3][:, None],'),
    dict(label='rho_divisors_R4',
         before='              DIV6=fnp.arange(1, 7, dtype=f32)[:, None, None], DIVZ=fnp.asarray([2, 4, 6, 24], dtype=f32)[:, None, None, None],',
         after='              DIV6=fnp.arange(1, 5, dtype=f32)[:, None, None], DIVZ=fnp.asarray([2, 4, 6, 24], dtype=f32)[:, None, None, None],'),
    dict(label='gaussian_product_scratch_R4',
         before='    views = dict(X1=fnp.reshape(arena[0:18 * nn], (6, 3, n, n)), X2=fnp.reshape(arena[0:45 * nn], (5, 3, 3, n, n)),',
         after='    views = dict(X1=fnp.reshape(arena[0:12 * nn], (4, 3, n, n)), X2=fnp.reshape(arena[0:45 * nn], (5, 3, 3, n, n)),'),
    dict(label='rho_comment',
         before="    _e2_hold.append(fnp.cumprod(P16, axis=0, out=P16))   # rho^r/r!, r=1..6",
         after="    _e2_hold.append(fnp.cumprod(P16, axis=0, out=P16))   # fixed Gaussian R4: rho^r/r!, r=1..4"),
]


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def functions(source):
    lines = source.splitlines(keepends=True)
    return [(n.name, ''.join(lines[n.lineno-1:n.end_lineno]))
            for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    policy_path = H2 / 'hill_climb_policy_20260928_v2.json'
    policy = json.loads(policy_path.read_text())
    assert policy['bank_gates'] == dict(adjusted_pct_at_most=-1, ci95_upper_below=0,
                                       wins_min_of40=28, max_per_network_regression_pct=10,
                                       serving_required=True)
    assert policy['resource_policy']['rss_bytes'] == 7500000000
    assert policy['resource_policy']['added_swap_bytes'] == 0
    parent = PARENT.read_bytes()
    original = parent.decode()
    candidate = original
    for edit in EDITS:
        assert candidate.count(edit['before']) == 1 and edit['after'] not in candidate, edit['label']
        candidate = candidate.replace(edit['before'], edit['after'], 1)
    ast.parse(candidate)
    restored = candidate
    for edit in reversed(EDITS):
        assert restored.count(edit['after']) == 1
        restored = restored.replace(edit['after'], edit['before'], 1)
    assert restored.encode() == parent
    old_functions, new_functions = functions(original), functions(candidate)
    assert len(old_functions) == len(new_functions)
    changed = []
    for (old_name, old), (new_name, new) in zip(old_functions, new_functions):
        assert old_name == new_name
        if ast.dump(ast.parse(old)) != ast.dump(ast.parse(new)):
            changed.append(old_name)
    assert changed == ['_e2_pair_state', '_e2_pair_arena']
    state = next(s for name, s in new_functions if name == '_e2_pair_state')
    assert "PB = fnp.empty((7, n, n), dtype=f32)" in state
    assert "FS = fnp.empty((3, 7, 3, n), dtype=f32)" in state
    assert "GS = fnp.empty((3, 7, 3, n), dtype=f32)" in state
    assert 'RR=PB[0:3][:, None], P12=PB[1:3][:, None]' in state
    with (HERE / 'candidate.py').open('xb') as f:
        f.write(candidate.encode())
    result = dict(parent=pin(PARENT), candidate=pin(HERE / 'candidate.py'),
                  builder=pin(Path(__file__)), protocol=pin(HERE / 'PROTOCOL.md'),
                  policy=pin(policy_path), edits=EDITS, source_inverse_exact=True,
                  numerical_edit_count=3, comment_edit_count=1,
                  changed_function_asts=changed, pair_arithmetic_ast_unchanged=True,
                  gaussian_order=dict(before=6, after=4, layers=list(range(1, 15))),
                  plane_bond_order=2, local_bond_order=2, derivative_rows=list(range(7)),
                  first_gaussian_source_unchanged=True, terminal_scalar_source_unchanged=True,
                  component_comparator=pin(PARENT), bank_comparator=pin(H2 / 'hill_climb_star_projection32_v1/candidate.py'),
                  source_only=True, numerical_execution=False, qualified=False, submitted=False)
    with (HERE / 'build.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['candidate']))


if __name__ == '__main__':
    main()
