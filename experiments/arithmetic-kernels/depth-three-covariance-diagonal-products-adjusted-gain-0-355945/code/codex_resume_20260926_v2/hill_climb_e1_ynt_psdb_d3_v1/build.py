"""Build only the root-selected D3 append; standard library, no model import."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
PARENT = D.parent / 'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
PARENT_SHA = '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
PRICE = D.parent / 'teammate_s33_psdb_depth234_pricing_v1.json'
PRICE_SHA = '762578cb8d6868338f860d16e85e785d5ae278de0b3edad5f03fe145a3b811ce'
ACTIVE = {(0, 128, 1024), (1, 1024, 128), (1, 1024, 256), (2, 128, 128), (2, 128, 256)}
KEYS = ((0,64,1024),(0,128,1024),(1,1024,64),(1,1024,128),
        (1,1024,256),(2,64,64),(2,64,128),(2,128,256),(2,128,128))

def pin(path):
    p = Path(path).resolve(); raw = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    assert pin(PRICE)['sha256'] == PRICE_SHA
    prices = {r['depth']:r for r in json.loads(PRICE.read_text())['fixed_price_table']}
    assert (prices[5]['helper_flops'], prices[3]['helper_flops']) == (707714016, 738303136)
    assert (prices[5]['helper_calls'], prices[3]['helper_calls']) == (422, 378)
    parent, fragment = PARENT.read_text(), (D / 'fragment.py').read_text()
    child = parent + '\n' + fragment
    pt, ft, ct = ast.parse(parent), ast.parse(fragment), ast.parse(child)
    assert [ast.dump(n) for n in ct.body[:len(pt.body)]] == [ast.dump(n) for n in pt.body]
    assert child[:-(len(fragment) + 1)] == parent
    assert len(ft.body) == 3
    assert [n.targets[0].id for n in ft.body[:2]] == ['_PSD5_DEPTHS', '_PSD5_REQUIRED']
    assert isinstance(ft.body[2], ast.FunctionDef) and ft.body[2].name == '_psdb_group'
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom, ast.AugAssign)) for n in ast.walk(ft))
    functions = {n.name:n for n in pt.body if isinstance(n, ast.FunctionDef)}
    lines = parent.splitlines()
    group = '\n'.join(lines[functions['_psdb_group'].lineno-1:functions['_psdb_group'].end_lineno])
    new_group = '\n'.join(fragment.splitlines()[ft.body[2].lineno-1:ft.body[2].end_lineno])
    assert group.count('_PSD5_DEPTHS.get(key) == 5') == 1
    assert new_group == group.replace('_PSD5_DEPTHS.get(key) == 5', '_PSD5_DEPTHS.get(key) == 3')
    assert '_PSD5_PARENT_GROUP = _psdb_group' in parent
    assert '_PSDB_GROUPS = ((0, 128, 256), (128, 128, 128))' in parent
    region = (90354432, 120472576, 120472576)
    starts = (0, region[0], region[0] + region[1])
    layout = []
    for kind, rows, cols in KEYS:
        depth = 3 if (kind, rows, cols) in ACTIVE else 6
        previous = 5 if (kind, rows, cols) in ACTIVE else 6
        u = 1 << depth
        assert rows % u == cols % u == 0
        size = 4 * 7**depth * (rows // u) * (cols // u)
        bind = 4 * (7**depth + 7 * 4**(depth-1)) * (rows // u) * (cols // u)
        assert size <= region[kind] and starts[kind] + size <= 391602176
        layout.append(dict(key=[kind, rows, cols], depth=depth, parent_depth=previous,
                           start_elements=starts[kind], tree_elements=size, bind_flops=bind))
    assert sum(r['depth'] == 3 for r in layout) == 5
    assert sum(r['depth'] == 6 for r in layout) == 4
    assert sum(r['bind_flops'] for r in layout) == 33781580
    history = [PARENT.parent / name for name in (
        'native2_assessment.json', 'native2_guard.json', 'boundary_assessment_extension_v2.json',
        'boundary_assess_guard_extension_v2.json', 'boundary_guard.json', 'closeout.json')]
    boundary = json.loads(history[2].read_text())
    assert boundary['passed'] is False and boundary['receipt_checks_passed'] is True
    with (D / 'candidate.py').open('x') as fp: fp.write(child)
    record = dict(parent=pin(PARENT), parent_build=pin(PARENT.parent / 'build.json'),
        candidate=pin(D / 'candidate.py'), builder=pin(__file__), fragment=pin(D / 'fragment.py'),
        primitive=pin(D / 'probe.py'), protocol=pin(D / 'PROTOCOL.md'), pricing=pin(PRICE),
        native_driver=pin(D / 'native_probe.py'), policy=pin(D.parent / 'hill_climb_policy_20260928_v2.json'),
        history=[pin(p) for p in history], append_only=True, source_inverse_exact=True,
        suffix_bytes=len(('\n' + fragment).encode()), parent_all_bytes_unchanged=True,
        setup_counts=dict(bound=682, lvfuse=649, psdb_fused=9), side_layout=layout,
        batch_side_bind_flops=33781580, arena_elements=391602176,
        root_selected_depth=3, outcome_informed_implementation_followon=True,
        source_proof=dict(dynamic_side_selection_unchanged=True, unchanged_group_core=True,
                          unchanged_ynt_and_e1=True, private_mixed_depth_group_rejected=True,
                          unused_four_keys_remain_depth6=True),
        source_only=True, numerical_execution=False, execution_authorized=False)
    with (D / 'build.json').open('x') as fp: json.dump(record, fp, indent=2); fp.write('\n')
    print(json.dumps({k:record[k] for k in ('candidate','builder','fragment','primitive','protocol')}))

if __name__ == '__main__': main()
