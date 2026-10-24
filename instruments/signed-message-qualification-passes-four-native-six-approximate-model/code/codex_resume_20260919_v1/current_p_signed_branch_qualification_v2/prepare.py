"""Build a zero branch and bind existing inputs without importing numerical libraries."""
import ast
import base64
from datetime import datetime, timezone
import zlib
from common import *

assert sha(PARENT) == SOURCE_PARENT_SHA
assert not (HERE / 'build.json').exists() and not (HERE / 'sources_sealed.json').exists()
text = PARENT.read_text()
tree = ast.parse(text)
gnn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gnn_step']
lit = [n.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_GNN_LITERAL']
assert len(gnn) == len(lit) == 1
old_gnn = ast.get_source_segment(text, gnn[0])
old_literal = ast.get_source_segment(text, lit[0])
needle = '    h=silu(h@P["enc0_w"].T+P["enc0_b"])'
replacement = '''    signed=fnp.clip(fnp.concatenate([m1,m2,m3],axis=1),-8,8)
    pre=h@P["enc0_w"].T+P["enc0_b"]
    h=silu(pre+signed@P["ungated_w"].T)'''
assert old_gnn.count(needle) == 1
new_gnn = old_gnn.replace(needle, replacement)
payload = ast.literal_eval(old_literal)
assert 'ungated_w' not in payload
payload['ungated_w'] = ([128, 48], base64.b64encode(zlib.compress(bytes(128 * 48 * 4), 9)).decode())
new_literal = repr(payload)
assert text.count(old_gnn) == text.count(old_literal) == 1
candidate = text.replace(old_gnn, new_gnn).replace(old_literal, new_literal)
assert candidate.replace(new_gnn, old_gnn).replace(new_literal, old_literal) == text
ast.parse(candidate)
with SOURCE.open('x') as f:
    f.write(candidate)
with (HERE / 'gnn_step.py').open('x') as f:
    f.write(new_gnn + '\n')
copies = []
for source, name in [(RELEASE / 'native.py', 'native.py'), (RELEASE / 'records.py', 'records.py'),
                     (D / 'current_full_lower_p_allocator_v1/guard.py', 'guard.py'),
                     (D / 'current_full_lower_p_allocator_v1/allocator.py', 'allocator.py')]:
    target = HERE / name
    with target.open('xb') as f:
        f.write(source.read_bytes())
    assert source.read_bytes() == target.read_bytes()
    copies.append(dict(original=pin(source), copy=pin(target), byte_exact=True))
model_source = D / 'current_full_lower_p_proposal_v1/model_v2.py'
model_text = model_source.read_text()
assert sha(model_source) == '278112f097d0748c255869104a22297b248ba1c424c0c4528b6b6df04942c00e'
assert model_text.count('== 45200 and') == 1
branch_text = model_text.replace('== 45200 and', '== 51344 and')
parent_binding = '\nSOURCE = PARENT\nPARENT_SHA = SOURCE_PARENT_SHA\nPKEYS = BASE_PKEYS\n'
assert model_text.count('from common import *\n') == 1
parent_text = model_text.replace('from common import *\n', 'from common import *\n' + parent_binding)
assert branch_text.replace('== 51344 and', '== 45200 and') == model_text
assert parent_text.replace(parent_binding, '') == model_text
for name, content in [('model_branch.py', branch_text), ('model_parent.py', parent_text)]:
    ast.parse(content)
    with (HERE / name).open('x') as f:
        f.write(content)
inputs = read(RELEASE / 'inputs.json')
references = read(RELEASE / 'release_inputs.json')
warm = references['warm_reference']
cold_summary = [r['candidate'] for r in references['rows'] if r['rank'] == 100][0]
cold_path = RELEASE / 'confirmation_100/call_00.json'
cold = read(cold_path)
assert all(cold[k] == v for k,v in cold_summary.items())
assert [r['rank'] for r in warm] == [35, 100, 35] and cold['rank'] == 100
used = [inputs['weights'][str(r)] for r in [35, 100]] + [pin(cold_path)]
for row in warm + [cold]:
    for key in ['prediction', 'bill', 'weights', 'input_bill', 'input_reset', 'lifecycle']:
        check(row[key])
        used.append(row[key])
for p in used:
    check(p)
write(HERE / 'inputs.json', dict(weights={str(r):inputs['weights'][str(r)] for r in [35, 100]},
    root_seeds={str(r):inputs['root_seeds'][str(r)] for r in [35, 100]},
    warm_reference=warm, cold100_reference=cold, score_labels_admitted=False,
    source_inputs=pin(RELEASE / 'inputs.json'), source_references=pin(RELEASE / 'release_inputs.json')))
history = D / 'full_lower_gaussian_gate_reuse_v1'
closed = read(history / 'closeout.json')
assert not closed['outcome']['passed'] and closed['outcome']['study_closed'] and not closed['uploaded']
for p in closed['artifact_pins'] + closed['external_pins']:
    check(p)
write(HERE / 'history_correction.json', dict(completed=True,
    previously_tested=True, prior_cdf_reuse_closed_negative=True, rerun=False,
    old_closeout=pin(history / 'closeout.json'), old_result=pin(history / 'RESULT.md'),
    old_protocol=pin(history / 'PROTOCOL.md'), verified_pins=len(closed['artifact_pins'])+len(closed['external_pins']),
    bounded_signed_branch_search_found_only_source_proposal=True, exhaustive_novelty_claim=False))
write(HERE / 'source_copies.json', dict(copies=copies, model_original=pin(model_source),
    branch_model=pin(HERE / 'model_branch.py'), parent_model=pin(HERE / 'model_parent.py'),
    branch_model_only_parameter_count_changed=True, parent_model_only_bindings_changed=True))
write(HERE / 'build.json', dict(passed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
    parent=pin(PARENT), candidate=pin(SOURCE), added_parameter='ungated_w', added_shape=[128,48],
    added_parameters=6144, total_allocated_trainable=51344, inactive_coordinates=66,
    only_gnn_and_literal_changed=True, whole_source_inverse_exact=True, initial_branch_zero=True,
    old_gnn_sha256=hashlib.sha256(old_gnn.encode()).hexdigest(), new_gnn=pin(HERE / 'gnn_step.py'),
    numerical_training_executed=False, targets_decoded=False))
external = [PARENT, model_source, SHARED, RELEASE / 'ready_closeout.json',
            A / 'birth_direction_fit_v1/native_v2.py',
            D / 'current_full_lower_p_allocator_v1/closeout.json',
            D / 'current_full_lower_p_allocator_v1/unit_result.json',
            D / 'current_full_lower_p_allocator_v1/guard_unit_result.json']
assert read(external[-2])['passed'] and read(external[-1])['passed']
files = sorted(p for p in HERE.iterdir() if p.is_file())
pins = [pin(p) for p in files + external] + used
unique = {p['path']:p for p in pins}
for p in unique.values():
    check(p)
for p in files:
    if p.suffix == '.py':
        ast.parse(p.read_text(), str(p))
write(HERE / 'sources_sealed.json', dict(pins=list(unique.values()), prepared=True,
    qualification_only=True, automatic_fit=False, automatic_upload=False))
resource_floor()
print(dict(prepared=True, candidate=pin(SOURCE), source_pins=len(unique), targets_decoded=False), flush=True)
