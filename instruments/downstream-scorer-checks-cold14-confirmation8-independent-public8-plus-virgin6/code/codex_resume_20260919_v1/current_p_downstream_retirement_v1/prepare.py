import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
OLD = D/'current_p_downstream_lifecycle_v1'
NATIVE = D/'current_p_native_retirement_v1'
QUAL = D/'current_p_native_allocator_qualification_v1'


def read(path):return json.loads(path.read_text())
def pin(path):
    path=path.resolve();b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def check(p):assert pin(Path(p['path'])) == {k:p[k] for k in ['path','bytes','sha256']},p['path']
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


assert pin(OLD/'sources_sealed.json')['sha256'] == '831fc7fcd1be66066cd893b502568bd0fcee249435e6338f6fe53aed7a1a41b9'
assert pin(NATIVE/'sources_sealed.json')['sha256'] == '2adae6bc685e560d967e68a4fcfbeb976694777b14d44b3dce8c0cfee37c3c9b'
assert read(QUAL/'controller_result.json')['passed']
assert not (NATIVE/'native40_score.json').exists() and not (OLD/'admission.json').exists()
for p in read(OLD/'sources_sealed.json')['pins']+read(NATIVE/'sources_sealed.json')['pins']:check(p)
copies = []
for name in ['native.py','records.py','qualify_control.py','seal_stage.py','score_stage.py','inputs.json',
             'ORIGINAL_PROTOCOL.md','audit_reference.json']:
    with (HERE/name).open('xb') as f:f.write((OLD/name).read_bytes())
    copies.append(dict(original=pin(OLD/name),copy=pin(HERE/name),byte_exact=True))
common = (OLD/'common.py').read_text()
before = "ENDPOINT = D / 'current_p_native_lifecycle_v1'"
after = "ENDPOINT = D / 'current_p_native_retirement_v1'"
assert common.count(before) == 1
new_common = common.replace(before,after)
assert new_common.replace(after,before) == common
with (HERE/'common.py').open('x') as f:f.write(new_common)
source = (OLD/'run.py').read_text()
changes = [
    ("controller['completed_native_calls'] == 85 and not controller['fit_or_candidate_changed']",
     "controller['completed_new_native_calls'] == 41 and controller['reused_qualified_calls'] == 44 and not controller['fit_or_candidate_changed']"),
    ("endpoint['native_calls'] == 85", "endpoint['native_records'] == 85 and endpoint['new_native_calls'] == 41 and endpoint['reused_qualified_calls'] == 44"),
    ("assert endpoint['closed_context_lifecycle_qualified'] and not endpoint['official_changed']",
     "assert endpoint['closed_context_lifecycle_qualified'] and not endpoint['official_changed']\n    assert endpoint['all_new_retirements_qualified'] and endpoint['historical_trained_prefix_exact_calls'] == 30"),
    ("[('null',3),('parent',12),('trained',0)]", "[('null',3),('parent',12),('trained',30)]"),
    ("    lifecycle = read(ENDPOINT/'lifecycle_receipts_sealed.json')",
     "    trained = read(ENDPOINT/'trained_lifecycle_qualified.json')\n    assert trained['all_retirements_qualified'] and len(trained['retirement_pins']) == 41\n    check(trained['allocator_qualification'])\n    allocator_dir = D/'current_p_native_allocator_qualification_v1'\n    allocator = read(allocator_dir/'controller_result.json')\n    assert allocator['completed'] and allocator['passed'] and allocator['completed_calls'] == 8\n    check(allocator['qualification'])\n    for name in ['native_baseline','qualify_baseline','native_relief','qualify_comparison']:\n        clean_watch(allocator_dir/f'{name}_watch.json')\n    copies = read(ENDPOINT/'source_copies.json')\n    assert copies['inverse_source_byte_exact'] and copies['paid_body_AST_exact']\n    assert copies['original_scoring_and_prediction_sealer_byte_exact'] and copies['reused_qualified_calls'] == 44\n    for pair in copies['copies']:\n        check(pair['original']);check(pair['copy'])\n        assert pair['byte_exact'] and pair['original']['sha256'] == pair['copy']['sha256']\n    lifecycle = read(ENDPOINT/'lifecycle_receipts_sealed.json')"),
]
modified = source
for before,after in changes:
    assert modified.count(before) == 1
    modified = modified.replace(before,after)
recovered = modified
for before,after in reversed(changes):
    assert recovered.count(after) == 1
    recovered = recovered.replace(after,before)
assert recovered == source
old_tree,new_tree = ast.parse(source),ast.parse(modified)
old_rest=[n for n in old_tree.body if not isinstance(n,ast.FunctionDef) or n.name != 'admit']
new_rest=[n for n in new_tree.body if not isinstance(n,ast.FunctionDef) or n.name != 'admit']
assert [ast.dump(n,include_attributes=False) for n in old_rest] == [ast.dump(n,include_attributes=False) for n in new_rest]
with (HERE/'run.py').open('x') as f:f.write(modified)
audit=read(HERE/'audit_reference.json')
assert audit['passed'] and len(audit['scalar_gate_cases']) == 17
assert not {'numpy','torch','flopscope'} & set(sys.modules)
write(HERE/'source_copies.json',dict(copies=copies,common_original=pin(OLD/'common.py'),common_successor=pin(HERE/'common.py'),
    common_inverse_exact=True,runner_original=pin(OLD/'run.py'),runner_successor=pin(HERE/'run.py'),runner_inverse_exact=True,
    only_endpoint_constant_and_admission_changed=True,all_other_runner_AST_exact=True,
    changes=[dict(before=a,after=b) for a,b in changes],scientific_code_rosters_gates_unchanged=True,
    numerical_imports=False,target_arrays_opened=False,estimator_calls=0,admission_executed=False,before_native40_score=True))
external=[OLD/'sources_sealed.json',OLD/'common.py',OLD/'run.py',NATIVE/'sources_sealed.json',
    NATIVE/'endpoint_sources_sealed.json',NATIVE/'source_copies.json',QUAL/'controller_result.json',QUAL/'qualification.json']
pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in external]+read(OLD/'sources_sealed.json')['pins']
write(HERE/'sources_sealed.json',dict(pins=list({p['path']:p for p in pins}.values()),metadata_only=True,
    before_native40_score=True,admission_executed=False,no_worker_launched=True,scientific_code_and_gates_byte_identical=True))
print(dict(prepared=True,sources=pin(HERE/'sources_sealed.json'),admission_executed=False),flush=True)
