"""Narrow, reversible adaptation of the archived measured-cost tools."""
import ast
from common import *

assert sha(SOURCE)==PARENT_SHA
proofs=[]
def derive(original_name,new_name,edits):
    original=D/original_name;old=original.read_text();new=old
    for before,after in edits:
        assert new.count(before)==1,(original_name,before,new.count(before))
        new=new.replace(before,after,1)
    restored=new
    for before,after in reversed(edits):
        assert restored.count(after)==1,(new_name,after)
        restored=restored.replace(after,before,1)
    assert restored==old;ast.parse(new)
    target=HERE/new_name
    with target.open('x') as f:f.write(new)
    proofs.append(dict(original=pin(original),generated=pin(target),exact_inverse=True,edits=[dict(before=a,after=b) for a,b in edits]))

derive('lazy_cost_census_v1/census.py','census.py',[
    ("SOURCE=D/'lazy_rotation_objective_v1/candidate.py'","SOURCE=D/'full_lower_covariance_edge_packed_v1/candidate.py'"),
    ("for row in seal['sources']+seal['external_pins']:","for row in seal['pins']:"),
    ("control_path=D/('lazy_rotation_v1/dev/result.json' if rank==35 else 'lazy_rotation_v1/cold100/result.json')", "control_path=D/('full_lower_multilayer_mean_observer_v1/parent_triple/result.json' if rank==35 else 'full_lower_multilayer_mean_observer_v1/parent_cold100/result.json')"),
    ("shape=(tuple(int(x) for x in left.shape),tuple(int(x) for x in right.shape))", "shape=(tuple(int(x) for x in left.shape),tuple(int(x) for x in right.shape))\n                input_dtypes=(str(left.dtype),str(right.dtype))"),
    ("flops=context._flops_used-before,\n                                                  caller=list(origin)","flops=context._flops_used-before,\n                                                  input_dtypes=list(input_dtypes),caller=list(origin)")])

derive('lazy_mixed_fmm_reprice_v3/reprice.py','reprice.py',[
    ("seal = read(HERE / 'sources_sealed.json')", "seal = read(HERE / 'reprice_inputs_sealed.json')"),
    ("inputs = read(HERE / 'inputs.json')", "inputs = read(HERE / 'reprice_inputs.json')"),
    ("assert not (HERE / 'result.json').exists()", "assert not (HERE / 'reprice_result.json').exists()"),
    ("assert len(calls) == 335", "assert len(calls) == row['expected_outermost_mm_calls']\n        assert all(c['input_dtypes']==['float32','float32'] for c in calls)"),
    ("assert sum(min(*x['left'], *x['right']) == 0 for x in calls) == 2", "assert sum(min(*x['left'], *x['right']) == 0 for x in calls) == row['expected_empty_calls']"),
    ("complete_cold_flops=measured,outermost_mm_calls=335", "complete_cold_flops=measured,outermost_mm_calls=len(calls)"),
    ("need=1-3e-9/3.8164492657795035e-9", "need=1-3e-9/inputs['official_adjusted']"),
    ("source_seal_sha256=sha(HERE/'sources_sealed.json')", "source_seal_sha256=sha(HERE/'reprice_inputs_sealed.json')"),
    ("with (HERE/'result.json').open('x') as f:", "with (HERE/'reprice_result.json').open('x') as f:")])

derive('lazy_mixed_fmm_reprice_v3/verify_paths.py','verify_paths.py',[
    ("data=json.loads((HERE/'result.json').read_text())", "data=json.loads((HERE/'reprice_result.json').read_text())"),
    ("inputs=json.loads((HERE/'inputs.json').read_text())", "inputs=json.loads((HERE/'reprice_inputs.json').read_text())"),
    ("assert len(row['details'])==335 and sorted(x['index'] for x in row['details'])==list(range(335))", "assert len(row['details'])==len(actual['outermost_mm_calls']) and sorted(x['index'] for x in row['details'])==list(range(len(actual['outermost_mm_calls'])))"),
    ("all_1340_call_records_reconciled=True", "all_call_records_reconciled=True,call_records_reconciled=sum(len(r['details']) for f in data['families'] for r in f['records'])"),
    ("result_sha256=sha(HERE/'result.json')", "result_sha256=sha(HERE/'reprice_result.json')")])

write(HERE/'build.json',dict(passed=True,proofs=proofs,parent=pin(SOURCE),algorithm_changes=False,qualification='Only parent/control/seal/metadata/call-count/output-name and current official-yardstick adaptations. Meter and integer dynamic program unchanged.'))
print('Three complete-source inverses pass; current-parent tools built.')
