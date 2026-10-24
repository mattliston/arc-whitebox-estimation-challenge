import ast
from common import *
seal_check();done('unit');done('solver_diagnostic')
failed=read(HERE/'solver_unit_watch.json')
assert failed['exit_code']==1 and failed['whole_group_retired'] and failed['max_swap_growth_bytes']==0
assert not list(HERE.glob('qual*.npz')) and not list(HERE.glob('train*.npz'))
insertion="seal_check()\nfor repair_pin in read(HERE/'repair_sources_sealed.json')['pins']:check(repair_pin)"
proofs=[]
def derive(old_name,new_name,edits):
    old=(HERE/old_name).read_text();new=old
    for a,b,count in edits:
        assert new.count(a)==count,(old_name,a,new.count(a));new=new.replace(a,b)
    recovered=new
    for a,b,count in reversed(edits):
        assert recovered.count(b)==count;recovered=recovered.replace(b,a)
    assert recovered==old;ast.parse(new)
    with (HERE/new_name).open('x') as f:f.write(new)
    proofs.append(dict(original=pin(HERE/old_name),derived=pin(HERE/new_name),whole_source_inverse=True,edits=edits))
derive('solver_unit.py','solver_unit_v2.py',[
    ("seal_check();rng=",insertion+"\nrng=",1),
    ("    alt=np.linalg.lstsq(aug,yy,rcond=None)[0]/cert['scale']\n    error=float(np.linalg.norm(beta-alt)/max(np.linalg.norm(beta),1e-30));assert error<=2e-10",
     "    scaled_alt=np.linalg.lstsq(aug,yy,rcond=None)[0];alt=scaled_alt/cert['scale']\n    error=float(np.linalg.norm(cert['scaled_beta']-scaled_alt)/max(np.linalg.norm(cert['scaled_beta']),1e-30));assert error<=2e-10\n    prediction_error=float(np.linalg.norm(X@beta-X@alt)/max(np.linalg.norm(X@beta),1e-30));assert prediction_error<=2e-10",1),
    ("augmented_lstsq_relative=error,normal_residual=", "augmented_scaled_lstsq_relative=error,prediction_relative=prediction_error,normal_residual=",1),
    ("HERE/'solver_unit.json'","HERE/'solver_unit_v2.json'",1)])
derive('capture.py','capture_v2.py',[
    ("done('solver_unit')","done('solver_unit_v2')",1),
    ("    args=ap.parse_args();seal_check();done('unit');done('solver_unit_v2')",
     "    args=ap.parse_args();seal_check();done('unit');done('solver_unit_v2')\n    for repair_pin in read(HERE/'repair_sources_sealed.json')['pins']:check(repair_pin)",1)])
derive('qualify.py','qualify_v2.py',[
    ("done('solver_unit')","done('solver_unit_v2')",1),
    ("seal_check();done('unit');done('solver_unit_v2')",insertion+"\ndone('unit');done('solver_unit_v2')",1)])
derive('run.py','run_v2.py',[
    ("seal_check()",insertion,1),
    ("guarded('unit',90,'unit.py')","done('unit')",1),
    ("guarded('solver_unit',90,'solver_unit.py')","guarded('solver_unit_v2',90,'solver_unit_v2.py')",1),
    ("'capture.py'","'capture_v2.py'",1),("'qualify.py'","'qualify_v2.py'",1)])
derive('preflight.py','preflight_v2.py',[
    ("preflight_resources.json","preflight_v2_resources.json",1),
    ("seal_check();rows=[]",insertion+"\nrows=[]",1),
    ("str(HERE/'run.py')","str(HERE/'run_v2.py')",1)])
write(HERE/'repair_build.json',dict(proofs=proofs,original_failure_preserved=True,native_calls_before_repair=0,
    targets_decoded_before_repair=False,solver_model_and_gates_unchanged=True))
names=['solver_unit_v2.py','capture_v2.py','qualify_v2.py','run_v2.py','preflight_v2.py','prepare_repair.py',
    'SOLVER_FIXTURE_REPAIR.md','repair_build.json','solver_diagnostic.py','solver_diagnostic.json',
    'solver_diagnostic_watch.json','solver_unit_watch.json','solver_unit.log']
write(HERE/'repair_sources_sealed.json',dict(pins=[pin(HERE/n) for n in names]))
print(dict(repair_sealed=True,whole_source_inverses=len(proofs),model_unchanged=True),flush=True)
