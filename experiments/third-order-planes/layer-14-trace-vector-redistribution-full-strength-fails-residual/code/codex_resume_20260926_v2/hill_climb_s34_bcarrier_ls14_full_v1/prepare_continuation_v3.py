
from pathlib import Path
import json, hashlib, ast, sys
D=Path(__file__).resolve().parent; H=D.parent
LIGHT=H.parents[4]/'.venv/bin/python'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
if sys.argv[1:]==['--admit-boundary']:
    cp=pin(D/'boundary_config_v3.json');c=json.loads(Path(cp['path']).read_text())
    np=pin(D/'native_screen_continuation_v3.json');nc=json.loads(Path(np['path']).read_text())
    rp=pin(nc['stages']['2']['result']);r=json.loads(Path(rp['path']).read_text())
    gp=pin(nc['stages']['2']['assessment_guard']);g=json.loads(Path(gp['path']).read_text())
    assert r['config']==np and r['passed_integrity'] and r['admitted'] and all(r['gates'].values())
    assert r['candidate']==c['candidate']
    assert g['command']==[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None and g['max_swap_growth_bytes']==0
    once(D/'boundary_admission_v3.json',dict(root_admitted=True,source_changes=False,serving_qualified=False,
        config=cp,assessor=c['engine'],candidate=c['candidate'],native_config=np,native2=rp,native2_guard=gp,source_proof=c['build']))
    print(json.dumps(pin(D/'boundary_admission_v3.json')));raise SystemExit
assert len(sys.argv)==1
p=D/'boundary_assess_s34_v2.py';old=p.read_text();new=old
for before,after in [
    ("V1/'paired_boundary_assessment_v2.json'","V1/'paired_boundary_assessment.json'"),
    ("V1/'paired_boundary_assess_guard_v2.json'","V1/'paired_boundary_assess_guard.json'")]:
    assert new.count(before)==1;new=new.replace(before,after,1)
for stem in ('boundary_admission','boundary_assessment','boundary_assess_guard'):
    before="'"+stem+"_v2.json'";after="'"+stem+"_v3.json'"
    assert new.count(before) in (1,2);new=new.replace(before,after)
assert "paired_boundary_assessment_v2" not in new and "paired_boundary_assess_guard_v2" not in new
ast.parse(new)
ap=D/'boundary_assess_s34_v3.py'
with ap.open('x') as f:f.write(new)
c=json.loads((D/'boundary_config_v2.json').read_text())
nc=json.loads((D/'native_screen_continuation_v2.json').read_text())
preserved=[pin(D/f) for f in ('boundary_assessment_v2.json','boundary_assess_guard_v2.json','boundary_config_v2.json','native_screen_continuation_v2.json')]
c.update(engine=pin(ap),native_config_path=str(D/'native_screen_continuation_v3.json'),
    technical_path_repair=dict(reason='Restore exact pinned paired control filenames changed by overly broad prior source replacement.',preserved=preserved,no_numerical_rerun=True))
once(D/'boundary_config_v3.json',c);cp=pin(D/'boundary_config_v3.json')
nc['stages']['2']['result']=str(D/'native2_continuation_v3_assessment.json')
nc['stages']['2']['assessment_guard']=str(D/'native2_continuation_v3_assess_guard.json')
req=nc['boundary_before8'];req.update(assessor=pin(ap),receipt=str(D/'boundary_assessment_v3.json'),
    assessment_guard=str(D/'boundary_assess_guard_v3.json'),
    assessment_command=[str(LIGHT),'-B',str(ap),cp['path'],cp['sha256']])
nc['provenance'] += [pin(__file__),pin(ap),cp,*preserved]
once(D/'native_screen_continuation_v3.json',nc)
print(json.dumps(dict(config=pin(D/'native_screen_continuation_v3.json'),boundary=cp,assessor=pin(ap))))

