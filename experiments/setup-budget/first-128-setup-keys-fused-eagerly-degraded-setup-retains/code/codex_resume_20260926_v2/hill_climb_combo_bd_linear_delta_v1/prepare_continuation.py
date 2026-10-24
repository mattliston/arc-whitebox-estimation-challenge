"""Reuse repaired BD linear boundary recipe for combined linear plus delta; saved N2 outputs, no model execution."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, sys
D=Path(__file__).resolve().parent
H=D.parent
T=H/'hill_climb_combo_bd_linear_v1'
LIGHT=H.parents[4]/'.venv/bin/python'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def pin(path):
    path=Path(path).resolve(); raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

def once(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n')

def admit():
    cp=pin(D/'boundary_config.json'); c=json.loads(Path(cp['path']).read_text())
    np=pin(D/'native_screen_continuation.json'); nc=json.loads(Path(np['path']).read_text())
    rp=pin(nc['stages']['2']['result']); r=json.loads(Path(rp['path']).read_text())
    gp=pin(nc['stages']['2']['assessment_guard']); g=json.loads(Path(gp['path']).read_text())
    assert r['config']==np and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==c['candidate'] and abs(r['raw_pct'])<=.01
    assert g['command']==[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None
    assert g['max_swap_growth_bytes']==0
    once(D/'boundary_admission.json',dict(root_admitted=True,source_changes=False,serving_qualified=False,
        config=cp,assessor=c['engine'],candidate=c['candidate'],native_config=np,native2=rp,
        native2_guard=gp,source_proof=c['build'],additional6_requires_boundary_pass=True,
        original_pilot_preserved=True,no_model_rerun_for_reassessment=True,
        official_residual_limit_s=.4,strict_residual_upper_bound=True,prospective_s34_reference=True))
    print(json.dumps(pin(D/'boundary_admission.json')))

def review_native8():
    np=pin(D/'native_screen_continuation.json');n=json.loads(Path(np['path']).read_text())
    rp=pin(n['stages']['8']['result']);r=json.loads(Path(rp['path']).read_text())
    gp=pin(n['stages']['8']['assessment_guard']);g=json.loads(Path(gp['path']).read_text())
    assert r['config']==np and r['candidate']==n['arms']['candidate']['source']
    assert r['passed_integrity'] and r['admitted'] and all(r['gates'].values())
    assert g['command']==[str(NATIVE),'-B',n['engine']['path'],np['path'],np['sha256'],'8']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None
    assert g['max_swap_growth_bytes']==0 and not g['remaining_rows']
    passed=abs(r['raw_pct'])<=.01
    once(D/'native8_root_review.json',dict(passed=passed,config=np,native8=rp,guard=gp,abs_raw_pct=abs(r['raw_pct']),max_abs_raw_pct=.01,no_model_rerun=True,serving_qualified=False))
    assert passed
    print(json.dumps(pin(D/'native8_root_review.json')))

def main():
    if sys.argv[1:]==['--review-native8']:
        review_native8(); return
    if sys.argv[1:]==['--admit-boundary']:
        admit(); return
    assert not sys.argv[1:]
    old_assessor=T/'boundary_assess_s34_v2.py'; old_config=T/'boundary_config_v2.json'
    assert pin(old_assessor)['sha256']=='5481fcc3cd1193757a22d7399c62bf92edbe6113b64944eeeafbf8f224f6380c'
    assert pin(old_config)['sha256']=='e3220f159483a9c6b070bb76f6e6d315d7b64e8a37714cc1b64342142abb12e2'
    pilot=pin(D/'native_screen.json')
    assert pilot['sha256']=='2b0c97146cb387d961c2c786841777e2ba41656346371d2398528a66025bc545'
    nc=json.loads(Path(pilot['path']).read_text()); original=copy.deepcopy(nc)
    r=json.loads((D/'native2_assessment.json').read_text())
    assert r['config']==pilot and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==pin(D/'candidate.py') and r['parity_layers']==[0]
    assert nc['parity_layers']==[0] and r['configured_parent_bytes_exact'] is True
    for size,wins in (('2',2),('8',6)):
        gates=nc['stages'][size]['gates']
        assert gates['adjusted_pct_at_most']==-.08 and gates['wins_min']==wins
        assert gates['max_per_network_regression_pct']==(.1 if size=='2' else None) and gates['strict_adjusted'] is False
    build=json.loads((D/'build.json').read_text())
    assert build['candidate']==pin(D/'candidate.py') and build['source_inverse_exact']
    source=(D/'candidate.py').read_text(); parent=Path(build['parent']['path']).read_text()
    inv=json.loads((D/'source_inverse.json').read_text())
    assert source.count(inv['after'])==1 and source.replace(inv['after'],inv['before'],1)==parent
    assert abs(r['raw_pct'])<=.01
    assert build['comparator']==nc['arms']['base']['source']==nc['arms']['parity_parent']['source']
    old=old_assessor.read_text(); new=old
    def replace_one(before,after):
        nonlocal new
        assert new.count(before)==1, before
        new=new.replace(before,after,1)
    replace_one('Prospective BD linear boundary2','Prospective BD linear plus delta boundary2')
    replace_one("'combo-bd-linear-s34-boundary-native2-v1'","'combo-bd-linear-delta-s34-boundary-native2-v1'")
    for name in ('boundary_admission','boundary_assessment','boundary_assess_guard'):
        before=repr(name+'_v2.json');assert before in new
        new=new.replace(before,repr(name+'.json'))
    start=new.index('    # Exact source inverse of the one BD algebra fusion; score reference is BD1078.')
    end=new.index("    assert nc['parity_layers']",start)
    proof="""    # Two exact source inverses: delta -> fixed linear fusion -> retained BD1078.
    assert a['source_proof'] == c['build']
    assert build['comparator'] == c['native_comparator'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']
    assert build['comparator']['sha256'] == '1078e51058e7674e9a322ebfee10ffa8cb124b70a68b1127cc6011ee274b0dde'
    source = q.checked(c['candidate']).decode()
    inv = json.loads(q.checked(build['source_inverse']))
    assert inv['parent'] == build['parent'] and inv['exact_inverse']
    assert source.count(inv['after']) == 1 and source.replace(inv['after'], inv['before'], 1) == q.checked(build['parent']).decode()
    lb = json.loads(q.checked(c['linear_build']))
    li = json.loads(q.checked(lb['source_inverse']))
    linear = q.checked(lb['candidate']).decode()
    assert lb['candidate'] == build['parent'] and lb['parent'] == build['comparator']
    assert li['parent'] == build['comparator'] and li['exact_inverse'] and li['fallback_verbatim']
    assert linear.count(li['after']) == 1 and linear.replace(li['after'], li['before'], 1) == q.checked(build['comparator']).decode()
    assert build['changed_parent_functions'] == lb['changed_parent_functions'] == ['_bd_update']
    retained = json.loads(q.checked(c['score_parent_retained']))
    assert retained['status'] == 'RETAINED_COMPONENT' and retained['candidate'] == build['comparator']
    parent_n8 = json.loads(q.checked(retained['native8']))
    parent_boundary = json.loads(q.checked(retained['boundary']))
    assert parent_n8['candidate'] == build['comparator'] and parent_n8['admitted'] and parent_n8['passed_integrity'] and all(parent_n8['gates'].values())
    assert parent_boundary['candidate'] == build['comparator'] and parent_boundary['passed'] and parent_boundary['receipt_checks_passed']
    increment = json.loads(q.checked(c['root_increment_review']))
    assert increment['passed'] and increment['abs_raw_pct'] <= increment['max_abs_raw_pct'] == .01
    assert increment['increment_vs_linear_pct'] <= -increment['minimum_increment_gain_pct'] and increment['minimum_increment_gain_pct'] == .015
    assert increment['assessment_sha256'] == nc['continuation']['pilot_result']['sha256']
    assert increment['linear_assessment_sha256'] == c['linear_native2']['sha256']
    q.checked(c['linear_native2'])
    assert abs(r['raw_pct']) <= .01
"""
    new=new[:start]+proof+new[end:]
    for name in ('paired_boundary_assessment.json','paired_boundary_assess_guard.json'):
        assert new.count(repr(name))==old.count(repr(name))==1
    compile(new,str(D/'boundary_assess_s34.py'),'exec')
    assessor=D/'boundary_assess_s34.py'
    with assessor.open('x') as f:f.write(new)
    diff=D/'boundary_adapter.diff'
    with diff.open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(old_assessor),tofile=str(assessor))))
    c=json.loads(old_config.read_text())
    c.update(schema='combo-bd-linear-delta-s34-boundary-native2-v1',engine=pin(assessor),candidate=pin(D/'candidate.py'),
        build=pin(D/'build.json'),source_protocol=pin(D/'protocol.json'),native_comparator=build['comparator'],
        native_config_path=str(D/'native_screen_continuation.json'),tag=D.name+'_boundary_feasibility_v1',
        setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']),native2_gates=copy.deepcopy(nc['stages']['2']['gates']),
        linear_build=pin(T/'build.json'),linear_native2=pin(T/'native2_assessment.json'),root_increment_review=pin(D/'root_increment_review.json'))
    assert c['setup_counts']==dict(bound=687,lvfuse=649,psdb_fused=9,pair_block_indices=1,read7_fused=4)
    assert c['parent']['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    c['interpretation']='Own combined BD linear plus delta boundary. Score comparator retained BD1078; original S34 residual and S32 runtime controls remain distinct. Fixed linear reassociation gates unchanged; mandatory native2 absolute raw change<=.01%.'
    bp=D/'boundary_config.json';once(bp,c);bp=pin(bp)
    nc['stages']['2']['result']=str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard']=str(D/'native2_continuation_assess_guard.json')
    req=nc['boundary_before8'];req.update(receipt=str(D/'boundary_assessment.json'),assessor=pin(assessor),
        assessment_guard=str(D/'boundary_assess_guard.json'),assessment_command=[str(LIGHT),'-B',str(assessor),bp['path'],bp['sha256']])
    nc['continuation']=dict(pilot_config=pilot,pilot_result=pin(D/'native2_assessment.json'),pilot_assessment_guard=pin(D/'native2_assess_guard.json'),
        no_model_rerun=True,same_saved_native2_outputs=True,no_score_gate_changes=True)
    nc['provenance']+=[pin(__file__),pin(assessor),pin(diff),bp,pin(old_assessor),pin(old_config),
        *[c[k] for k in ('score_parent_retained','s34_native_config','parent_boundary','s34_bank','s34_evaluation','linear_build','linear_native2','root_increment_review','metadata_repair')],
        *[nc['continuation'][k] for k in ('pilot_config','pilot_result','pilot_assessment_guard')]]
    nc['interpretation']='Combined BD linear plus delta continuation: unchanged saved N2, own boundary, fixed six and fixed linear N8 gates and native2 raw-change gate. No broad/bank qualification.'
    nc['prospective_scope']='saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert all(nc[k]==original[k] for k in ('engine','arms','parity_layers','future_only','raw_error_veto'))
    assert all(nc['stages'][s][k]==original['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
    np=D/'native_screen_continuation.json';once(np,nc);np=pin(np)
    b6=nc['arms']['candidate']['blocks']['6']
    commands=dict(native2_reassessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2'],
        admission=[str(LIGHT),'-B',str(Path(__file__).resolve()),'--admit-boundary'],
        boundary_census=[str(LIGHT),'-B',c['driver']['path'],c['candidate']['path'],'35,100',c['tag']],
        boundary_assessment=req['assessment_command'],
        native6=[str(NATIVE),'-B',b6['driver']['path'],c['candidate']['path'],','.join(map(str,b6['ranks'])),b6['tag']],
        native8_assessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'8'],
        mandatory_native8_raw_review=[str(LIGHT),'-B',str(Path(__file__).resolve()),'--review-native8'])
    once(D/'continuation_commands.json',dict(commands=commands,config=np,boundary_config=bp,assessor=pin(assessor),
        native2_guard=nc['stages']['2']['assessment_guard'],boundary_guard=str(D/'boundary_guard.json'),
        boundary_log=str(D/'boundary.log'),boundary_assessment_guard=req['assessment_guard'],native6_guard=b6['guard'],
        native8_guard=nc['stages']['8']['assessment_guard']))
    print(json.dumps(dict(config=np,boundary=bp,assessor=pin(assessor),commands=pin(D/'continuation_commands.json'))))

if __name__=='__main__':main()
