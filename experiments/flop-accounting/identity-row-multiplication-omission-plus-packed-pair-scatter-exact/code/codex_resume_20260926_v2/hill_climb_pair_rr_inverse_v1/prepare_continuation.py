"""Source-only RR+PI retention adapter; preserve the established normal resource formula."""
from pathlib import Path
import copy, difflib, hashlib, json, sys
D=Path(__file__).resolve().parent; H=D.parent
T=H/'hill_climb_response64_prejoin_v1'; BASE=H/'hill_climb_prefix128_fusion_v1'
LIGHT=H.parents[4]/'.venv/bin/python'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
def pin(path):
    path=Path(path).resolve();b=path.read_bytes();return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(path):return json.loads(Path(path).read_text())
def checked(p):assert pin(p['path'])==p;return read(p['path'])
def once(path,x):
    with Path(path).open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n')
def undo(source,inv):
    assert inv['exact_inverse']
    if inv.get('suffix'):
        assert source.endswith(inv['suffix']);source=source[:-len(inv['suffix'])]
    for e in reversed(inv['edits']):
        assert source.count(e['after'])==1;source=source.replace(e['after'],e['before'],1)
    assert source==Path(inv['parent']['path']).read_text() and pin(inv['parent']['path'])==inv['parent']
    return source

def admit():
    cp=pin(D/'boundary_config.json');c=checked(cp);np=pin(D/'native_screen_continuation.json');nc=checked(np)
    rp=pin(nc['stages']['2']['result']);r=checked(rp);gp=pin(nc['stages']['2']['assessment_guard']);g=checked(gp)
    assert r['config']==np and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==c['candidate'] and r['parity_layers']==list(range(16)) and r['all16_parent_bytes_exact'] and r['all16_predictions_finite']
    assert g['command']==[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None
    assert g['max_swap_growth_bytes']==0 and not g['remaining_rows'] and 0<g['max_rss_bytes']<7500000000 and 0<g['elapsed_s']<=120
    launch=checked(c['root_boundary_launch']);assert launch['candidate']==c['candidate'] and launch['primary']==nc['continuation']['pilot_result']
    assert launch['incremental']==c['incremental_review'] and checked(c['incremental_review'])['passed']
    once(D/'boundary_admission.json',dict(root_admitted=True,source_changes=False,serving_qualified=False,config=cp,assessor=c['engine'],candidate=c['candidate'],native_config=np,native2=rp,native2_guard=gp,source_proof=c['build'],root_boundary_launch=c['root_boundary_launch'],incremental_review=c['incremental_review'],additional6_requires_boundary_pass=True,original_pilot_preserved=True,no_model_rerun_for_reassessment=True,formal_receipt_created_before_census=False,chronology='Root admitted and launched the existing census from actual primary and auxiliary PASS; this formal binding follows that launch and does not authorize a rerun.',official_residual_limit_s=.4,strict_residual_upper_bound=True,retention_only=True))
    print(json.dumps(pin(D/'boundary_admission.json')))

def main():
    if sys.argv[1:]==['--admit-boundary']:admit();return
    assert not sys.argv[1:]
    pilot=pin(D/'native_screen.json');assert pilot['sha256']=='bef77c1cf96d906815f5160ca00e970386a171d9d8c2115c43e8761a45746d9b'
    nc=checked(pilot);original=copy.deepcopy(nc);r=read(D/'native2_assessment.json')
    assert r['config']==pilot and r['admitted'] and r['passed_integrity'] and all(r['gates'].values()) and r['all16_parent_bytes_exact']
    assert r['candidate']==pin(D/'candidate.py') and nc['parity_layers']==r['parity_layers']==list(range(16))
    for s,w,ci,add in [('2',2,None,False),('8',8,0,True)]:
        assert nc['stages'][s]['gates']==dict(adjusted_pct_at_most=-.02,strict_adjusted=False,wins_min=w,ci_upper_below=ci,max_per_network_regression_pct=0.,additional6_gain_required=add)
    build=read(D/'build.json');inv=checked(build['source_inverse']);rr=checked(inv['rr_to_p128'])
    source=undo(Path(build['candidate']['path']).read_text(),inv);undo(source,rr['source_inverse'])
    assert rr['source_inverse']['parent']==nc['arms']['base']['source']==pin(BASE/'candidate.py')
    pi=checked(inv['retained_pi_inverse']);assert inv['edits']==pi['edits'] and inv['suffix']==pi['suffix']
    auxp=pin(D/'native2_incremental_review_v1.json');aux=checked(auxp)
    assert aux['passed'] and aux['candidate']==r['candidate'] and aux['config']==pilot and aux['primary']==pin(D/'native2_assessment.json')
    assert aux['incremental_bill_pct']<0 and all(x['all16_file_exact'] and x['bill_pct']<0 for x in aux['per_network'])
    oldp=pin(T/'boundary_assess_s34.py');assert oldp['sha256']=='17fd367e47ddfcb04067f4889ba392245b195de1d20c2108fe02008d952fe328'
    old=Path(oldp['path']).read_text();new=old
    def rep(a,b):
        nonlocal new
        assert new.count(a)==1,a;new=new.replace(a,b,1)
    rep('Prospective response64 prejoin boundary2','Prospective RR plus PI boundary2')
    rep("'response64-prejoin-s34-boundary-native2-v1'","'rr-pi-s34-boundary-native2-v1'")
    rep("assert build['candidate'] == c['candidate'] and build['source_inverse']['exact_inverse'] is True","assert build['candidate'] == c['candidate']\n    assert json.loads(q.checked(build['source_inverse']))['exact_inverse'] is True")
    start=new.index('    # One response call plus helper suffix;');end=new.index("    ei = json.loads(q.checked(c['engine_inverse']))",start)
    proof='''    # Exact RR inverse followed by unchanged retained PI transplantation.
    assert a['source_proof'] == c['build']
    assert c['native_comparator'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']
    inv = json.loads(q.checked(build['source_inverse']))
    rr = json.loads(q.checked(inv['rr_to_p128']))
    assert inv['parent'] == rr['candidate'] and rr['source_inverse']['parent'] == c['native_comparator']
    source = q.checked(c['candidate']).decode()
    for inverse in (inv, rr['source_inverse']):
        assert inverse['exact_inverse']
        if inverse.get('suffix'):
            assert source.endswith(inverse['suffix'])
            source = source[:-len(inverse['suffix'])]
        for edit in reversed(inverse['edits']):
            assert source.count(edit['after']) == 1
            source = source.replace(edit['after'], edit['before'], 1)
        assert source == q.checked(inverse['parent']).decode()
    assert source == q.checked(c['native_comparator']).decode()
    assert c['native_comparator']['sha256'] == 'f9f4790b3c65f730404425e8923c767dc2f895fdcc1e14f14cc649286ad3b69f'
    pi = json.loads(q.checked(inv['retained_pi_inverse']))
    assert inv['edits'] == pi['edits'] and inv['suffix'] == pi['suffix']
    auxiliary = json.loads(q.checked(c['incremental_review']))
    assert auxiliary['passed'] and auxiliary['candidate'] == c['candidate']
    assert auxiliary['config'] == nc['continuation']['pilot_config'] and auxiliary['primary'] == nc['continuation']['pilot_result']
    assert auxiliary['incremental_bill_pct'] < 0
    assert all(x['all16_file_exact'] and x['bill_pct'] < 0 for x in auxiliary['per_network'])
    for key in ('checker', 'candidate_rows', 'candidate_manifest', 'primary'):
        q.checked(auxiliary[key])
    launch = json.loads(q.checked(c['root_boundary_launch']))
    assert a['root_boundary_launch'] == c['root_boundary_launch'] and a['incremental_review'] == c['incremental_review']
    assert launch['candidate'] == c['candidate'] and launch['primary'] == nc['continuation']['pilot_result'] and launch['incremental'] == c['incremental_review']
    assert launch['command'] == [str(q.LIGHT), '-B', c['driver']['path'], c['candidate']['path'], '35,100', c['tag']]
    assert launch['guard_record'] == str(d/'boundary_guard.json') and launch['seconds'] == 240 and launch['rss_bytes'] == 7500000000
    assert a['formal_receipt_created_before_census'] is False
'''
    new=new[:start]+proof+new[end:]
    rep("protocol['parent'] == c['native_comparator']","protocol['score_comparator'] == c['native_comparator']")
    rep("protocol['parity_layers'] == [0]","protocol['parity_layers'] == list(range(16))")
    rep("assert nc['parity_layers'] == r['parity_layers'] == [0]","assert nc['parity_layers'] == r['parity_layers'] == list(range(16))\n    assert r['all16_parent_bytes_exact']")
    start=new.index('    restored_arms = ');end=new.index("    for stage_id in ('2', '8'):",start)
    new=new[:start]+"    assert nc['arms'] == original_config['arms']\n"+new[end:]
    assert new[new.index('    control_pin = q.fixed'):]==old[old.index('    control_pin = q.fixed'):]
    for name in ('paired_boundary_assessment.json','paired_boundary_assess_guard.json'):assert new.count(repr(name))==old.count(repr(name))==1
    assessor=D/'boundary_assess_s34.py';compile(new,str(assessor),'exec');once(assessor,new)
    diff=D/'boundary_adapter.diff';once(diff,''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=oldp['path'],tofile=str(assessor))))
    c=read(T/'boundary_config.json')
    c.update(schema='rr-pi-s34-boundary-native2-v1',engine=pin(assessor),candidate=pin(D/'candidate.py'),build=pin(D/'build.json'),source_protocol=pin(D/'protocol.json'),native_config_path=str(D/'native_screen_continuation.json'),tag=D.name+'_boundary_feasibility_v1',setup_counts=nc['arms']['candidate']['setup']['counts'],native2_gates=nc['stages']['2']['gates'],native_engine=nc['engine'],incremental_review=auxp,root_boundary_launch=pin(D/'root_boundary_launch_v1.json'),official_score_bank=pin(H/'official_bank_S35_v1.json'))
    c['interpretation']='RR plus retained PI; actual official P128/S35 native score controls, all16 exact parity, unchanged exact .02% N2/N8 gates. Existing own normal census; unchanged S34 residual/S32 wall and CPU resource formula. Retention only; no broad, serving or release authority.'
    bp=D/'boundary_config.json';once(bp,c);bp=pin(bp)
    nc['stages']['2']['result']=str(D/'native2_continuation_assessment.json');nc['stages']['2']['assessment_guard']=str(D/'native2_continuation_assess_guard.json')
    req=nc['boundary_before8'];req.update(receipt=str(D/'boundary_assessment.json'),assessor=pin(assessor),assessment_guard=str(D/'boundary_assess_guard.json'),assessment_command=[str(LIGHT),'-B',str(assessor),bp['path'],bp['sha256']])
    nc['continuation']=dict(pilot_config=pilot,pilot_result=pin(D/'native2_assessment.json'),pilot_assessment_guard=pin(D/'native2_assess_guard.json'),no_model_rerun=True,same_saved_native2_outputs=True,no_score_gate_changes=True,no_control_changes=True)
    nc['provenance'] += [pin(__file__),pin(assessor),pin(diff),bp,oldp,pin(T/'boundary_config.json'),auxp,c['root_boundary_launch'],c['official_score_bank'],pilot,nc['continuation']['pilot_result'],nc['continuation']['pilot_assessment_guard']]
    nc['interpretation']=c['interpretation'];nc['prospective_scope']='retention_only_saved_native2_own_boundary_fixed6_native8';nc['no_further_stage_authority']=False
    assert nc['arms']==original['arms']
    for s in ('2','8'):
        for k in ('gates','assessment_limits'):assert nc['stages'][s][k]==original['stages'][s][k]
    np=D/'native_screen_continuation.json';once(np,nc);np=pin(np);b6=nc['arms']['candidate']['blocks']['6']
    commands=dict(native2_reassessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2'],admission=[str(LIGHT),'-B',str(Path(__file__).resolve()),'--admit-boundary'],boundary_census=[str(LIGHT),'-B',c['driver']['path'],c['candidate']['path'],'35,100',c['tag']],boundary_assessment=req['assessment_command'],native6=[str(NATIVE),'-B',b6['driver']['path'],c['candidate']['path'],','.join(map(str,b6['ranks'])),b6['tag']],native8_assessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'8'])
    assert commands['boundary_census']==checked(c['root_boundary_launch'])['command']
    once(D/'continuation_commands.json',dict(commands=commands,config=np,boundary_config=bp,assessor=pin(assessor),native2_guard=nc['stages']['2']['assessment_guard'],boundary_guard=str(D/'boundary_guard.json'),boundary_log=str(D/'boundary.log'),boundary_assessment_guard=req['assessment_guard'],native6_guard=b6['guard'],native8_guard=nc['stages']['8']['assessment_guard'],limits=dict(native2_reassessment=120,boundary_census=240,boundary_assessment=120,native6=300,native8_assessment=120,rss_bytes=7500000000,zero_added_swap=True),environment=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1'),boundary_already_run_do_not_repeat=True,retention_only=True))
    print(json.dumps(dict(config=np,boundary=bp,assessor=pin(assessor),commands=pin(D/'continuation_commands.json'))))
if __name__=='__main__':main()
