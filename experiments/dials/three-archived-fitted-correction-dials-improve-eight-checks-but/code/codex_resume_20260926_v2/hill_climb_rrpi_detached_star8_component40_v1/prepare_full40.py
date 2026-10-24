"""Conditional stdlib binding of existing fixed40 runners; no numerical jobs or launch."""
from pathlib import Path
import copy
import hashlib
import json
import sys
D=Path(__file__).resolve().parent
H=D.parent
F=H/'hill_climb_prefix128_fusion_v1'
E=D/'valid40_engine_v1'

def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def checked(r):
    assert pin(r['path'])==r,r['path']
    return json.loads(Path(r['path']).read_text())
def once(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def guard(p,cmd,seconds):
    r=pin(p);g=checked(r)
    assert g['command']==cmd and g['completed'] and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] and g['remaining_rows']==[] and g['max_swap_growth_bytes']==0
    assert 0<g['elapsed_s']<=seconds and 0<g['max_rss_bytes']<g['rss_trigger_bytes']==7500000000
    return r

def main():
    assert len(sys.argv)==2
    sp=pin(D/'full40_source_plan.json');assert sp['sha256']==sys.argv[1]
    plan=checked(sp);assert plan['schema']=='detached-star8-component40-source-v1'
    assert plan['prepare']==pin(__file__)
    for r in plan['pins']:assert pin(r['path'])==r,r['path']
    policy=checked(plan['root_policy']); component=checked(plan['component_plan'])
    assert policy['root_selected'] and policy['source_unchanged'] and component['research_permission']
    candidate=component['candidate']; assert candidate==policy['candidate'] and pin(candidate['path'])==candidate
    assert candidate['sha256']=='640f8ca12e9ea72b9cc534fe57e35e3d8973a8a32a2836e2b053c27ebdc72734'
    assert policy['component_gates']==dict(raw_pct_strictly_below=0.0,adjusted_pct_at_most=-.15,wins_min=24,ci_upper_below=0.0,max_per_network_regression_pct=10.0)
    assert policy['bank_gates']==dict(adjusted_pct_at_most=-1.0,wins_min=28,ci_upper_below=0.0,max_per_network_regression_pct=10.0)
    assert policy['own_boundary_required'] and policy['bank_requires_own_serving20_setup3_contract'] and policy['original_fixed40_blocks']
    assert policy['adaptive_reused_validation'] and policy['no_average_or_other_checkpoints']
    original=checked(component['original_failed_assessment']); followup=checked(component['followup_failed_assessment'])
    assert original['admitted'] is False and original['passed'] is False
    assert followup['passed'] is False and followup['retained_evidence_passed'] is False
    assert followup['original_failure_preserved'] and followup['original_admitted'] is False
    assert followup['receipt_checks_passed'] and followup['all16_finite'] and followup['layer0_byte_exact']
    assert candidate==original['candidate']==followup['candidate']
    assert policy['existing_failures_preserved']==[component['original_failed_assessment'],component['followup_failed_assessment']]
    rp=pin(D/'resource_plan.json');rplan=checked(rp)
    rr=pin(D/'resource_assessment.json');resource=checked(rr)
    assert resource['schema']=='detached-star8-component-resource-assessment-v1'
    assert resource['initial_VALID8_remains_failed'] is True
    assert resource['first_accuracy_assessment']==component['followup_failed_assessment'] and resource['original_failed_assessment']==component['original_failed_assessment']
    assert resource['resource_passed'] is True and resource['receipt_checks_passed'] is True
    assert resource['gates'] and all(resource['gates'].values())
    assert resource['candidate']==candidate==rplan['candidate']
    assert resource['component_plan']==plan['component_plan'] and resource['plan']==rp
    assert resource['assessor']==plan['resource_assessor']
    assert rplan['light']==plan['light']
    rg=guard(D/'resource_assessment_guard.json',[plan['light'],'-B',resource['assessor']['path'],rp['path'],rp['sha256']],120)
    assert resource['runtime']['gate_s']==105 and resource['runtime']['projected_worst_wall_s']<=105
    assert resource['runtime']['native2_boundary_disagreement_pp']<=5
    assert resource['residual']['gate_s']==.4 and resource['residual']['strict_upper_bound'] and resource['residual']['projected_s']<.4
    for r in resource['evidence']:assert pin(r['path'])==r
    old=checked(plan['prior_broad_config']);br=checked(plan['prior_broad_assessment']);bank=checked(plan['official_bank'])
    assert bank['submission_id']==333111 and bank['candidate']==old['candidate']['source']==original['baseline']==followup['baseline']
    assert bank['telemetry']['completed']==100 and bank['telemetry']['failed']==0
    assert bank['local_validation_reference']==plan['prior_broad_assessment']
    assert br['passed'] and br['receipt_checks_passed'] and br['validation_score_passed'] and br['candidate']==bank['candidate']
    assert list(old['candidate']['blocks'])==['b1','b2','b3','b4']
    assert [x['block'] for x in br['blocks']]==['b1','b2','b3','b4']
    score_reference=dict(submission_id=333111,source=bank['candidate'],bank_receipt=plan['official_bank'])
    arm=copy.deepcopy(old['candidate'])
    for b in br['blocks']:
        arm['blocks'][b['block']]['artifacts']=b['arms']['candidate']
        for r in b['arms']['candidate'].values():
            if isinstance(r,dict) and 'path' in r:assert pin(r['path'])==r
        for r in b['arms']['candidate']['predictions'].values():assert pin(r['path'])==r
    controls=dict(schema='saved-score-controls-s35-v1',score_reference=score_reference,arm=arm,
        assessment=plan['prior_broad_assessment'],provenance=[sp,plan['prior_broad_config'],plan['prior_broad_assessment'],plan['official_bank']])
    cp_path=D/'broad_config.json'
    for path in (D/'score_controls_s35.json',cp_path,D/'broad_admission.json',D/'full40_commands.json'):assert not path.exists()
    once(D/'score_controls_s35.json',controls)
    c=copy.deepcopy(old)
    for k in ('native_s34_controls','cheap_gate','incremental_reference','native_config'):c.pop(k,None)
    c.update(engine=plan['engine'],runner=plan['runner'],native_engine=plan['panel_helpers'],control_native_engine=plan['panel_helpers'],
        root_policy=plan['root_policy'],component_plan=plan['component_plan'],component_gates=policy['component_gates'],
        resource_plan=rp,resource_assessment=rr,resource_assessor=plan['resource_assessor'],resource_assessment_guard=rg,resource_runtime=plan['light'],
        controls=pin(D/'score_controls_s35.json'),score_reference=score_reference)
    c['candidate']['source']=candidate
    for key,b in c['candidate']['blocks'].items():
        assert 'artifacts' not in b
        b['tag']=D.name+'_valid40_'+key+'_v1';b['guard']=str(D/f'broad_{key}_guard.json')
    c['provenance']=plan['pins']+[sp,pin(__file__),rp,rr,rg,c['controls']]
    c['interpretation']='Fixed original update8 endpoint; research permission for original four10-case blocks against actual S35 controls after own normal resources. Earlier two/eight-case screen failures preserved. All40 rerun including first8. Component raw<0/adjusted<=-.15/24wins/negativeCI/maxreg10; bank remains >=1%/28wins/negativeCI/maxreg10 plus own serving. Adaptive reused validation; CI descriptive, no pristine coverage claim.'
    assert c['score_gates']==dict(adjusted_pct_at_most=-1,wins_min=28,ci_upper_below=0,max_per_network_regression_pct=10)
    once(cp_path,c);cp=pin(cp_path)
    once(D/'broad_admission.json',dict(passed=True,root_admitted=True,research_permission=True,research_only=True,config=cp,candidate=candidate,
         root_policy=plan['root_policy'],component_plan=plan['component_plan'],discovery_evidence=component['followup_failed_assessment'],
         original_screens_remain_failed=True,boundary=rr,boundary_guard=rg,
         baseline_submission=333111,bank_min_gain_pct=1.0,own_fixed40_and_serving_still_required=True,source_plan=sp))
    ap=pin(D/'broad_admission.json')
    once(D/'full40_commands.json',dict(config=cp,admission=ap,thread_env=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1'),
        run=[plan['runtime'],'-B',plan['runner']['path'],cp['path'],cp['sha256'],ap['sha256']],
        assess=[plan['runtime'],'-B',plan['engine']['path'],cp['path'],cp['sha256'],ap['sha256']],
        assessment_guard=str(D/'broad_assess_guard.json'),assessment_limits=dict(seconds=120,rss_bytes=7500000000),
        block_limits=dict(seconds=420,rss_bytes=7500000000),models_executed=False,serving_and_release_not_authorized_here=True))
    print(json.dumps(dict(config=cp,admission=ap,commands=pin(D/'full40_commands.json'))))

if __name__=='__main__':main()
