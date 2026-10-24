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
    plan=checked(sp);assert plan['schema']=='detached-star16-conditional-full40-source-v1'
    assert plan['prepare']==pin(__file__)
    for r in plan['pins']:assert pin(r['path'])==r,r['path']
    protocol=checked(plan['protocol']);assert protocol['updates']==16 and protocol['root_admitted'] is True
    nc_pin=pin(D/'native_config.json');nc=checked(nc_pin)
    n_pin=pin(D/'native_assessment.json');n=checked(n_pin)
    assert n['admitted'] is True and n['passed'] is True and n['valid8_admitted'] is True
    assert n['receipt_checks_passed'] and n['resource_gates_passed'] and n['stage']=='VALID8'
    assert n['all16_finite'] and n['layer0_byte_exact'] and n['single_process_all8_verified']
    assert n['gates8'] and all(n['gates8'].values())
    assert n['config']==nc_pin and nc['protocol']==plan['protocol'] and nc['training_updates']==16
    assert n['candidate']==nc['candidate']==pin(D/'candidate.py') and n['baseline']==protocol['official']
    assert n['assessor']==plan['validation_assessor']==nc['assessor']
    assert n['common']==plan['validation_common']==nc['common']
    m=n['metrics8']
    assert m['n']==8 and m['raw_pct']<=-.1 and m['adjusted_pct']<=-.1 and m['adjusted_wins']>=6
    assert m['max_per_network_adjusted_regression_pct']<=2 and m['raw_ci95_pct'][1]<0 and m['adjusted_ci95_pct'][1]<0
    ng=guard(D/'assessment_guard.json',[plan['runtime'],'-B',n['assessor']['path'],nc_pin['path'],nc_pin['sha256']],120)
    rp=pin(D/'resource_plan.json');rplan=checked(rp)
    rr=pin(D/'resource_assessment.json');resource=checked(rr)
    assert resource['resource_passed'] is True and resource['receipt_checks_passed'] is True
    assert resource['gates'] and all(resource['gates'].values())
    assert resource['candidate']==n['candidate']==rplan['candidate']
    assert resource['first_accuracy_assessment']==n_pin and resource['plan']==rp
    assert resource['assessor']==plan['resource_assessor']
    assert rplan['light']==plan['light']
    rg=guard(D/'resource_assessment_guard.json',[plan['light'],'-B',resource['assessor']['path'],rp['path'],rp['sha256']],120)
    assert resource['runtime']['gate_s']==105 and resource['runtime']['projected_worst_wall_s']<=105
    assert resource['runtime']['native2_boundary_disagreement_pp']<=5
    assert resource['residual']['gate_s']==.4 and resource['residual']['strict_upper_bound'] and resource['residual']['projected_s']<.4
    for r in resource['evidence']:assert pin(r['path'])==r
    old=checked(plan['prior_broad_config']);br=checked(plan['prior_broad_assessment']);bank=checked(plan['official_bank'])
    assert bank['submission_id']==333111 and bank['candidate']==old['candidate']['source']==protocol['official']
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
    for k in ('native_s34_controls','cheap_gate','incremental_reference'):c.pop(k,None)
    c.update(engine=plan['engine'],runner=plan['runner'],native_engine=plan['panel_helpers'],control_native_engine=plan['panel_helpers'],
        native_config=nc_pin,native_assessment=n_pin,validation_assessor=plan['validation_assessor'],validation_common=plan['validation_common'],validation_assessment_guard=ng,
        resource_plan=rp,resource_assessment=rr,resource_assessor=plan['resource_assessor'],resource_assessment_guard=rg,resource_runtime=plan['light'],
        controls=pin(D/'score_controls_s35.json'),score_reference=score_reference)
    c['candidate']['source']=n['candidate']
    for key,b in c['candidate']['blocks'].items():
        assert 'artifacts' not in b
        b['tag']=D.name+'_valid40_'+key+'_v1';b['guard']=str(D/f'broad_{key}_guard.json')
    c['provenance']=plan['pins']+[sp,pin(__file__),nc_pin,n_pin,ng,rp,rr,rg,c['controls']]
    c['interpretation']='Fixed update16 endpoint; new own40 in original four10-case blocks against actual S35 controls. All40 rerun, including first8. Requires passed ownVALID8 and normal resources. Original >=1%/28wins/negativeCI/maxreg10 and full serving requirements unchanged; adaptive development, no pristine CI claim.'
    assert c['score_gates']==dict(adjusted_pct_at_most=-1,wins_min=28,ci_upper_below=0,max_per_network_regression_pct=10)
    once(cp_path,c);cp=pin(cp_path)
    once(D/'broad_admission.json',dict(passed=True,root_admitted=True,config=cp,candidate=n['candidate'],training=n_pin,training_guard=ng,boundary=rr,boundary_guard=rg,
         baseline_submission=333111,bank_min_gain_pct=1.0,own_fixed40_and_serving_still_required=True,source_plan=sp))
    ap=pin(D/'broad_admission.json')
    once(D/'full40_commands.json',dict(config=cp,admission=ap,thread_env=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1'),
        run=[plan['runtime'],'-B',plan['runner']['path'],cp['path'],cp['sha256'],ap['sha256']],
        assess=[plan['runtime'],'-B',plan['engine']['path'],cp['path'],cp['sha256'],ap['sha256']],
        assessment_guard=str(D/'broad_assess_guard.json'),assessment_limits=dict(seconds=120,rss_bytes=7500000000),
        block_limits=dict(seconds=420,rss_bytes=7500000000),models_executed=False,serving_and_release_not_authorized_here=True))
    print(json.dumps(dict(config=cp,admission=ap,commands=pin(D/'full40_commands.json'))))

if __name__=='__main__':main()
