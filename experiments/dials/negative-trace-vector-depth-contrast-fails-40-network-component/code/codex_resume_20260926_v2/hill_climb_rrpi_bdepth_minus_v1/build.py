"""Stdlib-only fixed historical b-depth minus replay on retained RRPI.

No fit, model, target decoding, new coefficient, or baseline execution.
"""
from pathlib import Path
import ast, copy, difflib, hashlib, json

D=Path(__file__).resolve().parent
H=D.parent
G=H/'hill_climb_rrpi_native_joint3_recovery_v1'
U=H/'hill_climb_uniform_gnn_gain_v1'
PARENT=H/'hill_climb_pair_rr_inverse_v1/candidate.py'
PARENT_SHA='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
OLD=H/'candidate_old_both_tail_trace_bounded_v6.py'
OLD_SHA='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
MINUS=H/'candidate_trace_vector_depth_tilt_minus_v1.py'
MINUS_SHA='0e1bd0296b6d4f5729f23ffd8cd1544720ba37a89930ae4f09217eea34dcdaca'
BEFORE='                b = b * _S19_BSCALE'
AFTER='                b = b * _BTD_SCALE[l]'
FIT=[35,100,36,45]
CHECK=[52,58,64,70,76,80,84,88]
GATES=dict(raw_pct_strictly_below=0.,adjusted_pct_strictly_below=0.,all16_finite=True,layer0_exact=True,wins_and_ci_descriptive_only=True)

def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def checked(p):
    assert pin(p['path'])==p,p['path'];return Path(p['path']).read_bytes()

def read(p):return json.loads(Path(p).read_text())

def once(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def assignments(s):
    out={}
    for n in ast.parse(s).body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name):out[t.id]=ast.dump(n.value,include_attributes=False)
    return out

def main():
    assert pin(PARENT)['sha256']==PARENT_SHA and pin(OLD)['sha256']==OLD_SHA and pin(MINUS)['sha256']==MINUS_SHA
    parent=PARENT.read_text();old=OLD.read_text();minus=MINUS.read_text()
    assert old.count(BEFORE)==parent.count(BEFORE)==1
    old_body=old.replace(BEFORE,AFTER);assert minus.startswith(old_body)
    suffix=minus[len(old_body):];assert suffix.startswith('\n\n# Fixed trace-vector depth contrast; only metadata constants.\n_BTD_SCALE = ')
    manifest=read(H/'trace_vector_depth_tilt_manifest_v1.json')
    historical=next(a for a in manifest['arms'] if a['arm']=='minus')
    scales=ast.literal_eval(ast.parse(suffix).body[0].value)
    assert list(scales)==historical['scales'] and historical['theta']==-.02 and historical['sha256']==MINUS_SHA
    assert manifest['parent_sha256']==OLD_SHA and manifest['parent_bscale']==1.0383394097164294
    old_values=assignments(old);parent_values=assignments(parent)
    learned=[k for k in old_values if 'LITERAL' in k or k in ('_S19_BD_ADD','_S19_BSCALE','_S19_LAM','_S19_K4BAND','_S19_GN_TORCH','_K31R_TABLE','_K31R_TABLE_GN')]
    assert all(old_values[k]==parent_values[k] for k in learned)
    source=parent.replace(BEFORE,AFTER)+suffix
    assert source[:-len(suffix)].replace(AFTER,BEFORE)==parent
    source_values=assignments(source)
    assert all(source_values[k]==v for k,v in parent_values.items()) and set(source_values)-set(parent_values)=={'_BTD_SCALE'}
    compile(source,str(D/'candidate.py'),'exec')
    with (D/'candidate.py').open('x') as f:f.write(source)
    (D/'source.diff').write_text(''.join(difflib.unified_diff(parent.splitlines(True),source.splitlines(True),fromfile=str(PARENT),tofile=str(D/'candidate.py'))))
    closeout=read(H/'trace_vector_depth_tilt_closeout_v1.json')
    assert not closeout['information']['information_gate_passed']
    selection=dict(schema='fixed-bdepth-minus-current-replay-selection-v1',root_selected=True,parent=pin(PARENT),
        historical_minus=pin(MINUS),historical_protocol=pin(H/'trace_vector_depth_tilt_protocol_v1.md'),
        historical_closeout=pin(H/'trace_vector_depth_tilt_closeout_v1.json'),
        historical_information_gate_failed_preserved=True,no_retroactive_admission=True,
        selected_theta=-.02,selected_profile='(l-7)/7 for l=0..14',new_fit_or_extrapolation=False,
        rationale='Replay the actual positive fixed historical arm on current RRPI; prior derivative-quality and gain gates remain failed. Uniform GN3 bscale is a different direction.',
        fit_ranks=FIT,check_ranks=CHECK,gates=GATES,bank_gain_pct_min=1.,
        all_train_known_adaptive_not_pristine=True,no_variants_or_new_coefficients=True,
        check_only_after_actual_fit_raw_and_adjusted_gain=True,no_baseline_rerun=True)
    once(D/'root_selection.json',selection)
    inverse=dict(parent=pin(PARENT),candidate=pin(D/'candidate.py'),edits=[dict(before=BEFORE,after=AFTER)],suffix=suffix,whole_source_inverse_exact=True)
    once(D/'source_inverse.json',inverse)
    build=dict(passed=True,builder=pin(__file__),parent=pin(PARENT),candidate=pin(D/'candidate.py'),selection=pin(D/'root_selection.json'),
        source_inverse=pin(D/'source_inverse.json'),source_inverse_exact=True,historical_manifest=pin(H/'trace_vector_depth_tilt_manifest_v1.json'),
        historical_minus=pin(MINUS),historical_parent=pin(OLD),scales=list(scales),theta=-.02,
        historical_parameter_ast_equal=learned,all_existing_assignments_unchanged=True,
        only_existing_b_multiply_coefficient_changed=True,new_fit_performed=False,model_executed=False)
    once(D/'build.json',build)
    prior={};controls={}
    for stage,ranks in [('fit4',FIT),('check8',CHECK)]:
        p=G/('native_'+stage+'_plan.json');a=G/(stage+'_assessment.json');plan=read(p);assessment=read(a)
        assert assessment['passed_integrity'] and plan['ranks']==ranks and plan['parent']==pin(PARENT)
        arm=copy.deepcopy(plan['arms']['base']);artifacts=assessment['artifacts']['base']
        assert arm['source']==pin(PARENT) and arm['ranks']==ranks
        for item in artifacts.values():
            if 'path' in item:checked(item)
            else:
                for record in item.values():checked(record)
        m=json.loads(checked(artifacts['manifest']));g=json.loads(checked(artifacts['guard']))
        assert m['sha256']==PARENT_SHA and m['candidate']==str(PARENT) and m['ranks']==ranks
        assert g['command']==arm['command'] and g['completed'] and g['exit_code']==0 and g['error'] is None
        assert g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
        assert g['rss_trigger_bytes']==7500000000 and g['max_rss_bytes']<7500000000 and 0<g['elapsed_s']<=arm['limits']['seconds']
        controls[stage]=dict(arm=arm,artifacts=artifacts)
        prior[stage]=dict(plan=pin(p),assessment=pin(a))
    fp=read(G/'native_fit4_plan.json');cp=read(G/'native_check8_plan.json')
    helper=pin(U/'fit.py');assert helper['sha256']=='60f91d378fe4c7c1b12b12aa3a90b41acd73cf0f3714aba91eec85e13e08fd36'
    for stage,ranks,seconds in [('fit4',FIT,150),('check8',CHECK,300)]:
        tag=D.name+'_'+stage+'_v1';record=str(D/('native_'+stage+'_guard.json'))
        command=[fp['runtime'],'-B',fp['driver']['path'],str(D/'candidate.py'),','.join(map(str,ranks)),tag]
        arm=dict(source=pin(D/'candidate.py'),tag=tag,output=str(H/tag),ranks=ranks,driver=fp['driver'],guard=record,
            limits=dict(seconds=seconds,rss_bytes=7500000000,added_swap_bytes=0),command=command,
            guard_command=[fp['light'],'-B',fp['guard_source']['path'],'--record',record,'--seconds',str(seconds),'--rss-limit','7500000000',*command],log=str(D/('native_'+stage+'.log')))
        plan=dict(schema='fixed-bdepth-minus-native-'+stage+'-v1',stage=stage,ranks=ranks,fit_ranks=FIT,check_ranks=CHECK,gates=GATES,
            parent=pin(PARENT),candidate=pin(D/'candidate.py'),build=pin(D/'build.json'),source_inverse=pin(D/'source_inverse.json'),root_selection=pin(D/'root_selection.json'),
            runtime=fp['runtime'],light=fp['light'],driver=fp['driver'],guard_source=fp['guard_source'],helper=helper,
            engine=pin(H/'native_screen_engine_v1/screen.py'),setup_counts=fp['setup_counts'],data=dict(fit4=fp['data'],check8=cp['data']),controls=controls,prior=prior,arm=arm,
            assessor_path=str(D/'assess.py'),source_review_required_before_assessment=True,
            historical_information_gate_failed_preserved=True,all_train_known_adaptive_not_pristine=True,new_fit_performed=False,
            no_baseline_rerun=True,bank_gain_pct_min=1.,full40_and_resources_required_for_bank=True)
        if stage=='check8':plan.update(fit4_plan=pin(D/'native_fit4_plan.json'),fit4_assessment_path=str(D/'fit4_assessment.json'),fit4_assessment_guard=str(D/'fit4_assess_guard.json'),held_until_actual_fit4_raw_and_adjusted_gain=True)
        planpath=D/('native_'+stage+'_plan.json');once(planpath,plan)
        assess=[fp['runtime'],'-B',str(D/'assess.py'),str(planpath),pin(planpath)['sha256']]
        once(D/('commands_'+stage+'.json'),dict(plan=pin(planpath),thread_env=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1'),
            model=arm['guard_command'],model_log=arm['log'],assessment=[fp['light'],'-B',fp['guard_source']['path'],'--record',str(D/(stage+'_assess_guard.json')),'--seconds','120','--rss-limit','7500000000',*assess],
            assessment_log=str(D/(stage+'_assess.log')),conditional_on_actual_fit4_pass=stage=='check8',no_baseline_rerun=True,no_model_executed=True))
    print(json.dumps({k:pin(D/k) for k in ('candidate.py','build.json','source_inverse.json','native_fit4_plan.json','native_check8_plan.json','commands_fit4.json','commands_check8.json')}))

if __name__=='__main__':main()
