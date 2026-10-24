"""One fixed current-RRPI YNT-repair amplitude secant; stdlib preflight only."""
from pathlib import Path
import hashlib, importlib.util, json, math, sys
D=Path(__file__).resolve().parent
H=D.parent
C=H/'hill_climb_rrpi_k4_rowdiag_coadapt_v1'
FIT=[35,100,36,45]
CHECK=[52,58,64,70,76,80,84,88]
PARENT_SHA='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
GATES=dict(actual_fit_raw_pct_strictly_below=0.,actual_fit_adjusted_pct_strictly_below=0.,
           actual_check_raw_pct_strictly_below=0.,actual_check_adjusted_pct_strictly_below=0.,
           all16_finite=True,exact_prefix_layers=7,predicted_fit_gain_pct_min=.1)
BEFORE='return result if _YNT_REPAIR[0] is None else result + _YNT_REPAIR[0]'
AFTER='return result if _YNT_REPAIR[0] is None else result + (_YNT_REPAIR[0] if _YNT_REPAIR_GAIN == 1.0 else _YNT_REPAIR_GAIN * _YNT_REPAIR[0])'


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def tools(): return module(C/'common.py','ynt_amplitude_receipts')


def edit_source(source,alpha):
    assert math.isfinite(alpha) and -2<=alpha<=2
    assert hashlib.sha256(source.encode()).hexdigest()==PARENT_SHA
    assert source.count(BEFORE)==1 and AFTER not in source and '_YNT_REPAIR_GAIN' not in source
    gain=1.-float(alpha)
    suffix='\n\n_YNT_REPAIR_GAIN = '+repr(gain)+'\n'
    changed=source.replace(BEFORE,AFTER,1)+suffix
    assert changed[:-len(suffix)].replace(AFTER,BEFORE,1)==source
    compile(changed,'fixed_ynt_repair_amplitude','exec')
    return changed,dict(replacements=[dict(before=BEFORE,after=AFTER,count=1)],suffix=suffix,
        whole_source_inverse_exact=True,gain=gain,alpha0_parent_arithmetic_exact=True,
        gain0_preserves_trace_computation=True,old_tail_parent_and_crop_source_bytes_unchanged=True)


def configuration(path,digest):
    s=tools();pp=s.pin(path);assert pp['sha256']==digest;c=s.read(s.checked(pp))
    assert c['schema']=='ynt-repair-amplitude-secant-v1' and c['builder']==s.pin(__file__)
    assert c['fit_source']==s.pin(D/'fit.py') and c['assessor']==s.pin(D/'assess.py') and c['receipt_tools']==s.pin(C/'common.py')
    assert c['fit_ranks']==FIT and c['check_ranks']==CHECK and c['ridge_diagonal_fraction']==.1 and c['alpha_box']==[-2.,2.]
    assert c['gates']==GATES and c['one_endpoint_only'] and c['secant_not_exact_gradient'] and not c['validation_before_fixed_endpoint']
    assert c['parent']['sha256']==PARENT_SHA and c['direction_alpha']==1. and c['direction_gain']==0.
    for key in ('parent','helper','engine','driver','guard_source','receipt_tools'):s.checked(c[key])
    old=s.pin(H/'hill_climb_rrpi_smooth_kink_v1/prepare.py')
    assert old['sha256']=='626cff0af1a0250e0ce58b5833c09d49b10e3ae444cd5780bb033f81b82c4150'
    for key,value in module(old['path'],'ynt_actual_controls').common().items():assert c[key]==value
    expected,proof=edit_source(s.checked(c['parent']).read_text(),1.)
    assert expected.encode()==s.checked(c['direction_source']).read_bytes() and c['source_inverse']==proof
    assert c['direction_arm']==arm(c,c['direction_source'],'direction','fit4')
    return s,pp,c


def arm(c,source,name,stage):
    ranks=FIT if stage=='fit4' else CHECK;seconds=150 if stage=='fit4' else 300
    tag=D.name+'_'+name+'_'+stage+'_v1';guard=str(D/(name+'_'+stage+'_guard.json'))
    command=[c['runtime'],'-B',c['driver']['path'],source['path'],','.join(map(str,ranks)),tag]
    return dict(source=source,ranks=ranks,driver=c['driver'],tag=tag,output=str(H/tag),guard=guard,
                limits=dict(seconds=seconds,rss_bytes=7500000000,added_swap_bytes=0),command=command,
                guard_command=[c['light'],'-B',c['guard_source']['path'],'--record',guard,'--seconds',str(seconds),'--rss-limit','7500000000',*command],
                log=str(D/(name+'_'+stage+'.log')))



def endpoint_configuration(path,digest,stage):
    assert stage in ('fit4','check8');s,pp,c=configuration(path,digest)
    r=s.read(D/'fit_result.json')
    assert r['passed_integrity'] and r['endpoint_admitted'] and r['protocol']==pp and r['fitter']==c['fit_source']
    assert r['parent']==c['parent'] and r['direction_source']==c['direction_source']
    assert r['predicted_parent_weighted_loss_gain_pct']>=.1 and r['G']>0
    assert r['gain']==1.-r['alpha']
    s.guard(D/'fit_guard.json',[c['runtime'],'-B',c['fit_source']['path'],pp['path'],pp['sha256']],120)
    s.guard(c['direction_arm']['guard'],c['direction_arm']['command'],150)
    expected,proof=edit_source(s.checked(c['parent']).read_text(),r['alpha'])
    assert expected.encode()==s.checked(r['endpoint']).read_bytes() and r['source_inverse']==proof
    if stage=='check8':
        prior=s.read(D/'endpoint_fit4_assessment.json')
        assert prior['passed_integrity'] and prior['admitted'] and prior['protocol']==pp and prior['endpoint']==r['endpoint']
        assert prior['assessor']==c['assessor'] and prior['metrics']['raw_pct']<0 and prior['metrics']['adjusted_pct']<0
        a=arm(c,r['endpoint'],'endpoint','fit4');s.guard(a['guard'],a['command'],150)
        s.guard(D/'endpoint_fit4_assess_guard.json',[c['runtime'],'-B',c['assessor']['path'],pp['path'],pp['sha256'],'fit4'],120)
    return s,pp,c,r


def commands(s,pp,c,source,stage):
    a=arm(c,source,'endpoint',stage)
    assess=[c['runtime'],'-B',c['assessor']['path'],pp['path'],pp['sha256'],stage]
    return dict(protocol=pp,arm=a,assessment=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/('endpoint_'+stage+'_assess_guard.json')),
                '--seconds','120','--rss-limit','7500000000',*assess],thread_env=dict.fromkeys(s.THREADS,'1'),root_only=True,
                launch_prerequisite='Root calls prepare.endpoint_configuration(protocol_path, sha, '+repr(stage)+') before model launch.')



def main():
    assert sys.argv[1:]==['--activate'];s=tools()
    shared=module(H/'hill_climb_rrpi_smooth_kink_v1/prepare.py','ynt_saved_controls').common()
    parent=s.pin(H/'hill_climb_pair_rr_inverse_v1/candidate.py')
    source,proof=edit_source(s.checked(parent).read_text(),1.)
    with (D/'direction.py').open('x') as out:out.write(source)
    c=dict(schema='ynt-repair-amplitude-secant-v1',parent=parent,direction_source=s.pin(D/'direction.py'),source_inverse=proof,
           builder=s.pin(__file__),fit_source=s.pin(D/'fit.py'),assessor=s.pin(D/'assess.py'),receipt_tools=s.pin(C/'common.py'),
           fit_ranks=FIT,check_ranks=CHECK,alpha_box=[-2.,2.],ridge_diagonal_fraction=.1,gates=GATES,
           direction_alpha=1.,direction_gain=0.,direction_integrity_only_no_score_gate=True,
           formula='z=y_gain0-y_parent; G=sum(w*z*z); g=sum(w*z*(parent-truth)); alpha=clip(-g/(1.1*G),-2,2); gain=1-alpha',
           one_endpoint_only=True,secant_not_exact_gradient=True,forecast_includes_candidate_compute=False,
           validation_before_fixed_endpoint=False,bank_gain_pct_min=1.,full40_own_resources_serving_contract_required=True,
           no_baseline_rerun=True,no_head_layer_threshold_rank_or_other_parameter_fit=True,**shared)
    c['direction_arm']=arm(c,c['direction_source'],'direction','fit4')
    s.once(D/'protocol.json',c);pp=s.pin(D/'protocol.json')
    fit=[c['runtime'],'-B',c['fit_source']['path'],pp['path'],pp['sha256']]
    s.once(D/'commands.json',dict(protocol=pp,direction=c['direction_arm'],
         fit=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/'fit_guard.json'),'--seconds','120','--rss-limit','7500000000',*fit],
         thread_env=dict.fromkeys(s.THREADS,'1'),root_only=True,
         launch_prerequisite='Root calls prepare.configuration(protocol_path, sha) before gain0 FIT4 and saved-array fit; endpoint_configuration before endpoint models.'))
    print(json.dumps(dict(protocol=pp,direction=c['direction_source'])))


if __name__=='__main__':main()
