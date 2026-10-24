"""Inactive source proposal. Only root --activate may seal it after both head arms fail."""
from pathlib import Path
import importlib.util, json, math, sys

D=Path(__file__).resolve().parent
H=D.parent
C=H/'hill_climb_rrpi_k4_rowdiag_coadapt_v1'
K=H/'hill_climb_rrpi_k4_rowdiag_v1'
FIT=[35,100,36,45]
CHECK=[52,58,64,70,76,80,84,88]
GATES=dict(actual_fit_raw_pct_strictly_below=0.,actual_fit_adjusted_pct_strictly_below=0.,
           actual_check_raw_pct_strictly_below=0.,actual_check_adjusted_pct_strictly_below=0.,
           all16_finite=True,layer0_exact=True,predicted_fit_gain_pct_strictly_above=0.)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def tools(): return module(C/'common.py','amplitude_receipts')


def activation():
    s=tools();pp=s.pin(C/'protocol.json');assert pp['sha256']=='93eea5cc3a3ff6d7e889798ba57e78b73f466500b23e8b3801b3897537e06ef3'
    _,p=s.plan(pp['path'],pp['sha256']);cvp=s.pin(C/'cv_result.json');cv=s.read(s.checked(cvp))
    assert cv['completed'] and cv['protocol']==pp and cv['cv_gates']==p['cv_gates']
    assert cv['all_finite'] and cv['ridge']==.001 and cv['fit_count']==10
    s.checked(cv['fit_source']);cmd=[p['runtime'],'-B',cv['fit_source']['path'],'--cv',pp['path'],pp['sha256']]
    sp=s.pin(C/'fit_seal.json');seal=s.read(s.checked(sp));jp=s.pin(C/'fit_commands.json');jobs=s.read(s.checked(jp))
    assert seal['protocol']==pp and seal['fit_source']==cv['fit_source'] and seal['cv_gates']==p['cv_gates']
    assert jobs['protocol']==pp and jobs['seal']==sp and jobs['jobs']['cv']['command']==cmd
    cg=s.guard(C/'cv_guard.json',cmd,120)
    if not cv['physical_admitted'] and not cv['control_admitted']:
        return dict(reason='Neither head arm passed its fixed CV gate',evidence=[pp,sp,jp,cvp,cg])
    b=module(C/'prepare_check8.py','amplitude_head_check_binding');cp=s.pin(C/'native_check8_plan.json')
    _,c=b.configuration(cp['path'],cp['sha256']);ep=s.pin(C/'check8_assessment.json');e=s.read(s.checked(ep))
    assert e['passed_integrity'] and e['plan']==cp and e['binding']==c['binding'] and e['assessor']==c['assessor']
    assert e['eligible_arms'] and not any(e['eligible_arms'].values())
    eg=s.guard(C/'check8_assess_guard.json',[c['runtime'],'-B',e['assessor']['path'],cp['path'],cp['sha256']],120)
    return dict(reason='Every exported head arm failed actual CHECK8 admission',evidence=[pp,sp,jp,cvp,cg,cp,ep,eg])


def edit_source(source,alpha):
    assert math.isfinite(alpha) and -2<=alpha<=2
    before='_K4R_ALPHA = 1.0';after='_K4R_ALPHA = '+repr(float(alpha))
    assert source.count(before)==1
    changed=source.replace(before,after,1);assert changed.replace(after,before,1)==source
    compile(changed,'fixed_k4_amplitude','exec')
    return changed,dict(before=before,after=after,count=1,whole_source_inverse_exact=True)


def arm(c,source,name,stage):
    ranks=FIT if stage=='fit4' else CHECK;seconds=150 if stage=='fit4' else 300
    tag=D.name+'_'+name+'_'+stage+'_v1';guard=str(D/(name+'_'+stage+'_guard.json'))
    command=[c['runtime'],'-B',c['driver']['path'],source['path'],','.join(map(str,ranks)),tag]
    return dict(source=source,ranks=ranks,driver=c['driver'],tag=tag,output=str(H/tag),guard=guard,
                limits=dict(seconds=seconds,rss_bytes=7500000000,added_swap_bytes=0),command=command,
                guard_command=[c['light'],'-B',c['guard_source']['path'],'--record',guard,'--seconds',str(seconds),'--rss-limit','7500000000',*command],
                log=str(D/(name+'_'+stage+'.log')))


def configuration(path,digest):
    s=tools();pp=s.pin(path);assert pp['sha256']==digest;c=s.read(s.checked(pp))
    assert c['activation']==activation() and c['builder']==s.pin(__file__)
    assert c['fit_source']==s.pin(D/'fit.py') and c['assessor']==s.pin(D/'assess.py')
    assert c['fit_ranks']==FIT and c['check_ranks']==CHECK and c['ridge_diagonal_fraction']==.1 and c['box']==[-2.,2.]
    assert c['gates']==GATES and c['one_endpoint_only'] and not c['validation_before_fixed_endpoint']
    assert c['parent']['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
    for key in ('parent','rowdiag_source','helper','engine','driver','guard_source'):s.checked(c[key])
    assert c['rowdiag_source']['sha256']=='ebd4c32f26a3bcfec8e76f587db611ab5b5bdc053cb7d927eb61b40381389bc7'
    old=s.pin(H/'hill_climb_rrpi_smooth_kink_v1/prepare.py')
    assert old['sha256']=='626cff0af1a0250e0ce58b5833c09d49b10e3ae444cd5780bb033f81b82c4150'
    for key,value in module(old['path'],'amplitude_actual_controls').common().items():assert c[key]==value
    source=s.checked(c['rowdiag_source']).read_text()
    assert set(c['arms'])=={'minus','plus'}
    for name,alpha in [('minus',-1.),('plus',1.)]:
        proposed,proof=edit_source(source,alpha)
        assert proposed.encode()==s.checked(c['arms'][name]['source']).read_bytes() and proof==c['source_inverse'][name]
        assert c['arms'][name]==arm(c,c['arms'][name]['source'],name,'fit4')
    return s,pp,c


def endpoint_configuration(path,digest,stage):
    assert stage in ('fit4','check8');s,pp,c=configuration(path,digest)
    r=s.read(D/'fit_result.json')
    assert r['passed_integrity'] and r['endpoint_admitted'] and r['protocol']==pp and r['fitter']==c['fit_source']
    s.guard(D/'fit_guard.json',[c['runtime'],'-B',c['fit_source']['path'],pp['path'],pp['sha256']],120)
    expected,proof=edit_source(s.checked(c['rowdiag_source']).read_text(),r['alpha'])
    assert expected.encode()==s.checked(r['endpoint']).read_bytes() and r['source_inverse']==proof
    if stage=='check8':
        prior=s.read(D/'endpoint_fit4_assessment.json')
        assert prior['passed_integrity'] and prior['admitted'] and prior['protocol']==pp and prior['endpoint']==r['endpoint']
        assert prior['assessor']==c['assessor'] and prior['metrics']['raw_pct']<0 and prior['metrics']['adjusted_pct']<0
        s.guard(D/'endpoint_fit4_assess_guard.json',[c['runtime'],'-B',c['assessor']['path'],pp['path'],pp['sha256'],'fit4'],120)
    return s,pp,c,r


def commands(s,pp,c,source,stage):
    a=arm(c,source,'endpoint',stage)
    assess=[c['runtime'],'-B',c['assessor']['path'],pp['path'],pp['sha256'],stage]
    return dict(protocol=pp,arm=a,assessment=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/('endpoint_'+stage+'_assess_guard.json')),
                '--seconds','120','--rss-limit','7500000000',*assess],thread_env=dict.fromkeys(s.THREADS,'1'),root_only=True,
                launch_prerequisite='Root calls prepare.endpoint_configuration(protocol_path, sha, '+repr(stage)+') before model launch.')


def main():
    assert sys.argv[1:]==['--activate'];s=tools();admission=activation()
    helper=module(H/'hill_climb_rrpi_smooth_kink_v1/prepare.py','amplitude_saved_controls');shared=helper.common()
    parent=s.pin(H/'hill_climb_pair_rr_inverse_v1/candidate.py');rowdiag=s.pin(K/'candidate.py')
    assert rowdiag['sha256']=='ebd4c32f26a3bcfec8e76f587db611ab5b5bdc053cb7d927eb61b40381389bc7'
    c=dict(schema='conditional-k4-scalar-amplitude-v1',activation=admission,parent=parent,rowdiag_source=rowdiag,
           builder=s.pin(__file__),fit_source=s.pin(D/'fit.py'),assessor=s.pin(D/'assess.py'),
           fit_ranks=FIT,check_ranks=CHECK,box=[-2.,2.],ridge_diagonal_fraction=.1,gates=GATES,
           one_endpoint_only=True,validation_before_fixed_endpoint=False,bank_gain_pct_min=1.,**shared)
    c['arms']={};c['source_inverse']={}
    for name,alpha in [('minus',-1.),('plus',1.)]:
        source,proof=edit_source(s.checked(rowdiag).read_text(),alpha);p=D/('candidate_'+name+'.py')
        with p.open('x') as f:f.write(source)
        c['arms'][name]=arm(c,s.pin(p),name,'fit4');c['source_inverse'][name]=proof
    s.once(D/'protocol.json',c);pp=s.pin(D/'protocol.json')
    fit=[c['runtime'],'-B',c['fit_source']['path'],pp['path'],pp['sha256']]
    s.once(D/'discovery_commands.json',dict(protocol=pp,model_jobs=[c['arms'][n] for n in ('minus','plus')],
         fit=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/'fit_guard.json'),'--seconds','120','--rss-limit','7500000000',*fit],
         thread_env=dict.fromkeys(s.THREADS,'1'),root_only=True,
         launch_prerequisite='Root calls prepare.configuration(protocol_path, sha) before each model launch.'))


if __name__=='__main__':main()
