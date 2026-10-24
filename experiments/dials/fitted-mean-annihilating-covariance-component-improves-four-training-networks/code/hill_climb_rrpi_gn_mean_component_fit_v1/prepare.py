"""One explicit secant calibration following the closed fixed GN-perp failure."""
from pathlib import Path
import importlib.util, json, math, sys
D=Path(__file__).resolve().parent
H=D.parent
K=H/'hill_climb_rrpi_gn_perp_v1'
C=H/'hill_climb_rrpi_k4_rowdiag_coadapt_v1'
FIT=[35,100,36,45]
CHECK=[52,58,64,70,76,80,84,88]
GATES=dict(actual_fit_raw_pct_strictly_below=0.,actual_fit_adjusted_pct_strictly_below=0.,
           actual_check_raw_pct_strictly_below=0.,actual_check_adjusted_pct_strictly_below=0.,
           all16_finite=True,exact_prefix_layers=10,predicted_fit_gain_pct_strictly_above=0.)
PLUS_SHA='30ed7915c96276001681fbf10126026c49ec145e74038d04ce3c9131dc4cdd27'


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def tools(): return module(C/'common.py','amplitude_receipts')


def activation():
    s=tools(); builder=s.pin(K/'prepare.py')
    assert builder['sha256']=='37168834adc63f16e580422e642ad43e48dbc5e82d20882f7ad7622752d4b706'
    pp=s.pin(K/'native_fit4_plan.json'); assert pp['sha256']=='0ee25ec59089979818c701681f1d1b9c182352c8e8123330e7a6636a563b792d'
    _,p,_=module(builder['path'],'closed_gn_perp').configuration(pp['path'],pp['sha256'])
    rp=s.pin(K/'fit4_assessment.json');r=s.read(s.checked(rp))
    assert r['passed_integrity'] and not r['admitted'] and not r['check8_permitted']
    assert r['plan']==pp and r['candidate']==p['candidate'] and r['parent']==p['parent'] and r['assessor']==p['assessor']
    assert p['candidate']['sha256']==PLUS_SHA and r['ranks']==FIT
    mg=s.guard(K/'native_fit4_guard.json',p['arm']['command'],150)
    ag=s.guard(K/'fit4_assess_guard.json',[p['runtime'],'-B',p['assessor']['path'],pp['path'],pp['sha256']],120)
    return dict(reason='New explicit one-dimensional calibration after preserved fixed alpha1 failure',
                evidence=[builder,pp,rp,mg,ag],plus_arm=p['arm'],plus_artifacts=r['artifacts']['candidate'])


def edit_source(source,alpha):
    assert math.isfinite(alpha) and -4<=alpha<=1
    assert __import__('hashlib').sha256(source.encode()).hexdigest()==PLUS_SHA
    edits=[('def _gn_perp_host(layer, host, increment):',
            '_GN_PERP_ALPHA = '+repr(float(alpha))+'\n\ndef _gn_perp_host(layer, host, increment):'),
           ('def _gn_perp_host(layer, host, increment):\n    if layer not in (10, 11, 12):',
            'def _gn_perp_host(layer, host, increment):\n    if _GN_PERP_ALPHA == 0.0:\n        return increment\n    if layer not in (10, 11, 12):'),
           ('    fnp.fill_diagonal(Q, 0.0)\n    return Q\n',
            '    fnp.fill_diagonal(Q, 0.0)\n    if _GN_PERP_ALPHA == 1.0:\n        return Q\n    return increment + _GN_PERP_ALPHA * (Q - increment)\n')]
    changed=source
    for before,after in edits:
        assert changed.count(before)==1 and after not in changed
        changed=changed.replace(before,after,1)
    inverse=changed
    for before,after in reversed(edits):
        assert inverse.count(after)==1;inverse=inverse.replace(after,before,1)
    assert inverse==source
    compile(changed,'fixed_gn_mean_component_amplitude','exec')
    return changed,dict(replacements=[dict(before=a,after=b,count=1) for a,b in edits],whole_source_inverse_exact=True,
                        alpha0_returns_original_increment=True,alpha1_returns_original_projection=True)


def configuration(path,digest):
    s=tools();pp=s.pin(path);assert pp['sha256']==digest;c=s.read(s.checked(pp))
    assert c['activation']==activation() and c['builder']==s.pin(__file__)
    assert c['fit_source']==s.pin(D/'fit.py') and c['assessor']==s.pin(D/'assess.py') and c['receipt_tools']==s.pin(C/'common.py')
    assert c['fit_ranks']==FIT and c['check_ranks']==CHECK and c['ridge_diagonal_fraction']==.1 and c['box']==[-4.,1.]
    assert c['gates']==GATES and c['one_endpoint_only'] and c['secant_not_exact_gradient'] and not c['validation_before_fixed_endpoint']
    assert c['parent']['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
    assert c['projection_source']['sha256']==PLUS_SHA
    for key in ('parent','projection_source','helper','engine','driver','guard_source','receipt_tools'):s.checked(c[key])
    old=s.pin(H/'hill_climb_rrpi_smooth_kink_v1/prepare.py')
    assert old['sha256']=='626cff0af1a0250e0ce58b5833c09d49b10e3ae444cd5780bb033f81b82c4150'
    for key,value in module(old['path'],'mean_component_actual_controls').common().items():assert c[key]==value
    assert c['plus_arm']==c['activation']['plus_arm'] and c['plus_artifacts']==c['activation']['plus_artifacts']
    assert c['plus_arm']['source']==c['projection_source']
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
    s.guard(D/'fit_guard.json',[c['runtime'],'-B',c['fit_source']['path'],pp['path'],pp['sha256']],120)
    expected,proof=edit_source(s.checked(c['projection_source']).read_text(),r['alpha'])
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
    shared=module(H/'hill_climb_rrpi_smooth_kink_v1/prepare.py','mean_component_saved_controls').common()
    c=dict(schema='gn-mean-component-secant-calibration-v1',activation=admission,
           parent=s.pin(H/'hill_climb_pair_rr_inverse_v1/candidate.py'),projection_source=s.pin(K/'candidate.py'),
           builder=s.pin(__file__),fit_source=s.pin(D/'fit.py'),assessor=s.pin(D/'assess.py'),receipt_tools=s.pin(C/'common.py'),
           fit_ranks=FIT,check_ranks=CHECK,box=[-4.,1.],ridge_diagonal_fraction=.1,gates=GATES,
           one_endpoint_only=True,secant_not_exact_gradient=True,forecast_includes_candidate_compute=False,
           validation_before_fixed_endpoint=False,bank_gain_pct_min=1.,full40_own_resources_serving_contract_required=True,
           plus_arm=admission['plus_arm'],plus_artifacts=admission['plus_artifacts'],**shared)
    s.once(D/'protocol.json',c);pp=s.pin(D/'protocol.json')
    fit=[c['runtime'],'-B',c['fit_source']['path'],pp['path'],pp['sha256']]
    s.once(D/'fit_commands.json',dict(protocol=pp,model_jobs=[],
         fit=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/'fit_guard.json'),'--seconds','120','--rss-limit','7500000000',*fit],
         thread_env=dict.fromkeys(s.THREADS,'1'),root_only=True,
         launch_prerequisite='Root calls prepare.configuration(protocol_path, sha) before saved-array fit; endpoint_configuration before its model.'))


if __name__=='__main__':main()
