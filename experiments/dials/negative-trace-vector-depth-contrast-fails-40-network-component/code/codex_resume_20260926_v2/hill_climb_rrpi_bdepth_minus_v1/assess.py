"""Root-only paired fixed-step assessment; CLI PLAN_PATH SHA. No fit/model."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import sys

D = Path(__file__).resolve().parent
H = D.parent
G = H/'hill_climb_rrpi_native_joint3_recovery_v1'
HELPER = H/'hill_climb_uniform_gnn_gain_v1/fit.py'
HELPER_SHA = '60f91d378fe4c7c1b12b12aa3a90b41acd73cf0f3714aba91eec85e13e08fd36'
ENGINE = H/'native_screen_engine_v1/screen.py'
ENGINE_SHA = '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
PRIOR = {
    'fit4': ('43fc8ac2cd8a82835366b4144bdedb5c91780d7758f369a10353c7fb226bf10d', '57d52b3155bb09ab7add2952f2be234109e3e7b025b6b2616952307ed4e44338'),
    'check8': ('a00100251c915f57062a1fb84fab945d42b75440c91d2e376f2aa6cf6dc2201f', '9eeb31459174aa4ef6b5d7b8b3980f3954ac128224a3938e0c9f2619e952ad1a'),
}
GATES = dict(raw_pct_strictly_below=0., adjusted_pct_strictly_below=0., all16_finite=True,
             layer0_exact=True, wins_and_ci_descriptive_only=True)


def pin(path):
    p=Path(path).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def assess(path,digest):
    assert pin(HELPER)['sha256']==HELPER_SHA and pin(ENGINE)['sha256']==ENGINE_SHA
    spec=importlib.util.spec_from_file_location('bdepth_existing_helpers',HELPER)
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    q=f.module(ENGINE,'bdepth_existing_metrics')
    pp=q.pin(path);assert pp['sha256']==digest;c=json.loads(q.checked(pp))
    schemas={f'fixed-bdepth-minus-native-{s}-v1':s for s in ('fit4','check8')}
    stage=schemas[c['schema']];ranks=f.FIT if stage=='fit4' else f.CHECK
    assert c['ranks']==ranks and c['gates']==GATES
    assert c['stage']==stage and c['fit_ranks']==f.FIT and c['check_ranks']==f.CHECK
    assert c['helper']==q.pin(HELPER) and c['engine']==q.pin(ENGINE)
    assert c['assessor_path']==str(Path(__file__).resolve()) and c['source_review_required_before_assessment']
    review=json.loads(q.checked(q.pin(D/'source_review_assessor.json')))
    assert review['passed'] and review['assessor']==q.pin(__file__) and pp in review['plans']
    assert c['historical_information_gate_failed_preserved'] and c['new_fit_performed'] is False
    assert c['bank_gain_pct_min']==1 and c['full40_and_resources_required_for_bank']
    assert str(Path(sys.executable).absolute())==c['runtime']==str(q.NATIVE)
    assert q.np.__version__=='2.4.6' and all(os.environ.get(k)=='1' for k in q.THREADS)
    assert c['parent']['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
    assert c['driver']['sha256']=='9c30282361d9cef4e53a3ea95ff44cbe5f8e2ec3fb8cd1fb4f1ceb1ad01b40c7'
    assert c['guard_source']['sha256']=='314ed428e1986abf6b7c05daa5c0ea7c1b5e6ebe6472ba4a34c2ed59a80b5f76'
    for key in ('driver','guard_source','build','root_selection'):q.checked(c[key])
    build=json.loads(q.checked(c['build']));selection=json.loads(q.checked(c['root_selection']))
    assert build['passed'] and build['source_inverse_exact'] and build['new_fit_performed'] is False
    assert build['parent']==c['parent'] and build['candidate']==c['candidate'] and build['selection']==c['root_selection']
    assert build['source_inverse']==c['source_inverse'];q.checked(build['builder'])
    assert selection['root_selected'] and selection['parent']==c['parent'] and selection['gates']==GATES
    assert selection['selected_theta']==-.02 and selection['historical_information_gate_failed_preserved']
    q.checked(selection['historical_protocol']);q.checked(selection['historical_closeout'])
    for key,shas in PRIOR.items():
        lp=q.pin(G/('native_'+key+'_plan.json'));ap=q.pin(G/(key+'_assessment.json'))
        assert lp['sha256']==shas[0] and ap['sha256']==shas[1]
        assert c['prior'][key]==dict(plan=lp,assessment=ap)
        old=json.loads(q.checked(lp));receipt=json.loads(q.checked(ap))
        assert c['controls'][key]==dict(arm=old['arms']['base'],artifacts=receipt['artifacts']['base'])
        assert c['data'][key]==old['data'] and c['setup_counts']==old['setup_counts']
    # Literal replay of the historical fixed minus direction; all other bytes remain.
    parent=q.checked(c['parent']).decode();candidate=q.checked(c['candidate']).decode()
    before='                b = b * _S19_BSCALE';after='                b = b * _BTD_SCALE[l]'
    assert parent.count(before)==1 and after not in parent
    changed=parent.replace(before,after);assert candidate.startswith(changed)
    suffix=candidate[len(changed):];nodes=ast.parse(suffix).body
    assert len(nodes)==1 and isinstance(nodes[0],ast.Assign)
    assert len(nodes[0].targets)==1 and isinstance(nodes[0].targets[0],ast.Name) and nodes[0].targets[0].id=='_BTD_SCALE'
    expected=tuple(1.0383394097164294*(1+(-.02)*((l-7)/7)) for l in range(15))
    assert ast.literal_eval(nodes[0].value)==expected
    assert candidate[:-len(suffix)].replace(after,before)==parent
    inverse=json.loads(q.checked(c['source_inverse']))
    assert inverse==dict(parent=c['parent'],candidate=c['candidate'],edits=[dict(before=before,after=after)],suffix=suffix,whole_source_inverse_exact=True)
    historical_parent=q.checked(build['historical_parent']).decode()
    historical_minus=q.checked(build['historical_minus']).decode()
    assert build['historical_minus']==selection['historical_minus']
    assert historical_minus==historical_parent.replace(before,after)+suffix
    q.checked(build['historical_manifest']);assert build['scales']==list(expected)
    prerequisites=[]
    if stage=='check8':
        fp=q.pin(D/'fit4_assessment.json');prior=json.loads(q.checked(fp))
        assert c['fit4_assessment_path']==fp['path'] and c['fit4_assessment_guard']==str(D/'fit4_assess_guard.json')
        assert prior['passed_integrity'] and prior['admitted'] and prior['check8_permitted']
        assert prior['stage']=='fit4' and prior['candidate']==c['candidate'] and prior['parent']==c['parent']
        assert prior['assessor']==q.pin(__file__) and prior['root_selection']==c['root_selection'] and prior['build']==c['build']
        assert prior['metrics']['raw_pct']<0 and prior['metrics']['adjusted_pct']<0
        oldplan=json.loads(q.checked(prior['plan']))
        assert prior['plan']==c['fit4_plan'] and c['held_until_actual_fit4_raw_and_adjusted_gain']
        assert oldplan['schema']=='fixed-bdepth-minus-native-fit4-v1' and oldplan['gates']==GATES
        q.guard(D/'fit4_assess_guard.json',dict(seconds=120,rss_bytes=7500000000),
                [c['runtime'],'-B',str(Path(__file__).resolve()),prior['plan']['path'],prior['plan']['sha256']])
        prerequisites=[fp,q.pin(D/'fit4_assess_guard.json')]
    truth=f.truths(q,c,stage);control=c['controls'][stage]
    assert control['arm']['source']==c['parent'] and c['arm']['source']==c['candidate']
    base,ba,bp=f.panel(q,c,control['arm'],ranks,truth,control['artifacts'])
    rows,aa,ap=f.panel(q,c,c['arm'],ranks,truth)
    for rank in ranks:assert ba[rank][0].tobytes()==aa[rank][0].tobytes()
    metrics=q.metrics(base,rows)
    gates=dict(raw_gain=metrics['raw_pct']<0,adjusted_gain=metrics['adjusted_pct']<0,all16_finite=True,layer0_exact=True)
    return dict(schema='fixed-bdepth-minus-paired-assessment-v1',stage=stage,passed_integrity=True,
        admitted=all(gates.values()),check8_permitted=stage=='fit4' and all(gates.values()),
        component_point_passed=stage=='check8' and all(gates.values()),plan=pp,assessor=q.pin(__file__),
        helper=q.pin(HELPER),engine=q.pin(ENGINE),candidate=c['candidate'],parent=c['parent'],
        root_selection=c['root_selection'],build=c['build'],ranks=ranks,metrics=metrics,gates=gates,
        artifacts=dict(base=bp,candidate=ap),prerequisites=prerequisites,evidence=list(q.EVIDENCE.values()),
        wins_and_ci_descriptive_only=True,adaptive_reused_panels=True,old_failures_preserved=True,
        fitting_performed=False,retained=False,broad_admitted=False,bank_qualified=False,model_executed=False)


def main():
    assert len(sys.argv)==3
    result=assess(Path(sys.argv[1]).resolve(),sys.argv[2])
    out=D/(result['stage']+'_assessment.json')
    with out.open('x') as file:json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps({k:result[k] for k in ('stage','passed_integrity','admitted','metrics','gates')}))


if __name__=='__main__':main()
