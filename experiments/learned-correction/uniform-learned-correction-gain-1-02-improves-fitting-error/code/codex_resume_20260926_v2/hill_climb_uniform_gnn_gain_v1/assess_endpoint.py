"""Root-only replay of the one fixed endpoint. CLI: PROTOCOL SHA {fit4|check8}."""
from pathlib import Path
import importlib.util, json, sys

D=Path(__file__).resolve().parent

def main():
    assert len(sys.argv)==4 and sys.argv[3] in ('fit4','check8')
    path=Path(sys.argv[1]).resolve();digest=sys.argv[2];stage=sys.argv[3]
    # Protocol pins the source before importing its numerical replay helpers.
    import hashlib
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==digest;c=json.loads(raw)
    for key,file in [('fit_source',D/'fit.py'),('endpoint_assessor',Path(__file__).resolve())]:
        b=file.read_bytes();assert c[key]==dict(path=str(file),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    spec=importlib.util.spec_from_file_location('gain_fixed_fit',D/'fit.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    q,cp,c,builder=f.configuration(path,digest)
    fitpin=q.pin(D/'fit_result.json');fit=json.loads(q.checked(fitpin))
    assert fit['passed_integrity'] and fit['endpoint_admitted'] and fit['predicted_fit_adjusted_gain_pct']>0
    assert fit['protocol']==cp and fit['fitter']==c['fit_source'] and fit['parent']==c['parent'] and fit['fit_ranks']==f.FIT
    assert fit['check_targets_read'] is False and fit['one_endpoint_only']
    for evidence in fit['evidence']:q.checked(evidence)
    q.guard(D/'fit_guard.json',dict(seconds=120,rss_bytes=7500000000),[c['runtime'],'-B',c['fit_source']['path'],cp['path'],cp['sha256']])
    source,proof=builder.edit_source(q.checked(c['parent']).decode(),fit['effective_gain_multiplier'])
    assert fit['effective_gain_multiplier']==1+.02*fit['x'] and -1<=fit['x']<=1
    assert source.encode()==q.checked(fit['endpoint']) and proof==fit['source_inverse']
    alias=f.fit4_reuse(q,c,fit['endpoint'],fit['x'])
    assert fit['fit4_reuse']==alias
    prerequisites=[fitpin,q.pin(D/'fit_guard.json')]
    if stage=='check8':
        pp=q.pin(D/'endpoint_fit4_assessment.json');prior=json.loads(q.checked(pp))
        assert prior['passed_integrity'] and prior['admitted'] and prior['check8_permitted']
        assert prior['protocol']==cp and prior['fit_result']==fitpin and prior['candidate']==fit['endpoint'] and prior['parent']==c['parent']
        assert prior['assessor']==c['endpoint_assessor'] and prior['metrics']['raw_pct']<0 and prior['metrics']['adjusted_pct']<0
        q.guard(D/'endpoint_fit4_assess_guard.json',dict(seconds=120,rss_bytes=7500000000),
            [c['runtime'],'-B',c['endpoint_assessor']['path'],cp['path'],cp['sha256'],'fit4'])
        prerequisites.extend([pp,q.pin(D/'endpoint_fit4_assess_guard.json')])
    ranks=f.FIT if stage=='fit4' else f.CHECK;truth=f.truths(q,c,stage)
    control=c['controls'][stage];assert control['arm']['source']==c['parent']
    base,ba,bp=f.panel(q,c,control['arm'],ranks,truth,control['artifacts'])
    reuse=alias if stage=='fit4' else None
    executed_arm=f.endpoint_arm(c,fit['endpoint'],stage,reuse)
    expected=None if reuse is None else fit['artifacts'][reuse['arm_name']]
    rows,aa,ap=f.panel(q,c,executed_arm,ranks,truth,expected)
    for rank in ranks:assert ba[rank][0].tobytes()==aa[rank][0].tobytes()
    metrics=q.metrics(base,rows)
    gates=dict(raw_gain=metrics['raw_pct']<0,adjusted_gain=metrics['adjusted_pct']<0,all16_finite=True,layer0_exact=True)
    result=dict(schema='uniform-effective-gnn-gain-endpoint-assessment-v1',stage=stage,passed_integrity=True,
        admitted=all(gates.values()),check8_permitted=stage=='fit4' and all(gates.values()),
        component_point_passed=stage=='check8' and all(gates.values()),protocol=cp,assessor=q.pin(__file__),engine=q.pin(f.ENGINE),
        fit_result=fitpin,candidate=fit['endpoint'],parent=c['parent'],ranks=ranks,metrics=metrics,gates=gates,
        artifacts=dict(base=bp,candidate=ap),executed_source=executed_arm['source'],endpoint_alias=fit['endpoint'],
        reused_actual_fit4=reuse,prerequisites=prerequisites,evidence=list(q.EVIDENCE.values()),
        wins_and_ci_descriptive_only=True,adaptive_reused_train=True,old_failures_preserved=True,
        retained=False,broad_admitted=False,bank_qualified=False,submitted=False,model_executed=False)
    out=D/('endpoint_'+stage+'_assessment.json');q.once(out,result)
    if result['check8_permitted']:
        q.once(D/'endpoint_check8_commands.json',f.endpoint_commands(q,c,cp,fit['endpoint'],'check8',[q.pin(out),*prerequisites]))
    print(json.dumps({k:result[k] for k in ('stage','passed_integrity','admitted','metrics','gates')}))

if __name__=='__main__':main()
