"""Root-only paired CHECK8 component-point assessment; requires actual FIT4 pass."""
from pathlib import Path
import hashlib, importlib.util, io, json, math, os, sys
D=Path(__file__).resolve().parent
H=D.parent
ENGINE=H/'native_screen_engine_v1/screen.py'
ENGINE_SHA='1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
RANKS=[52,58,64,70,76,80,84,88]
GATES=dict(raw_pct_strictly_below=0.,adjusted_pct_strictly_below=0.,all16_finite=True,layer0_exact=True,wins_and_ci_descriptive_only=True)

def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def assess(path,digest):
    ep=pin(ENGINE);assert ep['sha256']==ENGINE_SHA
    spec=importlib.util.spec_from_file_location('joint3_existing_metrics',ENGINE)
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    cp=q.pin(path);assert cp['sha256']==digest
    c=json.loads(q.checked(cp));q.checked(ep);q.checked(q.pin(__file__))
    assert c['schema']=='fixed-archived-joint3-recovery-native-check8-plan-v1'
    assert c['ranks']==RANKS and c['gates']==GATES
    assert c['check_disjoint_from_fit4'] and c['old_forecast_failure_preserved'] and c['no_new_fit_or_variant']
    assert c['check_only_after_actual_train_gain'] and c['bank_gain_pct_min']==1
    assert c['check_ranks']==RANKS and c['fit_ranks']==[35,100,36,45] and not set(RANKS)&set(c['fit_ranks'])
    fit4=json.loads(q.checked(c['fit4_plan']))
    assert fit4['candidate']==c['candidate'] and fit4['parent']==c['parent'] and fit4['ranks']==c['fit_ranks']
    admission_pin=q.pin(c['fit4_assessment_path']);admission=json.loads(q.checked(admission_pin))
    assert admission['passed_integrity'] and admission['admitted'] and admission['check8_permitted']
    assert admission['plan']==c['fit4_plan'] and admission['assessor']==c['fit4_assessor'] and admission['engine']==ep
    assert admission['candidate']==c['candidate'] and admission['parent']==c['parent'] and admission['gates'] and all(admission['gates'].values())
    assert admission['metrics']['raw_pct']<0 and admission['metrics']['adjusted_pct']<0
    q.checked(c['fit4_assessor'])
    q.guard(c['fit4_assessment_guard'],dict(seconds=120,rss_bytes=7500000000),[c['runtime'],'-B',c['fit4_assessor']['path'],c['fit4_plan']['path'],c['fit4_plan']['sha256']])
    for evidence in admission['evidence']:q.checked(evidence)
    assert str(Path(sys.executable).absolute())==c['runtime']==str(q.NATIVE) and q.np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in q.THREADS)
    root=json.loads(q.checked(c['root_selection']));recovery=json.loads(q.checked(c['recovery_plan']));build=json.loads(q.checked(c['build']))
    assert root['root_selected'] and root['recovery_plan']==c['recovery_plan'] and root['fit_gates']==GATES
    assert root['values']==build['values']==recovery['values']
    assert build['passed'] and build['source_inverse_exact'] and build['old_forecast_failure_preserved'] and build['new_fit_performed'] is False
    assert build['selection']==c['root_selection'] and build['recovery_plan']==c['recovery_plan']
    assert build['candidate']==c['candidate'] and build['parent']==recovery['parent']==c['parent']
    q.checked(build['builder'])
    assert q.checked(c['candidate'])==q.checked(c['parent'])+build['suffix'].encode()
    assert c['parent']['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
    old=json.loads(q.checked(recovery['original_closeout']));fit=json.loads(q.checked(recovery['original_fit']))
    assert old['no_endpoint_exported'] and old['no_actual_endpoint_or_check_or_broad_admitted']
    assert old['fit_result']==recovery['original_fit']==build['original_fit']
    assert old['predicted_fit_adjusted_gain_pct']<old['gate_pct']
    assert c['driver']['sha256']=='9c30282361d9cef4e53a3ea95ff44cbe5f8e2ec3fb8cd1fb4f1ceb1ad01b40c7'
    assert c['guard_source']['sha256']=='314ed428e1986abf6b7c05daa5c0ea7c1b5e6ebe6472ba4a34c2ed59a80b5f76'
    q.checked(c['driver']);q.checked(c['guard_source'])
    data=json.loads(q.checked(c['original_data_protocol']))
    assert data['check_ranks']==RANKS
    authority=json.loads(q.checked(c['roles']));store=json.loads(q.checked(c['store_manifest']))
    assert c['roles']==data['roles'] and c['store_manifest']==data['store_manifest']
    assert set(RANKS)<=set(authority['train']) and not set(RANKS)&set(authority['valid'])
    assert [r['rank'] for r in c['data']]==RANKS
    for row in c['data']:
        for key in ('row_manifest','weights','truth'):q.checked(row[key])
        r=row['rank'];assert row['weights']==authority['weights'][str(r)]
        assert row['truth']['sha256']==authority['targets'][str(r)]['sha256']
        assert row['row_manifest']['sha256']==store['roles']['observer_fit']['row_manifests'][r]['sha256']
    assert set(c['arms'])=={'base','candidate'}
    rows={};arrays={};artifacts={}
    for role,source in [('base',c['parent']),('candidate',c['candidate'])]:
        arm=c['arms'][role];folder=Path(arm['output'])
        assert arm['source']==source and arm['driver']==c['driver'] and arm['ranks']==RANKS
        assert folder==H/arm['tag'] and folder.name==arm['tag']
        limits=arm['limits'];assert limits==dict(seconds=300,rss_bytes=7500000000,added_swap_bytes=0)
        command=[c['runtime'],'-B',c['driver']['path'],source['path'],','.join(map(str,RANKS)),arm['tag']]
        assert command==arm['command'];q.guard(arm['guard'],limits,command)
        expected=[c['light'],'-B',c['guard_source']['path'],'--record',arm['guard'],'--seconds','300','--rss-limit','7500000000',*command]
        assert arm['guard_command']==expected
        manifest_pin=q.pin(folder/'manifest.json');m=json.loads(q.checked(manifest_pin))
        assert m['candidate']==source['path'] and m['sha256']==source['sha256'] and m['ranks']==RANKS
        assert m['runtime']==c['runtime'] and m['numpy']=='2.4.6' and m['thread_env']==dict.fromkeys(q.THREADS,'1')
        assert 0<=m['setup']['setup_s']<=4.1
        for key,count in c['setup_counts'].items():assert [z[1] for z in m['setup']['pb_log'] if z[0]==key]==[count]
        row_pin=q.pin(folder/'rows.jsonl');rr=[json.loads(x) for x in q.checked(row_pin).decode().splitlines()]
        assert [r['rank'] for r in rr]==RANKS;aa={};predpins={}
        for r in rr:
            assert all(math.isfinite(r[k]) and r[k]>0 for k in ('mse','adjusted','util','cpu_s','wall_s'))
            assert type(r['flops']) is int and 0<r['flops']<=2**41 and r['util']==r['flops']/2**41
            assert math.isclose(r['adjusted'],r['mse']*max(.1,r['util']),rel_tol=1e-14)
            u=r['ubc'];assert u['retired'] and u['retained_cache_bytes']==u['live_cache_bytes']==0 and 0<=u['max_resident_bytes']<=512000000
            predpins[str(r['rank'])]=q.pin(folder/f"pred_{r['rank']}.npy")
            a=q.np.load(io.BytesIO(q.checked(predpins[str(r['rank'])])),allow_pickle=False)
            assert a.shape==(16,1024) and a.dtype==q.np.float64 and q.np.isfinite(a).all()
            assert q.np.array_equal(a,a.astype(q.np.float32).astype(q.np.float64))
            aa[r['rank']]=a
        rows[role]=rr;arrays[role]=aa
        artifacts[role]=dict(manifest=manifest_pin,rows=row_pin,guard=q.pin(arm['guard']),predictions=predpins)
    for rank in RANKS:assert arrays['base'][rank][0].tobytes()==arrays['candidate'][rank][0].tobytes()
    m=q.metrics(rows['base'],rows['candidate'])
    gates=dict(raw_gain=m['raw_pct']<0,adjusted_gain=m['adjusted_pct']<0,all16_finite=True,layer0_exact=True)
    for r in list(q.EVIDENCE.values()):q.checked(r)
    return dict(schema='archived-joint3-paired-check8-assessment-v1',passed_integrity=True,admitted=all(gates.values()),component_point_passed=all(gates.values()),component_retention_eligible=all(gates.values()),fit4_admission=admission_pin,fit4_assessment_guard=q.pin(c['fit4_assessment_guard']),
        plan=cp,assessor=q.pin(__file__),engine=ep,candidate=c['candidate'],parent=c['parent'],root_selection=c['root_selection'],
        ranks=RANKS,metrics=m,gates=gates,artifacts=artifacts,evidence=list(q.EVIDENCE.values()),
        old_forecast_failure_preserved=True,no_new_fit_or_search=True,check_disjoint_from_fit4=True,wins_and_ci_descriptive_only=True,
        historical_validation_reused_not_pristine=True,retained=False,broad_admitted=False,bank_qualified=False,submitted=False,targets_evaluated=False,model_executed=False)

def main():
    assert len(sys.argv)==3
    out=D/'check8_assessment.json';assert not out.exists()
    try:result=assess(Path(sys.argv[1]).resolve(),sys.argv[2])
    except Exception as exc:
        with out.open('x') as f:json.dump(dict(passed_integrity=False,admitted=False,component_point_passed=False,component_retention_eligible=False,assessor=pin(__file__),error=dict(type=type(exc).__name__,message=str(exc))),f,indent=2);f.write('\n')
        raise
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('passed_integrity','admitted','gates','metrics')}))

if __name__=='__main__':main()
