"""Root-only fixed one-coordinate fit. CLI: PROTOCOL_PATH PROTOCOL_SHA.

Reads only saved FIT4 predictions and TRAIN truths; never executes a model.
"""
from pathlib import Path
import importlib.util, io, json, math, os, sys

D=Path(__file__).resolve().parent
H=D.parent
ENGINE=H/'native_screen_engine_v1/screen.py'
ENGINE_SHA='1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
FIT=[35,100,36,45]
CHECK=[52,58,64,70,76,80,84,88]
GATES=dict(predicted_fit_gain_pct_strictly_above=0.,actual_fit_raw_pct_strictly_below=0.,
    actual_fit_adjusted_pct_strictly_below=0.,actual_check_raw_pct_strictly_below=0.,
    actual_check_adjusted_pct_strictly_below=0.,all16_finite=True,layer0_exact=True,wins_and_ci_descriptive_only=True)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def configuration(path,digest):
    import hashlib
    assert hashlib.sha256(ENGINE.read_bytes()).hexdigest()==ENGINE_SHA
    q=module(ENGINE,'gain_existing_metrics')
    cp=q.pin(path);assert cp['sha256']==digest
    c=json.loads(q.checked(cp))
    assert c['schema']=='uniform-effective-gnn-gain-native-fit4-v1'
    assert c['fit_ranks']==FIT and c['check_ranks']==CHECK and c['parity_layers']==[0]
    assert c['step_fraction']==.02 and c['ridge_diagonal_fraction']==.1 and c['box']==[-1.,1.]
    assert c['gates']==GATES
    assert c['one_endpoint_only'] and c['old_failures_preserved'] and c['bank_gain_pct_min']==1
    assert c['fit_source']==q.pin(__file__)
    q.checked(c['endpoint_assessor']);q.checked(c['builder']);q.checked(c['parent'])
    prep=json.loads(q.checked(c['preparation']))
    assert prep['builder']==c['builder'] and prep['fit_source']==c['fit_source'] and prep['endpoint_assessor']==c['endpoint_assessor']
    assert prep['gates']==GATES
    role='candidate' if c['parent_role']=='gn3' else 'base'
    for stage in ('fit4','check8'):
        prior_plan=json.loads(q.checked(prep[stage+'_plan']))
        prior_assessment=json.loads(q.checked(prep[stage+'_assessment']))
        assert c['data'][stage]==prior_plan['data']
        assert c['controls'][stage]==dict(arm=prior_plan['arms'][role],artifacts=prior_assessment['artifacts'][role])
    broad=json.loads(q.checked(c['parent_full40_assessment']))
    assert broad['receipt_checks_passed'] and broad['stage']=='VALID40' and broad['baseline_submission']==333111
    assert broad['candidate']==prep['parents']['gn3']
    assert c['parent_role']==('gn3' if broad['component_passed'] else 'rrpi')
    assert c['parent']==prep['parents'][c['parent_role']]
    q.guard(c['parent_full40_assessment_guard']['path'],dict(seconds=120,rss_bytes=7500000000),
        [c['runtime'],'-B',broad['assessor']['path'],broad['config']['path'],broad['config']['sha256'],broad['admission']['sha256']])
    assert q.pin(c['parent_full40_assessment_guard']['path'])==c['parent_full40_assessment_guard']
    assert str(Path(sys.executable).absolute())==c['runtime']==str(q.NATIVE) and q.np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in q.THREADS)
    assert c['driver']['sha256']=='9c30282361d9cef4e53a3ea95ff44cbe5f8e2ec3fb8cd1fb4f1ceb1ad01b40c7'
    assert c['guard_source']['sha256']=='314ed428e1986abf6b7c05daa5c0ea7c1b5e6ebe6472ba4a34c2ed59a80b5f76'
    q.checked(c['driver']);q.checked(c['guard_source'])
    builder=module(c['builder']['path'],'gain_source_builder')
    assert set(c['arms'])=={'minus','plus'}
    for name,multiplier in [('minus',.98),('plus',1.02)]:
        arm=c['arms'][name];assert arm['multiplier']==multiplier
        source,proof=builder.edit_source(q.checked(c['parent']).decode(),multiplier)
        assert source.encode()==q.checked(arm['source'])
        assert c['source_inverse'][name]==dict(parent=c['parent'],candidate=arm['source'],**proof)
    return q,cp,c,builder

def truths(q,c,stage):
    ranks=FIT if stage=='fit4' else CHECK
    data=c['data'][stage];assert [r['rank'] for r in data]==ranks
    result={}
    for item in data:
        for key in ('row_manifest','weights','truth'):q.checked(item[key])
        with q.np.load(io.BytesIO(q.checked(item['truth'])),allow_pickle=False) as z:
            a=q.np.asarray(z['mc_post'][15],dtype=q.np.float64)
        assert a.shape==(1024,) and q.np.isfinite(a).all()
        result[item['rank']]=a
    return result

def panel(q,c,arm,ranks,truth,expected=None):
    source=arm['source'];q.checked(source)
    folder=Path(arm['output']);assert folder==H/arm['tag'] and arm['ranks']==ranks and arm['driver']==c['driver']
    seconds=150 if ranks==FIT else 300
    assert arm['limits']==dict(seconds=seconds,rss_bytes=7500000000,added_swap_bytes=0)
    command=[c['runtime'],'-B',c['driver']['path'],source['path'],','.join(map(str,ranks)),arm['tag']]
    assert arm['command']==command
    assert arm['guard_command']==[c['light'],'-B',c['guard_source']['path'],'--record',arm['guard'],'--seconds',str(seconds),'--rss-limit','7500000000',*command]
    q.guard(arm['guard'],arm['limits'],command)
    artifacts=dict(manifest=q.pin(folder/'manifest.json'),rows=q.pin(folder/'rows.jsonl'),guard=q.pin(arm['guard']),
        predictions={str(r):q.pin(folder/f'pred_{r}.npy') for r in ranks})
    if expected is not None:assert artifacts==expected
    m=json.loads(q.checked(artifacts['manifest']))
    assert m['candidate']==source['path'] and m['sha256']==source['sha256'] and m['ranks']==ranks
    assert m['runtime']==c['runtime'] and m['numpy']=='2.4.6' and m['thread_env']==dict.fromkeys(q.THREADS,'1')
    assert 0<=m['setup']['setup_s']<=4.1
    for key,count in c['setup_counts'].items():assert [z[1] for z in m['setup']['pb_log'] if z[0]==key]==[count]
    rows=[json.loads(s) for s in q.checked(artifacts['rows']).decode().splitlines()]
    assert [r['rank'] for r in rows]==ranks
    arrays={}
    for row in rows:
        rank=row['rank']
        assert all(math.isfinite(row[k]) and row[k]>0 for k in ('mse','adjusted','util','cpu_s','wall_s'))
        assert type(row['flops']) is int and 0<row['flops']<=2**41 and row['util']==row['flops']/2**41
        assert math.isclose(row['adjusted'],row['mse']*max(.1,row['util']),rel_tol=1e-14)
        u=row['ubc'];assert u['retired'] and u['retained_cache_bytes']==u['live_cache_bytes']==0 and 0<=u['max_resident_bytes']<=512000000
        a=q.np.load(io.BytesIO(q.checked(artifacts['predictions'][str(rank)])),allow_pickle=False)
        assert a.shape==(16,1024) and a.dtype==q.np.float64 and q.np.isfinite(a).all()
        assert q.np.array_equal(a,a.astype(q.np.float32).astype(q.np.float64))
        assert math.isclose(float(q.np.mean((a[15]-truth[rank])**2)),row['mse'],rel_tol=1e-12)
        arrays[rank]=a
    return rows,arrays,artifacts

def fit4_reuse(q,c,endpoint,x):
    """A content alias never changes the executed path in native receipts."""
    if x not in (-1.,1.):return None
    name='minus' if x==-1. else 'plus';arm=c['arms'][name]
    raw=q.checked(endpoint)
    assert raw==q.checked(arm['source'])
    assert (endpoint['sha256'],endpoint['bytes'])==(arm['source']['sha256'],arm['source']['bytes'])
    # Driver has a fixed module name and no source-relative setup directory.
    text=raw.decode()
    for token in ('__file__','co_filename','__code__','inspect.','sys.argv','getcwd','submission_dir','scratch_dir'):
        assert token not in text
    driver=q.checked(c['driver']).decode()
    assert "types.ModuleType('candidate_eval')" in driver and 'scratch_dir=None,submission_dir=None' in driver
    return dict(arm_name=name,executed_source=arm['source'],endpoint_alias=endpoint,
        identical_source_bytes=True,source_path_independent=True,
        actual_manifest=q.pin(Path(arm['output'])/'manifest.json'),actual_guard=q.pin(arm['guard']),
        same_driver=c['driver'],same_cold_order=FIT,data=c['data']['fit4'],
        no_endpoint_named_native_manifest_or_guard_created=True)

def endpoint_arm(c,source,stage,reuse=None):
    if reuse is not None:
        assert stage=='fit4' and reuse['endpoint_alias']==source
        return c['arms'][reuse['arm_name']]
    ranks=FIT if stage=='fit4' else CHECK;seconds=150 if stage=='fit4' else 300
    tag=D.name+'_endpoint_'+stage+'_v1';guard=str(D/('endpoint_'+stage+'_guard.json'))
    command=[c['runtime'],'-B',c['driver']['path'],source['path'],','.join(map(str,ranks)),tag]
    return dict(source=source,ranks=ranks,driver=c['driver'],tag=tag,output=str(H/tag),guard=guard,
        limits=dict(seconds=seconds,rss_bytes=7500000000,added_swap_bytes=0),command=command,
        guard_command=[c['light'],'-B',c['guard_source']['path'],'--record',guard,'--seconds',str(seconds),'--rss-limit','7500000000',*command],
        log=str(D/('endpoint_'+stage+'.log')))

def endpoint_commands(q,c,cp,source,stage,prerequisites,reuse=None):
    arm=None if reuse is not None else endpoint_arm(c,source,stage)
    assess=[c['runtime'],'-B',c['endpoint_assessor']['path'],cp['path'],cp['sha256'],stage]
    record=str(D/('endpoint_'+stage+'_assess_guard.json'))
    return dict(protocol=cp,source=source,stage=stage,prerequisites=prerequisites,arm=arm,
        model_execution_required=reuse is None,reused_actual_fit4=reuse,
        assessment_command=assess,assessment_guard=record,
        guarded_assessment=[c['light'],'-B',c['guard_source']['path'],'--record',record,'--seconds','120','--rss-limit','7500000000',*assess],
        assessment_log=str(D/('endpoint_'+stage+'_assess.log')),thread_env=dict.fromkeys(q.THREADS,'1'),sequential=True)

def main():
    assert len(sys.argv)==3
    q,cp,c,builder=configuration(Path(sys.argv[1]).resolve(),sys.argv[2]);np=q.np
    truth=truths(q,c,'fit4');control=c['controls']['fit4'];assert control['arm']['source']==c['parent']
    base,yy,basepins=panel(q,c,control['arm'],FIT,truth,control['artifacts'])
    arm_arrays={};artifacts={'base':basepins};arm_metrics={}
    for name in ('minus','plus'):
        rows,arrays,records=panel(q,c,c['arms'][name],FIT,truth)
        for rank in FIT:assert arrays[rank][0].tobytes()==yy[rank][0].tobytes()
        arm_arrays[name]=arrays;artifacts[name]=records;arm_metrics[name]=q.metrics(base,rows)
    A=[];residual=[];even=[]
    for row in base:
        r=row['rank'];scale=math.sqrt(max(.1,row['util'])/1024)
        odd=(arm_arrays['plus'][r][15]-arm_arrays['minus'][r][15])/2
        even_row=(arm_arrays['plus'][r][15]+arm_arrays['minus'][r][15])/2-yy[r][15]
        A.append(odd*scale);residual.append((yy[r][15]-truth[r])*scale)
        even.append(dict(rank=r,odd_weighted_squared=float(np.sum((odd*scale)**2)),even_weighted_squared=float(np.sum((even_row*scale)**2))))
    A=np.concatenate(A);residual=np.concatenate(residual)
    G=float(A@A);g=float(A@residual);Q=1.1*G
    assert all(math.isfinite(v) for v in (G,g,Q)) and G>=0
    x=0. if G==0 else min(1.,max(-1.,-g/Q))
    den=float(residual@residual);pred=float((residual+x*A)@(residual+x*A));assert den>0 and math.isfinite(pred)
    gain=100*(1-pred/den);admitted=gain>0;multiplier=1+.02*x
    result=dict(schema='uniform-effective-gnn-gain-fit-v1',passed_integrity=True,endpoint_admitted=admitted,
        protocol=cp,fitter=q.pin(__file__),parent=c['parent'],fit_ranks=FIT,x=x,G=G,g=g,Q=Q,
        effective_gain_multiplier=multiplier,predicted_fit_adjusted_gain_pct=gain,
        even_diagnostics=even,arm_metrics=arm_metrics,artifacts=artifacts,endpoint=None,fit4_reuse=None,
        check_targets_read=False,exact_gradient_claim=False,model_executed=False,one_endpoint_only=True,
        adaptive_reused_train=True,bank_qualified=False)
    if admitted:
        source,proof=builder.edit_source(q.checked(c['parent']).decode(),multiplier)
        with (D/'endpoint.py').open('x') as f:f.write(source)
        result['endpoint']=q.pin(D/'endpoint.py');result['source_inverse']=proof
        result['fit4_reuse']=fit4_reuse(q,c,result['endpoint'],x)
    result['evidence']=list(q.EVIDENCE.values())
    q.once(D/'fit_result.json',result)
    if admitted:q.once(D/'endpoint_fit4_commands.json',endpoint_commands(q,c,cp,result['endpoint'],'fit4',[q.pin(D/'fit_result.json')],result['fit4_reuse']))
    print(json.dumps({k:result[k] for k in ('passed_integrity','endpoint_admitted','predicted_fit_adjusted_gain_pct','x','effective_gain_multiplier','endpoint')}))

if __name__=='__main__':main()
