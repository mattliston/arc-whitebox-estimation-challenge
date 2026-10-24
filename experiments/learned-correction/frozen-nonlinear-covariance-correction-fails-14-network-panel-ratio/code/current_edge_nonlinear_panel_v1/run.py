from common import *
seal_check();assert read(NATIVE/'score.json')['passed']
meta=read(HERE/'inputs.json');guarded_names=[];completed_stages=[]
for stage in ('panel','confirmation'):
    if stage=='confirmation' and not read(HERE/'panel_score.json')['passed']:break
    rows=[]
    for row in meta[stage]:
        rank=row['rank'];name=f'{stage}_{rank}'
        guarded(name,120,'native.py','--out',name,'--ranks',rank);guarded_names.append(name)
        result=read(HERE/name/'result.json')
        assert result['completed'] and result['candidate']==meta['candidate'] and not result['truth_accessed']
        assert result['full_estimator_calls']==1 and result['capture_additional_fnp_operations']==0
        assert len(result['records'])==1
        c=result['records'][0]
        assert c['rank']==rank and c['includes_initialization'] and not c['bank_parameters'] and not c['truth_accessed']
        assert c['weights']==row['weights'] and c['numeric_cache_retired']
        for k in ('prediction','capture','weights'):verify(c[k])
        rows.append(dict(**row,call=c,result=pin(HERE/name/'result.json'),watch=pin(HERE/f'{name}_watch.json')))
    write(HERE/f'{stage}_predictions_sealed.json',dict(stage=stage,rows=rows,candidate=meta['candidate'],
        inputs=pin(HERE/'inputs.json'),sources=pin(HERE/'sources_sealed.json'),before_targets=True,
        all_cold=True,no_new_fit=True))
    for action in ('qualify','score'):
        name=f'{stage}_{action}';guarded(name,90,f'{action}.py',stage);guarded_names.append(name)
    completed_stages.append(stage)
    print(dict(stage=stage,passed=read(HERE/f'{stage}_score.json')['passed']),flush=True)
passed=completed_stages==['panel','confirmation'] and read(HERE/'confirmation_score.json')['passed']
write(HERE/'completed.json',dict(completed=True,passed=passed,completed_stages=completed_stages,
    confirmation_executed='confirmation' in completed_stages,guarded_names=guarded_names,
    release_qualified=False,uploaded=False))
print(dict(completed=True,passed=passed,completed_stages=completed_stages),flush=True)
