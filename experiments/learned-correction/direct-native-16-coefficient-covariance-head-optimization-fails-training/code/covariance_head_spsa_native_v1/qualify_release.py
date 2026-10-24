import numpy as np
from common import HERE,PARENT,read,write,pin,verify,sha
for p in read(HERE/'release_sources_sealed.json')['pins']:verify(p)
done=read(HERE/'release_client_completed.json');checks=[];scopes=[]
def equivalent(c,n):
    assert c['rank']==n['rank'] and c['weights_sha256']==n['weights_sha256']
    q=np.load(c['output'],allow_pickle=False);p=np.load(n['output'],allow_pickle=False)
    assert q.shape==p.shape==(16,1024) and q.dtype==p.dtype==np.float32
    assert q.tobytes()==p.tobytes() and c['summary']['flops_used']==n['flops']
for scope in ['conservative','prediction_only']:
    pairs=[]
    for rank in [35,100]:
        pair={}
        for role in ['parent','candidate']:
            e=next(r for r in done['rows'] if (r['scope'],r['rank'],r['role'])==(scope,rank,role))
            verify(e['receipt']);verify(e['watch']);r=read(e['receipt']['path']);c=r['records'][0]
            source=PARENT if role=='parent' else HERE/'candidate.py';assert r['candidate_sha256']==sha(source)
            assert r['completed'] and not r['truth_accessed'] and not r['numpy_available']
            native_dir=(f'check_parent_{rank}' if role=='parent' else ('check_candidate_triple' if rank==35 else 'check_candidate_cold100'))
            n=read(HERE/native_dir/'result.json')['records'][0];equivalent(c,n);pair[role]=c
            checks.append(dict(scope=scope,rank=rank,role=role,all16_bytes_and_bill_exact=True,residual=c['summary']['residual_wall_time_s']))
        a,b=pair['candidate'],pair['parent']
        pairs.append(dict(rank=rank,candidate_wall=a['wall_s'],parent_wall=b['wall_s'],wall_ratio=a['wall_s']/b['wall_s'],
            candidate_residual=a['summary']['residual_wall_time_s'],parent_residual=b['summary']['residual_wall_time_s'],
            candidate_module_setup_outside_budget_s=a.get('module_setup_outside_budget_s'),parent_module_setup_outside_budget_s=b.get('module_setup_outside_budget_s')))
    ratio=sum(p['candidate_wall'] for p in pairs)/sum(p['parent_wall'] for p in pairs)
    passed=ratio<=1.075 and all(p['wall_ratio']<=1.10 and p['candidate_residual']<=.25 for p in pairs)
    scopes.append(dict(scope=scope,pairs=pairs,pooled_wall_ratio=ratio,passed=passed))
verify(done['warm']);warm=read(done['warm']['path']);native=read(HERE/'check_candidate_triple/result.json')['records']
assert warm['candidate_sha256']==sha(HERE/'candidate.py') and not warm['numpy_available']
for c,n in zip(warm['records'],native):equivalent(c,n)
assert len(warm['records'])==3 and warm['records'][0]['output_sha256']==warm['records'][2]['output_sha256']
warm_pass=all(r['summary']['residual_wall_time_s']<=.25 for r in warm['records'])
result=dict(passed=all(s['passed'] for s in scopes) and warm_pass,scopes=scopes,parity=checks,
    warm_bytes_and_bills_exact=True,warm_residual_pass=warm_pass,client_full_calls=11,
    official_completion_proven=False,causal_runtime_improvement_claimed=False)
write(HERE/'release_qualified.json',result);print(result,flush=True)
