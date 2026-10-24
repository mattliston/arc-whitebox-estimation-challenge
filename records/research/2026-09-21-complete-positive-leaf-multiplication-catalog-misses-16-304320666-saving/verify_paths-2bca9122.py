"""Independent replay of every emitted scalar schedule; no candidate search."""
import hashlib,json,resource
from pathlib import Path

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'reprice_result.json').read_text())
inputs=json.loads((HERE/'reprice_inputs.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert data['completed']
by_rank={r['rank']:json.loads(Path(r['path']).read_text()) for r in inputs['censuses']}

def replay(shape,path,programs):
    m,k,n=shape
    if path=='gemm':
        return 2*m*k*n-m*n,4*(m*k+k*n+m*n)
    op,tail=path.split(' / ',1)
    peeled=op.endswith('[peel]')
    if peeled:op=op[:-6]
    p=programs[op]
    u,v,w=p['shape']
    a,b,c=tuple(x//d if peeled else -(-x//d) for x,d in zip(shape,(u,v,w)))
    assert min(a,b,c)>=8
    child_cost,child_memory=replay((a,b,c),tail,programs)
    aa,bb,cc=p['arithmetic']
    cost=p['rank']*child_cost+aa*a*b+bb*b*c+cc*a*c
    memory=p['rank']*child_memory
    if peeled:
        mm,kk,nn=a*u,b*v,c*w
        assert (mm,kk,nn)!=(m,k,n)
        # Independently count the fringe as the difference between whole
        # classical multiplication and the divisible core's classical bill.
        cost += (2*m*k*n-m*n)-(2*mm*kk*nn-mm*nn)
        for x,y,z in ((mm,k-kk,nn),(mm,k,n-nn),(m-mm,k,n)):
            if min(x,y,z)>0:memory=max(memory,4*(x*y+y*z+x*z))
    return cost,memory

verified=0
for family in data['families']:
    programs={p['name']:p for p in family['programs']}
    assert len(programs)==len(family['programs'])
    for row in family['records']:
        actual=by_rank[row['rank']]
        assert row['complete_cold_flops']==actual['total_flops']
        assert len(row['details'])==len(actual['outermost_mm_calls']) and sorted(x['index'] for x in row['details'])==list(range(len(actual['outermost_mm_calls'])))
        saves=[0,0]
        for x in row['details']:
            old=actual['outermost_mm_calls'][x['index']]
            assert x['shape']==[*old['left'],old['right'][1]] and x['actual_call_flops']==old['flops']
            for j,name in enumerate(('optimistic_best','leaf_storage_limited_best')):
                entry=x[name]
                saving=x['optimistic_saved' if j==0 else 'leaf_storage_limited_saved']
                if entry is None:
                    assert j==1 and saving==0
                    continue
                cost,memory,path=entry
                if path=='unchanged_empty_parent':
                    assert min(x['shape'])==0 and cost==old['flops']==0 and memory==0
                else:
                    assert path.count(' / ')<=7
                    assert (cost,memory)==replay(x['shape'],path,programs)
                if j==1:assert memory<=data['necessary_simultaneous_leaf_bytes_limit']
                assert saving==max(0,old['flops']-cost)
                saves[j]+=saving
                verified+=1
        assert saves==[row['optimistic_saved'],row['leaf_storage_limited_saved']]
        assert row['optimistic_new_total']==actual['total_flops']-saves[0]
        assert row['leaf_storage_limited_new_total']==actual['total_flops']-saves[1]
        assert row['optimistic_fraction']==saves[0]/actual['total_flops']
        assert row['leaf_storage_limited_fraction']==saves[1]/actual['total_flops']
    threshold=data['required_uniform_fraction_to_cross_3e9_at_unchanged_raw']
    assert family['passes_necessary_3e9_gate']==all(r['optimistic_fraction']>=threshold for r in family['records'])
    assert family['leaf_storage_limit_passes_necessary_3e9_gate']==all(r['leaf_storage_limited_fraction']>=threshold for r in family['records'])

# The second catalog contains the first, so it cannot worsen either optimum.
for small,big in zip(data['families'][0]['records'],data['families'][1]['records']):
    old={x['index']:x for x in small['details']}
    for x in big['details']:
        for field in ('optimistic_saved','leaf_storage_limited_saved'):
            assert x[field]>=old[x['index']][field]

result=dict(passed=True,schedules_replayed=verified,all_call_records_reconciled=True,call_records_reconciled=sum(len(r['details']) for f in data['families'] for r in f['records']),
            catalog_superset_monotonicity=True,scalar_fringe_check_independent_of_DP=True,
            result_sha256=sha(HERE/'reprice_result.json'),verification_source_sha256=sha(Path(__file__)),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'paths_verified.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
