"""Independent integer replay and exhaustive binary-support inequality."""
from fractions import Fraction
import itertools,math,resource
from common import *
seal_check();done('analyze');result=read(HERE/'result.json');cases=0
for m,k,n in ((1,1,1),(1,2,2),(2,2,2),(2,3,2)):
    for abits in itertools.product((0,1),repeat=m*k):
        for bbits in itertools.product((0,1),repeat=k*n):
            ts=[sum(abits[i*k+h]*bbits[h*n+j] for h in range(k)) for i in range(m) for j in range(n)]
            T=sum(ts);Q=sum(t>0 for t in ts)
            exact=sum(2*t-1 for t in ts if t)
            lower=T+max(0,T-m*n)
            factored=sum(sum(abits[i*k+h] for i in range(m))*sum(bbits[h*n+j] for j in range(n)) for h in range(k))
            assert factored==T and exact==2*T-Q and lower<=exact<=m*n*(2*k-1)
            cases+=1
meta=read(HERE/'inputs.json');threshold=1-Fraction(3,10**9)/Fraction('3.5844144212297072e-9');checks=[]
for row in result['rows']:
    rank=row['rank'];t=read(HERE/f'rank{rank:04d}/trace.json');ref=next(x for x in meta['rows'] if x['rank']==rank)
    assert t['outer_calls']==read(ref['attribution']['path'])['outermost_mm_calls']
    old=next(x for x in read(meta['previous_bound']['path'])['rows'] if x['rank']==rank)
    oldsites={(x['function'],x['line']):x['flops'] for x in old['site_bills']};newsites={k:0 for k in SITES}
    dense=lower=terms=0
    for i,x in enumerate(t['leaves']):
        assert x['index']==i and 0<=x['outer_index']<295 and 0<=x['layer']<16
        a=x['left_shape'];b=x['right_shape'];m,k,n=a[-2],a[-1],b[-1];batch=math.prod(a[:-2])
        assert a[:-2]==b[:-2] and k==b[-2] and batch==x['batches']
        formula=batch*m*n*(2*k-1);assert formula==x['actual_flops']==x['dense_flops']
        assert x['dense_terms']==batch*m*k*n and 0<=x['nonzero_terms']<=x['dense_terms']
        # Aggregating after summing can only loosen the per-batch lower bound.
        assert x['nonzero_terms']+max(0,x['nonzero_terms']-batch*m*n)<=x['retained_lower_flops']<=formula
        assert 0<=x['zero_operand_batches']<=batch and x['max_operand_chunk_bytes']<=1024*1024
        newsites[(x['function'],x['line'])]+=formula;dense+=formula;lower+=x['retained_lower_flops'];terms+=x['nonzero_terms']
    assert newsites==oldsites and dense==old['retained']==row['original_leaf_flops']
    remaining=old['complete_cold_flops']-old['inside']+lower
    removed=old['complete_cold_flops']-remaining
    assert remaining==row['optimistic_remaining_complete_flops'] and dense-lower==row['extra_zero_term_allowance']
    exact=Fraction(removed,old['complete_cold_flops'])
    assert [exact.numerator,exact.denominator]==row['saving_fraction_exact']
    assert (exact>=threshold)==row['passes_uniform_3e9_planning_gate']
    checks.append(dict(rank=rank,leaf_calls=len(t['leaves']),dense_flops=dense,retained_lower=lower,nonzero_terms=terms,passed=True))
assert result['both_pass_planning_gate']==all(x['passes_uniform_3e9_planning_gate'] for x in result['rows'])
write(HERE/'audit_result.json',dict(passed=True,exhaustive_binary_cases=cases,rows=checks,
    source_and_outer_census_reconciled=True,integer_gate_reproduced=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,exhaustive_binary_cases=cases,rows=checks),flush=True)
