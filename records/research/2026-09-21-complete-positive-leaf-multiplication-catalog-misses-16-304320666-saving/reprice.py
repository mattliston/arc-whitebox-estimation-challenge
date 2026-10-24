"""Finite-family scalar cost envelope on the exact current Lazy census."""
from functools import lru_cache
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal = read(HERE / 'reprice_inputs_sealed.json')
for row in seal['pins']:
    assert sha(row['path']) == row['sha256'], row['path']
inputs = read(HERE / 'reprice_inputs.json')
assert not (HERE / 'reprice_result.json').exists()
ARENA = 1168900096
DEPTH = 14
MIN_LEAF = 1

def gemm(m, k, n):
    return 2*m*k*n - m*n

def fringe_parts(m, k, n, dims, shape):
    a, b, c = dims
    m0, k0, n0 = (x*d for x, d in zip(dims, shape))
    rm, rk, rn = m-m0, k-k0, n-n0
    cost, memory = 2*m0*n0*rk, 0
    if rk:
        memory = max(memory, 4*(m0*rk + rk*n0 + m0*n0))
    if rn:
        cost += gemm(m0, k, rn)
        memory = max(memory, 4*(m0*k + k*rn + m0*rn))
    if rm:
        cost += gemm(rm, k, n)
        memory = max(memory, 4*(rm*k + k*n + rm*n))
    return cost, memory, (rm, rk, rn)

def scalar_checks():
    count = 0
    for m in range(1, 14):
        for k in range(1, 14):
            for n in range(1, 14):
                for shape in ((2, 2, 2), (3, 3, 6), (3, 6, 3), (4, 4, 4)):
                    dims = tuple(x//d for x, d in zip((m, k, n), shape))
                    if min(dims) == 0:
                        continue
                    m0, k0, n0 = (x*d for x, d in zip(dims, shape))
                    extra, _, _ = fringe_parts(m, k, n, dims, shape)
                    assert gemm(m0, k0, n0) + extra == gemm(m, k, n)
                    count += 1
    for depth in range(6):
        m, k, n = 32*3, 32*5, 32*7
        a, b, c = m//2**depth, k//2**depth, n//2**depth
        value = gemm(a, b, c)
        for _ in range(depth):
            value = 7*value + 7*(a*b + b*c + a*c)
            a, b, c = 2*a, 2*b, 2*c
        growth = Fraction(7, 4)**depth
        formula = 2*m*k*n*Fraction(7, 8)**depth - m*n*growth
        formula += Fraction(7, 3)*(m*k + k*n + m*n)*(growth-1)
        assert value == formula
    return dict(classical_partition_cases=count, winograd_depth_cases=6, passed=True)

def pareto(options):
    result, best = [], float('inf')
    for item in sorted(options, key=lambda x: (x[1], x[0], x[2])):
        if item[0] < best:
            result.append(item)
            best = item[0]
    kept = [item for item in result if item[1] <= ARENA]
    if result[-1] not in kept:
        kept.append(result[-1])
    return tuple(kept)

def calculate(programs):
    @lru_cache(None)
    def frontier(m, k, n):
        options = [(gemm(m,k,n), 4*(m*k+k*n+m*n), 'gemm')]
        if max(m,k,n) > 1:
            for p in programs:
                shape, rank, (ca,cb,cc) = p['shape'], p['rank'], p['arithmetic']
                dims = tuple((x+d-1)//d for x,d in zip((m,k,n),shape))
                if min(dims) < MIN_LEAF:
                    continue
                a,b,c = dims
                linear = ca*a*b+cb*b*c+cc*a*c
                for cost,memory,child in frontier(a,b,c):
                    options.append((rank*cost+linear,rank*memory,p['name']+' / '+child))
                dims = tuple(x//d for x,d in zip((m,k,n),shape))
                if min(dims) < MIN_LEAF:
                    continue
                extra, memory_extra, remainders = fringe_parts(m,k,n,dims,shape)
                if not any(remainders):
                    continue
                a,b,c = dims
                linear = ca*a*b+cb*b*c+cc*a*c
                for cost,memory,child in frontier(a,b,c):
                    options.append((rank*cost+linear+extra,max(rank*memory,memory_extra),p['name']+'[peel] / '+child))
        return pareto(options)
    records = []
    for row in inputs['censuses']:
        data = read(row['path'])
        calls = data['outermost_mm_calls']
        assert len(calls) == row['expected_outermost_mm_calls']
        assert all(c['input_dtypes']==['float32','float32'] for c in calls)
        assert sum(min(*x['left'], *x['right']) == 0 for x in calls) == row['expected_empty_calls']
        assert sum(x['flops'] for x in calls) == data['outermost_mm_sum']
        assert data['total_flops'] == row['expected_total_flops']
        details, saved, limited = [], 0, 0
        for i, call in enumerate(calls):
            m,k = call['left']
            k2,n = call['right']
            assert k==k2 and min(m,k,n)>=0
            choices = frontier(m,k,n) if min(m,k,n)>0 else ((call['flops'],0,'unchanged_empty_parent'),)
            best = min(choices,key=lambda x:x[0])
            fits = [x for x in choices if x[1]<=ARENA]
            small = min(fits,key=lambda x:x[0]) if fits else None
            gain = max(0,call['flops']-best[0])
            mgain = max(0,call['flops']-small[0]) if small else 0
            assert mgain <= gain
            saved += gain
            limited += mgain
            details.append(dict(index=i,shape=(m,k,n),caller=call['caller'],actual_call_flops=call['flops'],
                                optimistic_best=best,leaf_storage_limited_best=small,
                                optimistic_saved=gain,leaf_storage_limited_saved=mgain))
        measured = data['total_flops']
        records.append(dict(rank=row['rank'],complete_cold_flops=measured,outermost_mm_calls=len(calls),
                            optimistic_saved=saved,optimistic_fraction=saved/measured,
                            optimistic_new_total=measured-saved,
                            leaf_storage_limited_saved=limited,leaf_storage_limited_fraction=limited/measured,
                            leaf_storage_limited_new_total=measured-limited,
                            details=sorted(details,key=lambda x:(-x['optimistic_saved'],x['index']))))
    states=frontier.cache_info().currsize
    frontier.cache_clear()
    return records, states

start=time.monotonic()
checks=scalar_checks()
smirnov=read(inputs['smirnov_certificate']['path'])
catalog=read(inputs['catalog_certificate']['path'])
assert smirnov['passed'] and catalog['passed']
base=[dict(name=p['name'],shape=p['shape'],rank=40,arithmetic=p['packed_linear_operations']) for p in smirnov['records']]
assert len(base)==6 and all(p['exact_basis_products']==p['shape'][0]*p['shape'][1]**2*p['shape'][2] and p['passed'] for p in smirnov['records'])
base.append(dict(name='bank_winograd',shape=(2,2,2),rank=7,arithmetic=(7,7,7)))
extra=[dict(name=p['name'],shape=p['shape'],rank=p['rank'],arithmetic=p['arithmetic']) for p in catalog['records']]
assert len(extra)==36
extra += [dict(name='DPS2026_accurate48',shape=(4,4,4),rank=48,arithmetic=(84,76,124)),
          dict(name='DPS2026_accurate48_transpose',shape=(4,4,4),rank=48,arithmetic=(76,84,124))]
need=1-3e-9/inputs['official_adjusted']
families=[]
for name,programs in [('executable_smirnov_and_bank_winograd',base),('extended_optimistic_catalog',base+extra)]:
    began=time.monotonic()
    records,states=calculate(programs)
    family=dict(name=name,programs=programs,records=records,states=states,wall_s=time.monotonic()-began,
                passes_necessary_3e9_gate=all(r['optimistic_fraction']>=need for r in records),
                leaf_storage_limit_passes_necessary_3e9_gate=all(r['leaf_storage_limited_fraction']>=need for r in records))
    families.append(family)
    print(json.dumps({k:v for k,v in family.items() if k not in ('records','programs')}),flush=True)
result=dict(completed=True,families=families,checks=checks,maximum_stages=DEPTH,minimum_leaf_dimension=MIN_LEAF,
            necessary_simultaneous_leaf_bytes_limit=ARENA,required_uniform_fraction_to_cross_3e9_at_unchanged_raw=need,
            new_model_calls=0,truth_arrays_read=0,fits=0,candidate=False,native_runtime_qualification=False,
            wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            source_seal_sha256=sha(HERE/'reprice_inputs_sealed.json'),
            qualification='Finite-family optimistic arithmetic only. Leaf storage is not total workspace. No feasible saving, native bill, runtime, accuracy or universal bound established.')
with (HERE/'reprice_result.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps({**{k:v for k,v in result.items() if k!='families'},'families':[
    {**{k:v for k,v in x.items() if k not in ('programs','records')},'records':[
        {k:v for k,v in r.items() if k!='details'} for r in x['records']]} for x in families]}),flush=True)
