"""Bounded cost/memory Pareto DP; no numerical bank run or new candidate.

Mix the certified six rank40 orientations, bank-style Winograd, and the
2026 primary rational rank48 SLP with known84/76/124 arithmetic. At most
five recursion stages and child dimensions >=8. Include every child pad.
"""
from functools import lru_cache
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'mixed_fmm_envelope_v1.json';assert not OUT.exists()
certificate=json.loads((HERE/'smirnov_cyclic_certificate_v1.json').read_text());assert certificate['passed']
census=json.loads((HERE/'bank_matmul_census_v1.json').read_text())
bank=json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']
ARENA=1168900096
programs=[dict(name=p['name'],shape=p['shape'],rank=40,arithmetic=p['scalar_linear_arithmetic']) for p in certificate['programs']]
# The current packed Winograd input stages bill4 arithmetic+3 copy calls
# per block; C has7 arithmetic, already verified in measured coefficients.
programs.append(dict(name='bank_winograd',shape=(2,2,2),rank=7,arithmetic=(7,7,7)))
programs.append(dict(name='DPS2026_accurate48',shape=(4,4,4),rank=48,arithmetic=(84,76,124)))
programs.append(dict(name='DPS2026_accurate48_transpose',shape=(4,4,4),rank=48,arithmetic=(76,84,124)))


def pareto(options):
    ordered=sorted(options,key=lambda x:(x[1],x[0]));result=[];best=float('inf')
    for item in ordered:
        if item[0]<best:result.append(item);best=item[0]
    return tuple(result)


@lru_cache(None)
def frontier(m,k,n,stages):
    # Necessary simultaneous final-batch leaf A/B/C storage only. Not the
    # full arena (old/new transform parity, scratch, outputs are excluded).
    options=[(2*m*k*n-m*n,4*(m*k+k*n+m*n),'gemm')]
    if stages and min(m,k,n)>=16:
        for p in programs:
            dims=tuple((x+d-1)//d for x,d in zip((m,k,n),p['shape']))
            if min(dims)<8:continue
            a,b,c=dims;ca,cb,cc=p['arithmetic']
            linear=ca*a*b+cb*b*c+cc*a*c
            for cost,memory,child in frontier(a,b,c,stages-1):
                options.append((p['rank']*cost+linear,p['rank']*memory,p['name']+' / '+child))
    return pareto(options)


start=time.time();records=[]
for row in census['records']:
    saved=limited=0.;details=[]
    for key,item in row['shapes'].items():
        shape,ad,bd=key.split(';');m,k,n=map(int,shape.split(','))
        if min(m,k,n)==0 or ad!='float32' or bd!='float32':continue
        actual=item['flops']/item['calls']
        choices=frontier(m,k,n,5);best=min(choices,key=lambda x:x[0])
        fits=[x for x in choices if x[1]<=ARENA]
        small=min(fits,key=lambda x:x[0]) if fits else (float('inf'),0,'none')
        gain=max(0.,actual-best[0])*item['calls'];mgain=max(0.,actual-small[0])*item['calls']
        saved+=gain;limited+=mgain
        if gain:details.append(dict(shape=(m,k,n),calls=item['calls'],actual_per_call=actual,best=best,arena_lower_bound_best=small,
                                    optimistic_saved=gain,arena_lower_bound_saved=mgain))
    measured=next(r['flops'] for r in bank if r['rank']==row['rank'])
    records.append(dict(rank=row['rank'],remote_bank_flops=measured,optimistic_saved=saved,optimistic_fraction=saved/measured,
                        arena_lower_bound_saved=limited,arena_lower_bound_fraction=limited/measured,
                        details=sorted(details,key=lambda x:-x['optimistic_saved'])))
result=dict(records=records,programs=programs,maximum_stages=5,minimum_leaf_dimension=8,arena_bytes=ARENA,
            required_fraction_to_cross_4e9_at_unchanged_raw=1-4e-9/4.982231460183304e-9,
            dps2026_primary='https://arxiv.org/html/2603.18699v3#S3',
            states=frontier.cache_info().currsize,wall_s=time.time()-start,
            qualification='Bounded optimistic arithmetic envelope: child padding included; Smirnov/DPS literal copies, allocations and remote overhead excluded. Leaf-ABC memory is only a necessary full-batch lower bound. Not a native bill, runtime certificate, candidate or universal bound.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**result,'records':[{k:v for k,v in r.items() if k!='details'} for r in records]},indent=2))
