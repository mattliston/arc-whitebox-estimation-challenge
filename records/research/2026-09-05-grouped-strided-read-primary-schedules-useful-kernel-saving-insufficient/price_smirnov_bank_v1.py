"""Whole-bank optimistic envelope for the audited primary Smirnov SLP.

Uses new actual leaf-call census, not the old Gaussian-first instrument.
Choose per shape among 1/2 primary levels, its transpose, and D0/3/4/5
Winograd children. No cyclic algorithm or unpriced arithmetic is invented.
"""
from functools import lru_cache
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;OUT=HERE/'smirnov_bank_cost_v1.json';assert not OUT.exists()
audit=json.loads((HERE/'smirnov_slp_audit_v1.json').read_text())
assert audit['max_exact_error']=='0' and audit['multiplications']==40
counts=audit['scalar_linear_arithmetic']
coef=json.loads((HERE/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
census=json.loads((HERE/'bank_matmul_census_v1.json').read_text())
bank=json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']
ARENA=1168900096


def leaf(m,k,n,d):
    if not d:return 2*m*k*n-m*n,m,k,n
    u=2**d;m,k,n=((x+u-1)//u*u for x in (m,k,n))
    return sum(a*b for a,b in zip(coef[str(d)],(m*k*n,m*k,k*n,m*n))),m//u,k//u,n//u


@lru_cache(None)
def hybrid(m,k,n,s,d,transpose):
    if not s:
        cost,a,b,c=leaf(m,k,n,d)
        # Necessary simultaneous full-batch leaf A,B,C storage only;
        # excludes transform intermediates, old parity and outputs.
        return cost,4*(7**d)*(a*b+b*c+a*c)
    div=(6,3,3) if transpose else (3,3,6)
    a,b,c=((x+t-1)//t for x,t in zip((m,k,n),div))
    ca,cb=(counts['B'],counts['A']) if transpose else (counts['A'],counts['B'])
    cost,memory=hybrid(a,b,c,s-1,d,transpose)
    return 40*cost+ca*a*b+cb*b*c+counts['C']*a*c,40*memory


records=[]
for row in census['records']:
    details=[];saved=0.;within_arena_saved=0.;covered=0.
    for key,item in row['shapes'].items():
        shape,ad,bd=key.split(';');m,k,n=map(int,shape.split(','))
        if not min(m,k,n) or ad!='float32' or bd!='float32':continue
        actual=item['flops']/item['calls'];covered+=item['flops']
        options=[]
        for s in (1,2):
            for d in (0,3,4,5):
                for tr in (False,True):
                    cost,memory=hybrid(m,k,n,s,d,tr)
                    options.append(dict(stages=s,winograd_depth=d,transpose=tr,arithmetic=cost,leaf_ABC_bytes=memory))
        best=min(options,key=lambda o:o['arithmetic'])
        gain=max(0.,actual-best['arithmetic'])*item['calls'];saved+=gain
        legal=[o for o in options if o['leaf_ABC_bytes']<=ARENA]
        better=min(legal,key=lambda o:o['arithmetic']) if legal else None
        limited=max(0.,actual-better['arithmetic'])*item['calls'] if better else 0.
        within_arena_saved+=limited
        if gain:
            details.append(dict(shape=(m,k,n),calls=item['calls'],actual_per_call=actual,
                                best=best,saving= gain,within_arena_lower_bound_best=better,
                                within_arena_optimistic_saving=limited))
    measured=next(x['flops'] for x in bank if x['rank']==row['rank'])
    records.append(dict(rank=row['rank'],native_census_flops=row['flops'],remote_measured_bank_flops=measured,
                        covered_matmul_flops=covered,optimistic_saved=saved,optimistic_fraction=saved/measured,
                        existing_arena_optimistic_saved=within_arena_saved,
                        existing_arena_optimistic_fraction=within_arena_saved/measured,
                        details=sorted(details,key=lambda x:-x['saving'])))
result=dict(records=records,existing_arena_bytes=ARENA,
            necessary_fraction_to_cross_4e9_at_unchanged_raw=1-4e-9/4.982231460183304e-9,
            qualification='Optimistic arithmetic envelope only: all SLP operations and child padding included; copy/allocation/remote overhead omitted. Full-batch ABC is only a memory lower bound, not a complete live-memory plan. Not universal over cyclic schemes or tiled execution.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**result,'records':[{k:v for k,v in r.items() if k!='details'} for r in records]},indent=2))
