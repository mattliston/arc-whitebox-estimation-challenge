"""Whole-block, fixed-q4 opportunity bill using the bank kernel schedule."""
import json
from pathlib import Path
import numpy as np
from flopscope.accounting import qr_cost

HERE=Path(__file__).resolve().parent;OUT=HERE/'cp_only_one_high_cost_v1.json';assert not OUT.exists()
assert json.loads((HERE/'cp_only_one_high_certificate_v1.json').read_text())['passed']
coef=json.loads((HERE/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
bank=json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']
def mm(m,k,n):
    if not m or not k or not n:return 0.
    if n>8192:return sum(mm(m,k,min(8192,n-j)) for j in range(0,n,8192))+m*n
    if k>8192:return sum(mm(m,min(8192,k-j),n) for j in range(0,k,8192))+(len(range(0,k,8192))-1)*m*n
    plain=2*m*k*n-m*n
    if any(t%32 for t in (m,k,n)):return float(plain)
    depths=(3,4,5) if max(m,k,n)<=1024 else (4,)
    return min(plain,*(sum(a*b for a,b in zip(coef[str(d)],(m*k*n,m*k,k*n,m*n))) for d in depths))
records=[];n=1024;q=4
for rank in (35,100):
    folder=HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
    x=np.load(folder/'x.npy');active=np.load(folder/'alive.npy').sum(1).astype(int)
    births=((np.count_nonzero(np.minimum(x[:,:,5],1-x[:,:,5])>=.01,axis=1)+31)//32*32).clip(max=n)
    R=0;old=0.;new=0.;layers=[]
    for l in range(16):
        a=int(active[l]);prev=n if l==0 else int(active[l-1]);b=int(births[l])
        removed=mm(n,prev,R)+8*n*R
        if l<15:removed+=mm(a,R,a)+4*a*R+2*n*R
        transport=0 if l==0 else q*(mm(prev,prev,n)+(0 if l==15 else mm(n,prev,n)))
        geometry=0 if l==15 else 4*(2*mm(a,a,q)+2*qr_cost(a,q))
        born=0 if l==15 else q*mm(a,b,a)
        # Full-width f64 polynomial envelope for pre-plane, inherited query,
        # one-high coefficients, coordinate repair, masks and slice sums.
        polynomial=2*((40*q*q+50*q)*n*n+40*q*n*b) if l<15 else 2*(8*q*n*n+10*q*q*n)
        # All large f32 products' input/output conversion/copy allowance.
        copies=(12*q*n*n+4*q*n*b) if l<15 else 6*q*n*n
        basis_push=2*mm(n,prev,q)
        added=transport+geometry+born+polynomial+copies+basis_push
        old+=removed;new+=added;R+=2*b
        layers.append(dict(layer=l,removed_cp_block=removed,new_block=added,transport=transport,
                           geometry=geometry,birth=born,polynomial_envelope=polynomial,copy_cast_allowance=copies,basis_push=basis_push))
    bill=next(r['flops'] for r in bank if r['rank']==rank);saving=(old-new)/bill
    records.append(dict(rank=rank,bank_flops=bill,removed_cp_block=old,new_block=new,
                        projected_flops=bill-old+new,projected_fraction_saved=saving,gate=saving>=.25,layers=layers))
record=dict(records=records,passed=all(r['gate'] for r in records),
            qualification='Source-blind shape opportunity price, fixed original active/birth tapes, f32 qualified bank kernel schedule, f64 explicit polynomial envelope and cast/copy allowance. Existing star/trace/cov/readout costs remain in measured baseline. Full candidate billing, changed selections, memory/runtime and f32 recursive accuracy unresolved.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({**record,'records':[{k:v for k,v in r.items() if k!='layers'} for r in records]},indent=2))
