"""Optimistic complete-block opportunity price for a q4 one-high carrier."""
import json
from pathlib import Path
import numpy as np
from flopscope.accounting import qr_cost,eigh_cost

HERE=Path(__file__).resolve().parent;OUT=HERE/'one_high_third_cost_v1.json';assert not OUT.exists()
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
rows=[];q=4;n=1024
for rank in (35,100):
    folder=HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
    x=np.load(folder/'x.npy');active=np.load(folder/'alive.npy').sum(1).astype(int)
    births=((np.count_nonzero(np.minimum(x[:,:,5],1-x[:,:,5])>=.01,axis=1)+31)//32*32).clip(max=n)
    R=0;old=0.;new=0.;layers=[]
    for l in range(16):
        a=int(active[l]);prev=n if l==0 else int(active[l-1]);nb=2*int(births[l])
        before=mm(n,prev,R)+8*n*R
        if l<15:before+=mm(a,R,a)+4*a*R+2*n*R
        # Carry q full matrices. Final layer needs only their W diagonals,
        # not q complete two-sided output matrices.
        transport=0 if l==0 else q*(mm(n,prev,prev)+(0 if l==15 else mm(n,prev,n)))
        geometry=0 if l==15 else 4*mm(a,a,q)+4*2*qr_cost(a,q)+2*eigh_cost(q)
        # Project only NEW paired cubes, not the whole historic CP bank.
        birth=0 if l==15 else q*mm(a,nb,a)+mm(q,a,nb)
        # Re-query inherited q-matrix sector and star, subtract duplicate
        # two-/three-high sectors, construct full repeated planes and
        # coordinate repair. Generous polynomial allowance, not measured.
        reproject=0 if l==15 else 2*(30*q*q*a*a+30*q*a*a+50*q*a*nb)
        scalar=2*(12*q*n*n+30*q*n)
        after=transport+geometry+birth+reproject+scalar
        old+=before;new+=after;R+=nb
        layers.append(dict(layer=l,old_cp_block=before,new_projected_block=after,transport=transport,geometry=geometry,birth_queries=birth,reprojection_allowance=reproject,read_allowance=scalar))
    bill=next(r['flops'] for r in bank if r['rank']==rank);fraction=(old-new)/bill
    rows.append(dict(rank=rank,current_measured_flops=bill,replaced_cp_flops=old,new_optimistic_block_flops=new,
                     projected_fraction_saved=fraction,gate_pass=fraction>=.15,layers=layers))
result=dict(rows=rows,passed=all(r['gate_pass'] for r in rows),qualification='Optimistic shape-only opportunity estimate for q4 one-high full-matrix carrier, not actual bill or established recursive implementation. Leaves existing star/trace costs in base; counts new star re-projection. Fixed active/birth tapes. No targets.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({**result,'rows':[{k:v for k,v in r.items() if k!='layers'} for r in rows]},indent=2))
