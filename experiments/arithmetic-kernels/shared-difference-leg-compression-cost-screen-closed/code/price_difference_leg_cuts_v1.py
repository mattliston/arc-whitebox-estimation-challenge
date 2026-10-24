"""Target-free cost screen, not a carrier implementation or measured gain.

One proposed geometry only: retain the common leg of paired signed cubes;
compress only their difference leg to rank128 at cuts8/12, ages>=4.
Unlike the closed per-cohort P/Q/SVD family, basis changes occur twice in
total and are shared across old cohorts. Include acquisition and repeated-
plane restoration before contemplating any accuracy experiment.
"""
import json
from pathlib import Path
import numpy as np
from flopscope.accounting import eigh_cost

here=Path(__file__).resolve().parent
out=here/'difference_leg_cuts_cost_v1.json'
assert not out.exists()
coef=json.loads((here/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
bank=json.loads((here/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']

def mm(m,k,n):
    if not m or not k or not n:return 0.
    if n>8192:return sum(mm(m,k,min(8192,n-j)) for j in range(0,n,8192))+m*n
    if k>8192:return sum(mm(m,min(8192,k-j),n) for j in range(0,k,8192))+(len(range(0,k,8192))-1)*m*n
    plain=2*m*k*n-m*n
    if any(t%32 for t in (m,k,n)):return float(plain)
    depths=(3,4,5) if max(m,k,n)<=1024 else (4,)
    prices=[sum(a*b for a,b in zip(coef[str(d)],(m*k*n,m*k,k*n,m*n))) for d in depths]
    return min(plain,*prices)

def mixed_read(a,k,r):
    return mm(a,k,r)+mm(a,r,a)+mm(a,k,a)+12*a*k+3*a*a

rows=[]
for rank in (35,100):
    folder=here/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
    x=np.load(folder/'x.npy');alive=np.load(folder/'alive.npy')
    active=alive.sum(1).astype(int)
    births=((np.count_nonzero(np.minimum(x[:,:,5],1-x[:,:,5])>=.01,axis=1)+31)//32*32).clip(max=1024)
    group=0;r=128;old_total=0.;new_total=0.;acquisition=0.;layers=[];cuts=[]
    for l in range(16):
        if group and l:
            a=int(active[l]);b=int(active[l-1]);k=group
            old=mm(1024,b,2*k)
            new=mm(1024,b,k)+mm(1024,b,r)+mm(1024,r,k)+12*1024*k
            if l<15:
                old+=mm(a,2*k,a)+6*a*k
                new+=mixed_read(a,k,r)
            else:
                # Conservative extra arithmetic of the mixed scalar read.
                new+=10*1024*k
            old_total+=old;new_total+=new
            layers.append(dict(layer=l,old_pairs=k,old_flops=old,new_flops=new,saving=old-new))
        if l in (8,12):
            group=int(births[:l-3].sum());a=int(active[l]);k=group
            # Optimistically assume the old cohort plane is already retained
            # from its ordinary read. New plane restoration is fully priced.
            cost=mm(a,k,a)+eigh_cost(a)+mm(r,a,k)+mm(a,r,k)+mixed_read(a,k,r)+20*a*k
            acquisition+=cost
            cuts.append(dict(layer=l,pairs=k,active=a,acquisition_and_restoration=cost))
    total=next(z['flops'] for z in bank if z['rank']==rank)
    net=old_total-new_total-acquisition
    rows.append(dict(rank=rank,active=active.tolist(),births=births.tolist(),layers=layers,cuts=cuts,
                     old_affected_block_flops=old_total,new_affected_block_flops=new_total,
                     acquisition_flops=acquisition,optimistic_net_saving=net,
                     current_first_context_measured_flops=total,optimistic_total_fraction_saved=net/total))
passed=all(z['optimistic_total_fraction_saved']>=.15 for z in rows)
receipt=dict(rows=rows,cost_information_gate_pass=passed,
             qualification='Analytical projection from existing metered kernel polynomials and public eig cost, current 35/100 shape tapes. Not measured candidate FLOPs, runtime, or raw accuracy. Old-plane reuse is optimistic; reject if even this fails15%.')
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(cost_information_gate_pass=passed,rows=[{k:v for k,v in z.items() if k not in ('active','births','layers','cuts')} for z in rows]),indent=2))
