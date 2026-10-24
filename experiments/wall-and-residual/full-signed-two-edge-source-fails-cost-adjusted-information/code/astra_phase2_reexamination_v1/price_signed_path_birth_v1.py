"""Incremental CP cost on the fixed parent tapes; no kernel or truth use."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'signed_path_birth_cost_v1.json'
assert not OUT.exists()
assert json.loads((HERE/'signed_path_birth_certificate_v1.json').read_text())['passed']
price=(HERE/'price_cp_only_one_high_v1.py').read_text().split('records=[];n=1024;q=4',1)[0]
price=price.replace("OUT=HERE/'cp_only_one_high_cost_v1.json';assert not OUT.exists()",'OUT=None')
env=dict(__file__=str(HERE/'price_cp_only_one_high_v1.py'))
exec(compile(price,str(HERE/'price_cp_only_one_high_v1.py'),'exec'),env)
mm,bank=env['mm'],env['bank']
records=[]
for rank in (35,100):
    folder=HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
    x=np.load(folder/'x.npy')
    active=np.load(folder/'alive.npy').sum(1).astype(int)
    births=((np.count_nonzero(np.minimum(x[:,:,5],1-x[:,:,5])>=.01,axis=1)+31)//32*32).clip(max=1024)
    oldR=newR=0
    oldcost=newcost=0.
    layers=[]
    for l in range(16):
        n=1024;a=int(active[l]);prev=n if l==0 else int(active[l-1])
        young=0 if l==0 else int(births[l-1])
        old=mm(n,prev,oldR-2*young)+mm(n,prev,young)+4*n*young+8*n*oldR
        new=mm(n,prev,newR-4*young)+2*mm(n,prev,young)+8*n*young+8*n*newR
        if l<15:
            old+=mm(a,oldR,a)+4*a*oldR+2*n*oldR
            new+=mm(a,newR,a)+4*a*newR+2*n*newR
            # Conservative extra float64 polynomial/cast allowance for
            # both spectral factors and added birth masks/assembly.
            new+=160*n*n+32*n*int(births[l])
        oldcost+=old;newcost+=new
        layers.append(dict(layer=l,parent_cp=old,signed_cp=new,parent_columns=oldR,signed_columns=newR))
        oldR+=2*int(births[l]);newR+=4*int(births[l])
    bill=next(r['flops'] for r in bank if r['rank']==rank)
    projected=bill-oldcost+newcost
    records.append(dict(rank=rank,bank_flops=bill,old_cp_block=oldcost,new_cp_block=newcost,
                        projected_flops=projected,projected_multiplier=projected/bill,
                        layers=layers))
record=dict(records=records,
    qualification='Fixed source-blind active/birth tapes, current bank kernel polynomials and both youngest-pair sharing branches; explicit extra polynomial envelope. Projection only, not measured whole bill, changed-state selection, memory or runtime. Cost requires a large accuracy gain.')
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({**record,'records':[{k:v for k,v in r.items() if k!='layers'} for r in records]},indent=2))
