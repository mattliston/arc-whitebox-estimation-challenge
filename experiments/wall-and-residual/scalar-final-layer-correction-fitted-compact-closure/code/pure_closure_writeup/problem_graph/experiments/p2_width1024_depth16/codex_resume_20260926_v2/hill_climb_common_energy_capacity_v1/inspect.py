from pathlib import Path
import json,hashlib
import numpy as np
S=Path(__file__).resolve().parent;H=S.parent
rows=[]
for rank in [35,100]:
 for layer in [10,14]:
  p=H/'local_common_trace_capture_v1'/f'cell_{rank:04d}_{layer:02d}.npz'
  with np.load(p,allow_pickle=False) as z:
   cohorts=z['cohorts'].tolist();split=int(z['split'])
   parts=[('F',0)] if 'F' in z.files else [('base',0),('join',split)]
   for name,offset in parts:
    F=z[name];R,N=F.shape;energy=np.zeros(R,dtype=np.float64);covered=0
    for start,width in cohorts:
     start,width=int(start),int(width)
     if start<offset or start+width>offset+N:continue
     a=start-offset;half=width//2
     Fu=(F[:,a:a+half]+F[:,a+half:a+width])*.5
     energy+=np.sum(Fu.astype(np.float64)**2,axis=1);covered+=width
    assert covered==N
    order=np.argsort(-energy,kind='stable');total=energy.sum()
    for k in [192,256]:
     u=min(k,R);selected=np.sort(order[:u]);leading=np.arange(u)
     rows.append(dict(rank=rank,layer=layer,block=name,shape=[R,N],keep=u,leading_missing_fraction=float(1-energy[:u].sum()/total),selected_missing_fraction=float(1-energy[selected].sum()/total),changed_coordinates=int(np.sum(selected!=leading))))
    del F,Fu
r=dict(rows=rows,no_targets_or_endpoint_scores=True,limitation='Coefficient Frobenius energy only; propagated query basis, signed moments and endpoint accuracy are not measured.')
with(S/'result.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
