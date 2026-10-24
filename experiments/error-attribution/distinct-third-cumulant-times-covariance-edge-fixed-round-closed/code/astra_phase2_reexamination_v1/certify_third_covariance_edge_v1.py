"""Primary all-distinct K3-edge coefficient and exact query certificate."""
from functools import lru_cache
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from mlp_kprop.diagslice import eval_part,zero_repeated
from mlp_kprop.harmonic import HTensor
from mlp_kprop.partitions import check_vec_partition
from mlp_kprop.wick import relu_wick_coef
import mlp_kprop.kprop_harmonic as kp
from third_covariance_edge_v1 import ThirdCore,row

HERE=Path(__file__).resolve().parent
OUT=HERE/'third_covariance_edge_certificate_v1.json';assert not OUT.exists()
torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
terms=[]
for part,blocks in kp.get_all_terms(3,d_max=4,augment=True):
    if part==(1,1,1) and len(blocks)==2 and (1,1,1) in blocks and sorted(sum(b) for b in blocks)==[2,3]:
        terms.append(blocks)
assert len(terms)==3,terms
records=[]
for n,seed in ((8,9051723),(16,9051724)):
    rng=np.random.default_rng(seed)
    p=rng.normal(size=(n,n+3))*.2;lam=rng.normal(size=n+3)*.03
    U=rng.normal(size=(n,n))*.03;V=rng.normal(size=(n,n))*.2
    b=rng.normal(size=n)*.01;x=rng.normal(size=(n,n));B=x@x.T/n
    cp=np.einsum('ir,jr,kr,r->ijk',p,p,p,lam)
    star=np.einsum('ir,jr,kr->ijk',U,V,V);star=(star+star.transpose(1,0,2)+star.transpose(2,1,0))/3
    trace=np.einsum('i,jk->ijk',b,B);trace=trace+trace.transpose(1,0,2)+trace.transpose(2,1,0)
    T=cp+star+trace
    K=np.array([[T[i,i,j] for j in range(n)] for i in range(n)])
    td=zero_repeated(torch.from_numpy(T)).numpy()
    x=rng.normal(size=(n,n));C=x@x.T/n+np.eye(n)
    mu=rng.normal(size=n)*.5
    tower={1:HTensor(torch.from_numpy(mu),0),2:HTensor(torch.from_numpy(C),0),3:HTensor(torch.from_numpy(T),0)}
    @lru_cache(None)
    def wick(k,power):return relu_wick_coef(torch.from_numpy(mu),torch.from_numpy(C.diagonal().copy()),k,power)
    target=torch.zeros((n,n,n))
    for blocks in terms:target+=kp.multiply_wicks(eval_part(tower,blocks,3),check_vec_partition(blocks,3),(1,1,1),wick)
    target=zero_repeated(target).numpy()
    g=wick(1,1).numpy();d=wick(2,1).numpy()
    edge=C*np.outer(d,d)
    direct=td*(edge[:,:,None]*g[None,None,:]+edge[:,None,:]*g[None,:,None]+edge[None,:,:]*g[:,None,None])
    W=rng.normal(size=(n,n))*.2
    core=ThirdCore(p,lam,U,V,b,B,K)
    modes=rng.normal(size=(n,4))*.1;Clow=modes@modes.T
    elow=Clow*np.outer(d,d)
    lowtensor=td*(elow[:,:,None]*g[None,None,:]+elow[:,None,:]*g[None,:,None]+elow[None,:,:]*g[:,None,None])
    qerr=0.;lerr=0.
    for index in range(n):
        pred,low=row(core,C,g,d,W,index,modes)
        reference=np.einsum('ijk,i,j,kb->b',direct,W[:,index],W[:,index],W)
        rl=np.einsum('ijk,i,j,kb->b',lowtensor,W[:,index],W[:,index],W)
        qerr=max(qerr,float(np.max(np.abs(pred-reference))))
        lerr=max(lerr,float(np.max(np.abs(low-rl))))
    coeferr=float(np.linalg.norm(direct-target)/np.linalg.norm(target))
    record=dict(width=n,primary_coefficient_relative=coeferr,query_max=qerr,low_edge_query_max=lerr,
                passed=coeferr<1e-11 and max(qerr,lerr)<1e-12)
    records.append(record)
result=dict(passed=all(r['passed'] for r in records),primary_terms=terms,records=records,
            qualification='Independent primary diagram sum and small dense current CP/star/trace query, including exact repeated removal. No wide source accuracy or deployability claim.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
assert result['passed']
