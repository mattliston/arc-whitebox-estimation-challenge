"""Independent explicit tensors certify the fixed sketch identities."""
import hashlib,json,resource
from pathlib import Path
import numpy as np
from algebra import contract,plane,project

HERE=Path(__file__).resolve().parent
rng=np.random.Generator(np.random.PCG64(2026092002))
rows=[]
for name,n,c,separation in [('ordinary',7,9,1.),('near_cancel',8,6,1e-6),('zero',6,5,0.)]:
    base=rng.normal(size=(n,c))
    delta=rng.normal(size=(n,c))*separation
    p=np.concatenate([base+delta,base-delta],axis=1)
    coef=rng.normal(size=c)
    lam=np.concatenate([coef,-coef])
    a,b=rng.normal(size=(2,n,11))
    q=np.linalg.qr(rng.normal(size=(n,4)))[0]
    tensor=np.zeros((n,n,n))
    singles=[]
    for k in range(p.shape[1]):
        v=p[:,k]
        one=float(lam[k])*v[:,None,None]*v[None,:,None]*v[None,None,:]
        tensor+=one
        singles.append(one.reshape(n,n*n))
    direct=np.einsum('ijk,jr,kr->ir',tensor,a,b)
    actual=contract(p,lam,a,b)
    error=float(np.linalg.norm(actual-direct)/max(1.,float(np.linalg.norm(direct))))
    projected=np.einsum('ia,jb,kc,abc->ijk',q@q.T,q@q.T,q@q.T,tensor)
    pp=project(p,q)
    projected_query=contract(pp,lam,a,b)
    reference=np.einsum('ijk,jr,kr->ir',projected,a,b)
    projected_error=float(np.linalg.norm(projected_query-reference)/max(1.,float(np.linalg.norm(reference))))
    repeated=np.stack([tensor[i,i,:] for i in range(n)])
    plane_error=float(np.linalg.norm(plane(p,lam)-repeated)/max(1.,float(np.linalg.norm(repeated))))
    aa=np.repeat(np.eye(n),n,axis=1)
    bb=np.tile(np.eye(n),(1,n))
    z=contract(p,lam,aa,bb)
    gram=z@z.T
    unfolding=tensor.reshape(n,n*n)
    gram_reference=unfolding@unfolding.T
    gram_error=float(np.linalg.norm(gram-gram_reference)/max(1.,float(np.linalg.norm(gram_reference))))
    sum_single=sum(x@x.T for x in singles)
    difference=float(np.linalg.norm(gram_reference-sum_single)/max(1.,float(np.linalg.norm(gram_reference))))
    assert max(error,projected_error,plane_error,gram_error)<=1e-11
    if name=='ordinary':assert difference>.05
    rows.append(dict(case=name,query_error=error,projection_error=projected_error,
                     plane_error=plane_error,full_gram_error=gram_error,
                     sum_individual_gram_relative_difference=difference))
record=dict(passed=True,cases=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            algebra_sha256=hashlib.sha256((HERE/'algebra.py').read_bytes()).hexdigest(),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'unit.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record))
