"""Independent small tensor and complete terminal-dual certificates."""
from common import HERE,write,seal_check
seal_check()
import numpy as np
from threadpoolctl import threadpool_info
from feature import kernel
from tangent import terminal
assert all(r['num_threads']==1 for r in threadpool_info())
rng=np.random.Generator(np.random.PCG64DXSM(2026092012));rows=[]
for n,count,m in ((5,3,4),(8,5,7),(11,7,9)):
    for masking in (False,True):
        indices=rng.permutation(n)[:count]
        alive=(rng.uniform(size=n)>.25) if masking else np.ones(n,dtype=bool)
        b=(rng.normal(size=(n,count))*.2).astype(np.float32);b[indices,np.arange(count)]=0
        t=(rng.normal(size=count)*.4).astype(np.float32);s=(rng.normal(size=count)*.4).astype(np.float32)
        w=(rng.normal(size=(n,m))*.3).astype(np.float32);q=rng.normal(size=count)
        bm=b.astype(np.float64)*alive[:,None];u=bm.copy();v=bm.copy()
        u[indices,np.arange(count)]+=t.astype(np.float64)*alive[indices]
        v[indices,np.arange(count)]+=s.astype(np.float64)*alive[indices]
        dense=np.einsum('aj,bj,cj,j->abc',u,u,u,q)+np.einsum('aj,bj,cj,j->abc',v,v,v,q)
        ia,ib,ic=np.indices((n,n,n));dense[(ia==ib)|(ia==ic)|(ib==ic)]=0
        wd=w.astype(np.float64)
        expected=np.einsum('ai,bi,ci,abc->i',wd,wd,wd,dense,optimize=True)
        D=kernel(w,b,t,s,indices,alive);got=D@q
        err=float(np.max(np.abs(got-expected))/max(1.,float(np.max(np.abs(expected)))))
        assert err<=2e-12
        assert np.array_equal(D,kernel(w,b,t,s,indices,alive))
        rows.append(dict(n=n,selected=count,outputs=m,masked=masking,scaled_error=err,repeat_exact=True))
n=12;k=5;W=rng.normal(size=(n,n))/np.sqrt(n)
v=rng.uniform(.8,1.2,size=n);av=.8*v
a=dict(W=W,v=v,active_v=av,mu=rng.normal(size=n)*.15,k3base=rng.normal(size=n)*.003,bpre=rng.normal(size=n)*.002,radial=np.asarray(.01))
for name in ('m1','m2','m3'):a[name]=rng.normal(size=(n,2))*.2
for name in ('e1','e2','e3'):a[name]=rng.uniform(.01,.4,size=(n,2))
a.update(P_centers=np.zeros((16,30)),P_scales=np.ones((16,30))*2,
    P_enc0_w=rng.normal(size=(8,42))*.03,P_enc0_b=rng.normal(size=8)*.02,
    P_enc2_w=rng.normal(size=(7,8))*.1,P_enc2_b=rng.normal(size=7)*.02,
    P_enc4_w=rng.normal(size=(6,7))*.1,P_enc4_b=rng.normal(size=6)*.02,
    P_head=rng.normal(size=(16,6))*.3,P_head_b=rng.normal(size=16)*.01,
    P_gain=np.zeros(16),P_rms=np.ones(16)*.02)
K=rng.normal(size=(n,k))*.01;J=terminal(a,K).d;base=a['k3base']+3*a['active_v']*a['bpre']
cols=[]
for i in range(k):
    plus=terminal(a,K*0,k3_override=base+1e-5*K[:,i]).v
    minus=terminal(a,K*0,k3_override=base-1e-5*K[:,i]).v
    cols.append((plus-minus)/2e-5)
FD=np.stack(cols,axis=1)
error=float(np.linalg.norm(J-FD)/max(np.linalg.norm(FD),1e-30));assert error<=1e-6,error
result=dict(passed=True,tensor_fixtures=rows,complete_terminal_relative_error=error,
    all_population_feature_terms_retained=True,truth_accessed=False,new_estimator_calls=0)
write(HERE/'unit.json',result);print(__import__('json').dumps(result))

