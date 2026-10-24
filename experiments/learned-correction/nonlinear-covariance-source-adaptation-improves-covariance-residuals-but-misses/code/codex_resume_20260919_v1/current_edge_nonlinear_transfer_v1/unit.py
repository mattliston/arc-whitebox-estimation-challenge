import resource
import numpy as np
from threadpoolctl import threadpool_info
from common import *
from model import forward,objective,reference
verify();assert all(t['num_threads']==1 for t in threadpool_info())
rng=np.random.default_rng(810926);records=[]
shapes={'left':(14,16),'right':(14,16),'edge':(3,16),'bias':(16,),'output':(16,)}
for n in (4,7,16):
    d=dict(F=rng.normal(size=(n,14)),R=rng.normal(size=(n,n)),S=rng.normal(size=(n,n)),
        pf=rng.uniform(.1,.4,n),gf=rng.uniform(.1,.9,n),sf=rng.uniform(.5,2,n),factor=np.array(.01),
        W=rng.normal(size=(n,n)),response=rng.uniform(.1,1,n))
    p={k:rng.normal(0,.12,size=s) for k,s in shapes.items()};p0={k:v+rng.normal(0,.01,size=v.shape) for k,v in p.items()}
    d['base']=forward(d,p0);u=rng.normal(0,.001,(n,n));d['U']=(u+u.T)/2
    nc=float(np.mean(d['U']**2));y=np.sum(d['W']*(d['U']@d['W']),axis=0)*d['response'];nv=float(np.mean(y*y))
    actual=forward(d,p);expected=reference(d,p);fr=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected));assert fr<=2e-10
    assert np.array_equal(actual,actual.T) and np.array_equal(forward(d,p0)-d['base'],np.zeros_like(actual))
    perm=rng.permutation(n);dp={k:v for k,v in d.items()}
    for k in ('F','pf','gf','sf'):dp[k]=d[k][perm]
    for k in ('R','S'):dp[k]=d[k][np.ix_(perm,perm)]
    assert np.linalg.norm(forward(dp,p)-actual[np.ix_(perm,perm)])<=2e-10*np.linalg.norm(actual)
    loss,g,_=objective(d,p,p0,nc,nv)
    def independent_loss(q):
        e=reference(d,q)-d['base']-d['U'];v=np.array([d['W'][:,j]@e@d['W'][:,j]*d['response'][j] for j in range(n)])
        return np.sum(e*e)/(n*n*nc)+np.sum(v*v)/(n*nv)+.001*sum(np.sum((q[k]-p0[k])**2) for k in KEYS)/528
    assert abs(loss-independent_loss(p))<=2e-10*max(abs(loss),1)
    errors=[]
    for k in KEYS:
        q={a:v.astype(np.complex128) for a,v in p.items()};ref=np.zeros_like(p[k])
        for idx in np.ndindex(q[k].shape):
            q[k][idx]+=1e-30j;ref[idx]=independent_loss(q).imag/1e-30;q[k][idx]-=1e-30j
        er=float(np.linalg.norm(g[k]-ref)/max(np.linalg.norm(ref),1e-30));assert er<=2e-10,(n,k,er);errors.append(er)
    norm=sum(float(np.sum(v*v)) for v in g.values())**.5
    step=1e-5/max(norm,1);q={k:v-step*g[k] for k,v in p.items()};assert independent_loss(q)<independent_loss(p)
    d32={k:np.asarray(v,dtype=np.float32) for k,v in d.items()};p32={k:v.astype(np.float32) for k,v in p.items()}
    f32=forward(d32,p32);_,g32,_=objective(d32,p32,p0,nc,nv)
    e32=float(np.linalg.norm(f32-actual)/np.linalg.norm(actual));ge32=max(float(np.linalg.norm(g32[k]-g[k])/np.linalg.norm(g[k])) for k in KEYS)
    assert max(e32,ge32)<=3e-5
    records.append(dict(n=n,parameters_checked=528,forward_relative=fr,max_complex_step_gradient_relative=max(errors),fp32_forward_relative=e32,fp32_gradient_relative=ge32,null_exact=True,symmetry_exact=True,permutation_passed=True,descent_passed=True))
write(HERE/'unit_result.json',dict(passed=True,fixtures=records,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,fixtures=records),flush=True)
