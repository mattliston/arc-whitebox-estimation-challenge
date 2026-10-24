import resource,time
from common import *
import numpy as np
import math
def ndtr(x):return np.array([.5*math.erfc(-float(v)/math.sqrt(2.)) for v in x],dtype=np.float64)
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from head import features,correction,predict_s
from math_core import initialize,forward,backward,losses,consumer

assert all(x['num_threads']==1 for x in threadpool_info());verify();start=time.monotonic()
rng=np.random.default_rng(172091);records=[]
for n in (17,70):
    q=rng.normal(size=(n,7));C=q@q.T/n+np.eye(n);mu=rng.normal(0,.3,n)
    K=rng.normal(0,.02,(n,n));W=rng.normal(0,.04,(n,n)).astype(np.float32);H=rng.normal(size=(n,64)).astype(np.float32)
    t=initialize(712031);t['output']=rng.normal(0,.1,32).astype(np.float32)
    t.update(feature_scale=np.ones(140,np.float32),rho_scale=np.array(.2,np.float32),third_scale=np.array(.05,np.float32),target_scale=np.array(.005,np.float32))
    expected=correction(np,ndtr,9,mu,C,K,.02,W,H,t)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        args=[fnp.asarray(x) for x in (mu,C,K,W,H)];ft={k:fnp.asarray(v) for k,v in t.items()}
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        result=correction(fnp,flops.stats.norm.cdf,9,*args[:3],.02,*args[3:],ft)
        repeat=correction(fnp,flops.stats.norm.cdf,9,*args[:3],.02,*args[3:],ft)
    actual=np.asarray(result);error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    assert error<=3e-5 and np.array_equal(actual,np.asarray(repeat)) and np.array_equal(np.diag(actual),np.zeros(n))
    pi=rng.permutation(n);pj=rng.permutation(n)
    perm=correction(np,ndtr,9,mu[pi],C[np.ix_(pi,pi)],K[np.ix_(pi,pi)],.02,W[np.ix_(pj,pi)],H[pj],t)
    perr=float(np.linalg.norm(perm-expected[np.ix_(pi,pi)])/np.linalg.norm(expected));assert perr<=3e-5
    t['output']*=0;zero=correction(np,ndtr,9,mu,C,K,.02,W,H,t);assert np.array_equal(zero,np.zeros_like(zero))
    records.append(dict(width=n,fnp_relative=error,permutation_relative=perr,repeat_exact=True,zero_exact=True,flops=ctx.summary_dict()['flops_used']))

n=9;F=rng.normal(0,.3,(n,140));R=rng.normal(0,.2,(n,n));S=rng.normal(0,.2,(n,n))
t={k:v.astype(np.float64) for k,v in initialize().items()};t['output']=rng.normal(0,.1,32)
c=dict(L=rng.normal(0,.2,(n,n)),Wnext=rng.normal(0,.2,(n,n)),gnext=rng.uniform(.1,.3,n),target=rng.normal(0,.1,(n,n)),weight=rng.uniform(.2,2,(n,n)))
for k in ('L','weight','target'):np.fill_diagonal(c[k],0)
energies=losses(c,np.zeros((n,n)));pred=forward(F,R,S,t,.2)
value,terms,grad=losses(c,pred,energies,True);der=backward(F,R,S,t,.2,grad);checks=[]
for j in range(12):
    direction={k:rng.normal(size=t[k].shape) for k in KEYS}
    norm=np.sqrt(sum(np.sum(x*x) for x in direction.values()));direction={k:x/norm for k,x in direction.items()}
    exact=sum(float(np.sum(der[k]*direction[k])) for k in KEYS);eps=1e-5
    vals=[]
    for sign in (-1,1):
        tt={k:t[k]+sign*eps*direction[k] for k in KEYS}
        vals.append(losses(c,forward(F,R,S,tt,.2),energies)[0])
    finite=(vals[1]-vals[0])/(2*eps);err=abs(finite-exact)/max(1e-7,abs(exact),abs(finite));assert err<=3e-6,(j,err)
    checks.append(dict(direction=j,analytic=exact,finite=finite,relative=err))
C,N=consumer(c,pred);direct=np.array([sum(c['Wnext'][i,k]*(c['L'][i,j]*pred[i,j]+c['L'][j,i]*pred[j,i])*c['Wnext'][j,k] for i in range(n) for j in range(n) if i!=j)*c['gnext'][k] for k in range(n)])
assert np.allclose(N,direct,rtol=1e-12,atol=1e-14)

# Shape-only component price, including one cold parameter ingress per layer
# (deliberately overcounts the future once-per-predict loader).
n=1024;q=rng.normal(0,.1,(n,7));C=q@q.T+np.eye(n);mu=rng.normal(0,.3,n)
K=rng.normal(0,.02,(n,n));W=rng.normal(0,.04,(n,n)).astype(np.float32);H=rng.normal(size=(n,64)).astype(np.float32)
t=initialize();t['output']=rng.normal(0,.1,32).astype(np.float32)
t.update(feature_scale=np.ones(140,np.float32),rho_scale=np.array(.2,np.float32),third_scale=np.array(.05,np.float32),target_scale=np.array(.005,np.float32))
with flops.BudgetContext(flop_budget=2**41,quiet=True):args=[fnp.asarray(x) for x in (mu,C,K,W,H)]
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    ft={k:fnp.asarray(v) for k,v in t.items()}
    delta=correction(fnp,flops.stats.norm.cdf,9,*args[:3],.02,*args[3:],ft)
    corrected=args[2]+delta
bill=ctx.summary_dict()['flops_used'];assert bill*7<=15_000_000_000,bill
assert np.isfinite(np.asarray(corrected)).all()
write(HERE/'unit_result.json',dict(passed=True,kernel=records,gradient_checks=checks,consumer_direct_passed=True,
      component_flops_one_layer_cold=bill,conservative_seven_layer_flops=bill*7,price_cap=15_000_000_000,
      actual_final_estimator_billed=False,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(read(HERE/'unit_result.json'),flush=True)
