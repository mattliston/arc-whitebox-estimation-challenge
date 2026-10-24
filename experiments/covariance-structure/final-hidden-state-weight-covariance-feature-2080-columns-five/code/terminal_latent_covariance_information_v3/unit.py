"""Native features versus independently materialized signed covariance reads."""
import json,math,resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from feature import latent_covariance
from fit_library import load_library
from common import HERE,seal_check,write
seal_check();assert all(x['num_threads']==1 for x in threadpool_info())
rng=np.random.Generator(np.random.PCG64DXSM(2026092006));records=[]
for n,k in [(7,3),(9,5)]:
    C=np.eye(n)*rng.uniform(.4,2,size=n);H=rng.normal(size=(n,k)).astype(np.float32);W=(rng.normal(size=(n,n))/np.sqrt(n)).astype(np.float32)
    mu=rng.normal(size=n)*.1;v=rng.uniform(.3,1.4,size=n)
    theta=rng.normal(size=k*(k+1)//2);S=np.zeros((k,k));ix=0
    for i in range(k):
        for j in range(i,k):S[i,j]=S[j,i]=theta[ix];ix+=1
    for kind,alive in [('full',np.ones(n,dtype=bool)),('empty',np.zeros(n,dtype=bool)),('mixed',np.arange(n)%3!=0),('signed_cold',np.arange(n)%3!=0)]:
        Cuse=C.copy()
        if kind=='signed_cold':
            ii=np.flatnonzero(~alive);Cuse[ii,ii]*=-.3
        with flops.BudgetContext(flop_budget=1_000_000_000,quiet=True) as ctx:
            out=latent_covariance(fnp,*[fnp.asarray(x) for x in [Cuse,H,alive,W,mu,v]])
        got=np.asarray(out).astype(np.float64)@theta
        U=(np.sqrt(np.where(alive,np.diag(Cuse),0))[:,None]*H).astype(np.float32).astype(np.float64)
        masked=U@S@U.T
        cold=np.flatnonzero(~alive)
        Hc=H[cold].astype(np.float64)
        masked[cold,cold]=np.diag(Cuse)[cold]*np.sum(Hc*(Hc@S),axis=1)
        reference=np.diag(W.astype(np.float64).T@masked@W.astype(np.float64))
        sigma=np.sqrt(v);reference=reference*np.exp(-.5*(mu/sigma)**2)/(math.sqrt(2*math.pi)*2*sigma)
        rel=float(np.linalg.norm(got-reference)/max(np.linalg.norm(reference),1e-30));assert rel<=3e-6
        records.append(dict(n=n,channels=k,mask=kind,relative_error=rel,flops=ctx.summary_dict()['flops_used']))
rejected=[]
for bad in ['negative_active_covdiag','zero_active_covdiag','nonfinite_H','low_prevariance']:
    args=[C.copy(),H.copy(),alive.copy(),W.copy(),mu.copy(),v.copy()]
    active_index=int(np.flatnonzero(alive)[0])
    if bad=='negative_active_covdiag':args[0][active_index,active_index]=-1
    if bad=='zero_active_covdiag':args[0][active_index,active_index]=0
    if bad=='nonfinite_H':args[1][0,0]=np.nan
    if bad=='low_prevariance':args[-1][0]=1e-11
    try:
        with flops.BudgetContext(flop_budget=1_000_000_000,quiet=True):latent_covariance(fnp,*[fnp.asarray(x) for x in args])
    except AssertionError:rejected.append(bad)
    else:raise AssertionError('invalid domain accepted: '+bad)
lib,equations=load_library();cert=lib.certificate()
result=dict(passed=True,records=records,domain_rejections=rejected,fit_certificate=cert,equations=equations,truth_accessed=False,model_calls=0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE/'unit.json',result);print(json.dumps(result),flush=True)
