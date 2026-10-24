import resource
import numpy as np
from threadpoolctl import threadpool_info
from common import *
from metrics import stats,arm_stats
from tangent import tangent_cov
seal_check();assert all(t['num_threads']==1 for t in threadpool_info())
rng=np.random.default_rng(830926);count=0;maximum=0.
s=np.array([3.,0,0]);a=s+np.array([0.,2,0]);b=s+np.array([0.,0,2])
q=stats(a,b);assert q['cross_energy']==3 and q['dispersion_energy']==4/3
assert stats(np.zeros(3),np.zeros(3))['dispersion_fraction'] is None
assert stats(s,-s)['dispersion_fraction']==2
for n in (3,5,9):
    for scale in (0.,1.,.003):
        a=rng.normal(size=(n,n))*scale;b=rng.normal(size=(n,n))*scale;d=rng.normal(size=(n,n))*scale
        if scale:arm_stats(a,b,d)
        q=stats(a,b);mask=~np.eye(n,dtype=bool);diag=stats(np.diag(a),np.diag(b));off=stats(a[mask],b[mask])
        for key in ('energy_A','energy_B','observed_energy','cross_energy','dispersion_energy'):
            x=(diag[key]*n+off[key]*(n*n-n))/(n*n)
            assert abs(x-q[key])<=2e-10*max(q['observed_energy'],1e-30)
        count+=1
for n in (3,5,7):
    W=rng.normal(size=(n,n));e=rng.normal(size=n);E=rng.normal(size=(n,n));E=(E+E.T)*.5
    pre=dict(sigma=rng.uniform(.7,1.5,n),gate=rng.uniform(.1,.9,n),pdf=rng.uniform(.05,.4,n),alpha=rng.normal(size=n),mg=rng.uniform(.1,1,n),C_pre=rng.normal(size=(n,n)))
    value=tangent_cov(pre,e,E,W);u=np.empty(n);V=np.empty((n,n))
    for i in range(n):
        u[i]=sum(W[k,i]*e[k] for k in range(n))
        for j in range(n):V[i,j]=sum(W[k,i]*E[k,l]*W[l,j] for k in range(n) for l in range(n))
    ref=np.empty((n,n));dg=pre['pdf']*u/pre['sigma']-pre['alpha']*pre['pdf']*np.diag(V)/(2*pre['sigma']**2)
    for i in range(n):
        for j in range(n):ref[i,j]=pre['gate'][i]*pre['gate'][j]*V[i,j]+pre['C_pre'][i,j]*(dg[i]*pre['gate'][j]+pre['gate'][i]*dg[j])
        ref[i,i]=(pre['gate'][i]-pre['mg'][i]*pre['pdf'][i]/pre['sigma'][i])*V[i,i]+2*pre['mg'][i]*(1-pre['gate'][i])*u[i]
    ref=(ref+ref.T)*.5;err=relative(value,ref);assert err<=2e-10;maximum=max(maximum,err)
write(HERE/'unit.json',dict(passed=True,decomposition_cases=count+3,tangent_fixtures=3,maximum_tangent_relative=maximum,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,maximum_tangent_relative=maximum),flush=True)
