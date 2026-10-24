import math,resource,time
from common import *
import numpy as np
from threadpoolctl import threadpool_info
from head import features
verify();verify('fit_sources_sealed.json');assert all(x['num_threads']==1 for x in threadpool_info())
assert read(HERE/'inputs_completed.json')['completed'];start=time.monotonic()
ordinal=int(sys.argv[1]);rec=read(HERE/f'capture_mini{ordinal:04d}.json');assert rec['passed'] and rec['truth_values_decoded']==0
check(rec['capture']);check(rec['weights']);W=np.load(rec['weights']['path'],allow_pickle=False)
records=[];out=HERE/f'inputs_mini{ordinal:04d}';out.mkdir(exist_ok=False)
def cdf(x):return np.array([.5*math.erfc(-float(v)/math.sqrt(2)) for v in x],np.float64)
with np.load(rec['capture']['path'],allow_pickle=False) as z:
    for l in LAYERS:
        C=z[f'C_{l:02d}'];K=z[f'K_{l:02d}'];mu=z[f'mu_{l:02d}'];H=z[f'H_{l:02d}'];radial=z[f'radial_{l:02d}'].item()
        F,R,S,s=features(np,cdf,l,mu,C,K,radial,W[l],H)
        assert F.shape==(1024,140) and F.dtype==np.float32
        assert np.array_equal(s,z[f'sigma_{l:02d}'])
        a=z[f'alpha_{l:02d}'];p=z[f'pdf_{l:02d}'];g=z[f'gate_{l:02d}'];rho=C/np.outer(s,s);np.fill_diagonal(rho,0)
        L=.5*np.outer(s,s)*(p[:,None]*g[None,:]+rho*(-a*p)[:,None]*p[None,:]+.5*rho*rho*((a*a-1)*p)[:,None]*(-a*p)[None,:])
        np.fill_diagonal(L,0);weight=L*L+(L.T)*(L.T);weight/=np.sum(weight)/(1024*1023)
        data=dict(F=F,R=R,S=S,sigma=s,L=L,weight=weight,Wnext=W[l+1],gnext=z[f'pdf_{l+1:02d}']/(2*z[f'sigma_{l+1:02d}']))
        assert all(np.isfinite(v).all() for v in data.values())
        path=out/f'layer{l:02d}.npz';np.savez(path,**data);records.append(dict(layer=l,input=pin(path)))
write(HERE/f'inputs_mini{ordinal:04d}.json',dict(passed=True,ordinal=ordinal,records=records,capture=pin(HERE/f'capture_mini{ordinal:04d}.json'),
      target_values_decoded=0,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(ordinal=ordinal,inputs_frozen=True,layers=len(records)),flush=True)
