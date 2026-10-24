"""Target-free independent finite-difference qualification."""
import time,resource
from common import HERE,verify_seal,write
def main():
    import numpy as np
    from math import erf,sqrt
    def ndtr(x):return np.asarray([.5*(1+erf(float(v)/sqrt(2))) for v in x])
    from common import tangent
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify_seal();verify_seal('dependency_fix_sealed.json');start=time.monotonic();rng=np.random.default_rng(2026092052);records=[]
    def gaussian(mu,c):
        v=np.diag(c);s=np.sqrt(v);a=mu/s;p=np.exp(-a*a/2)/np.sqrt(2*np.pi);g=ndtr(a);m=s*p+mu*g
        cov=c*g[:,None]*g[None,:]
        np.fill_diagonal(cov,(v+mu*mu)*g+mu*s*p-m*m)
        return m,cov
    def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
    for n in (6,11):
        q=rng.normal(size=(n,n));C=q@q.T/n+np.eye(n)*.5;W=rng.normal(size=(n,n))/np.sqrt(n)
        mu=rng.uniform(-2,2,n)*np.sqrt(np.diag(C));s=np.sqrt(np.diag(C));a=mu/s;p=np.exp(-a*a/2)/np.sqrt(2*np.pi);g=ndtr(a)
        pre=dict(C_pre=C,sigma=s,alpha=a,pdf=p,gate=g,mg=s*p+mu*g)
        e=rng.normal(size=n)*.1;q=rng.normal(size=(n,n))*.03;E=(q+q.T)/2
        out=tangent(pre,e,E,W);u=W.T@e;V=W.T@E@W;eps=1e-5
        plus=gaussian(mu+eps*u,C+eps*V);minus=gaussian(mu-eps*u,C-eps*V)
        fd=[(x-y)/(2*eps) for x,y in zip(plus,minus)]
        errors=[rel(x,y) for x,y in zip(out,fd)]
        x=tangent(pre,e,np.zeros_like(E),W);y=tangent(pre,np.zeros_like(e),E,W)
        linear=[rel(v,a+b) for v,a,b in zip(out,x,y)]
        rec=dict(n=n,tangent_relative=errors,linearity_relative=linear,passed=max(errors)<=1e-6 and max(linear)<=1e-12)
        records.append(rec)
    result=dict(passed=all(x['passed'] for x in records),records=records,truth_accessed=False,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'unit.json',result);print(result,flush=True);assert result['passed']
if __name__=='__main__':main()
