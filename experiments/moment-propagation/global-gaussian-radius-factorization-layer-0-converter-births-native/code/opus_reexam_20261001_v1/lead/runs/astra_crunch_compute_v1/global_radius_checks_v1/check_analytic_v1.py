"""Deterministic sphere/ReLU moment identity checks; no model or Monte Carlo."""
import hashlib, json, math, resource
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fn

D = Path(__file__).resolve().parent

def constants(n):
    c = math.exp(.5 * math.log(2/n) + math.lgamma((n+1)/2) - math.lgamma(n/2)) if n < 100000 else 1-1/(4*n)+1/(32*n*n)+5/(128*n**3)
    return (1., c, 1., (1+1/n)*c, 1+2/n)

def radial(n, k):
    if n == 2:
        return 2**(k/2)
    return math.exp((k/2)*math.log(n)+math.lgamma(1+k/2)+math.lgamma(n/2)-math.lgamma((n+k)/2))

nodes, weights = np.polynomial.legendre.leggauss(48)

def pair_raw(n, p, q, rho):
    a = math.acos(rho)
    lo, hi = a-math.pi/2, math.pi/2
    theta = (hi-lo)/2*nodes+(hi+lo)/2
    integral = (hi-lo)/2*np.dot(weights, np.maximum(0,np.cos(theta))**p*np.maximum(0,np.cos(theta-a))**q)
    rad = 2**((p+q)/2)*math.gamma(1+(p+q)/2) if n is None else radial(n,p+q)
    return rad*integral/(2*math.pi)

def uni_raw(n,k):
    if n is None:
        return 2**(k/2-1)*math.gamma((k+1)/2)/math.sqrt(math.pi)
    return .5*math.exp(k/2*math.log(n)+math.lgamma(n/2)+math.lgamma((k+1)/2)-.5*math.log(math.pi)-math.lgamma((n+k)/2))

def convert(f,m,r2,r3,r4,cov,pk21,pk22,n):
    _,c,_,a3,a4=constants(n)
    ms=m/c
    cs=cov+(1-1/(c*c))*f.outer(m,m)
    p21=pk21/a3+f.outer(r2,m/a3-ms)
    p22=pk22/a4+(1/a4-1)*f.outer(r2,r2)
    return ms,r2,r3/a3,r4/a4,cs,p21,p22

def recenter(f,m,r2,r3,r4,cov,p21,p22):
    v=r2-m*m
    k3=r3-3*m*r2+2*m**3
    k4=r4-4*m*r3+6*m*m*r2-3*m**4-3*v*v
    k21=p21-2*m[:,None]*cov
    k22=p22-2*m[:,None]*p21.T-2*m[None,:]*p21+4*f.outer(m,m)*cov-2*cov*cov
    return v,k3,k4,k21,k22

results=[]
for n in (2,3,16,1024):
    for rho in (-1.,-.9,0.,.3,.95,1.):
        scale=np.array([.3,2.7])
        ug=[np.array([uni_raw(None,k)*s**k for s in scale]) for k in range(1,5)]
        us=[np.array([uni_raw(n,k)*s**k for s in scale]) for k in range(1,5)]
        def rawmat(dim,p,q):
            out=np.empty((2,2))
            for i in range(2):
                for j in range(2):
                    out[i,j]=(uni_raw(dim,p+q) if i==j else pair_raw(dim,p,q,rho))*scale[i]**p*scale[j]**q
            return out
        g11,g21,g22=[rawmat(None,*pq) for pq in ((1,1),(2,1),(2,2))]
        s11,s21,s22=[rawmat(n,*pq) for pq in ((1,1),(2,1),(2,2))]
        mg,r2g,r3g,r4g=ug
        cg=g11-np.outer(mg,mg);pg21=g21-np.outer(r2g,mg);pg22=g22-np.outer(r2g,r2g)
        with fs.BudgetContext(flop_budget=2**60,quiet=True):
            args=[fn.asarray(x) for x in (mg,r2g,r3g,r4g,cg,pg21,pg22)]
            converted=convert(fn,*args,n)
            centered=recenter(fn,*converted)
        cc=[np.asarray(x) for x in converted]; yy=[np.asarray(x) for x in centered]
        ms,r2s,r3s,r4s=us
        cs=s11-np.outer(ms,ms)
        k21=s21-2*ms[:,None]*s11-ms[None,:]*r2s[:,None]+2*ms[:,None]**2*ms[None,:]
        central22=s22-2*ms[None,:]*s21-2*ms[:,None]*s21.T+4*np.outer(ms,ms)*s11+np.outer(ms*ms,r2s)+np.outer(r2s,ms*ms)-3*np.outer(ms*ms,ms*ms)
        vs=r2s-ms*ms
        k22=central22-np.outer(vs,vs)-2*cs*cs
        refs=us+[cs,s21-np.outer(r2s,ms),s22-np.outer(r2s,r2s)]
        errs=[float(np.max(np.abs(x-y))) for x,y in zip(cc,refs)]
        assert max(errs)<3e-9,(n,rho,errs)
        assert np.max(np.abs(yy[3]-k21))<3e-9
        assert np.max(np.abs(yy[4]-k22))<3e-9
        assert np.max(np.abs(np.diag(yy[3])-yy[1]))<3e-10
        assert np.max(np.abs(np.diag(yy[4])-yy[2]))<3e-9
        assert np.max(np.abs(cs-cs.T))<1e-12 and np.max(np.abs(yy[4]-yy[4].T))<1e-12
        assert all(np.isfinite(x).all() for x in cc+yy)
        results.append(dict(n=n,rho=rho,max_raw_error=max(errs),k21_error=float(np.max(np.abs(yy[3]-k21))),k22_error=float(np.max(np.abs(yy[4]-k22)))))

limit=[dict(n=n,a=constants(n),max_departure=max(abs(x-1) for x in constants(n))) for n in (1024,16384,1048576)]
assert all(limit[i]['max_departure']>limit[i+1]['max_departure'] for i in range(2))
out=dict(status='PASS',implementation='actual public flopscope NumPy on 2x2 arrays; deterministic 48-point angular quadrature',cases=results,constants1024=constants(1024),asymptotic=limit,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['Algebra and L0 homogeneous raw-moment conversion only.','Does not prove finite Hermite truncation exact, birth/radial ownership consistency, later-layer accuracy, or client/server timing.'])
(D/'ANALYTIC_v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='cases'}))
