import math,importlib.util,time,resource,json
from common import HERE,write,verify,pin
def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify();started=time.monotonic()
    spec=importlib.util.spec_from_file_location('ec_unit_kernel',HERE/'kernel.py');k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)
    cdf=lambda x:np.fromiter((.5*math.erfc(-float(v)/math.sqrt(2)) for v in np.asarray(x).flat),dtype=np.float64,count=np.asarray(x).size).reshape(np.asarray(x).shape)
    nodes,weights=np.polynomial.legendre.leggauss(32);k._ec_nodes=nodes;k._ec_weights=weights;k.fnp=np;k._ec_cdf=cdf
    write(HERE/'gl32.json',dict(nodes=nodes.tolist(),weights=weights.tolist(),construction='numpy.polynomial.legendre.leggauss(32)',numpy_version=np.__version__))
    xn,wn=np.polynomial.legendre.leggauss(128);xi,wi=np.polynomial.legendre.leggauss(200)
    cases=[];maxprob=0.;maxmoment=0.
    for alpha,rhos in [([-8.,-3.,0.,2.,8.],[0.,.3,-.3,.85,-.85]),([-2.,-.3,1.2,2.],[.1,-.2,.4,-.4])]:
        a=np.array(alpha);n=len(a);pdf=np.exp(-a*a/2)/np.sqrt(2*np.pi);g=cdf(a)
        for rho_value in rhos:
            rho=np.full((n,n),rho_value);geometry=k._ec_geometry(a,rho,pdf,g);M=geometry[0]
            t=(xn+1)*rho_value/2;den=1-t*t
            density=np.exp(-.5*(a[:,None,None]**2+a[None,:,None]**2-2*t[None,None,:]*a[:,None,None]*a[None,:,None])/den)/(2*np.pi*np.sqrt(den))
            pref=g[:,None]*g[None,:]+rho_value*np.sum(wn*density,axis=2)/2
            perr=float(np.max(np.abs(M[(0,0)]-pref)));maxprob=max(maxprob,perr);assert perr<=2e-11,(rho_value,perr)
            # Direct conditional quadrature over positive physical X.
            upper=np.maximum(16.,a+16.);xx=(xi[None,:]+1)*upper[:,None]/2
            xw=wi[None,:]*upper[:,None]/2*np.exp(-(xx-a[:,None])**2/2)/np.sqrt(2*np.pi)
            cy=a[None,:,None]+rho_value*(xx[:,None,:]-a[:,None,None]);sd=np.sqrt(1-rho_value*rho_value);z=cy/sd
            cg=cdf(z);cp=np.exp(-z*z/2)/np.sqrt(2*np.pi)
            cm={0:cg,1:sd*cp+cy*cg,2:(cy*cy+sd*sd)*cg+cy*sd*cp}
            errors={}
            for A,B in [(1,1),(2,1),(2,2)]:
                ref=np.sum(xw[:,None,:]*xx[:,None,:]**A*cm[B],axis=2)
                err=float(np.max(np.abs(M[(A,B)]-ref)/(1+np.abs(ref))));errors[str((A,B))]=err;maxmoment=max(maxmoment,err)
                assert err<=2e-10,(rho_value,A,B,err)
            cases.append(dict(alpha=alpha,rho=rho_value,probability_error=perr,moment_errors=errors))
    def derivatives(a,order):
        pdf=np.exp(-a*a/2)/np.sqrt(2*np.pi);g=cdf(a);A=[pdf+a*g,g];h0=np.ones_like(a);h1=a.copy()
        for r in range(2,order+1):
            if r==2:h=h0
            elif r==3:h=h1
            else:h=a*h1-(r-3)*h0;h0,h1=h1,h
            A.append((-1)**(r-2)*h*pdf)
        B=[(a*a+1)*g+a*pdf,2*A[0],2*g]
        for r in range(3,order+1):B.append(2*A[r-1])
        return {1:A,2:B}
    a=np.array([-2.,-.3,0.,1.2,2.]);pdf=np.exp(-a*a/2)/np.sqrt(2*np.pi);g=cdf(a);D=derivatives(a,84)
    def series(A,B,p,q,rho):
        ans=np.zeros_like(rho);rr=np.ones_like(rho)
        for r in range(81):
            ans+=rr*(D[A][r+p][:,None]*D[B][r+q][None,:]);rr*=rho/(r+1)
        return ans
    derivative_max=0.;pair_max=0.;fnpm=0.;bills=[]
    rng=np.random.Generator(np.random.PCG64DXSM(2026092055))
    for rv in [0.,.4,-.4,None]:
        rho=np.full((5,5),rv) if rv is not None else .4*np.sin(3*np.arange(5)[:,None]+np.arange(5)[None,:])
        geo=k._ec_geometry(a,rho,pdf,g)
        for A,B in [(1,1),(2,1),(2,2)]:
            for p,q in [(3,0),(0,3),(2,1),(1,2),(4,0),(0,4),(3,1),(1,3),(2,2)]:
                got=k._ec_derivative(A,B,p,q,geo);ref=series(A,B,p,q,rho)
                err=float(np.max(np.abs(got-ref)/(1+np.abs(ref))));derivative_max=max(derivative_max,err)
                assert err<=2e-10,(A,B,p,q,rv,err)
        T21=rng.standard_normal((5,5))*.01;T22=rng.standard_normal((5,5))*.002;T22=(T22+T22.T)/2
        T31=rng.standard_normal((5,5))*.002;t3=rng.standard_normal(5)*.01;t4=rng.standard_normal(5)*.002
        result=k._ec_pairs(a,rho,pdf,g,T21,T22,T31,t3,t4)
        for idx,(A,B) in enumerate([(1,1),(2,1),(2,2)]):
            def s(p,q):return series(A,B,p,q,rho)
            ref=s(0,0)-D[A][0][:,None]*D[B][0][None,:]
            ref+=(T21*s(2,1)+T21.T*s(1,2))/2+T22*s(2,2)/4+(T31*s(3,1)+T31.T*s(1,3))/6
            ref+=(t3[:,None]*(s(3,0)-D[A][3][:,None]*D[B][0][None,:])+t3[None,:]*(s(0,3)-D[A][0][:,None]*D[B][3][None,:]))/6
            ref+=(t4[:,None]*(s(4,0)-D[A][4][:,None]*D[B][0][None,:])+t4[None,:]*(s(0,4)-D[A][0][:,None]*D[B][4][None,:]))/24
            err=float(np.max(np.abs(result[idx]-ref)/(1+np.abs(ref))));pair_max=max(pair_max,err);assert err<=2e-10
        k.fnp=fnp;k._ec_cdf=flops.stats.norm.cdf
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            args=[fnp.asarray(x) for x in [a,rho,pdf,g,T21,T22,T31,t3,t4]];native=k._ec_pairs(*args)
        for x,y in zip(native,result):
            err=float(np.max(np.abs(np.asarray(x)-y)/(1+np.abs(y))));fnpm=max(fnpm,err);assert err<=2e-10
        bills.append(ctx.summary_dict()['flops_used']);k.fnp=np;k._ec_cdf=cdf
    rejected=0
    for bad in [np.nan,np.inf,.851,-.851]:
        try:k._ec_geometry(a,np.full((5,5),bad),pdf,g)
        except ValueError:rejected+=1
    assert rejected==4
    for rv in [0.,.4,-.4]:
        z=np.zeros((5,5));v=np.zeros(5);ans=k._ec_pairs(a,np.full((5,5),rv),pdf,g,z,z,z,v,v)
        assert max(float(np.max(np.abs(ans[i]-ans[i].T))) for i in [0,2])<=2e-10
    write(HERE/'unit.json',dict(passed=True,cases=cases,max_probability_error=maxprob,max_moment_error=maxmoment,max_derivative_error=derivative_max,
        max_pair_error=pair_max,max_fnp_error=fnpm,fnp_tiny_bills=bills,domain_rejections=rejected,gl32=pin(HERE/'gl32.json'),truth_accessed=False,
        wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(passed=True,max_probability_error=maxprob,max_moment_error=maxmoment,max_derivative_error=derivative_max,max_pair_error=pair_max,max_fnp_error=fnpm)),flush=True)
if __name__=='__main__':main()
