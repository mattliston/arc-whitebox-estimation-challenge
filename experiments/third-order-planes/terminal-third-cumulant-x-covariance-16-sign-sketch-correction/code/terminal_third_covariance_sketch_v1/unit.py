"""Enumerated-sign source/trace certificate against explicit dense cubic tensor."""
import json,resource,itertools
from common import HERE,verify,write

def main():
    verify()
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from feature_fnp import feature
    from reference import reference
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    rng=np.random.default_rng(2060);results=[]
    for case in range(4):
        n=4;R=6
        p=rng.normal(size=(n,R)).astype(np.float32);lam=rng.normal(size=R).astype(np.float32)
        w=rng.normal(size=(n,n)).astype(np.float32)/2
        A=(.1*rng.normal(size=(n,n))).astype(np.float32)
        v=rng.normal(size=(n,n));B=(v@v.T/n).astype(np.float32)
        v=rng.normal(size=(n,n));C=(v@v.T/n).astype(np.float32)
        b=(.01*rng.normal(size=n)).astype(np.float32)
        g=np.linspace(.2,.8,n).astype(np.float32);d=np.linspace(.3,.1,n).astype(np.float32)
        if case==1:g[1]=0;d[1]=0
        if case==2:A[:]=0;b[:]=0
        if case==3:p[:]=0;lam[:]=0
        v=rng.normal(size=(n,n));Cpost=v@v.T/n
        wn=(rng.normal(size=(n,n))/2).astype(np.float32)
        omega=np.array(list(itertools.product([-1.,1.],repeat=n)),dtype=np.float32).T
        T=np.einsum('ia,ja,ka,a->ijk',p.astype(float),p.astype(float),p.astype(float),lam.astype(float),optimize=True)
        S=np.zeros((n,n,n));a64=A.astype(float)
        for i in range(n):
            for j in range(n):
                for k in range(n):S[i,j,k]=((a64[k,i] if i==j else 0)+(a64[j,i] if i==k else 0)+(a64[i,j] if j==k else 0))/3
        T+=np.einsum('ai,bj,ck,abc->ijk',w.astype(float),w.astype(float),w.astype(float),S,optimize=True)
        T+=np.einsum('i,jk->ijk',b.astype(float),B.astype(float))+np.einsum('j,ik->ijk',b.astype(float),B.astype(float))+np.einsum('k,ij->ijk',b.astype(float),B.astype(float))
        td=np.array([T[i,i,i] for i in range(n)],dtype=np.float32)
        pkt=dict(p=p,lam=lam,w=w,A=A,B=B,b=b,C=C,g=g,d=d,Cpost=Cpost,tdiag=td)
        nv=np.diag(wn.astype(float).T@Cpost@wn.astype(float)).copy()
        values=[];bills=[]
        for rep in range(2):
            with flops.BudgetContext(flop_budget=2**34,quiet=True) as ctx:
                fpkt={k:fnp.asarray(v) for k,v in pkt.items()}
                zn=fnp.asarray(g)[:,None]*fnp.asarray(wn)
                pushed=(fnp.asarray(p).T@zn).T
                delta,detail=feature(fpkt,fnp.asarray(wn),pushed,fnp.asarray(omega),fnp.asarray(nv))
            values.append(np.asarray(delta));bills.append(ctx.summary_dict()['flops_used'])
        assert values[0].tobytes()==values[1].tobytes() and bills[0]==bills[1]>0
        def ray(v):
            z=g.astype(float)*v
            return 12*np.dot(d.astype(float)*v*(C.astype(float)@z),np.einsum('iab,a,b->i',T,z,z))
        direct=np.array([ray(wn[:,j].astype(float)) for j in range(n)])
        tr=0.
        for i in range(n):
            for j in range(n):
                if i==j:tr+=ray(np.eye(n)[i])
                else:tr+=(ray(np.eye(n)[i]+np.eye(n)[j])+ray(np.eye(n)[i]-np.eye(n)[j])-2*ray(np.eye(n)[i])-2*ray(np.eye(n)[j]))/12
        den=np.trace(Cpost)**2+2*np.sum(Cpost*Cpost)
        expected=direct-(3*tr/den)*nv**2
        ref,rd=reference(pkt,wn,omega,nv)
        rel=float(np.linalg.norm(values[0]-expected)/max(np.linalg.norm(expected),1e-30))
        rr=float(np.linalg.norm(ref-expected)/max(np.linalg.norm(expected),1e-30))
        trace_rel=abs(rd['trace3']-3*tr)/max(abs(3*tr),1e-30)
        assert rel<=2e-5 and rr<=1e-10 and trace_rel<=1e-10,(case,rel,rr,trace_rel)
        results.append(dict(case=case,relative=rel,reference_relative=rr,trace_relative=trace_rel,flops=bills[0],repeat_exact=True))
    out=dict(passed=True,cases=results,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'unit.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
