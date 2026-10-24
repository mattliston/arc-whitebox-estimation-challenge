"""Small explicit dense-tensor checks, including cold coordinates and cancellation."""
import json,resource
from common import HERE,verify,write,pin

def main():
    verify("unit_v2_sealed.json")
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    from feature_fnp import slices,probes
    from reference import reference
    assert all(x['num_threads']==1 for x in threadpool_info())
    rng=np.random.default_rng(2059);rows=[]
    for case in range(4):
        n=7;q=16;cohorts=[(0,6),(6,4)]
        w=(rng.normal(size=(n,n))/n**.5).astype(np.float32)
        pp=rng.normal(size=(n,10)).astype(np.float32)
        lam=np.concatenate([np.array([.2,1,3]),-np.array([.2,1,3]),[.7,2],[-.7,-2]]).astype(np.float32)
        if case==3:
            pp[:,3:6]=pp[:,:3]+np.float32(1e-4)*pp[:,3:6]
            pp[:,8:10]=pp[:,6:8]+np.float32(1e-4)*pp[:,8:10]
            lam*=1000
        base=rng.normal(size=(n,n));C=(base@base.T/n).astype(np.float64)
        A=(.1*rng.normal(size=(n,n))).astype(np.float32)
        b=(.1*rng.normal(size=n)).astype(np.float64)
        alive=np.ones(n,dtype=bool)
        if case in [1,3]:alive[[1,5]]=False
        if case==2:A[:]=0;b[:]=0
        sig=np.linspace(.7,1.3,n)
        G=rng.choice(np.array([-1,1],dtype=np.float32),size=(n,q))/np.float32(n**.5)
        H=rng.choice(np.array([-1,1],dtype=np.float32),size=(n,q))/np.float32(n**.5)
        args=[pp,lam,w,C,alive,A,b,sig]
        outputs=[];bills=[]
        for repeat in range(2):
            with flops.BudgetContext(flop_budget=2**34,quiet=True) as ctx:
                z,energy=slices(*[fnp.asarray(a) for a in args],cohorts,fnp.asarray(G),fnp.asarray(H))
            outputs.append((np.asarray(z).copy(),np.asarray(energy).copy()));bills.append(ctx.summary_dict()['flops_used'])
        assert bills[0]==bills[1]>0 and all(x.tobytes()==y.tobytes() for x,y in zip(*outputs))
        # Explicitly expand every cubic coefficient. No production helper used.
        pf=pp.astype(np.float64)
        T=np.einsum('ia,ja,ka,a->ijk',pf,pf,pf,lam.astype(np.float64),optimize=True)
        a=A.astype(np.float64);diag=np.diag(a).copy();a*=alive[:,None]*alive[None,:];np.fill_diagonal(a,diag)
        S=np.zeros((n,n,n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    S[i,j,k]=((a[k,i] if i==j else 0)+(a[j,i] if i==k else 0)+(a[i,j] if j==k else 0))/3
        wf=w.astype(np.float64)
        T+=np.einsum('ai,bj,ck,abc->ijk',wf,wf,wf,S,optimize=True)
        c=C.astype(np.float32);c=((c+c.T)*np.float32(.5)).astype(np.float64)*alive[:,None]*alive[None,:]
        B=wf.T@c@wf;bf=b.astype(np.float32).astype(np.float64)
        T+=np.einsum('i,jk->ijk',bf,B)+np.einsum('j,ik->ijk',bf,B)+np.einsum('k,ij->ijk',bf,B)
        sf=sig.astype(np.float32).astype(np.float64)
        dense=np.einsum('iab,aj,bj->ij',T,G.astype(np.float64)/sf[:,None],H.astype(np.float64)/sf[:,None],optimize=True)/sf[:,None]
        ref=reference(*args,cohorts,G,H)
        er=float(np.linalg.norm(outputs[0][0]-dense)/max(np.linalg.norm(dense),1e-30))
        rf=float(np.linalg.norm(ref-dense)/max(np.linalg.norm(dense),1e-30))
        ee=float(np.linalg.norm(outputs[0][1]-np.mean(dense*dense,axis=1))/max(np.linalg.norm(np.mean(dense*dense,axis=1)),1e-30))
        assert er<=1e-5 and ee<=2e-5 and rf<=1e-10,(case,er,ee,rf)
        rows.append(dict(case=case,z_relative=er,energy_relative=ee,reference_dense_relative=rf,flops=bills[0],repeat_exact=True))
    with flops.BudgetContext(flop_budget=2**30,quiet=True) as ctx:g,h=probes()
    assert g.shape==h.shape==(1024,16) and not np.array_equal(g,h)
    r=dict(passed=True,cases=rows,probe_setup_flops=ctx.summary_dict()['flops_used'],truth_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'unit.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
