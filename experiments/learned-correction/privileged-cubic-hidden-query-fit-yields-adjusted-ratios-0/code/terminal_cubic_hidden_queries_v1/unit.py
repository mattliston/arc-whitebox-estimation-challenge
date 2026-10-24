import json,resource
from common import *
from helper import cubic_hidden_features
def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    from capture import observed
    assert all(x['num_threads']==1 for x in threadpool_info());verify();_,proof=observed()
    rng=np.random.default_rng(26092171);rows=[]
    for n,R in ((3,0),(5,7),(8,9)):
        P=rng.normal(size=(n,R)).astype(np.float32)*.2;lam=rng.normal(size=R).astype(np.float32)
        if R:P[:,1]=P[:,0];lam[1]=-lam[0]
        A=rng.normal(size=(n,n)).astype(np.float32)*.1;C=rng.normal(size=(n,n))*.1;W=rng.normal(size=(n,n)).astype(np.float32)
        b=rng.normal(size=n)*.02;H=rng.normal(size=(n,4)).astype(np.float32);H[:,2]=0;H[:,3]=1
        sigma=np.exp(rng.normal(size=n)*.2);alpha=rng.normal(size=n);pdf=np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)
        args=dict(P=P,lam=lam,A=A,C=C,W=W,b=b,H=H,sigma=sigma,alpha=alpha,pdf=pdf)
        before={k:v.tobytes() for k,v in args.items()};actuals=[];bills=[]
        for rep in range(2):
            with flops.BudgetContext(flop_budget=2**30,quiet=True) as ctx:
                aa={k:fnp.asarray(v) for k,v in args.items()};out=cubic_hidden_features(fnp,**aa)
            actuals.append([np.asarray(v).copy() for v in out]);bills.append(ctx.summary_dict()['flops_used'])
        assert bills[0]==bills[1] and all(np.array_equal(a,b_) for a,b_ in zip(*actuals))
        assert before=={k:v.tobytes() for k,v in args.items()}
        V=W.astype(np.float64).T;U=V@A.astype(np.float64);cc=C.astype(np.float32).astype(np.float64);B=V@((cc+cc.T)*.5)@V.T
        pp=P.astype(np.float64);ll=lam.astype(np.float64)
        T=np.einsum('ir,jr,kr,r->ijk',pp,pp,pp,ll,optimize=True)
        T+=(np.einsum('ia,ja,ka->ijk',U,V,V)+np.einsum('ia,ja,ka->ijk',V,U,V)+np.einsum('ia,ja,ka->ijk',V,V,U))/3
        T+=np.einsum('i,jk->ijk',b,B)+np.einsum('j,ik->ijk',b,B)+np.einsum('k,ij->ijk',b,B)
        X=np.column_stack([np.ones(n),H.astype(np.float64)]);q=X/np.maximum(np.linalg.norm(X,axis=0),1e-12);r=q/sigma[:,None]
        query=np.einsum('ijk,jc,kc->ic',T,r,r,optimize=True);feature=(-alpha*pdf/6)[:,None]*query
        err=float(np.linalg.norm(actuals[0][0]-feature)/max(np.linalg.norm(feature),1e-30));qerr=float(np.linalg.norm(actuals[0][1]-query)/max(np.linalg.norm(query),1e-30));assert max(err,qerr)<=2e-10,(err,qerr)
        assert np.array_equal(actuals[0][0][:,3],np.zeros(n))
        perm=rng.permutation(n);pa={k:(v[perm] if k in ('P','b','H','sigma','alpha','pdf') else v[:,perm] if k=='W' else v) for k,v in args.items()}
        permuted=cubic_hidden_features(np,**pa)[0];perr=float(np.linalg.norm(permuted-feature[perm])/max(np.linalg.norm(feature),1e-30));assert perr<=2e-10,perr
        rows.append(dict(n=n,cp_columns=R,feature_relative=err,query_relative=qerr,permutation_relative=perr,repeat_bytes_and_bill_exact=True,flops=bills[0]))
    write(HERE/'unit.json',dict(passed=True,rows=rows,source_proof=proof,targets_opened=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));print(json.dumps(dict(passed=True,rows=rows)),flush=True)
if __name__=='__main__':main()
