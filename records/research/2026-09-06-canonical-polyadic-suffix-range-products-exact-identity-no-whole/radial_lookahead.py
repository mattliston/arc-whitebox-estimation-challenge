"""Future-query-aware CP weight compression, never using target means.

Match contractions of the all-distinct tensor along known future linearized
network directions. Current repeated entries are restored separately. This
is a bounded approximation test, not a cost-certified estimator.
"""
import inspect
import numpy as np
import k3_radial_numpy as base

def forecast(W):
    depth,n,_=W.shape;m=np.zeros(n);C=np.eye(n);gates=[];sensitivity=[]
    for w0 in W:
        w=w0.astype(np.float64);mu=w.T@m;Cpre=w.T@C@w
        s,a,pdf,g,mg,r2,*_=base.gaussian_raw(mu,np.diag(Cpre))
        d=pdf/s;C=np.outer(g,g)*Cpre+.5*np.outer(d,d)*Cpre*Cpre
        np.fill_diagonal(C,r2-mg*mg);m=mg
        gates.append(g);sensitivity.append(np.abs(a*pdf/(s*s)))
    suffix=[None]*depth;suffix[-2]=W[-1].T.astype(np.float64)
    for l in range(depth-3,-1,-1):suffix[l]=suffix[l+1]@(gates[l+1][:,None]*W[l+1].T)
    return suffix,sensitivity

def distinct_queries(J,p,q,lam):
    j=J.astype(np.float32);j2=j*j
    P=j@p;Q=j@q
    A=j2@(p*p);B=j2@(p*q);D=(j2*j)@(p*p*q)
    return ((3*(P*P-A)*Q-6*B*P+6*D)*lam).astype(np.float64)

def make_predict(fit=True,cap=2048,stages=(3,7,11)):
    log=[]
    def compress(p,q,lam,l,W,suffix,sensitivity):
        pn=np.sum(p*p,axis=0);qn=np.sum(q*q,axis=0);pq=np.sum(p*q,axis=0)
        energy=lam*lam*3*(pn*pn*qn+2*pn*pq*pq)
        keep=np.argsort(-energy,kind="stable")[:cap]
        weights=np.ones(cap)
        if fit:
            now=np.argsort(-sensitivity[l+1])[:256]
            end=np.argsort(-sensitivity[-1])[:256]
            J=np.concatenate([W[l+1].T[now],suffix[l][end]],axis=0)
            F=distinct_queries(J,p,q,lam)
            norm=np.sqrt(np.mean(F*F,axis=1));F/=np.maximum(norm[:,None],1e-18)
            A=F[:,keep];target=np.sum(F,axis=1);rhs=target-np.sum(A,axis=1)
            gram=A@A.T;ridge=1e-3*np.mean(np.diag(gram))
            weights+=A.T@np.linalg.solve(gram+ridge*np.eye(len(gram)),rhs)
            before=float(np.linalg.norm(rhs)/max(np.linalg.norm(target),1e-30))
            after=float(np.linalg.norm(A@weights-target)/max(np.linalg.norm(target),1e-30))
        else:before=after=None
        log.append(dict(layer=l,original=len(lam),kept=cap,relative_query_before=before,relative_query_after=after,
                        weight_min=float(weights.min()),weight_max=float(weights.max())))
        return p[:,keep],q[:,keep],(lam[keep]*weights).astype(np.float32)
    src=inspect.getsource(base.predict)
    src=src.replace('    depth,n,_ = W.shape','    depth,n,_ = W.shape\n    suffix,sensitivity=forecast(W)')
    old='''        if cap and len(lam)>cap:'''
    new='''        if l in stages and len(lam)>target_cap:
            p,q,lam=compress(p,q,lam,l,W,suffix,sensitivity)
            cp_post=cp_plane(p,q,lam)
        if cap and len(lam)>cap:'''
    assert old in src;src=src.replace(old,new)
    env=dict(vars(base));env.update(forecast=forecast,compress=compress,stages=stages,target_cap=cap)
    exec(src,env)
    return env["predict"],log
