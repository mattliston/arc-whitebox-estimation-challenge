"""Pure paid FNP feature definition; NumPy may stand in only for reference checks."""
def cubic_hidden_features(fnp,P,lam,A,C,W,b,H,sigma,alpha,pdf):
    P=P.astype(fnp.float64);lam=lam.astype(fnp.float64)
    A=A.astype(fnp.float64);C=C.astype(fnp.float32).astype(fnp.float64)
    C=(C+C.T)*.5;W=W.astype(fnp.float64);b=b.astype(fnp.float64)
    X=fnp.concatenate([fnp.ones((int(H.shape[0]),1),dtype=fnp.float64),H.astype(fnp.float64)],axis=1)
    norms=fnp.sqrt(fnp.sum(X*X,axis=0));q=X/fnp.maximum(norms,1e-12)[None,:]
    rays=q/sigma.astype(fnp.float64)[:,None]
    projection=P.T@rays
    cp=P@(lam[:,None]*projection*projection)
    z=W@rays
    star=(W.T@(A@(z*z)+2*z*(A.T@z)))/3
    Br=W.T@(C@z)
    trace=b[:,None]*fnp.sum(rays*Br,axis=0)[None,:]+2*Br*fnp.sum(b[:,None]*rays,axis=0)[None,:]
    query=cp+star+trace
    feature=(-alpha.astype(fnp.float64)*pdf.astype(fnp.float64)/6)[:,None]*query
    return feature,query,q,norms
