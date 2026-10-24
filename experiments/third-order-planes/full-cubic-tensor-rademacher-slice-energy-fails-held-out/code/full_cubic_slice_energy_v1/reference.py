"""Research-only FP64 contractions, independent of the pair factorization."""
def reference(pp,lam,w,C,alive,A,b,sigma,cohorts,G,H):
    import numpy as np
    w=np.asarray(w,dtype=np.float64)
    # These input casts explicitly define the producer's auxiliary tensor.
    sigma=np.asarray(sigma,dtype=np.float32).astype(np.float64)
    G=np.asarray(G,dtype=np.float64)/sigma[:,None]
    H=np.asarray(H,dtype=np.float64)/sigma[:,None]
    out=np.zeros_like(G)
    for start,width in cohorts:
        p=np.asarray(pp[:,start:start+width],dtype=np.float64)
        c=np.asarray(lam[start:start+width],dtype=np.float64)
        out+=p@(c[:,None]*(p.T@G)*(p.T@H))
    mask=np.asarray(alive,dtype=bool)
    a=np.asarray(A,dtype=np.float32).astype(np.float64)
    diag=np.diag(a).copy();a*=mask[:,None]*mask[None,:];np.fill_diagonal(a,diag)
    x=w@G;y=w@H
    out+=w.T@(a@(x*y)+x*(a.T@y)+y*(a.T@x))/3
    # Symmetrization is rounded in FP32 in the producer, then interpreted exactly.
    c=np.asarray(C,dtype=np.float32)
    c=((c+c.T)*np.float32(.5)).astype(np.float64)*mask[:,None]*mask[None,:]
    b=np.asarray(b,dtype=np.float32).astype(np.float64)
    bg=w.T@(c@x);bh=w.T@(c@y)
    out+=b[:,None]*np.sum(G*bh,axis=0)[None,:]+bg*(b@H)[None,:]+bh*(b@G)[None,:]
    return out/sigma[:,None]
