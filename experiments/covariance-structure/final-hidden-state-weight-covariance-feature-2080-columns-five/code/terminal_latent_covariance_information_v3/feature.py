"""Counted upper-triangle bilinear covariance-read features."""

def latent_covariance(fnp,C,H,alive,W,mu,v):
    n,h=int(H.shape[0]),int(H.shape[1])
    assert tuple(C.shape)==(n,n) and tuple(W.shape)==(n,n) and tuple(alive.shape)==(n,)
    assert tuple(mu.shape)==tuple(v.shape)==(n,)
    cv=fnp.diag(C)
    assert bool(fnp.all(fnp.isfinite(cv))) and bool(fnp.all(cv[alive]>0))
    assert bool(fnp.all(fnp.isfinite(H))) and bool(fnp.all(fnp.isfinite(W)))
    assert bool(fnp.all(fnp.isfinite(mu))) and bool(fnp.all(fnp.isfinite(v))) and bool(fnp.all(v>=1e-10))
    ii=fnp.asarray([i for i in range(h) for j in range(i,h)],dtype=fnp.int64)
    jj=fnp.asarray([j for i in range(h) for j in range(i,h)],dtype=fnp.int64)
    multiplicity=fnp.asarray([1 if i==j else 2 for i in range(h) for j in range(i,h)],dtype=fnp.float32)
    u=(fnp.sqrt(fnp.where(alive,cv,0)).astype(fnp.float64)[:,None]*H.astype(fnp.float64)).astype(fnp.float32)
    w=W.astype(fnp.float32)
    z=w.T@u
    value=(z[:,ii]*z[:,jj])*multiplicity[None,:]
    cold=fnp.nonzero(~alive)[0]
    if int(cold.shape[0]):
        wc=w[cold,:];hc=H[cold,:].astype(fnp.float64)
        tail=((cv[cold,None]*hc[:,ii])*hc[:,jj]).astype(fnp.float32)*multiplicity[None,:]
        value=value+(wc*wc).T@tail
    s=fnp.sqrt(v.astype(fnp.float64));a=mu.astype(fnp.float64)/s
    pdf=fnp.exp(-a*a/2)/fnp.sqrt(fnp.asarray(2*3.141592653589793,dtype=fnp.float64))
    out=((pdf/(2*s))[:,None]*value.astype(fnp.float64)).astype(fnp.float32)
    assert bool(fnp.all(fnp.isfinite(out)))
    return out
