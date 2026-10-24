"""Appended inside the native participant; uses only metered FNP operations."""

def _fm_predict(xp,F,R,S,t):
    F=xp.clip(F/t['feature_scale'],-8,8)
    R=R/t['rho_scale'];S=S/t['third_scale']
    left=F@t['left.weight'].T;right=F@t['right.weight'].T
    edge=t['edge.weight'];raw=xp.empty((len(F),len(F)),dtype=xp.float32)
    for lo in range(0,len(F),64):
        hi=min(lo+64,len(F))
        h=left[lo:hi,None,:]+right[None,:,:]+t['bias']
        h=h+R[lo:hi,:,None]*edge[:,0]
        h=h+S[lo:hi,:,None]*edge[:,1]
        h=h+S.T[lo:hi,:,None]*edge[:,2]
        value=xp.tanh(h)@t['output.weight'].T+t['output.bias']
        raw[lo:hi,:]=value[:,:,0]
    D=(raw+raw.T)*(.5*t['off_scale'])
    diagonal=xp.tanh(F@t['diagonal.0.weight'].T+t['diagonal.0.bias'])@t['diagonal.2.weight'].T+t['diagonal.2.bias']
    D[xp.arange(len(F)),xp.arange(len(F))]=diagonal[:,0]*t['diag_scale']
    return D

def _fm_moments(l,mu,C,K,radial,W,H):
    global _FM_PARAMETERS
    if _FM_PARAMETERS is None:
        _FM_PARAMETERS=_unpack_static_fp32(_FM_LITERAL)
    F,R,S,s=_fm_features(fnp,flops.stats.norm.cdf,l,mu,C,K,radial,W,H)
    D=_fm_predict(fnp,F,R,S,_FM_PARAMETERS).astype(fnp.float64)
    a=fnp.diag(D);v=fnp.diag(C);ss=fnp.outer(s,s)
    rawR=C/ss;Rs=(C+C.T)/(2*ss);fnp.fill_diagonal(Rs,1.)
    q31=(D+a[:,None]*Rs)/2
    q22=(a[:,None]+a[None,:]+4*Rs*D)/6
    delta4=a*s**4
    row_scale=_k31_row_scale(mu,fnp.diag(K),v)
    shift=delta4/(v*v)
    if row_scale is not None:
        shift=shift*row_scale
    # The parent's default31 changes when delta4 is added. Remove just that
    # overlap from the explicit extra31 argument; do not double-count it.
    extra31=(q31-shift[:,None]*rawR)*((v*s)[:,None]*s[None,:])
    delta22=q22*fnp.outer(v,v)
    fnp.fill_diagonal(extra31,0.);fnp.fill_diagonal(delta22,0.)
    return delta4,extra31,delta22
