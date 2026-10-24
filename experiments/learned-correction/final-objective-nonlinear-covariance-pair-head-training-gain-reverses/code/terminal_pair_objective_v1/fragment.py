"""Final covariance correction; all inference arithmetic is metered."""

def _tp_covariance(C,x,K,s,H):
    global _TP_PARAMETERS
    if _TP_PARAMETERS is None:
        _TP_PARAMETERS=_unpack_static_fp32(_TP_LITERAL)
    p=_TP_PARAMETERS
    F=fnp.concatenate((x,H),axis=1)
    F=fnp.clip(F/p['feature_scale'],-8,8)
    R=(C/fnp.outer(s,s)).astype(fnp.float32)/p['rho_scale']
    S=(K/fnp.outer(s*s,s)).astype(fnp.float32)/p['third_scale']
    left=F@p['left.weight'].T;right=F@p['right.weight'].T
    edge=p['edge.weight'];raw=fnp.empty(C.shape,dtype=fnp.float32)
    for lo in range(0,len(F),64):
        hi=min(lo+64,len(F))
        h=left[lo:hi,None,:]+right[None,:,:]+p['bias']
        h=h+R[lo:hi,:,None]*edge[:,0]
        h=h+S[lo:hi,:,None]*edge[:,1]
        h=h+S.T[lo:hi,:,None]*edge[:,2]
        value=fnp.tanh(h)@p['output.weight'].T+p['output.bias']
        fnp.copyto(raw[lo:hi,:],value[:,:,0])
    D=(raw+raw.T).astype(fnp.float64)*(.001*fnp.outer(s,s))
    fnp.fill_diagonal(D,0.)
    return C+D
