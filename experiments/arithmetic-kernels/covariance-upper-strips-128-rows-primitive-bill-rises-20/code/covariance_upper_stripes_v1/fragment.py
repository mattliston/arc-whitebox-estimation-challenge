def _cov_upper_stripes(A, B, block=128):
    """Upper block rows plus transposed strict lower blocks; no input writes."""
    m, k = int(A.shape[0]), int(A.shape[1])
    if tuple(B.shape) != (k,m) or str(A.dtype) != 'float32' or str(B.dtype) != 'float32':
        raise ValueError('Upper covariance stripe shapes or dtype')
    if block <= 0 or m % block:
        raise ValueError('Upper covariance stripe partition')
    out = fnp.empty((m,m),dtype=fnp.float32)
    for lo in range(0,m,block):
        hi = lo+block
        hold = _E2Hold()
        aa,bb = hold(A[lo:hi,:]),hold(B[:,lo:])
        value = _mm(aa,bb)
        hold(fnp.copyto(hold(out[lo:hi,lo:]),value))
        if hi < m:
            hold(fnp.copyto(hold(out[hi:,lo:hi]),hold(hold(value[:,block:]).T)))
        del value,aa,bb,hold
    return out
