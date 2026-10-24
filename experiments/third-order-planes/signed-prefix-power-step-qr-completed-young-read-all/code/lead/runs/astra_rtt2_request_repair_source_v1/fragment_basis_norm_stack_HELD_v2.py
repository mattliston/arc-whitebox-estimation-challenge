# Exact RTT2 basis reuse: no in-place writes; q is x before the first projection.
def _rtt_basis(vc, mu, hold):
    m = int(vc.shape[0]);assert tuple(mu.shape)==(m,)
    basis=[]
    for x in (vc,mu):
        x=x if str(x.dtype)=='float64' else hold(x.astype(fnp.float64));q=x
        original=float(hold(fnp.linalg.norm(x)))
        for _ in range(2):
            for b in basis:q=hold(q-hold(b*hold(fnp.sum(hold(b*q)))))
        norm=original if not basis else float(hold(fnp.linalg.norm(q)))
        assert original<float('inf') and norm<float('inf')
        if norm>2.220446049250313e-16*max(m,2)*original:
            basis.append(hold(q/norm))
    if not basis:return hold(fnp.zeros((m,0),dtype=fnp.float32))
    return hold(fnp.stack(basis,axis=1,dtype=fnp.float32))
