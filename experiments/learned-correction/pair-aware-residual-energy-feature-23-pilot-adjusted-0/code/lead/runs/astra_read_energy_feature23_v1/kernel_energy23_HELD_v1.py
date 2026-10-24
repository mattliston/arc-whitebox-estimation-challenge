# Fixed pair-aware residual energy; no cross estimator or fitted scale.
def _re23_energy(A, Q, scores, weighted, rows, cols, variance, blocks, hold):
    m, k = map(int, A.shape)
    t = int(weighted.shape[0])
    assert tuple(weighted.shape) == (t, k) and tuple(Q.shape)[0] == m
    assert sum(2*h for _,h in blocks) == k
    residual = hold(A - hold(Q @ scores)) if int(Q.shape[1]) else A
    r64 = hold(residual.astype(fnp.float64))
    invscale = hold(variance[rows])
    R = hold(r64 / hold(invscale[:, None]))
    values, ir, ic = fnp.intersect1d(rows, cols, assume_unique=True, return_indices=True)
    values, ir, ic = hold(values), hold(ir), hold(ic)
    result = hold(fnp.zeros(t, dtype=fnp.float64))
    diagonal = hold(fnp.zeros(int(ir.shape[0]), dtype=fnp.float64))
    for start, width in blocks:
        rp, rn = hold(R[:,start:start+width]), hold(R[:,start+width:start+2*width])
        common = hold(.5 * hold(rp + rn))
        difference = hold(rp - rn)
        ct, dt = hold(common.T), hold(difference.T)
        cc = hold(fnp.einsum('ij,ij->i', ct, ct, dtype=fnp.float64, optimize=False))
        dd = hold(fnp.einsum('ij,ij->i', dt, dt, dtype=fnp.float64, optimize=False))
        cd = hold(fnp.einsum('ij,ij->i', ct, dt, dtype=fnp.float64, optimize=False))
        # weighted already contains the signed lambda, exactly as used by RTT.
        bp = hold(hold(weighted[:,start:start+width]).astype(fnp.float64))
        bn = hold(hold(weighted[:,start+width:start+2*width]).astype(fnp.float64))
        bc = hold(.5 * hold(bp - bn))
        bd = hold(bp + bn)
        energy = hold(hold(hold(bd*bd) @ cc) + hold(hold(bc*bc) @ dd))
        energy = hold(energy + hold(2.0 * hold(hold(bd*bc) @ cd)))
        result = hold(result + energy)
        # The production K21 diagonal is restored exactly; omit each pair's
        # diagonal square before summing pairs (not the square of their sum).
        if int(ir.shape[0]):
            diag = hold(hold(hold(common[ir])*hold(bd[ic])) + hold(hold(difference[ir])*hold(bc[ic])))
            diagonal = hold(diagonal + hold(fnp.einsum('ij,ij->i',diag,diag,dtype=fnp.float64,optimize=False)))
    if int(ic.shape[0]):
        result[ic] = hold(hold(result[ic]) - diagonal)
    return hold(result / hold(variance[cols]))
