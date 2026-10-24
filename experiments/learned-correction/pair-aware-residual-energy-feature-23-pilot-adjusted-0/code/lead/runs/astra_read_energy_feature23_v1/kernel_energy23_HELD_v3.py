# Fixed pair-aware residual energy, vectorized by authoritative pair partner.
def _re23_energy(A, Q, scores, weighted, rows, cols, variance, blocks, hold):
    m, k = map(int, A.shape)
    t = int(weighted.shape[0])
    assert tuple(weighted.shape) == (t,k) and int(Q.shape[0]) == m
    assert sum(2*h for _,h in blocks) == k
    partner = list(range(k))
    for start, width in blocks:
        partner[start:start+width] = range(start+width,start+2*width)
        partner[start+width:start+2*width] = range(start,start+width)
    partner = hold(fnp.asarray(partner,dtype=fnp.int64))
    residual = hold(A-hold(Q@scores)) if int(Q.shape[1]) else A
    # F64 variance promotes the division; no redundant full residual cast.
    R = hold(residual / hold(hold(variance[rows])[:,None]))
    Rp = hold(R[:,partner])
    Rt, Rpt = hold(R.T), hold(Rp.T)
    norm = hold(fnp.einsum('ij,ij->i',Rt,Rt,dtype=fnp.float64,optimize=False))
    cross = hold(fnp.einsum('ij,ij->i',Rt,Rpt,dtype=fnp.float64,optimize=False))
    B = hold(weighted.astype(fnp.float64))
    Bp = hold(B[:,partner])
    result = hold(hold(hold(B*B)@norm) + hold(hold(B*Bp)@cross))
    values, ir, ic = fnp.intersect1d(rows,cols,assume_unique=True,return_indices=True)
    values, ir, ic = hold(values),hold(ir),hold(ic)
    if int(ir.shape[0]):
        diag = hold(hold(R[ir])*hold(B[ic]))
        diagp = hold(diag[:,partner])
        diagonal = hold(hold(fnp.einsum('ij,ij->i',diag,diag,dtype=fnp.float64,optimize=False)) + hold(fnp.einsum('ij,ij->i',diag,diagp,dtype=fnp.float64,optimize=False)))
        fnp.put(result,ic,hold(hold(result[ic])-diagonal))
    return hold(result/hold(variance[cols]))
