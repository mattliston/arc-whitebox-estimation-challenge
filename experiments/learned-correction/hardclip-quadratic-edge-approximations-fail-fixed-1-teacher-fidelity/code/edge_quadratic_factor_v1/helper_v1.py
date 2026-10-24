# Source-only quadratic edge prototype. Coefficients are not fitted here.
# q1 lives in t['output']; q2/constant are frozen literals if ever exported.
def _quadratic_edge_raw(left, right, t, k, pf, gf, hold):
    H = _E2Hold()
    I, J = _E2_LAYER['ecut_rows'], _E2_LAYER['ecut_cols']
    n, ni, nj = int(pf.shape[0]), int(I.shape[0]), int(J.shape[0])
    if 'quadratic_coefficients_v1' not in k:
        q2 = fnp.asarray(_EDGE_QUADRATIC_LITERAL, dtype=fnp.float32)
        e = fnp.stack((k['e0'], k['e1'], k['e2']))
        qe = e * q2
        ce = 2.0 * qe
        c1 = e @ t['output']
        c2 = qe @ e.T
        upper = fnp.asarray(((1., 2., 2.), (0., 1., 2.), (0., 0., 1.)), dtype=fnp.float32)
        c2 = c2 * upper
        k['quadratic_coefficients_v1'] = (q2, ce, c1, c2)
    q2, ce, c1, c2 = k['quadratic_coefficients_v1']
    Ic = H(I[:, None])
    planes = (H(k['R'][Ic, J]), H(k['S'][Ic, J]), H(k['S'][J, Ic]))
    A = H(H(left[I]) + t['bias'])
    B = H(right[J])
    qa = H(H(H(A * A) @ q2) + H(A @ t['output']))
    qb = H(H(H(B * B) @ q2) + H(B @ t['output']))
    hold.append(fnp.add(qa, _EDGE_QUADRATIC_CONSTANT, out=qa))
    result = H(H(A * q2) @ B.T)
    hold.append(fnp.multiply(result, 2.0, out=result))
    hold.append(fnp.add(result, H(qa[:, None]), out=result))
    hold.append(fnp.add(result, H(qb[None, :]), out=result))
    ca = H(H(A @ ce.T) + c1)
    cb = H(B @ ce.T)
    # Upper-triangular expansion includes the exact 2*off-diagonal coefficients.
    for p in range(3):
        term = H(H(ca[:, p, None]) + H(cb[None, :, p]))
        for q in range(p, 3):
            hold.append(fnp.add(term, H(planes[q] * c2[p, q]), out=term))
        hold.append(fnp.multiply(term, planes[p], out=term))
        hold.append(fnp.add(result, term, out=result))
    # Fixed physical function bound from the original teacher output L1 norm.
    # This is a mathematical activation clamp, never a runtime/resource governor.
    hold.append(fnp.clip(result, -_EDGE_QUADRATIC_OUTPUT_BOUND, _EDGE_QUADRATIC_OUTPUT_BOUND, out=result))
    block = fnp.einsum('ij,i,j->ij', result, H(pf[I]), H(gf[J]), optimize=False)
    raw = fnp.zeros((n, n), dtype=fnp.float32)
    H(fnp.put(raw, H(H(Ic * n) + J), block))
    return raw

