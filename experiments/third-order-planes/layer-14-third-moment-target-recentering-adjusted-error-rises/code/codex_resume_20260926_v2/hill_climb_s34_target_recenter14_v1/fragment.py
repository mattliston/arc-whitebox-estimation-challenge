# R3[i,j] = E[X_i^2 X_j] = central_K21[i,j] + F(mean,cov)[i,j].
def _tr14_raw_offset(m, C):
    h = _E2Hold()
    diagonal = h(fnp.diag(C))
    first = h(fnp.outer(diagonal, m))
    second = h(fnp.einsum('i,ij->ij', h(2 * m), C, optimize=False))
    third = h(fnp.outer(h(m * m), m))
    return h(first + second) + third


def _target_recenter14(target, old_mean, old_cov, new_mean, new_cov):
    h = _E2Hold()
    assert all(str(x.dtype) == 'float64' for x in (target, old_mean, old_cov, new_mean, new_cov))
    old_offset = h(_tr14_raw_offset(old_mean, old_cov))
    new_offset = h(_tr14_raw_offset(new_mean, new_cov))
    return target + h(old_offset - new_offset)
