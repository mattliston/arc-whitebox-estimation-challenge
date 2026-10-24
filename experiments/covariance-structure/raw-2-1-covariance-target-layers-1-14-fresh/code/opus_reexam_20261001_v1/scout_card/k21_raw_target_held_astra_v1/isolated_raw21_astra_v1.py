"""HELD hypothesis kernels. No imports, globals, model or call-site edits.

K_ij denotes E[(X_i-m_i)^2 (X_j-m_j)]. The intended invariant is the
implied raw R21 under replacement of the first TWO supplied moments;
it is not a translation of a probability distribution. n is small in tests.
"""


def raw21_actual_covariance(xp, target, host, dm, covariance_before, covariance_after):
    """Stable raw-R21 identity using the actually supplied covariance pair.

    Does not infer an ideal kappa from rounded covariance. Inputs unchanged.
    Returning a new target also invalidates source's identity-keyed ps cache.
    """
    mean = host + dm
    dc = covariance_after - covariance_before
    diag0 = xp.diag(covariance_before)
    diagdc = xp.diag(dc)
    shift_square = (2 * host) * dm + dm * dm
    a = -(diag0 + host * host)
    b = -(diagdc + shift_square)
    correction = (-2 * dm)[:, None] * covariance_before
    correction = correction - (2 * mean)[:, None] * dc
    correction = correction + a[:, None] * dm[None, :]
    correction = correction + b[:, None] * mean[None, :]
    return target + correction


def raw21_ideal_kappa(xp, target, host, dm, covariance_before, kappa):
    """Cheaper symbolic specialization, NOT selected for a rounded native port.

    Requires C1=C0-kappa*(host outer dm+dm outer host+dm outer dm) exactly.
    """
    hd = host * dm
    dd = dm * dm
    a = ((2 * kappa - 1) * (host * host)
         + (6 * kappa - 2) * hd + (3 * kappa - 1) * dd
         - xp.diag(covariance_before))
    b = (4 * kappa - 2) * hd + (3 * kappa - 1) * dd
    correction = (-2 * dm)[:, None] * covariance_before
    correction = correction + a[:, None] * dm[None, :]
    correction = correction + b[:, None] * host[None, :]
    return target + correction
