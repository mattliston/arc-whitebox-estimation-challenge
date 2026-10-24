"""Paid-array implementation of a supplied leading cumulant recurrence.

No host imports, caches or numerical work at module entry. All factors and
operations are owned by the caller's charged context. Native qualification
and integration are separate requirements.
"""
K3 = [[0, 2, 0], [2, 0, 6], [0, 6, 0], [0, 8, 0], [6, 0, 36], [0, 36, 0]]
K4 = [[0, 0, 6, 8, 0, 36], [0, 8, 0, 0, 48, 0],
      [6, 0, 36, 48, 0, 324], [8, 0, 48, 48, 0, 360],
      [0, 48, 0, 0, 360, 0], [36, 0, 324, 360, 0, 3240]]
MULTIPLICITY = [1, 2, 2, 1, 2, 1]


def source(fnp, mm, a, b, c, beta, gamma):
    n, r = int(a.shape[0]), int(a.shape[1])
    if (tuple(b.shape) != (n, r) or tuple(c.shape) != (n, r)
            or tuple(beta.shape) != (r,) or tuple(gamma.shape) != (r,)):
        raise ValueError('incompatible source factor shapes')
    b1 = b * beta[None, :]
    c1 = c * gamma[None, :]
    z = fnp.stack((a, b1, c1), axis=2).reshape(n, 3*r)
    u = fnp.zeros((n, 6*r), dtype=fnp.float32)
    covariance = 2*mm(b1, b1.T) + 6*mm(c1, c1.T)
    return (z, u), covariance


def causal_source(fnp, mm, wt, h, variance, theta, phi, basis):
    n = int(h.shape[0])
    if (tuple(wt.shape) != (n, n) or tuple(h.shape) != (n, 64)
            or tuple(variance.shape) != (n,) or tuple(theta.shape) != (16,)
            or tuple(phi.shape) != (16,) or tuple(basis.shape) != (64, 48)):
        raise ValueError('causal supplier requires64 hidden channels and16 sectors')
    projected = mm(wt, h)
    selected = mm(projected, basis)
    # Incoming covariance, before inserting this cohort. The caller owns its
    # positive-variance domain checks. No floor is added to the draft formula.
    sigma = fnp.sqrt(variance).astype(fnp.float32)
    factors = sigma[:, None] * fnp.tanh(selected) / 4
    beta = .05 * fnp.tanh(theta)
    gamma = .025 * fnp.tanh(phi)
    return source(fnp, mm, factors[:, :16], factors[:, 16:32],
                  factors[:, 32:48], beta, gamma)


def append(fnp, old, new):
    if old is None:
        return new
    return (fnp.concatenate((old[0], new[0]), axis=1),
            fnp.concatenate((old[1], new[1]), axis=1))


def affine(fnp, mm, state, left):
    z, u = state
    n, columns = int(left.shape[0]), int(z.shape[1])
    pushed = mm(left, fnp.concatenate((z, u), axis=1))
    return pushed[:, :columns], pushed[:, columns:]


def _products(fnp, z):
    n, columns = int(z.shape[0]), int(z.shape[1])
    x = z.reshape(n, columns//3, 3)
    a, b, c = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    return fnp.stack((a*a, a*b, a*c, b*b, b*c, c*c), axis=2)


def gate(fnp, state, g, d):
    z, u = state
    n, columns = int(z.shape[0]), int(z.shape[1])
    products = _products(fnp, z).reshape(n, 2*columns)
    return g[:, None]*z, g[:, None]*u + d[:, None]*products


def _symmetric_product(fnp, packed, z):
    return fnp.stack((packed[:, :, 0]*z[:, :, 0] + packed[:, :, 1]*z[:, :, 1] + packed[:, :, 2]*z[:, :, 2],
                      packed[:, :, 1]*z[:, :, 0] + packed[:, :, 3]*z[:, :, 1] + packed[:, :, 4]*z[:, :, 2],
                      packed[:, :, 2]*z[:, :, 0] + packed[:, :, 4]*z[:, :, 1] + packed[:, :, 5]*z[:, :, 2]), axis=2)


def _read_factors(fnp, state, want31=True):
    z, u = state
    n, r = int(z.shape[0]), int(z.shape[1])//3
    x = z.reshape(n, r, 3)
    mult = fnp.asarray(MULTIPLICITY, dtype=fnp.float32)
    q = _products(fnp, z) * mult
    up = u.reshape(n, r, 6) * mult
    m = fnp.asarray(K4, dtype=fnp.float32)
    qm = (q.reshape(n*r, 6) @ m).reshape(n, r, 6)
    um = (up.reshape(n*r, 6) @ m).reshape(n, r, 6)
    direct = (q.reshape(n*r, 6) @ fnp.asarray(K3, dtype=fnp.float32)).reshape(n, r, 3)
    left21 = direct + _symmetric_product(fnp, um, x)
    left31 = _symmetric_product(fnp, qm, x).reshape(n, 3*r) if want31 else None
    return (z, q.reshape(n, 6*r), up.reshape(n, 6*r),
            qm.reshape(n, 6*r), left21.reshape(n, 3*r), left31)


def planes(fnp, mm, state):
    z, q, up, qm, left21, left31 = _read_factors(fnp, state)
    k21 = mm(left21, z.T) + .5*mm(qm, up.T)
    k31 = mm(left31, z.T)
    k22 = mm(qm, q.T)
    k22 = .5*(k22+k22.T)
    return k21, k31, k22


def marginals(fnp, state):
    z, q, up, qm, left21, unused_left31 = _read_factors(fnp, state, want31=False)
    k3 = fnp.sum(left21*z, axis=1) + .5*fnp.sum(qm*up, axis=1)
    k4 = fnp.sum(qm*q, axis=1)
    return k3, k4


def normalized31_offset(fnp, c0, k40, c1, q31, q4):
    """Off-diagonal correction when the host automatically builds t4*rho."""
    v0, v1 = fnp.diag(c0), fnp.diag(c1)
    s1 = fnp.sqrt(v1)
    rho1 = c1 / (s1[:, None]*s1[None, :])
    den = (v1*s1)[:, None]*s1[None, :]
    offset = (q31/den - (q4/(v1*v1))[:, None]*rho1
              + (k40[:, None]/den)*(c0/v0[:, None] - c1/v1[:, None]))
    fnp.fill_diagonal(offset, 0)
    return offset


def radial_residual(fnp, radial_total, q22_post, covariance_response):
    denominator = fnp.trace(covariance_response)**2 + 2*fnp.sum(covariance_response*covariance_response)
    return radial_total - 3*fnp.sum(q22_post)/denominator


def third_residual(fnp, target, inherited, trace_post, cp_post, explicit_post):
    innovation = fnp.diag(target)-fnp.diag(inherited)-fnp.diag(trace_post)-fnp.diag(explicit_post)
    rest = target-cp_post-trace_post-explicit_post
    return innovation, rest
