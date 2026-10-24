def _r44_marginal(s, a, pdf, g, mg, k4, v):
    A, unused = derivatives(a, pdf, g, mg / s, 8)
    r = k4 / (v * v)
    q = r * r / 1152
    return (q * s * A[8], q * (s * s) * (2 * A[7]),
            q * (s * s * s) * (6 * A[6]),
            q * (s * s * s * s) * (24 * A[5]))


def _r44_pair(F, G, rho, t4):
    # The actual propagated t4 is one common radial scalar. Averaging only
    # removes its per-coordinate roundoff from k4/v^2 normalization.
    radial = fnp.sum(t4, keepdims=True, dtype=fnp.float64) / int(t4.shape[0])
    radial = radial.astype(fnp.float32)
    q = radial * radial / 1152
    rho2 = rho * rho
    powers = (fnp.ones_like(rho), rho, rho2)
    ans = fnp.zeros((int(F.shape[0]), int(rho.shape[0]), int(rho.shape[1])),
                    dtype=fnp.float32)
    # (x derivative count, correlation degree, coefficient). The remaining
    # y derivative count is 8-x; Mehler t raises both counts by t.
    for x, b, coefficient in ((0,0,1), (2,0,4), (4,0,6), (6,0,4), (8,0,1),
                               (1,1,8), (3,1,24), (5,1,24), (7,1,8),
                               (2,2,24), (4,2,48), (6,2,24)):
        for t in range(3 - b):
            if b == 0 and t == 0 and (x == 0 or x == 8):
                continue
            factor = coefficient / (2 if t == 2 else 1)
            term = F[:, x+t, :, None] * G[:, 8-x+t, None, :]
            ans = ans + factor * powers[b+t][None, :, :] * term
    ans = ans - 2 * (F[:, 4, :, None] * G[:, 4, None, :])
    return q * ans
