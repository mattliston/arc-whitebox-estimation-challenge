"""Direct common K31 coefficients; all float64 work is O(number of pairs).

No globals are changed. The caller supplies the live per-cohort strengths in
quartic and existing float32 sums/cross-products. Large polynomial arrays remain
float32. Formula is valid for either sign of the first member's lambda.
"""
def _o21_k31_common64(lam_plus, quartic, square_plus, square_minus, cross, n, hold):
    inv = hold(1.0 / hold(lam_plus.astype(fnp.float64)))
    e1, e2, e3, e4 = hold(fnp.unstack(hold(quartic.astype(fnp.float64)), axis=0))
    g = hold(hold(lam_plus > 0.0).astype(fnp.float64))
    t = hold(1.0 - g)
    sign = hold(hold(2.0 * g) - 1.0)
    u = hold(hold(square_plus.astype(fnp.float64)) * (1.0 / n))
    qm = hold(hold(square_minus.astype(fnp.float64)) * (1.0 / n))
    pq = hold(hold(cross.astype(fnp.float64)) * (1.0 / n))
    v = hold(u - pq)
    w = hold(hold(u + qm) - hold(2.0 * pq))
    te1 = hold(t * e1)
    te2 = hold(t * e2)
    A = hold(hold(hold(sign * e1) * inv) * (1.0 / 12.0))
    B = hold(hold(hold(hold(3.0 * te1) + e2) * inv) * (1.0 / 6.0))
    C = hold(hold(hold(hold(hold(-3.0 * te1) - hold(2.0 * te2)) + hold(sign * e3)) * inv) * 0.25)
    D = hold(hold(hold(hold(te1 + te2) + hold(t * e3) + e4) * inv) * (1.0 / 3.0))
    E = hold(hold(hold(-6.0 * A) * v) - hold(B * w))
    F = hold(hold(hold(hold(-3.0 * A) * u) - hold(hold(2.0 * B) * v)) - hold(C * w))
    G = hold(hold(hold(-B * u) - hold(hold(2.0 * C) * v)) - hold(hold(3.0 * D) * w))
    return tuple(hold(x.astype(fnp.float32)) for x in (A, B, C, D, E, F, G))
