"""Inert FNP source fragment for the fixed four-direction newborn update.

The builder appends this fragment to the frozen participant.  It imports
nothing and performs no array arithmetic at module import.  ``fnp`` is the
participant's existing metered namespace.  All incoming physical quantities
are the current participant's native FP64 values; coefficients are frozen
FP32 values, and the newborn common bases and a/t/s are rounded to FP32.
"""

_BD_COEFFICIENTS = None


def _bd_coefficients():
    global _BD_COEFFICIENTS
    if _BD_COEFFICIENTS is None:
        _BD_COEFFICIENTS = fnp.asarray(_BD_COEFFICIENT_LITERAL,
                                      dtype=fnp.float32)
    return _BD_COEFFICIENTS


def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
    """Return (new full stored base32, stored-pair repeated-plane delta64).

    ``bp`` is the ORIGINAL off-diagonal g_i*Cpre_ij.  Orthogonalization is
    unmasked, columnwise, before the prescribed native alive-row mask.
    Below the 1e-20 norm-squared floor this is a regularized residual,
    not a claim of exact Euclidean orthogonality.
    No input is mutated.  The returned delta's diagonal is explicitly zero.
    This is added to old ``born``; it does not replace the old beta term.
    """
    n = int(bp.shape[0])
    c = coefficients.astype(fnp.float64)
    alpha = aux['alpha']
    sigma = aux['sigma']
    g = aux['gate']
    eta = alpha / sigma
    half_d = 0.5 * d
    negative_half_g = -0.5 * g
    negative_half_d = -half_d
    sigma3 = sigma * sigma * sigma
    skew = aux['pre_k3'] / sigma3
    gate_jet = skew * (alpha * alpha - 1) * aux['pdf'] / 6
    denominator = fnp.maximum(fnp.sum(bp * bp, axis=0), 1e-20)
    change = fnp.zeros((n, n), dtype=fnp.float64)
    for j in range(4):
        if j == 0:
            direction = half_d[:, None] * K21
        elif j == 1:
            direction = (negative_half_g[:, None] * K21.T) * eta[None, :]
        elif j == 2:
            direction = (negative_half_d[:, None] * (Cpre * Cpre)) * eta[None, :]
        else:
            direction = Cpre * gate_jet[:, None]
        fnp.fill_diagonal(direction, 0.0)
        projection = fnp.sum(bp * direction, axis=0) / denominator
        direction = direction - bp * projection[None, :]
        change = change + c[j] * direction
        del direction, projection
    fnp.fill_diagonal(change, 0.0)

    old32 = (bp * alive[:, None]).astype(fnp.float32)
    new32 = ((bp + change) * alive[:, None]).astype(fnp.float32)
    del change
    old = old32.astype(fnp.float64)
    new = new32.astype(fnp.float64)
    difference = new - old
    del old32, new

    # These are precisely the scalar values stored in lam/newborn, extended
    # to all columns; births selects the exact same padded selected set.
    a64 = a.astype(fnp.float32).astype(fnp.float64)
    t64 = (t * alive).astype(fnp.float32).astype(fnp.float64)
    s64 = (s * alive).astype(fnp.float32).astype(fnp.float64)
    weight = (a64 * (t64 - s64)) * births
    quadratic_weight = (a64 * (t64 * t64 - s64 * s64)) * births
    delta = ((2 * old * difference + difference * difference) * weight[None, :]
             + quadratic_weight[:, None] * difference.T)
    fnp.fill_diagonal(delta, 0.0)
    return new32, delta
