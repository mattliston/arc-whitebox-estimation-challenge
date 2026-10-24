_E2_BD_STATE = {}   # E2: persistent (4,n,n) float64 direction stack, its views and the flat indices of its four diagonals


def _e2_bd_state(n):
    st = _E2_BD_STATE.get(n)
    if st is None:
        D = fnp.empty((4, n, n), dtype=fnp.float64)
        st = dict(D=D, Dv=fnp.unstack(D), D12=D[1:3], diag=fnp.arange(4 * n) % n * (n + 1) + fnp.arange(4 * n) // n * (n * n),
                  old32=fnp.empty((n, n), dtype=fnp.float32))
        _E2_BD_STATE[n] = st
    return st


def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
    """Return (new full stored base32, stored-pair repeated-plane delta64).

    ``bp`` is the ORIGINAL off-diagonal g_i*Cpre_ij.  Orthogonalization is
    unmasked, columnwise, before the prescribed native alive-row mask.
    Below the 1e-20 norm-squared floor this is a regularized residual,
    not a claim of exact Euclidean orthogonality.
    No input is mutated.  The returned delta's diagonal is explicitly zero.
    This is added to old ``born``; it does not replace the old beta term.
    E2: the four directions live in one persistent (4,n,n) stack; projection, orthogonalization and the ordered
    coefficient sum run once over the stack (same per-element float64 operations and order as the j-loop).
    """
    n = int(bp.shape[0])
    st = _e2_bd_state(n)
    D = st['D']; D0, D1, D2, D3 = st['Dv']
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
    fnp.multiply(half_d[:, None], K21, out=D0)
    fnp.multiply(negative_half_g[:, None], K21.T, out=D1)
    fnp.multiply(negative_half_d[:, None], _E2_LAYER['CC'] if _E2_LAYER.get('Cpre') is Cpre and 'CC' in _E2_LAYER else Cpre * Cpre, out=D2)
    fnp.multiply(st['D12'], eta, out=st['D12'])
    fnp.multiply(Cpre, gate_jet[:, None], out=D3)
    fnp.put(D, st['diag'], 0.0)
    projection = fnp.sum(bp * D, axis=1) / denominator
    fnp.subtract(D, bp * projection[:, None, :], out=D)
    fnp.multiply(D, c[:, None, None] if c.ndim == 1 else c[:, None, :], out=D)
    change = fnp.sum(D, axis=0)
    fnp.fill_diagonal(change, 0.0)

    alive_column = alive[:, None]
    old32 = st['old32']
    fnp.multiply(bp, alive_column, out=old32)
    new32 = fnp.empty((n, n), dtype=fnp.float32)
    fnp.multiply(bp + change, alive_column, out=new32)
    del change
    old = old32.astype(fnp.float64)
    new = new32.astype(fnp.float64)
    difference = new - old
    del new

    # These are precisely the scalar values stored in lam/newborn, extended
    # to all columns; births selects the exact same padded selected set.
    a64, t64, s64 = fnp.unstack(fnp.stack((a, t * alive, s * alive), dtype=fnp.float32).astype(fnp.float64))
    weight = (a64 * (t64 - s64)) * births
    quadratic_weight = (a64 * (t64 * t64 - s64 * s64)) * births
    delta = ((2 * old * difference + difference * difference) * weight
             + quadratic_weight[:, None] * difference.T)
    fnp.fill_diagonal(delta, 0.0)
    return new32, delta
