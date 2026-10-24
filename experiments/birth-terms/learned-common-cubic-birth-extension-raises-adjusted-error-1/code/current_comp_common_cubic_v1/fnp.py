"""Inert common-cubic participant fragment, using only the owned fnp API.

The reviewed source constructor supplies the immutable _CC_LITERAL.  This
fragment performs no imports, allocation, or numerical work at module load.
Every array operation below executes within the participant's predict call.
"""

_CC_PARAMETERS = None


def _cc_parameters():
    global _CC_PARAMETERS
    if _CC_PARAMETERS is None:
        _CC_PARAMETERS = fnp.asarray(_CC_LITERAL, dtype=fnp.float32)
    return _CC_PARAMETERS


def _cc_response(H, Q):
    """The original NODE normalization, with one new FP32 scalar readout."""
    n = int(H.shape[0])
    if H.shape != (n, 64) or Q.shape != (65,):
        raise ValueError('Typed common-cubic hidden/head shapes')
    if str(H.dtype) != 'float32' or str(Q.dtype) != 'float32':
        raise TypeError('Common-cubic hidden state and Q must be FP32')
    centered = H - fnp.mean(H, axis=0)
    scale = fnp.sqrt(fnp.mean(centered * centered, axis=0) + 1e-6)
    standardized = fnp.clip(centered / scale, -8, 8)
    z = standardized @ Q[:64] + Q[64]
    response = fnp.tanh(z)
    if str(response.dtype) != 'float32' or not bool(fnp.all(fnp.isfinite(response))):
        raise ValueError('Finite FP32 common-cubic response')
    return response


def _cc_weights(a32, t32, s32, response):
    """Evaluate both stored weights, including their separate FP32 rounding."""
    n = int(a32.shape[0])
    if any(x.shape != (n,) or str(x.dtype) != 'float32'
           for x in (a32, t32, s32, response)):
        raise ValueError('Typed stored common-cubic weight inputs')
    denominator = fnp.maximum(fnp.maximum(fnp.abs(t32), fnp.abs(s32)), 1e-20)
    q = ((.25 * a32) * response) / denominator
    plus = a32 + ((-q) * s32)
    minus = (-a32) + (q * t32)
    if any(str(x.dtype) != 'float32' for x in (denominator, q, plus, minus)):
        raise TypeError('The complete common-cubic slot arithmetic must stay FP32')
    if not bool(fnp.all(fnp.isfinite(plus))) or not bool(fnp.all(fnp.isfinite(minus))):
        raise ValueError('Nonfinite common-cubic stored weights')
    return plus, minus


def _cc_plane(B32, a32, t32, s32, plus, minus, selected, births):
    """FP64 increment of the ACTUAL stored pair, including every diagonal.

    B32 is the parent's full zero-diagonal, alive-masked BD/NODE base.
    Selected columns keep the original stable padded order.  All d0..d3
    are retained; ideal A1 cancellation is never substituted for d1.
    """
    n = int(B32.shape[0])
    if B32.shape != (n, n) or str(B32.dtype) != 'float32':
        raise ValueError('Original stored common-cubic base')
    if births.shape != (n,) or len(selected.shape) != 1:
        raise ValueError('Original birth selection shapes')
    if any(x.shape != (n,) or str(x.dtype) != 'float32'
           for x in (a32, t32, s32, plus, minus)):
        raise ValueError('Original and new stored slot shapes/dtypes')
    if not bool(fnp.all(fnp.diag(B32) == 0)):
        raise ValueError('The original stored common base must have zero diagonal')
    a = a32.astype(fnp.float64)
    t = t32.astype(fnp.float64)
    s = s32.astype(fnp.float64)
    delta_plus = plus.astype(fnp.float64) - a
    delta_minus = minus.astype(fnp.float64) + a
    t2 = t * t
    s2 = s * s
    d0 = (delta_plus + delta_minus) * births
    d1 = (delta_plus * t + delta_minus * s) * births
    d2 = (delta_plus * t2 + delta_minus * s2) * births
    d3 = (delta_plus * (t2 * t) + delta_minus * (s2 * s)) * births
    B = B32.astype(fnp.float64)
    B2 = B * B
    Bs = B[:, selected]
    Bs2 = B2[:, selected]
    d0s = d0[selected]
    weighted_square = Bs2 * d0s[None, :]
    # Intentionally direct FP64 matmul: never the parent's FP32 _mm.
    increment = weighted_square @ Bs.T
    del weighted_square
    increment = increment + B2 * d1[None, :]
    increment = increment + d2[:, None] * B.T
    diagonal = fnp.sum((Bs2 * Bs) * d0s[None, :], axis=1) + d3
    fnp.fill_diagonal(increment, diagonal)
    if str(increment.dtype) != 'float64' or not bool(fnp.all(fnp.isfinite(increment))):
        raise ValueError('Finite FP64 stored common-cubic plane increment')
    return increment


def _cc_birth(B32, a, t, s, alive, births, selected, response):
    """Return both full stored weight vectors and their additive plane repair."""
    n = int(B32.shape[0])
    if any(x.shape != (n,) for x in (a, t, s, alive, births, response)):
        raise ValueError('Typed common-cubic birth inputs')
    if any(str(x.dtype) != 'float64' for x in (a, t, s)):
        raise TypeError('Original native birth parameters must be FP64')
    # These are exactly the parent's stored per-coordinate conventions;
    # the original selected t/s expressions remain unchanged in newborn.
    a32 = a.astype(fnp.float32)
    t32 = (t * alive).astype(fnp.float32)
    s32 = (s * alive).astype(fnp.float32)
    plus, minus = _cc_weights(a32, t32, s32, response)
    increment = _cc_plane(B32, a32, t32, s32, plus, minus, selected, births)
    return plus, minus, increment
