# Fixed symmetric covariance projection, only for the five selected strip outputs.
# Both operands are float32 and the arithmetic remains float32. The add owns its
# output; the scalar multiplication reuses that owned allocation.
_O18_MARGIN_SYM_PARENT = _o18_margin_two_strip
_O18_MARGIN_SYMMETRIZE = True

def _o18_margin_two_strip(A, B):
    result = _O18_MARGIN_SYM_PARENT(A, B)
    transposed = result.T
    symmetric = fnp.add(result, transposed)
    return fnp.multiply(symmetric, 0.5, out=symmetric)
