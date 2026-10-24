# Fixed server-native preservation of the existing paid overflow predicate.
_O21_SLR_CHOLESKY_PARENT = _slr_virtual_rotation_envelope
def _slr_virtual_rotation_envelope(overlap, base):
    """Paid conservative overflow guard for the deliberately unformed L@F0.

    At depth d<=6, each input transform and output recombination grows maxabs
    by at most4**d. A leaf has r/2**d terms, so 2**(5*d) times r*max(L)*max(F0)
    bounds every scalar intermediate. Both maxima also individually must be
    <=2**80, so even a huge/tiny pair cannot overflow an input transform.
    The product gate leaves >=18 exponent bits for products/recombination.
    All scalar values remain typed one-element FP64 arrays on native and RPC.
    """
    _e2_H = _E2Hold(); _e2_append_inline = _e2_H.append; _e2_value_inline = None   # E2: temporaries of this call, released together (one hidden free trip)
    largest_map = (_e2_append_inline((_e2_value_inline := fnp.max(_e2_append_inline((_e2_value_inline := fnp.abs(overlap))) or _e2_value_inline, keepdims=True))) or _e2_value_inline).astype(fnp.float64)
    largest_base = (_e2_append_inline((_e2_value_inline := fnp.max(_e2_append_inline((_e2_value_inline := fnp.abs(base))) or _e2_value_inline, keepdims=True))) or _e2_value_inline).astype(fnp.float64)
    width = fnp.full((1, 1), 384.0, dtype=fnp.float64)
    limit = fnp.full((1, 1), 1208925819614629174706176.0, dtype=fnp.float64)
    bound = (_e2_append_inline((_e2_value_inline := (largest_map * largest_base))) or _e2_value_inline) * width
    allowed = (_e2_append_inline((_e2_value_inline := ((largest_map <= limit) & (largest_base <= limit)))) or _e2_value_inline) & (bound <= limit)
    # The (1,1) Boolean guard becomes a paid SPD check executed by the backend.
    # False -> [[0.0]] is singular and raises before this function returns.
    # True -> [[1.0]] succeeds. No array/scalar is materialized in Python.
    try:
        fnp.linalg.cholesky(allowed.astype(fnp.float64))
    except Exception as _slr_error:
        if (type(_slr_error).__name__ in ('LinAlgError', 'FlopscopeServerError')
                and str(_slr_error) == 'Matrix is not positive definite'):
            _slr_require(False, 'virtual rotation overflow envelope')
        raise
    del largest_map, largest_base, width, limit, bound, allowed
