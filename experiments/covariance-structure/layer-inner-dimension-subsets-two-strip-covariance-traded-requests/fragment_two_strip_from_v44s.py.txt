"""Distinct SAFE policy: two D6 covariance strips, same statistical model.

Only admitted 1024-output covariance sandwiches change geometry.  This is
rounding-level algebra, not a byte-identity claim or a score improvement.
"""
assert _ASTRA_DENSE_ON and _ASTRA_FHM_ON and _GOP2_ON and _GTR_ENABLED
assert _ASTRA_IOP_ON is False and _LVFS_LIMIT == 10 ** 15
assert all(globals().get(name) is value for name, value in _ASTRA_DENSE_CONTEXT)
_GSAFE_PARENT_SYM = _sym_upper
_GSAFE_PARENT_PSDM = _psdm_sym_upper
_GSAFE_INNERS = (896, 960, 1024)
_GSAFE_NEW_PB_KEYS = ((0, 6, 512, 896), (2, 6, 512, 1024), (0, 6, 512, 960))
assert len(_PB_SINGLE) == 649
assert not any(key in _PB_SINGLE for key in _GSAFE_NEW_PB_KEYS)
# Preserve all old relative ordering and every first128 eager/FNV position.
# The three new keys are deliberately bound before the old long-tail keys.
_PB_SINGLE = tuple(_PB_SINGLE[:128]) + _GSAFE_NEW_PB_KEYS + tuple(_PB_SINGLE[128:])


def _gsafe_fits(A, B, inner):
    return (_AB_ON and _WINOGRAD and _SCRATCH is not None
            and inner in _GSAFE_INNERS
            and len(A.shape) == len(B.shape) == 2
            and tuple(A.shape) == (1024, inner)
            and tuple(B.shape) == (inner, 1024)
            and str(A.dtype) == str(B.dtype) == 'float32'
            and _ipt_fits(6, 512, inner, 1024)
            and _ipt_fits(6, 512, inner, 512))


def _gsafe_two_strip(A, B):
    # Upper blocks: [G00 G01] and G11; preserve G10=G01.T as in parent.
    # Each product completes before the next arena use. No input/cache writes.
    keep = []
    _TR_LAST[0] = None
    try:
        out = fnp.empty((1024, 1024), dtype=fnp.float32)
        a0, a1, b1 = A[:512, :], A[512:, :], B[:, 512:]
        top = fnp.empty((512, 1024), dtype=fnp.float32)
        bottom = fnp.empty((512, 512), dtype=fnp.float32)
        keep.extend((a0, a1, b1, top, bottom))
        _ipt_product_ab(a0, B, top, 6)
        upper, lower, cross = out[:512, :], out[512:, :512], top[:, 512:]
        cross_t = cross.T
        keep.extend((upper, lower, cross, cross_t))
        keep.append(fnp.copyto(upper, top))
        keep.append(fnp.copyto(lower, cross_t))
        _ipt_product_ab(a1, b1, bottom, 6)
        last = out[512:, 512:]
        keep.extend((last, fnp.copyto(last, bottom)))
        return out
    finally:
        _TR_LAST[0] = None
        keep.clear()


def _sym_upper(A, B):
    if _gsafe_fits(A, B, 1024):
        return _gsafe_two_strip(A, B)
    return _GSAFE_PARENT_SYM(A, B)


def _psdm_sym_upper(A, B, m):
    if _gsafe_fits(A, B, m):
        result = _gsafe_two_strip(A, B)
        _PSDM_STATS['inner'] += 1
        return result
    return _GSAFE_PARENT_PSDM(A, B, m)


# Only these two dynamic covariance functions change. Retain every other
# captured dense/FHM/GOP function generation and all outer prediction wrappers.
_ASTRA_DENSE_CONTEXT = tuple((name, globals()[name])
                            for name, _value in _ASTRA_DENSE_CONTEXT)
