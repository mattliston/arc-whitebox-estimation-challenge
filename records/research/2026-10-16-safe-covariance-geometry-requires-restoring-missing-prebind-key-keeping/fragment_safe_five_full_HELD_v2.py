"""Prospective SAFE request margin: first five eligible covariance sandwiches.

Same real-arithmetic covariance and the existing upper-to-lower block mirror.
The full D6 product changes floating-point evaluation order and costs more;
fresh SAFE production gates and ten-network physics parity remain mandatory.
"""
assert all(globals().get(name) is value for name, value in _ASTRA_DENSE_CONTEXT)
_GSAFE5_PARENT_TWO_STRIP = _gsafe_two_strip
_GSAFE5_PARENT_PREDICT = Estimator.predict
_GSAFE5_LIMIT = 5
_GSAFE5_ACTIVE = False
_GSAFE5_STATS = dict(full=0, two_strip=0, retired=True)
_GSAFE5_FULL_KEYS = tuple(sorted({key for inner in _GSAFE_INNERS for key in
    ((0, 6, 1024, inner), (1, 6, inner, 1024), (2, 6, 1024, 1024))}))
assert all(key in _PB_SINGLE for key in _GSAFE5_FULL_KEYS)


def _gsafe5_full(A, B):
    keep = []
    _TR_LAST[0] = None
    try:
        out = fnp.empty((1024, 1024), dtype=fnp.float32)
        _ipt_product_ab(A, B, out, 6)
        # Retain precisely the two-strip policy's block semantics: both
        # diagonal 512-square blocks are full products; only G10 is mirrored.
        upper, lower = out[:512, 512:], out[512:, :512]
        upper_t = upper.T
        keep.extend((upper, lower, upper_t))
        keep.append(fnp.copyto(lower, upper_t))
        return out
    finally:
        _TR_LAST[0] = None
        keep.clear()


def _gsafe_two_strip(A, B):
    inner = int(A.shape[1])
    if (_GSAFE5_ACTIVE and _GSAFE5_STATS['full'] < _GSAFE5_LIMIT
            and _gsafe_fits(A, B, inner)
            and _ipt_fits(6, 1024, inner, 1024)):
        _GSAFE5_STATS['full'] += 1
        return _gsafe5_full(A, B)
    if _GSAFE5_ACTIVE:
        _GSAFE5_STATS['two_strip'] += 1
    return _GSAFE5_PARENT_TWO_STRIP(A, B)


def _gsafe5_predict(self, *args, **kwargs):
    global _GSAFE5_ACTIVE
    assert not _GSAFE5_ACTIVE
    _GSAFE5_ACTIVE = True
    _GSAFE5_STATS.clear()
    _GSAFE5_STATS.update(full=0, two_strip=0, retired=False)
    try:
        return _GSAFE5_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _GSAFE5_ACTIVE = False
        _GSAFE5_STATS['retired'] = True


Estimator.predict = _gsafe5_predict
