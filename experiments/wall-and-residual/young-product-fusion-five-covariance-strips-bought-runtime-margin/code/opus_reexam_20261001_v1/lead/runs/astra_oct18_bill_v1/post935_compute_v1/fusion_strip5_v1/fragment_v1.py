# Fixed runtime-cushion policy: first five covariance products with inner896/960.
# Literal historical two-strip kernel; all directions and symmetric outputs retained.
# The parent649-key prebind order/classes are untouched; new aliases bind lazily.
_O18_MARGIN_PARENT_PSDM = _psdm_sym_upper
_O18_MARGIN_PARENT_PREDICT = Estimator.predict
_O18_MARGIN_INNERS = (896, 960)
_O18_MARGIN_STATE = dict(left=5, calls=0, inners=[], retired=True)

def _o18_margin_fits(A, B, inner):
    return (_AB_ON and _WINOGRAD and _SCRATCH is not None
            and inner in _O18_MARGIN_INNERS
            and len(A.shape) == len(B.shape) == 2
            and tuple(A.shape) == (1024, inner)
            and tuple(B.shape) == (inner, 1024)
            and str(A.dtype) == str(B.dtype) == 'float32'
            and _ipt_fits(6, 512, inner, 1024)
            and _ipt_fits(6, 512, inner, 512))

def _o18_margin_two_strip(A, B):
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


def _psdm_sym_upper(A, B, m):
    if _O18_MARGIN_STATE['left'] > 0 and _o18_margin_fits(A, B, m):
        _O18_MARGIN_STATE['left'] -= 1
        _O18_MARGIN_STATE['calls'] += 1
        _O18_MARGIN_STATE['inners'].append(int(m))
        result = _o18_margin_two_strip(A, B)
        _PSDM_STATS['inner'] += 1
        return result
    return _O18_MARGIN_PARENT_PSDM(A, B, m)

_ASTRA_DENSE_CONTEXT = tuple((name, globals()[name])
                            for name, _value in _ASTRA_DENSE_CONTEXT)

def _o18_margin_predict(self, *args, **kwargs):
    _O18_MARGIN_STATE.update(left=5, calls=0, inners=[], retired=False)
    try:
        return _O18_MARGIN_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _O18_MARGIN_STATE['retired'] = True

Estimator.predict = _o18_margin_predict
