# Algebraically exact first covariance product: diagonal plus off-diagonal.
# Fixed hidden layers 1..14, same selected rows and same parent kernels.
_O18_DS_STATS = dict(calls=0, layers=[], retired=True)
def _o18_diag_split_first(C, w):
    previous = int(_PHYS_LAYER[0])
    if not 0 <= previous <= 13:
        return _mm(C, w)
    if C.dtype != 'float32' or w.dtype != 'float32':
        raise TypeError('diagonal split requires original float32 factors')
    diagonal = fnp.diagonal(C).copy()
    off = C.copy()
    fnp.fill_diagonal(off, 0.0)
    answer = _mm(off, w)
    correction = diagonal[:, None] * w
    # Owned result: no subsequent kernel can mutate the returned plane.
    answer = fnp.add(answer, correction)
    _O18_DS_STATS['calls'] += 1
    _O18_DS_STATS['layers'].append(previous + 1)
    return answer

def _flms_cov_star(w, C, A, retain32=False):
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if (w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    hold = _E2Hold()
    wt = w.T
    C32 = _ps32(C)
    A32 = A if (A.dtype) == 'float32' else A.astype(fnp.float32)
    wt2 = wt * wt
    P1, T2 = _ss_factors(wt, wt2, A32, hold)
    first = _o18_diag_split_first(C32, w)
    sub = _astra_star_scaled_product(T2, P1, hold)
    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)
    diagonal = fnp.sum(hold(P1 * T2), axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    _SS_STATS['planes'] += 1
    if retain32:
        fnp.fill_diagonal(sub, diagonal)
        return Cpre, result, cov, sub
    return Cpre, result


def _dc_cov_star(w, C, A, gate, floor, retain32=False):
    roster = _dcv_roster(gate, floor)
    if roster is None:
        return _flms_cov_star(w, C, A, retain32)
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if (w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    hold = _E2Hold()
    wt = w.T
    C32 = _ps32(C)
    A32 = A if (A.dtype) == 'float32' else A.astype(fnp.float32)
    wt2 = wt * wt
    P1, T2 = _ss_factors(wt, wt2, A32, hold)
    m = _DCV_STATS['m']
    if _DCV_SYM == 'pad' and int(w.shape[0]) == 1024 and int(w.shape[1]) == 1024 and _psdm_fits(m):
        live = hold(roster[:m])
        wa = hold(_xsg_rows(13378, w, live))
        first = _o18_diag_split_first(hold(C32[fnp.ix_(live, live)]), wa)
        sub = _astra_star_scaled_product(T2, P1, hold)
        cov = _psdm_sym_upper(hold(wa.T), first, m)
    elif _DCV_SYM == 'pad':
        wp = hold(w[roster])
        wa = hold(wp[:m])
        live = hold(roster[:m])
        first = _o18_diag_split_first(hold(C32[fnp.ix_(live, live)]), wa)
        sub = _astra_star_scaled_product(T2, P1, hold)
        cov = _sym_upper(hold(wp.T), hold(fnp.concatenate((first, hold(fnp.zeros((int(w.shape[0]) - m, int(w.shape[1])), dtype=fnp.float32))))))
        _PSDM_STATS['padded'] += 1
    else:
        wa = hold(w[roster])
        first = _o18_diag_split_first(hold(C32[fnp.ix_(roster, roster)]), wa)
        sub = _astra_star_scaled_product(T2, P1, hold)
        cov = _dcv_tiles(hold(wa.T), first, m) if _DCV_SYM == 'tiles' else _sym_upper(hold(wa.T), first)
    Cpre = cov.astype(fnp.float64)
    diagonal = fnp.sum(hold(P1 * T2), axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    _SS_STATS['planes'] += 1
    if retain32:
        fnp.fill_diagonal(sub, diagonal)
        return Cpre, result, cov, sub
    return Cpre, result


_ASTRA_DENSE_CONTEXT = tuple((name, globals()[name])
                            for name, _value in _ASTRA_DENSE_CONTEXT)
_O18_DS_PARENT_PREDICT = Estimator.predict
def _o18_ds_predict(self, *args, **kwargs):
    _O18_DS_STATS.update(calls=0, layers=[], retired=False)
    try:
        return _O18_DS_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _O18_DS_STATS['retired'] = True
Estimator.predict = _o18_ds_predict
