# Direct hidden covariance diagonal, fixed layers1..14 only.
# _PHYS_LAYER holds the PREVIOUS layer while covariance is pushed.
_O18_CDIAG_STATS = dict(calls=0, layers=[], retired=True)
def _o18_direct_covdiag(w, first, hold):
    previous = int(_PHYS_LAYER[0])
    if not 0 <= previous <= 13:
        return None
    # Keep the parent's float32 factors; accumulate their elementwise product
    # in float64. Compute before any later kernel can reuse scratch storage.
    value = fnp.sum(hold(w * first), axis=0, dtype=fnp.float64)
    _O18_CDIAG_STATS['calls'] += 1
    _O18_CDIAG_STATS['layers'].append(previous + 1)
    return value

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
    first = _mm(C32, w)
    _o18_covdiag = _o18_direct_covdiag(w, first, hold)
    sub = _astra_star_scaled_product(T2, P1, hold)
    cov = _sym_upper(wt, first)
    Cpre = cov.astype(fnp.float64)
    if _o18_covdiag is not None:
        fnp.fill_diagonal(Cpre, _o18_covdiag)
        fnp.fill_diagonal(cov, _o18_covdiag)
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
        first = _mm(hold(C32[fnp.ix_(live, live)]), wa)
        _o18_covdiag = _o18_direct_covdiag(wa, first, hold)
        sub = _astra_star_scaled_product(T2, P1, hold)
        cov = _psdm_sym_upper(hold(wa.T), first, m)
    elif _DCV_SYM == 'pad':
        wp = hold(w[roster])
        wa = hold(wp[:m])
        live = hold(roster[:m])
        first = _mm(hold(C32[fnp.ix_(live, live)]), wa)
        _o18_covdiag = _o18_direct_covdiag(wa, first, hold)
        sub = _astra_star_scaled_product(T2, P1, hold)
        cov = _sym_upper(hold(wp.T), hold(fnp.concatenate((first, hold(fnp.zeros((int(w.shape[0]) - m, int(w.shape[1])), dtype=fnp.float32))))))
        _PSDM_STATS['padded'] += 1
    else:
        wa = hold(w[roster])
        first = _mm(hold(C32[fnp.ix_(roster, roster)]), wa)
        _o18_covdiag = _o18_direct_covdiag(wa, first, hold)
        sub = _astra_star_scaled_product(T2, P1, hold)
        cov = _dcv_tiles(hold(wa.T), first, m) if _DCV_SYM == 'tiles' else _sym_upper(hold(wa.T), first)
    Cpre = cov.astype(fnp.float64)
    if _o18_covdiag is not None:
        fnp.fill_diagonal(Cpre, _o18_covdiag)
        fnp.fill_diagonal(cov, _o18_covdiag)
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
_O18_CDIAG_PARENT_PREDICT = Estimator.predict
def _o18_cdiag_predict(self, *args, **kwargs):
    _O18_CDIAG_STATS.update(calls=0, layers=[], retired=False)
    try:
        return _O18_CDIAG_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _O18_CDIAG_STATS['retired'] = True
Estimator.predict = _o18_cdiag_predict
