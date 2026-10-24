# Exact public-dtype cast elisions; no parameter, kernel, or reduction-order changes.

def _lr_factor(A32):
    cn = fnp.sum(fnp.multiply(A32, A32, dtype=fnp.float64), axis=0)
    cols = fnp.sort(fnp.argsort(-cn, kind='stable')[:_LR_K])
    Q = fnp.linalg.qr(A32[:, cols])[0]
    for _ in range(_LR_Q if _LRN_POWER else 0):
        Q = fnp.linalg.qr(_tsr_mm(A32, fnp.linalg.qr(_tsr_mm(A32.T, Q))[0]))[0]
    B = _tsr_mm(Q.T, A32)
    if _LR_R is None:
        return Q, B
    U, S, Vt = fnp.linalg.svd(B, full_matrices=False)
    return Q @ U[:, :_LR_R], S[:_LR_R, None] * Vt[:_LR_R]


def _gn_dcov(l, mu, Cpre, K21, k4, aux):
    pr = _E2_LAYER.get('pair')
    if pr is None or pr.get('aux') is not aux or aux['pre_cov'] is not Cpre or aux.get('pre_k3') is None:
        return _G5_OLD_GN_DCOV(l, mu, Cpre, K21, k4, aux)
    _g5st = _e2_pair_state(int(mu.shape[0]))
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    th = _GN_KNOBS
    s = aux['sigma']; a = aux['alpha']
    v = aux['pre_v']   # G5: = diag(Cpre), the pair map's
    _e1gn = _e1gn_state(_g5st, int(mu.shape[0]))
    ss = pr['ss']   # G5: = outer(s, s)
    rho = _g5st['P16'][0]   # G5: = (Cpre / ss) with a zero diagonal, cast: the pair map's rho.astype(float32) / 1
    # E1: the GN plane read uses the same retained r=0,1 terms.
    k3 = aux['pre_k3']   # G5: = diag(K21)
    kink = (_e2_H(fnp.abs(a)) < 1.5).astype(fnp.float64)
    dK21 = _e2_H(_g5_rowmul(_ps32(_e2_H(_g5_schain(th['iso'], kink, k3)) / v), _ps_pre32(Cpre))) + _e2_H(_g5_rowmul(_ps32(_e2_H(th['k21n']) * _e2_H(1 - kink)), _ps_k2132(K21)))
    _e2_H(fnp.divide(dK21, _e2_H(_g5_outer(_ps32(v), _ps32(s))), out=_e1gn['D0']))
    t4 = _e2_H(fnp.broadcast_to(k4, (len(mu),))) / _e2_H(v ** 2)
    r31 = ((_k31_row_scale_gn(mu, k3, v) if _S19_GN_TORCH else _k31_row_scale(mu, k3, v))
           if '_k31_row_scale' in _e2_H(globals()) else None)   # PT-33 S19: torch GN remainder keeps the configure-time 0.85/0.5
    _e2_H(fnp.multiply(_e2_H(_e2_H(_g5_chain(t4, r31, _g5c(th['k31'], 'float64')) if r31 is not None else _e2_H(th['k31']) * t4).astype(fnp.float32)), _e1gn['rhoT'], out=_e1gn['D1T']))   # same weight_i * rho_ij, no einsum out-copy
    X = _e1gn_contract(_e1gn, rho, _e2_H)
    dcov = _e2_H(_e2_H(X + _e2_H(X.T)) * pr['ss32'])
    fnp.fill_diagonal(dcov, 0)
    return dcov


def _flms_covariance_diagonal(w, C):
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance diagonal shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    pushed = _mm(_e2_H(_ps32(C)), w)
    v = fnp.sum(_e2_H(fnp.multiply(w, pushed, dtype=fnp.float64)), axis=0)
    return v, v
