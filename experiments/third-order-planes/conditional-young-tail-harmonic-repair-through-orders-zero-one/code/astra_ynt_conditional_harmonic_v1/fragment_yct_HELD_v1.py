"""Fixed mean-conditioned degree-three harmonic repair of the existing YNT tail.
No roster, tensor coefficient, layer band, or inherited trace changes.
"""
_YCT_ON = True
_YCT_PRE_MEAN = [None]
_YCT_ACTIVE = [False]
_YCT_STATS = dict(calls=0,layers=[],tail_rows=0,retired=True,mean_clear=True)
_YCT_PARENT_TRACE = _ynt_trace
_YCT_PARENT_PREDICT = Estimator.predict

def _ynt_trace(w, p, previous, newborn, coefficient, tail):
    if not _YCT_ON:
        return _YCT_PARENT_TRACE(w,p,previous,newborn,coefficient,tail)
    assert _YCT_ACTIVE[0] and _YCT_PRE_MEAN[0] is not None
    hold = _E2Hold()
    q, selected, base, t, s = newborn
    q, births = int(q), int(selected.shape[0])
    n = int(w.shape[0])
    assert n == 1024 and tuple(_YCT_PRE_MEAN[0].shape) == (n,)
    assert tuple(p.shape) == (int(previous.shape[0]),q)
    assert tuple(coefficient.shape) == (q+2*births,)
    # Original first-chaos trace, with its original operations and dtypes.
    normB = _idc_get('colsq', base, 1, lambda: fnp.sum(base * base, axis=0))
    ct = coefficient[q:q+births] * (normB + t*t)
    cs = coefficient[q+births:] * (normB + s*s)
    tau = base @ (ct+cs)
    if q:
        P = p[:, :q]
        carried = P @ (coefficient[:q] * fnp.sum(P*P,axis=0))
        hold(fnp.put(tau,previous,tau[previous]+carried))
    hold(fnp.put(tau,selected,tau[selected]+t*ct+s*cs))
    wt = w[:,tail]
    r2 = fnp.sum(wt*wt,axis=0)
    original = ((3*r2/(n+2))*(wt.T@tau)).astype(fnp.float64)
    # T(u,u,.) from the same implicit carried/newborn CP tensor.
    mean = _YCT_PRE_MEAN[0].astype(fnp.float32)
    mean2 = fnp.sum(mean*mean)
    assert float(mean2)>0.0
    u = mean/fnp.sqrt(mean2)
    h = base.T@u
    us = u[selected]
    hp = h+t*us
    hm = h+s*us
    cp = coefficient[q:q+births]*(hp*hp)
    cm = coefficient[q+births:]*(hm*hm)
    tuu = base@(cp+cm)
    if q:
        h_old = P.T@u[previous]
        carried_uu = P@(coefficient[:q]*(h_old*h_old))
        hold(fnp.put(tuu,previous,tuu[previous]+carried_uu))
    hold(fnp.put(tuu,selected,tuu[selected]+t*cp+s*cm))
    # Orthogonal m=0 and m=1 components of the degree-three spherical harmonic.
    tu = fnp.sum(tau*u)
    a = fnp.sum(tuu*u)-3*tu/(n+2)
    b0 = tuu-tau/(n+2)
    b = b0-u*fnp.sum(b0*u)
    z = wt.T@u
    wb = wt.T@b
    h0 = a*((n+2)*(z*z*z)-3*z*r2)/(n-1)
    h1 = 3*((n+2)*(z*z)-r2)*wb/(n+1)
    result = original+(h0+h1).astype(fnp.float64)
    _YCT_STATS['calls'] += 1
    _YCT_STATS['layers'].append(int(_UVP_CUR[0]))
    _YCT_STATS['tail_rows'] += int(tail.shape[0])
    return result

def _yct_predict(self,*args,**kwargs):
    global _YCT_STATS
    assert not _YCT_ACTIVE[0] and _YCT_PRE_MEAN[0] is None
    _YCT_STATS=dict(calls=0,layers=[],tail_rows=0,retired=False,mean_clear=False)
    _YCT_ACTIVE[0]=True
    try:
        return _YCT_PARENT_PREDICT(self,*args,**kwargs)
    finally:
        _YCT_PRE_MEAN[0]=None
        _YCT_ACTIVE[0]=False
        _YCT_STATS['retired']=True
        _YCT_STATS['mean_clear']=True

Estimator.predict=_yct_predict
