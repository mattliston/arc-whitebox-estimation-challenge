# One fixed conditional-moment completion of GOP2's log-energy input.
# The Gaussian score surrogate is an approximation, not a Jensen guarantee.
_GCL_STATS=dict(layers=[],retired=True)

def _gcl_completion(T,centered,V,values,coefficients,projected,Q,mass):
    n=int(T.shape[1]);assert T.shape==(3,n,n)
    # Top-two source scores and the remaining source-feature matrix.
    X=centered.astype(fnp.float64);VV=V.astype(fnp.float64)
    Z=coefficients.astype(fnp.float64);QQ=Q.astype(fnp.float64)
    tail=X-fnp.matmul(Z,VV.transpose(0,2,1))
    energy=fnp.sum(tail*tail,axis=1)
    eigen=fnp.maximum(values[:,-2:],1e-30)
    z=Z/fnp.sqrt(eigen[:,None,:])
    # Exact independent-weight second and third cumulants for W^(2p).
    wvar=2.0/n
    kap2=fnp.asarray([2*wvar**2,96*wvar**4,10170*wvar**6],dtype=fnp.float64)
    kap3=fnp.asarray([8*wvar**3,9504*wvar**6,33998400*wvar**9],dtype=fnp.float64)
    u=projected.astype(fnp.float64)/fnp.sqrt(kap2[:,None,None]*eigen[:,None,:])
    zz=fnp.stack((z[:,:,0]*z[:,:,0],2*z[:,:,0]*z[:,:,1],z[:,:,1]*z[:,:,1]),axis=2)
    H=fnp.matmul(zz.transpose(0,2,1),tail)
    uu=fnp.stack((u[:,:,0]*u[:,:,0]-1,u[:,:,0]*u[:,:,1],u[:,:,1]*u[:,:,1]-1),axis=2)
    mean=(kap3/(2*kap2))[:,None,None]*fnp.matmul(uu,H)
    # The actual output projection removes the retained score subspace.
    mean=mean-fnp.matmul(QQ,fnp.matmul(QQ.transpose(0,2,1),mean))
    # Realized row variance, then diag[(I-QQ')diag(nu)(I-QQ')].
    second=fnp.sum(T*T,axis=2).astype(fnp.float64)
    mm=mass.astype(fnp.float64)
    nu=fnp.maximum(second-mm*mm/n,0.0)/(n-1)
    leverage=fnp.sum(QQ*QQ,axis=2)
    G=fnp.matmul(QQ.transpose(0,2,1),nu[:,:,None]*QQ)
    vn=nu*(1-2*leverage)+fnp.sum(fnp.matmul(QQ,G)*QQ,axis=2)
    variance=fnp.maximum(vn,0.0)[:,:,None]*energy[:,None,:]
    assert bool(fnp.all(fnp.isfinite(mean))) and bool(fnp.all(fnp.isfinite(variance)))
    return mean,variance

def _gcl_log_adjust(E,mean,variance,out):
    # E is exactly the parent's already-clamped energy; parent log is in out.
    den=1+E.astype(fnp.float64)
    correction=mean/den-(variance+mean*mean)/(2*den*den)
    answer=fnp.maximum(out.astype(fnp.float64)+correction,0.0)
    assert bool(fnp.all(fnp.isfinite(answer)))
    fnp.copyto(out,answer.astype(fnp.float32))

def _gcl_contract(T, X):
    # Fixed two-dimensional output range; exact full mean component.
    mu = fnp.mean(X, axis=1)
    centered = fnp.subtract(X, mu[:, None, :])
    wide = centered.astype(fnp.float64)
    gram = fnp.matmul(wide.transpose(0, 2, 1), wide)
    values, vectors = fnp.linalg.eigh(gram)
    V = vectors[:, :, -2:].astype(fnp.float32)
    coefficients = fnp.matmul(centered, V)
    projected = fnp.matmul(T, coefficients)
    Q, _ = fnp.linalg.qr(projected.astype(fnp.float64), mode='reduced')
    Q = Q.astype(fnp.float32)
    back = fnp.matmul(Q.transpose(0, 2, 1), T)
    fitted = fnp.matmul(back, centered)
    residual = fnp.matmul(Q, fitted)
    mass = fnp.sum(T, axis=2)
    answer = fnp.add(residual, fnp.multiply(mass[:, :, None], mu[:, None, :]))
    mean, variance = _gcl_completion(T,centered,V,values,coefficients,projected,Q,mass)
    return answer, mean, variance

def _gcl_gnn_step(l,x,sigma,W,state,P):
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    _e2_hold = []   # E2: out= alias handles, released together at exit (one hidden free trip)
    if l<1:
        return (_e2_H(fnp.zeros_like(sigma)), _e2_H(state * 0))
    k = _e2_gnn_consts(P, int(_e2_H(sigma.shape[0])))
    c = _e2_gnn_layer(k, l)
    Hx = k['Hx']
    _e2_hold.append(fnp.clip(((x if str(x.dtype) == 'float32' else x.astype(fnp.float32)) - c['centers']) / c['scales'], -8, 8, out=Hx))
    if l == 15:
        _e2_hold.append(fnp.multiply(Hx, k['mask'], out=Hx))
    M = _e2_messages(k, W, state)
    _e2_hold.append(fnp.power(k['SRC3'], 2, out=k['SQ']))
    E, _gcl_mean, _gcl_var = _gcl_contract(_e2_H(_e2_H(_e2_wpow(W, need6=True))['T135']), k['SQ'])
    _e2_hold.append(fnp.multiply(k['Hg'], M, out=k['HM03']))
    _e2_hold.append(fnp.maximum(E, 0, out=E))
    _e2_hold.append(fnp.log1p(E, out=k['HM36']))
    _gcl_log_adjust(E,_gcl_mean,_gcl_var,k['HM36'])
    _GCL_STATS['layers'].append(int(l))
    _e2_hold.append(fnp.clip(k['Hm'], -8, 8, out=k['Hm']))
    h = silu(_e2_H(_e2_H(k['H'] @ k['enc0_wT']) + _e2_H(_hc_bias(l, P['enc0_b']))))   # opus-engineer hc: enc0_b + z u' when the layer's read made the new column
    h = silu(_e2_H(_e2_H(h @ k['enc2_wT']) + P['enc2_b']))
    state = _e2_H(h @ k['enc4_wT']) + P['enc4_b']
    delta = _e2_H(_e2_H(_e2_H(sigma.astype(fnp.float32)) * c['rms']) * _e2_H(_e2_H(state @ c['head']) + c['head_b'])) * c['gain']
    return delta,state

_GCL_PARENT_PREDICT=Estimator.predict
def _gcl_predict(self,*args,**kwargs):
    global _GCL_STATS
    _GCL_STATS=dict(layers=[],retired=False)
    try:
        answer=_GCL_PARENT_PREDICT(self,*args,**kwargs)
        assert _GCL_STATS['layers']==list(range(1,16))
        return answer
    finally:
        _GCL_STATS['retired']=True
_gop2_gnn_step=_gcl_gnn_step
Estimator.predict=_gcl_predict
