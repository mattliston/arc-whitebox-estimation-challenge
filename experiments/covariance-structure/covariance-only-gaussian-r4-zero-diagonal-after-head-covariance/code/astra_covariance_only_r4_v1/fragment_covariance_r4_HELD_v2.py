# Fixed covariance-only fourth Gaussian Hermite term, hidden layers1–14.
# Runs AFTER current learned head/FHM recentering; current m, target/radial,
# scalar moments and all pair-map outputs remain untouched. Downstream
# covariance consumers see the corrected C with its coherent float32 cache.
_CR4_STATS=dict(calls=0,layers=[],elements=0,retired=False)
_CR4_PARENT_PREDICT=Estimator.predict

def _cr4_covariance(l,C,Cpre,aux):
    s=aux['sigma'];a=aux['alpha'];pdf=aux['pdf']
    h=(pdf*(a*a-1))/(s*s*s)
    delta=fnp.multiply(Cpre,Cpre)
    fnp.multiply(delta,delta,out=delta)
    delta=fnp.multiply(delta,fnp.outer(h,h))
    fnp.divide(delta,24.,out=delta)
    fnp.fill_diagonal(delta,0.)
    diagonal=fnp.diag(C)
    corrected=fnp.add(C,delta)
    fnp.fill_diagonal(corrected,diagonal)
    corrected32=corrected.astype(fnp.float32)
    _CR4_STATS['calls']+=1
    _CR4_STATS['layers'].append(int(l))
    _CR4_STATS['elements']+=int(C.size)
    return corrected,corrected32

def _cr4_predict(self,*args,**kwargs):
    _CR4_STATS.clear();_CR4_STATS.update(calls=0,layers=[],elements=0,retired=False)
    try:
        return _CR4_PARENT_PREDICT(self,*args,**kwargs)
    finally:
        _CR4_STATS['retired']=True

Estimator.predict=_cr4_predict
