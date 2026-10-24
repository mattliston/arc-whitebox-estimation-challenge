
# Fixed prefix residual-energy restoration of feature22; all affected hidden layers.
_RE22_ACTIVE=[False]
_RE22_PENDING=[None]
_RE22_STATS={}
_RE22_PARENT_FEATURES=features
_RE22_PARENT_PREDICT=Estimator.predict

def _re22_energy(prefix,Q,rows,prefix_cols,tail_cols,variance,hold):
    m,p=map(int,prefix.shape)
    assert int(Q.shape[0])==m and tuple(rows.shape)==(m,)
    assert tuple(prefix_cols.shape)==(p,) and int(tail_cols.shape[0])>0
    assert str(prefix.dtype)==str(Q.dtype)=='float32'
    v=hold(variance.astype(fnp.float32))
    residual=hold(prefix-hold(Q@hold(Q.T@prefix)))
    values,ir,ic=fnp.intersect1d(rows,prefix_cols,assume_unique=True,return_indices=True)
    values,ir,ic=hold(values),hold(ir),hold(ic)
    if int(ir.shape[0]):fnp.put(residual,hold(hold(ir*p)+ic),fnp.float32(0))
    scale=hold(fnp.sqrt(hold(v[prefix_cols])))
    normalized=hold(residual/hold(hold(v[rows,None])*hold(scale[None,:])))
    energy=hold(fnp.einsum('ij,ij->i',normalized,normalized,optimize=False))
    ratio=hold(hold(fnp.sum(hold(1/hold(v[tail_cols]))))/hold(fnp.sum(hold(1/hold(v[prefix_cols])))))
    return energy*ratio

def _re22_capture(prefix,Q,rows,cols,cut,hold):
    if not _RE22_ACTIVE[0]:return
    assert _RE22_PENDING[0] is None
    layer=int(_RS_LAYER[0]);assert 1<=layer<=14
    energy=_re22_energy(prefix,Q,rows,hold(cols[:cut]),hold(cols[cut:]),_E2_LAYER['vc'],hold)
    _RE22_PENDING[0]=(layer,rows,energy)
    _RE22_STATS['reads'].append(dict(layer=layer,m=int(prefix.shape[0]),prefix=int(prefix.shape[1]),tail=int(cols.shape[0])-int(cut),qrank=int(Q.shape[1])))

def _re22_features(layer,*args,**kwargs):
    result=_RE22_PARENT_FEATURES(layer,*args,**kwargs)
    pending=_RE22_PENDING[0]
    if not _RE22_ACTIVE[0]:
        assert pending is None
        return result
    if pending is None:return result
    l,rows,energy=pending;assert l==int(layer)
    hold=_E2Hold()
    prior=hold(result[rows,22])
    restored=hold(fnp.sqrt(hold(fnp.maximum(hold(hold(prior*prior)+energy),fnp.float32(0)))))
    fnp.put(result,hold(hold(rows*30)+22),restored)
    _RE22_STATS['feature_layers'].append(int(layer))
    _RE22_PENDING[0]=None
    return result

def _re22_clear():
    _RE22_PENDING[0]=None
    _RE22_ACTIVE[0]=False

def _re22_predict(self,*args,**kwargs):
    assert not _RE22_ACTIVE[0] and _RE22_PENDING[0] is None
    _RE22_ACTIVE[0]=True
    _RE22_STATS.clear();_RE22_STATS.update(reads=[],feature_layers=[],retired=False)
    try:
        result=_RE22_PARENT_PREDICT(self,*args,**kwargs)
        assert _RE22_PENDING[0] is None
        return result
    finally:
        _re22_clear();_RE22_STATS['retired']=True

features=_re22_features
Estimator.predict=_re22_predict
