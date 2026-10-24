"""Fixed L1--14 innovation to the existing norm-coupling pair carrier.

No new cumulant plane is added to the current layer. The law's isotropic-input
diagonal prediction is replaced by the already computed post22 row sums.
Its explicit anisotropic carry and all existing diagonal subtractions remain.
The residual's radial component is removed on represented birth hubs.
"""

_NCR_ACTIVE = [False]
_NCR_PAIR = [None]
_NCR_STATS = dict(captures=0, updates=0, layers=[], retired=True)
_NCR_BINARY = _predict_binary


def _ncr_project(row_sum, iso_diagonal, radial_diagonal, eligible, xp):
    delta = (row_sum-iso_diagonal)*eligible
    radial = radial_diagonal*eligible
    den = xp.sum(radial)
    # An empty roster produces zero. A degenerate radial template uses the
    # constant template on the same roster; neither branch changes the arm.
    safe = xp.isfinite(den) & (xp.abs(den)>1e-30)
    radial = xp.where(safe, radial, eligible)
    den = xp.sum(radial)
    factor = xp.sum(delta)/xp.where(xp.abs(den)>1e-30, den, 1.0)
    return delta-factor*radial


def _ncr_weights(delta, gap, birth, kap_next, s_new, strength, xp):
    z = (2.0*kap_next*strength/s_new)*delta/(gap*birth)
    return xp.stack((xp.concatenate((z,-z)),xp.concatenate((-z,z)))).astype(xp.float32)


def _ncr_capture(post22, covariance, aux):
    if not _NCR_ACTIVE[0]:
        return
    layer = int(_PHYS_LAYER[0])
    if not 1<=layer<=14 or _NCR_PAIR[0] is not None:
        raise RuntimeError('norm-coupling residual capture placement')
    row_sum = fnp.sum(post22,axis=1)
    radial = fnp.trace(covariance)*fnp.diag(covariance)+2*fnp.sum(covariance*covariance,axis=1)
    _NCR_PAIR[0] = (layer,aux,row_sum,radial)
    _NCR_STATS['captures'] += 1


def _ncr_birth_delta(layer,aux,hubs,iso_diagonal,delta,safe_birth):
    saved = _NCR_PAIR[0]
    if saved is None or saved[0]!=layer or saved[1] is not aux:
        raise RuntimeError('norm-coupling residual source/consumer mismatch')
    eligible = fnp.zeros(int(iso_diagonal.shape[0]),dtype=fnp.float64)
    fnp.put(eligible,hubs,_E2_LAYER['ott_live_mask'][hubs])
    correction = _ncr_project(saved[2],iso_diagonal,saved[3],eligible,fnp)
    kap_next = (_NP_STRENGTH[layer+1] or 1.0)/int(iso_diagonal.shape[0])
    s_new = 1.0 if _S19_LAM is None else _S19_LAM[1]
    result = _ncr_weights(correction[hubs],delta[hubs],safe_birth[hubs],kap_next,s_new,_NP_DIAG,fnp)
    scale = float(fnp.sum(fnp.abs(correction)))
    balance = float(fnp.abs(fnp.sum(correction)))
    if not balance<=1e-10*max(scale,1e-30):
        raise RuntimeError('norm-coupling radial balance')
    _NCR_STATS['layers'].append(dict(layer=int(layer),hubs=int(hubs.shape[0]),
        represented_hubs=int(fnp.sum(eligible)),relative_trace_residual=balance/max(scale,1e-30)))
    _NCR_STATS['updates'] += 1
    _NCR_PAIR[0] = None
    return result


def _predict_binary(mlp,*args,**kwargs):
    if int(mlp.width)!=1024 or int(mlp.depth)!=16 or kwargs.get('final_hook') is None:
        return _NCR_BINARY(mlp,*args,**kwargs)
    if _NCR_ACTIVE[0] or _NCR_PAIR[0] is not None:
        raise RuntimeError('norm-coupling residual state leaked')
    _NCR_STATS.clear()
    _NCR_STATS.update(captures=0,updates=0,layers=[],retired=False)
    _NCR_ACTIVE[0] = True
    try:
        return _NCR_BINARY(mlp,*args,**kwargs)
    finally:
        _NCR_ACTIVE[0] = False
        _NCR_PAIR[0] = None
        _NCR_STATS['retired'] = True
