# Read-action basis on exact current YR7; UC8 coherent248+8 maps are reused.
_RAP4_CONTEXT = [None]
_RAP4_FULL = {}
_RAP4_STATS = dict(builds=[],probe_rows=[],retired=True)
_RAP4_PARENT_RECON = _slr_reconstruct
_RAP4_PARENT_LAZY = _astra_bcc_lazy
_RAP4_PARENT_PREDICT = _predict_binary


def _slr_reconstruct(U,state):
    saved = _RAP4_CONTEXT[0]
    _RAP4_CONTEXT[0] = (U,state)
    try:
        return _RAP4_PARENT_RECON(U,state)
    finally:
        _RAP4_CONTEXT[0] = saved


def _astra_bcc_lazy(mapped,U,rows,uv,columns):
    # Full pre-crop bases already paid by the original lazy reconstruction.
    if _RAP4_FULL:
        raise RuntimeError('RAP4 overlapping lazy metric context')
    _RAP4_FULL[id(uv[1])] = (uv[1],mapped,0)
    _RAP4_FULL[id(uv[2])] = (uv[2],U,2*int(uv[1][0].shape[1]))
    try:
        return _RAP4_PARENT_LAZY(mapped,U,rows,uv,columns)
    finally:
        _RAP4_FULL.clear()


def _uvp_split(F,blocks):
    uv = _UC8_PARENT_SPLIT(F,blocks)
    if not _UC8_ACTIVE or _UVP_CUR[0] not in (10,11,12,13,14):
        return uv
    if int(F.shape[0])<320 or uv[3]!=256:
        raise RuntimeError('RAP4 fixed common256/omitted64 frame')
    _UC8_CACHE[id(uv)] = dict(uv=uv,F=F,entry=None)
    return uv


def _uc8_entry(uv):
    if not _UC8_ACTIVE or _UVP_CUR[0] not in (10,11,12,13,14):
        return None
    pending = _UC8_CACHE.get(id(uv))
    if pending is None or pending['uv'] is not uv:
        raise RuntimeError('RAP4 immutable frame ownership missing')
    if pending['entry'] is not None:
        return pending['entry']
    if _UVP_CUR[0] not in (10,13):
        raise RuntimeError('RAP4 metric must be frozen at frame birth')
    full = _RAP4_FULL.get(id(uv))
    if full is not None:
        if full[0] is not uv:
            raise RuntimeError('RAP4 lazy identity mismatch')
        U = full[1]; offset = full[2]
    else:
        context = _RAP4_CONTEXT[0]
        if context is None or context[1]['lazy'] is not None:
            raise RuntimeError('RAP4 full-row materialized context missing')
        U = context[0]; offset = 0
    F = pending['F'];Fu,Fv,blocks,ru = uv
    if int(U.shape[1]) != int(F.shape[0]) or not 72<=int(U.shape[0])<=1024:
        raise RuntimeError('RAP4 full represented row universe')
    h = _E2Hold();P=int(Fu.shape[1])
    tail=fnp.empty((64,P),dtype=fnp.float32);b=0
    for a,k in blocks:
        h(fnp.add(F[256:320,a:a+k],F[256:320,a+k:a+2*k],out=tail[:,b:b+k]));b+=k
    h(fnp.multiply(tail,0.5,out=tail))
    ctx = _OTT_CONTEXT[0]
    if ctx is None:
        raise RuntimeError('RAP4 missing birth-layer signed coefficients')
    probe,global_rows = _rap4_probe_rows(U,ctx)
    lam_pair = fnp.empty(P,dtype=ctx['lam'].dtype);b=0
    for a,k in blocks:
        plus=ctx['lam'][offset+a:offset+a+k]
        minus=ctx['lam'][offset+a+k:offset+a+2*k]
        if not bool(fnp.all(plus == -minus)):
            raise RuntimeError('RAP4 signed pair coefficient mismatch')
        h(fnp.add(plus,0.0,out=lam_pair[b:b+k]));b+=k
    if b != P:
        raise RuntimeError('RAP4 pair layout')
    compact,keep,joint,A = _rap4_factor(Fu,tail,Fv,U,probe,lam_pair)
    cuv=(compact,Fv,blocks,ru)
    pending['entry']=(uv,cuv,keep,joint,A)
    pending['F']=None
    _UC8_STATS['builds'].append([int(_UVP_CUR[0]),int(F.shape[0]),P,248,72,8])
    _RAP4_STATS['builds'].append([int(_UVP_CUR[0]),int(U.shape[0]),int(U.shape[1]),P,4])
    _RAP4_STATS['probe_rows'].append([int(x) for x in global_rows[probe]])
    return pending['entry']


def _predict_binary(mlp,*args,**kwargs):
    if _RAP4_CONTEXT[0] is not None or _RAP4_FULL:
        raise RuntimeError('RAP4 stale metric context')
    _RAP4_STATS.update(builds=[],probe_rows=[],retired=False)
    try:
        return _RAP4_PARENT_PREDICT(mlp,*args,**kwargs)
    finally:
        _RAP4_CONTEXT[0]=None
        _RAP4_FULL.clear()
        _RAP4_STATS['retired']=True
