# Fixed common256 budget: retain248 coordinates, compress joint72 into8.
_UC8_ACTIVE = False
_UC8_CACHE = {}
_UC8_STATS = dict(builds=[], reconstruction=[], read=[], retired=True)
_UC8_PARENT_SPLIT = _uvp_split
_UC8_PARENT_PREDICT = _predict_binary


def _uvp_split(F, blocks):
    uv = _UC8_PARENT_SPLIT(F, blocks)
    if not _UC8_ACTIVE or _UVP_CUR[0] not in (10,11,12,13,14):
        return uv
    if int(F.shape[0]) < 320 or uv[3] != 256:
        raise RuntimeError('UC8 requires common256 and the omitted64 coordinates')
    hold = _E2Hold()
    Fu, Fv, blocks, ru = uv
    energy = hold(fnp.sum(hold(Fu * Fu), axis=1))
    order = hold(fnp.argsort(energy, kind='stable'))
    drop = fnp.sort(order[:8]); keep = fnp.sort(order[8:])
    joint = fnp.concatenate([drop, fnp.arange(256,320)])
    P = int(Fu.shape[1])
    tail = fnp.empty((64,P),dtype=fnp.float32)
    b = 0
    for a,h in blocks:
        hold(fnp.add(F[256:320,a:a+h], F[256:320,a+h:a+2*h], out=tail[:,b:b+h]))
        b += h
    hold(fnp.multiply(tail,0.5,out=tail))
    T = hold(fnp.concatenate([Fu[drop], tail]))
    gram = hold(T @ T.T)
    values, vectors = fnp.linalg.eigh(gram.astype(fnp.float64))
    V = vectors[:,-8:][:,::-1].astype(fnp.float32)
    B = hold(V.T @ T)
    compact = fnp.concatenate([Fu[keep],B])
    cuv = (compact,Fv,blocks,ru)
    _UC8_CACHE[id(uv)] = (uv,cuv,keep,joint,V)
    _UC8_STATS['builds'].append([int(_UVP_CUR[0]),int(F.shape[0]),P,248,72,8])
    return uv


def _uc8_entry(uv):
    if not _UC8_ACTIVE or _UVP_CUR[0] not in (10,11,12,13,14):
        return None
    entry = _UC8_CACHE.get(id(uv))
    if entry is None or entry[0] is not uv:
        raise RuntimeError('UC8 immutable frame ownership missing')
    return entry


def _uc8_product(left,right,entry,leg):
    uv,cuv,keep,joint,V = entry
    run = _UBC_RUN[0]
    product = _ubc_product if leg == 'u' else _ubx_product
    if run is None or run['uv'] is not uv:
        return product(left,right,cuv,leg)
    # The values differ from original Fu despite identical256 product shapes.
    # Tag only the private common-factor trees; original v/vt and L15 retain
    # their original identities. The L15 original-u cold work remains paid.
    saved_uv,saved_generation = run['uv'],run['generation']
    run['uv'] = cuv
    run['generation'] = (saved_generation,'uc8-joint72-rank8')
    try:
        return product(left,right,cuv,leg)
    finally:
        run['uv'],run['generation'] = saved_uv,saved_generation


def _uc8_recon(U,uv):
    entry = _uc8_entry(uv)
    if entry is None:
        return _ubc_product(U[:,:uv[3]],uv[0],uv,'u')
    _,cuv,keep,joint,V = entry
    hold = _E2Hold()
    left = hold(fnp.concatenate([U[:,keep],hold(U[:,joint] @ V)],axis=1))
    answer = _uc8_product(left,cuv[0],entry,'u')
    _UC8_STATS['reconstruction'].append([int(_UVP_CUR[0]),int(U.shape[0]),int(cuv[0].shape[1])])
    return answer


def _uc8_inner(inner,Su,uv):
    entry = _uc8_entry(uv)
    if entry is None:
        inner_u = _ubx_product(Su,uv[0].T,uv,'ut')
        _BH_HOLD.append(fnp.add(inner[:,:uv[3]],inner_u,out=inner[:,:uv[3]]))
        _BH_HOLD.clear()
        return inner
    _,cuv,keep,joint,V = entry
    hold = _E2Hold()
    inner_u = hold(_uc8_product(Su,cuv[0].T,entry,'ut'))
    row = hold(fnp.arange(int(inner.shape[0])) * int(inner.shape[1]))
    keep_index = hold(row[:,None] + keep)
    joint_index = hold(row[:,None] + joint)
    hold(fnp.put(inner,keep_index,hold(inner[:,keep]+inner_u[:,:248])))
    hold(fnp.put(inner,joint_index,hold(inner[:,joint]+hold(inner_u[:,248:] @ V.T))))
    _UC8_STATS['read'].append([int(_UVP_CUR[0]),int(Su.shape[0]),int(cuv[0].shape[1])])
    return inner


def _predict_binary(mlp,*args,**kwargs):
    global _UC8_ACTIVE
    if _UC8_ACTIVE:
        raise RuntimeError('UC8 nested prediction')
    _UC8_CACHE.clear()
    _UC8_STATS.update(builds=[],reconstruction=[],read=[],retired=False)
    _UC8_ACTIVE = int(mlp.width)==1024 and int(mlp.depth)==16 and kwargs.get('final_hook') is not None
    try:
        return _UC8_PARENT_PREDICT(mlp,*args,**kwargs)
    finally:
        _UC8_ACTIVE=False
        _UC8_CACHE.clear()
        _UC8_STATS['retired']=True
