# Fixed rank-eight additive common-factor restoration, layers 10..14.
# Existing U256 and full difference-factor products retain their inputs/routes.
_UT8_ACTIVE = False
_UT8_CACHE = {}
_UT8_STATS = {'builds': [], 'reconstruction': [], 'read': [], 'retired': True}
_UT8_PARENT_SPLIT = _uvp_split
_UT8_PARENT_PREDICT = _predict_binary


def _uvp_split(F, blocks):
    uv = _UT8_PARENT_SPLIT(F, blocks)
    if not _UT8_ACTIVE or _UVP_CUR[0] not in (10, 11, 12, 13, 14):
        return uv
    if int(F.shape[0]) < 320 or uv[3] != 256:
        raise RuntimeError('UT8 requires the incumbent common256 frame and 64 omitted rows')
    P = int(uv[0].shape[1])
    tail = fnp.empty((64, P), dtype=fnp.float32)
    b = 0
    for a, h in blocks:
        fnp.add(F[256:320, a:a+h], F[256:320, a+h:a+2*h], out=tail[:, b:b+h])
        b += h
    fnp.multiply(tail, 0.5, out=tail)
    gram = tail @ tail.T
    values, vectors = fnp.linalg.eigh(gram.astype(fnp.float64))
    # Descending eigh order; arbitrary eigenvector signs cancel between V and B.
    V = vectors[:, -8:][:, ::-1].astype(fnp.float32)
    B = V.T @ tail
    _UT8_CACHE[id(uv)] = (uv, V, B)
    _UT8_STATS['builds'].append([int(_UVP_CUR[0]), int(F.shape[0]), P, 64, 8])
    return uv


def _ut8_recon(Pu, U, uv):
    if not _UT8_ACTIVE or _UVP_CUR[0] not in (10, 11, 12, 13, 14):
        return Pu
    found = _UT8_CACHE.get(id(uv))
    if found is None or found[0] is not uv:
        raise RuntimeError('UT8 immutable frame ownership missing')
    _, V, B = found
    extra = (U[:, 256:320] @ V) @ B
    _UT8_STATS['reconstruction'].append([int(_UVP_CUR[0]), int(U.shape[0]), int(B.shape[1])])
    return Pu + extra


def _ut8_inner(inner, Su, uv):
    if not _UT8_ACTIVE or _UVP_CUR[0] not in (10, 11, 12, 13, 14):
        return inner
    found = _UT8_CACHE.get(id(uv))
    if found is None or found[0] is not uv:
        raise RuntimeError('UT8 transpose immutable frame ownership missing')
    _, V, B = found
    extra = (Su @ B.T) @ V.T
    fnp.add(inner[:, 256:320], extra, out=inner[:, 256:320])
    _UT8_STATS['read'].append([int(_UVP_CUR[0]), int(Su.shape[0]), int(B.shape[1])])
    return inner


def _predict_binary(mlp, *args, **kwargs):
    global _UT8_ACTIVE
    if _UT8_ACTIVE:
        raise RuntimeError('UT8 nested prediction')
    _UT8_CACHE.clear()
    _UT8_STATS.update(builds=[], reconstruction=[], read=[], retired=False)
    _UT8_ACTIVE = int(mlp.width) == 1024 and int(mlp.depth) == 16 and kwargs.get('final_hook') is not None
    try:
        return _UT8_PARENT_PREDICT(mlp, *args, **kwargs)
    finally:
        _UT8_ACTIVE = False
        _UT8_CACHE.clear()
        _UT8_STATS['retired'] = True
