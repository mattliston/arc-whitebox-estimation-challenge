# Local-only principal common128 coordinates. Every carried factor is retained.
_LCR_RU = 128
_LOR_PARENT_UV_STATE = _uvp_state
_LOR_PARENT_UV_RECON = _uvp_recon
_LOR_PARENT_UV_INNER = _uvp_inner
_LOR_STATS = dict(splits=0, cache_hits=0, recon=0, inner=0)


def _lor_split(F, blocks, lam_cols):
    rank = int(F.shape[0])
    ru = min(_LCR_RU, rank)
    pairs = sum(h for _, h in blocks)
    _slr_require(int(lam_cols.shape[0]) == int(F.shape[1]), 'local optimal signed weights align')
    Fu = fnp.empty((rank, pairs), dtype=fnp.float32)
    Fv2 = fnp.empty((rank, pairs), dtype=fnp.float32)
    weights = fnp.empty(pairs, dtype=fnp.float64)
    handles = []
    offset = 0
    for start, half in blocks:
        out = slice(offset, offset + half)
        handles.append(fnp.add(F[:, start:start + half], F[:, start + half:start + 2 * half], out=Fu[:, out]))
        handles.append(fnp.subtract(F[:, start:start + half], F[:, start + half:start + 2 * half], out=Fv2[:, out]))
        lp = lam_cols[start:start + half].astype(fnp.float64)
        lm = lam_cols[start + half:start + 2 * half].astype(fnp.float64)
        handles.append(fnp.copyto(weights[out], (lp * lp + lm * lm) * 0.5))
        offset += half
    handles.append(fnp.multiply(Fu, 0.5, out=Fu))
    handles.append(fnp.multiply(Fv2, 0.5, out=Fv2))
    handles.clear()
    u64 = Fu.astype(fnp.float64)
    v64 = Fv2.astype(fnp.float64)
    weights = weights * fnp.sum(u64 * u64, axis=0) * fnp.sum(v64 * v64, axis=0)
    del u64, v64
    # A common normalization changes only Gram scale, not its principal span.
    weights = weights / fnp.maximum(fnp.max(weights), 1e-30)
    weighted = Fu * fnp.sqrt(weights).astype(fnp.float32)[None, :]
    gram = _shared_old384_mm(weighted, weighted.T).astype(fnp.float64)
    gram = (gram + gram.T) * 0.5
    values, vectors = fnp.linalg.eigh(gram)
    basis = vectors[:, -ru:].astype(fnp.float32)
    projected = _shared_old384_mm(basis.T, Fu)
    del weights, weighted, gram, values, vectors, Fu
    _LOR_STATS['splits'] += 1
    return (projected, Fv2, blocks, ru, basis)


def _lor_cached_split(state, F, blocks, lam_cols):
    # Full F is unchanged between joins. Uniform old-lambda carry rescaling
    # does not change the weighted Gram span. L13's lazy base reuses L10 F.
    cache = state.setdefault('lor_blocks', [])
    for old_F, old_blocks, uv in cache:
        if old_F is F and old_blocks == blocks:
            _LOR_STATS['cache_hits'] += 1
            return uv
    uv = _lor_split(F, blocks, lam_cols)
    cache.append((F, blocks, uv))
    _slr_require(len(cache) <= 3, 'bounded local optimal factor cache')
    return uv


def _uvp_state(state):
    if _UVP_CUR[0] not in _LCR_LAYERS:
        return _LOR_PARENT_UV_STATE(state)
    if state.get('lcr_ru') == _LCR_RU and state.get('uvp') is not None:
        return state['uvp']
    context = _OTT_CONTEXT[0]
    _slr_require(context is not None, 'local optimal split first use occurs during push')
    columns = int(state['columns'])
    lam_cols = context['lam'][:columns]
    group = state['lazy']
    if group is None:
        uv = ('m', _lor_cached_split(state, state['F'], _uvp_blocks(state, 0, columns), lam_cols))
    else:
        old = int(group['prefix'])
        uv = ('l',
              _lor_cached_split(state, group['base'], _uvp_blocks(state, 0, old), lam_cols[:old]),
              _lor_cached_split(state, group['join'], _uvp_blocks(state, old, columns), lam_cols[old:]))
    state['uvp'] = uv
    state['lcr_ru'] = _LCR_RU
    return uv


def _uvp_recon(U, uv):
    if len(uv) == 4:
        return _LOR_PARENT_UV_RECON(U, uv)
    Fu, Fv2, blocks, ru, basis = uv
    Pu = _shared_old384_mm(_shared_old384_mm(U, basis), Fu)
    Pv = _shared_old384_mm(U, Fv2)
    out = fnp.empty((int(U.shape[0]), 2 * int(Fu.shape[1])), dtype=fnp.float32)
    handles = []
    offset = 0
    for start, half in blocks:
        handles.append(fnp.add(Pu[:, offset:offset + half], Pv[:, offset:offset + half], out=out[:, start:start + half]))
        handles.append(fnp.subtract(Pu[:, offset:offset + half], Pv[:, offset:offset + half], out=out[:, start + half:start + 2 * half]))
        offset += half
    _LOR_STATS['recon'] += 1
    return out


def _uvp_inner(left_sq, lam_cols, uv):
    if len(uv) == 4:
        return _LOR_PARENT_UV_INNER(left_sq, lam_cols, uv)
    Fu, Fv2, blocks, ru, basis = uv
    rows, pairs = int(left_sq.shape[0]), int(Fu.shape[1])
    weighted = left_sq * lam_cols[None, :]
    Su = fnp.empty((rows, pairs), dtype=fnp.float32)
    Dv = fnp.empty((rows, pairs), dtype=fnp.float32)
    handles = []
    offset = 0
    for start, half in blocks:
        handles.append(fnp.add(weighted[:, start:start + half], weighted[:, start + half:start + 2 * half], out=Su[:, offset:offset + half]))
        handles.append(fnp.subtract(weighted[:, start:start + half], weighted[:, start + half:start + 2 * half], out=Dv[:, offset:offset + half]))
        offset += half
    handles.clear()
    del weighted
    inner = _shared_old384_mm(Dv, Fv2.T)
    common = _shared_old384_mm(_shared_old384_mm(Su, Fu.T), basis.T)
    _LOR_STATS['inner'] += 1
    return inner + common
