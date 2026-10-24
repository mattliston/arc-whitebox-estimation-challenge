# Fixed difference factor192 only on the energy-ordered L10 rank320 base. New approximation; no qualification.
_DVC_RV = 192
_DVC_STATS = dict(splits=0, recon=0, read=0, rank=320, ru=256, rv=192)
_DVC_PARENT_SPLIT = _uvp_split
_DVC_PARENT_RECON = _uvp_recon
_DVC_PARENT_INNER = _uvp_inner
_DVC_PARENT_STATE = _uvp_state
_DVC_PARENT_FORWARD_B = _ipt_forward_b_ab


def _uvp_split(F, blocks):
    uv = _DVC_PARENT_SPLIT(F, blocks)
    if int(F.shape[0]) != 320:
        return uv
    Fu, Fv2, layout, ru = uv
    if ru != 256 or tuple(Fv2.shape) != (320, int(Fu.shape[1])):
        raise RuntimeError('DVC fixed L10-base shape')
    _DVC_STATS['splits'] += 1
    return (Fu, fnp.copy(Fv2[:_DVC_RV]), layout, ru)


def _uvp_recon(U, uv):
    Fu, Fv2, blocks, ru = uv
    if int(Fv2.shape[0]) != _DVC_RV:
        return _DVC_PARENT_RECON(U, uv)
    if int(U.shape[1]) != 320 or ru != 256:
        raise RuntimeError('DVC fixed reconstructed basis')
    hold = _E2Hold()
    Pu = hold(_ubc_product(hold(U[:, :ru]), Fu, uv, 'u'))
    Pv = hold(_ubc_product(hold(U[:, :_DVC_RV]), Fv2, uv, 'v'))
    out = fnp.empty((int(U.shape[0]), 2 * int(Fu.shape[1])), dtype=fnp.float32)
    b = 0
    for a, h in blocks:
        hold(fnp.add(hold(Pu[:, b:b+h]), hold(Pv[:, b:b+h]), out=hold(out[:, a:a+h])))
        hold(fnp.subtract(hold(Pu[:, b:b+h]), hold(Pv[:, b:b+h]), out=hold(out[:, a+h:a+2*h])))
        b += h
    _DVC_STATS['recon'] += 1
    return out



def _uvp_inner(left_sq, lam_cols, uv):
    """(L^2 lam) F^T with F in u/v form: rows x rank. = the parent's arithmetic; the two products through _ubx_product."""
    Fu, Fv2, blocks, ru = uv
    if int(Fv2.shape[0]) != _DVC_RV:
        return _DVC_PARENT_INNER(left_sq, lam_cols, uv)
    if ru != 256:
        raise RuntimeError('DVC fixed read basis')
    rows = int(left_sq.shape[0])
    P = int(Fu.shape[1])
    a_ = left_sq * lam_cols[None, :]
    Su = fnp.empty((rows, P), dtype=fnp.float32)
    Dv = fnp.empty((rows, P), dtype=fnp.float32)
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(a_[:, a:a + h], a_[:, a + h:a + 2 * h], out=Su[:, b:b + h]))
        h_(fnp.subtract(a_[:, a:a + h], a_[:, a + h:a + 2 * h], out=Dv[:, b:b + h]))
        b += h
    _BH_HOLD.clear()
    del a_
    reduced = _ubx_product(Dv, Fv2.T, uv, 'vt')
    inner = fnp.zeros((rows, 320), dtype=fnp.float32)
    _BH_HOLD.append(fnp.copyto(inner[:, :_DVC_RV], reduced))
    del reduced
    inner_u = _ubx_product(Su, Fu.T, uv, 'ut')
    _BH_HOLD.append(fnp.add(inner[:, :ru], inner_u, out=inner[:, :ru]))
    _BH_HOLD.clear()
    del Su, Dv, inner_u
    _DVC_STATS['read'] += 1
    return inner

def _uvp_state(state):
    uv = _UBC_PARENT_STATE(state)
    run = _UBC_RUN[0]
    if run is None:
        return uv
    group = state['lazy']
    if group is None:
        joins = state.get('joins', ())
        admitted = bool(joins) and joins[-1]['layer'] == 10 and state['rank'] == 320
        owner = state['F']
    else:
        admitted = group['layer'] == 13 and group['base_rank'] == 320
        owner = group['base']
    if not admitted or _UVP_CUR[0] not in (10, 11, 12, 13, 14, 15):
        _ubc_drop(run)
        return uv
    base = uv[1]
    Fu, Fv, blocks, ru = base
    P = int(Fu.shape[1])
    generation = (tuple(blocks), int(ru), tuple(owner.shape), tuple(Fv.shape))
    if not (ru == 256 and 4096 < P <= 6144 and tuple(Fu.shape) == (256, P)
            and tuple(Fv.shape) == (_DVC_RV, P) and tuple(owner.shape) == (320, 2 * P)
            and str(owner.dtype) == str(Fu.dtype) == str(Fv.dtype) == 'float32'):
        _ubc_drop(run)
        return uv
    if run['owner'] is not owner or run['generation'] != generation:
        _ubc_drop(run)
        run['owner'], run['generation'] = owner, generation
        run['stats']['generations'] += 1
    # At join13 uv arrays are recreated from the SAME unmodified base F.
    run['uv'] = base
    return uv

def _ipt_forward_b_ab(b, L, k, n):
    rc = _UBX_RC[0]
    if rc is not None:
        j = rc['n']
        rc['n'] = j + 1
        key = rc['tag'] + (j, L, k, n, tuple(int(x) for x in b.shape), str(b.dtype))
        found = _UBX_RCACHE['cache'].get(key)
        if found is not None:
            _UBX_RC_STATS['hits'] += 1
            return found
        result = _UBC_PARENT_B(b, L, k, n)
        count = 1
        for x in result.shape:
            count *= int(x)
        if str(result.dtype) == 'float32' and _UBX_RCACHE['bytes'] + 4 * count <= _UBX_RC_LIMIT:
            _UBX_RCACHE['cache'][key] = fnp.copy(result)     # counted copy, as UBC
            _UBX_RCACHE['bytes'] += 4 * count
            _UBX_RC_STATS['stores'] += 1
        else:
            _UBX_RC_STATS['uncached'] += 1
        return result
    if _UBX_RCACHE['cache'] and _UVP_CUR[0] is not None and _UVP_CUR[0] >= 10:   # the L7 frame's trees are dead after its layers
        _UBX_RCACHE.update(owner=None, run=None, cache={}, bytes=0)
    # = UBC's _ipt_forward_b_ab with the u leg's tail admitted and four stored trees
    run, context = _UBC_RUN[0], _UBC_CONTEXT[0]
    key = None
    if (run is not None and context is not None and context['usable']
            and context['uv'] is run['uv'] and context['generation'] == run['generation']):
        token = _ubc_find(context, b)
        if token is not None:
            offset, valid, stored, rank = token
            leg = context['leg']
            width = int(run['uv'][0].shape[1])
            main = offset == 0 and valid == stored == n == 4096
            tail = (leg in ('u', 'v') and offset == 4096 and valid == width - 4096
                    and stored == n == ((valid + 63) // 64) * 64)
            if (leg in ('ut', 'vt') and L == 5 and k == rank == width
                    and offset == 0 and valid == stored == n == (256 if leg == 'ut' else int(run['uv'][1].shape[0]))
                    and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32'):
                key = (run['generation'], leg, offset, valid, L, k, n, 'float32')
            elif (L == 6 and k == rank == (256 if leg == 'u' else int(run['uv'][1].shape[0]))
                    and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32'
                    and (main or tail)):
                key = (run['generation'], leg, offset, valid, L, k, n, 'float32')
    if key is None:
        return _UBC_PARENT_B(b, L, k, n)
    run['stats']['eligible'] += 1
    found = run['cache'].get(key)
    if found is not None:
        run['stats']['hits'] += 1
        slots = run['stats']['hit_slots']
        slots[key[1] + ':' + str(key[2])] = slots.get(key[1] + ':' + str(key[2]), 0) + 1
        return found
    run['stats']['misses'] += 1
    result = _UBC_PARENT_B(b, L, k, n)
    count = 7 ** L * (k // (1 << L)) * (n // (1 << L))
    size = count * 4
    if (len(run['cache']) < 4 and run['bytes'] + size <= _UBC_LIMIT
            and tuple(result.shape) == (7,) * L + (k // (1 << L), n // (1 << L))
            and str(result.dtype) == 'float32'):
        saved = fnp.copy(result)
        run['cache'][key] = saved
        run['bytes'] += size
        run['stats']['copies'] += 1
        run['stats']['stores'] += 1
        run['stats']['copied_bytes'] += size
        run['stats']['copy_elements'] += count
        run['stats']['peak_cache_bytes'] = max(run['stats']['peak_cache_bytes'], run['bytes'])
        run['stats']['max_resident_bytes'] = run['stats']['peak_cache_bytes']
    else:
        run['stats']['uncached'] += 1
    return result
