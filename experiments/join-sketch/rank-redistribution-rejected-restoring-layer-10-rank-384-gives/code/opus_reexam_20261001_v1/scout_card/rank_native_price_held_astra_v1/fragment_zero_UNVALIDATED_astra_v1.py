# HELD rank price candidate; no validation or submission authorization.
assert _ASTRA_COMPUTE_ON is True
assert _K3LB_PARENT_LEFT is _astra_compute_left_v1
assert _R320_RULE == {7: 384, 10: 320, 13: 384}, 'Unexpected price parent rank map'
_R320_RULE = {7: 384, 10: 320, 13: 384}

# v2: false flag delegates before any state access; non-rank states retain original path.
# Astra UB384 v1 SOURCE-ONLY, UNVALIDATED compute route; separate from capacity physics screen.
# Appended after existing UBXv2/v3 and after the rank384 literal. No kernels, plans,
# numerical operations, owner/generation keys, copies or byte caps are changed.
# A true flag admits the same immutable L10 prefix at384 and its v/vt B trees.
# A false flag delegates both wrappers to the original functions (native identity
# still must be certified). The320 route always delegates unchanged.
_ASTRA_UB384_ON = False
_ASTRA_UB384_PARENT_STATE = _uvp_state
_ASTRA_UB384_PARENT_B = _ipt_forward_b_ab

def _uvp_state(state):
    if not _ASTRA_UB384_ON:
        return _ASTRA_UB384_PARENT_STATE(state)
    group_check = state.get('lazy')
    rank_check = state.get('rank') if group_check is None else group_check.get('base_rank')
    if rank_check != 384:
        return _ASTRA_UB384_PARENT_STATE(state)
    uv = _UBC_PARENT_STATE(state)
    run = _UBC_RUN[0]
    if run is None:
        return uv
    group = state['lazy']
    if group is None:
        joins = state.get('joins', ())
        admitted = bool(joins) and joins[-1]['layer'] == 10 and state['rank'] == 384
        owner = state['F']
    else:
        admitted = group['layer'] == 13 and group['base_rank'] == 384
        owner = group['base']
    if not admitted or _UVP_CUR[0] not in (10, 11, 12, 13, 14, 15):
        _ubc_drop(run)
        return uv
    base = uv[1]
    Fu, Fv, blocks, ru = base
    P = int(Fu.shape[1])
    generation = (tuple(blocks), int(ru), tuple(owner.shape))
    if not (ru == 256 and 4096 < P <= 6144 and tuple(Fu.shape) == (256, P)
            and tuple(Fv.shape) == (384, P) and tuple(owner.shape) == (384, 2 * P)
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
    run_check = _UBC_RUN[0]
    if (not _ASTRA_UB384_ON or run_check is None or run_check.get('owner') is None
            or int(run_check['owner'].shape[0]) != 384):
        return _ASTRA_UB384_PARENT_B(b, L, k, n)
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
                    and offset == 0 and valid == stored == n == (256 if leg == 'ut' else 384)
                    and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32'):
                key = (run['generation'], leg, offset, valid, L, k, n, 'float32')
            elif (L == 6 and k == rank == (256 if leg == 'u' else 384)
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
