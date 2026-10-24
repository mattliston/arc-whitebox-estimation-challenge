# HELD prospective approximation, not a continuation/promotion of either failed case.
# Parent: full384 state/UB384 on the exact current c149 bank+quarter source.
# One fixed arm: prefix reconstruction v-coordinate320 on L10..13. All common
# coordinates, all full384 reads, full carried state, full tail repairs, recent
# L13 block and late L14/L15 reconstruction retain the parent route.
# Coordinates are current index-sorted QR axes, NOT an importance order or old320 span.
_ASTRA_RP_ON = True
_ASTRA_RP_LAYERS = (10, 11, 12, 13)
_ASTRA_RP_PREFIX = [None]
_ASTRA_RP_PARENT_STATE = _uvp_state
_ASTRA_RP_PARENT_PRODUCT = _ubc_product
_ASTRA_RP_PARENT_B = _ASTRA_CLUSTER_PARENT_B
_ASTRA_RP_PARENT_PREDICT = _predict_binary
_ASTRA_RP_STATS = {}


def _astra_rp_retire_early(run):
    # Own entries only. The late384 layout replaces dead early320 entries under
    # the same four-slot /512MB parent cap; full u256 entries remain resident.
    if run is None:
        return
    dead = [key for key in run['cache'] if len(key) == 8 and key[1] == 'astra_rp_v320']
    for key in dead:
        value = run['cache'].pop(key)
        size = 4
        for extent in value.shape:
            size *= int(extent)
        run['bytes'] -= size
        _ASTRA_RP_STATS['retired_early_bytes'] += size
    if run['bytes'] < 0:
        raise RuntimeError('recon-prefix cache byte accounting')


def _uvp_state(state):
    uv = _ASTRA_RP_PARENT_STATE(state)
    if not _ASTRA_RP_ON:
        return uv
    layer = _UVP_CUR[0]
    if layer in (14, 15):
        _astra_rp_retire_early(_UBC_RUN[0])
    group = state.get('lazy')
    if group is None:
        joins = state.get('joins', ())
        eligible = bool(joins) and joins[-1]['layer'] == 10 and state['rank'] == 384
        owner = state.get('F')
    else:
        eligible = group['layer'] == 13 and group['base_rank'] == 384
        owner = group['base']
    _ASTRA_RP_PREFIX[0] = None
    if layer in _ASTRA_RP_LAYERS and eligible:
        prefix = uv[1]
        Fu, Fv, blocks, ru = prefix
        width = int(Fu.shape[1])
        if not (ru == 256 and tuple(Fu.shape) == (256, width)
                and tuple(Fv.shape) == (384, width)
                and tuple(owner.shape) == (384, 2 * width)
                and str(owner.dtype) == str(Fu.dtype) == str(Fv.dtype) == 'float32'):
            raise RuntimeError('recon-prefix full384 state shape')
        _ASTRA_RP_PREFIX[0] = (prefix, owner)
    return uv


def _ubc_product(left, right, uv, leg):
    prefix = _ASTRA_RP_PREFIX[0]
    if (not _ASTRA_RP_ON or _UVP_CUR[0] not in _ASTRA_RP_LAYERS
            or leg != 'v' or prefix is None or uv is not prefix[0]):
        return _ASTRA_RP_PARENT_PRODUCT(left, right, uv, leg)
    if (right is not uv[1] or int(left.shape[1]) != 384
            or int(right.shape[0]) != 384
            or str(left.dtype) != 'float32' or str(right.dtype) != 'float32'):
        raise RuntimeError('recon-prefix exact operand ownership')
    if _UBC_CONTEXT[0] is not None:
        raise RuntimeError('recon-prefix nested product context')
    a = left[:, :320]
    b = right[:320, :]
    run = _UBC_RUN[0]
    context = None
    if (run is not None and run['uv'] is uv and run['owner'] is prefix[1]):
        width = int(b.shape[1])
        context = dict(leg='astra_rp_v320', uv=uv, generation=run['generation'],
                       ancestry=[(b, (0, width, width, 320))], usable=True)
        _UBC_CONTEXT[0] = context
    _ASTRA_RP_STATS['prefix_calls'] += 1
    _ASTRA_RP_STATS['events'].append((_UVP_CUR[0], int(left.shape[0]), int(right.shape[1]), 384, 320))
    try:
        # Original counted planner/kernels and current quarter loaders remain.
        return _shared_old384_mm(a, b)
    finally:
        if context is not None:
            context['ancestry'].clear()
            _UBC_CONTEXT[0] = None


def _astra_rp_forward_b(b, L, k, n):
    context = _UBC_CONTEXT[0]
    if not (_ASTRA_RP_ON and context is not None and context['leg'] == 'astra_rp_v320'):
        return _ASTRA_RP_PARENT_B(b, L, k, n)
    run = _UBC_RUN[0]
    key = None
    if (run is not None and context['usable'] and context['uv'] is run['uv']
            and context['generation'] == run['generation']):
        token = _ubc_find(context, b)
        if token is not None:
            offset, valid, stored, rank = token
            width = int(run['uv'][0].shape[1])
            main = offset == 0 and valid == stored == n == 4096
            tail = (offset == 4096 and valid == width - 4096
                    and stored == n == ((valid + 63) // 64) * 64)
            if (L == 6 and k == rank == 320 and tuple(b.shape) == (k, n)
                    and str(b.dtype) == 'float32' and (main or tail)):
                key = (run['generation'], 'astra_rp_v320', offset, valid, L, k, n, 'float32')
    if key is None:
        return _UBC_PARENT_B(b, L, k, n)
    stats = run['stats']
    stats['eligible'] += 1
    found = run['cache'].get(key)
    if found is not None:
        stats['hits'] += 1
        slot = key[1] + ':' + str(key[2])
        stats['hit_slots'][slot] = stats['hit_slots'].get(slot, 0) + 1
        _ASTRA_RP_STATS['cache_hits'] += 1
        return found
    stats['misses'] += 1
    result = _UBC_PARENT_B(b, L, k, n)
    count = 7 ** L * (k // (1 << L)) * (n // (1 << L))
    size = 4 * count
    if (len(run['cache']) < 4 and run['bytes'] + size <= _UBC_LIMIT
            and tuple(result.shape) == (7,) * L + (k // (1 << L), n // (1 << L))
            and str(result.dtype) == 'float32'):
        run['cache'][key] = fnp.copy(result)
        run['bytes'] += size
        stats['copies'] += 1
        stats['stores'] += 1
        stats['copied_bytes'] += size
        stats['copy_elements'] += count
        stats['peak_cache_bytes'] = max(stats['peak_cache_bytes'], run['bytes'])
        stats['max_resident_bytes'] = stats['peak_cache_bytes']
        _ASTRA_RP_STATS['cache_stores'] += 1
        _ASTRA_RP_STATS['copied_elements'] += count
    else:
        stats['uncached'] += 1
        _ASTRA_RP_STATS['uncached'] += 1
    return result


# Keep the outer current UBcluster wrapper and its direct/RC ownership intact.
_ASTRA_CLUSTER_PARENT_B = _astra_rp_forward_b


def _predict_binary(*args, **kwargs):
    if not _ASTRA_RP_ON:
        return _ASTRA_RP_PARENT_PREDICT(*args, **kwargs)
    if _ASTRA_RP_PREFIX[0] is not None:
        raise RuntimeError('recon-prefix stale prediction context')
    _ASTRA_RP_STATS.clear()
    _ASTRA_RP_STATS.update(prefix_calls=0, events=[], cache_hits=0, cache_stores=0,
                           copied_elements=0, uncached=0, retired_early_bytes=0)
    try:
        return _ASTRA_RP_PARENT_PREDICT(*args, **kwargs)
    finally:
        _ASTRA_RP_PREFIX[0] = None
