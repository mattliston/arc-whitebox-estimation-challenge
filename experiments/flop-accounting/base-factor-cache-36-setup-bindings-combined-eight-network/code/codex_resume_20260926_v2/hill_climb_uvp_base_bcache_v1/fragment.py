# Exact, prediction-local UVP base B-transform cache. Source edits in build.py
# convey actual slice/padding ancestry; no inferred offsets or value hashes.
_UBC_ENABLED = True
_UBC_LIMIT = 512000000
_UBC_RUN = [None]
_UBC_CONTEXT = [None]
_UBC_STATS = {}


def _ubc_drop(run):
    run['cache'].clear()
    run['bytes'] = 0
    run['owner'] = None
    run['uv'] = None
    run['generation'] = None


def _ubc_find(context, value):
    for owner, token in context['ancestry']:
        if owner is value:
            return token
    return None


def _ubc_record(context, value, token):
    if len(context['ancestry']) >= 16:
        context['usable'] = False
        return
    context['ancestry'].append((value, token))


def _ubc_slice(b, start, stop):
    # Exactly the source's original one native basic slice.
    piece = b[:, start:stop]
    context = _UBC_CONTEXT[0]
    if context is not None and context['usable']:
        token = _ubc_find(context, b)
        if token is not None:
            offset, valid, stored, rank = token
            if (0 <= start < stop <= stored and tuple(b.shape) == (rank, stored)
                    and tuple(piece.shape) == (rank, stop - start)):
                kept = max(0, min(valid, stop) - start)
                _ubc_record(context, piece, (offset + start, kept, stop - start, rank))
    return piece


def _ubc_padding(b, padded, k, n, K, N):
    # Called ONLY after original zeros + copyto. Row padding is not admitted.
    context = _UBC_CONTEXT[0]
    if context is None or not context['usable'] or padded is b:
        return
    token = _ubc_find(context, b)
    if token is None:
        return
    offset, valid, stored, rank = token
    if (k == K == rank and n == stored and N >= n
            and tuple(b.shape) == (k, n) and tuple(padded.shape) == (K, N)
            and str(b.dtype) == str(padded.dtype) == 'float32'):
        _ubc_record(context, padded, (offset, valid, N, rank))


_UBC_PARENT_STATE = _uvp_state


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
    generation = (tuple(blocks), int(ru), tuple(owner.shape))
    if not (ru == 256 and 4096 < P <= 6144 and tuple(Fu.shape) == (256, P)
            and tuple(Fv.shape) == (320, P) and tuple(owner.shape) == (320, 2 * P)
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


def _ubc_product(left, right, uv, leg):
    run = _UBC_RUN[0]
    slot = 0 if leg == 'u' else 1
    if (run is None or run['uv'] is not uv or right is not uv[slot]
            or _UBC_CONTEXT[0] is not None):
        return _shared_old384_mm(left, right)
    rank, width = map(int, right.shape)
    context = dict(leg=leg, uv=uv, generation=run['generation'],
                   ancestry=[(right, (0, width, width, rank))], usable=True)
    _UBC_CONTEXT[0] = context
    try:
        return _shared_old384_mm(left, right)
    finally:
        context['ancestry'].clear()
        _UBC_CONTEXT[0] = None


_UBC_PARENT_B = _ipt_forward_b_ab


def _ipt_forward_b_ab(b, L, k, n):
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
            tail = (leg == 'v' and offset == 4096 and valid == width - 4096
                    and stored == n == ((valid + 63) // 64) * 64)
            if (L == 6 and k == rank == (256 if leg == 'u' else 320)
                    and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32'
                    and (main or tail)):
                key = (run['generation'], leg, offset, valid, L, k, n, 'float32')
    if key is None:
        return _UBC_PARENT_B(b, L, k, n)
    run['stats']['eligible'] += 1
    found = run['cache'].get(key)
    if found is not None:
        run['stats']['hits'] += 1
        run['stats']['hit_slots'][key[1] + ':' + str(key[2])] += 1
        return found
    run['stats']['misses'] += 1
    result = _UBC_PARENT_B(b, L, k, n)
    count = 7 ** L * (k // (1 << L)) * (n // (1 << L))
    size = count * 4
    if (len(run['cache']) < 3 and run['bytes'] + size <= _UBC_LIMIT
            and tuple(result.shape) == (7,) * L + (k // (1 << L), n // (1 << L))
            and str(result.dtype) == 'float32'):
        # Ordinary counted FNP copy: one FLOP per FP32 element, independent
        # storage. Leaf/unwind reads this array and writes only its C region.
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


_UBC_PARENT_PREDICT = _predict_binary


def _predict_binary(*args, **kwargs):
    global _UBC_STATS
    if not _UBC_ENABLED or _UBC_RUN[0] is not None:
        return _UBC_PARENT_PREDICT(*args, **kwargs)
    stats = dict(generations=0, eligible=0, hits=0, misses=0, stores=0, copies=0, copy_elements=0,
                 copied_bytes=0, max_resident_bytes=0,
                 uncached=0, peak_cache_bytes=0, hit_slots={'u:0': 0, 'v:0': 0, 'v:4096': 0})
    run = dict(owner=None, uv=None, generation=None, cache={}, bytes=0, stats=stats)
    _UBC_RUN[0] = run
    try:
        return _UBC_PARENT_PREDICT(*args, **kwargs)
    finally:
        _UBC_CONTEXT[0] = None
        _ubc_drop(run)
        _UBC_RUN[0] = None
        _UBC_STATS = dict(stats, retired=True, retained_cache_bytes=0, live_cache_bytes=0)
