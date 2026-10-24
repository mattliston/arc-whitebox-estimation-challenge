# v3: preserve the parent lazy-quad binding; count both new owned reshapes.
# Experimental exact UBC admission: build a B tree in its retained storage.
# No new rank/shape admission. Only the existing rank320 U/V depth6 packets.
# Every load, basis change and level operation keeps its operands/order/mode.
# The arena tree -> independent cache copy is removed, not made uncounted.
# Native identity, bill, request and footprint gates have NOT been run.
_ASTRA_UBOWN_ON = True
_ASTRA_UBOWN_PARENT_B = _ipt_forward_b_ab
_ASTRA_UBOWN_STATS = dict(stores=0, avoided_copy_elements=0, binding_reshape_elements=0, owned_quad_elements=0, net_bill_credit=0, owned_bytes=0)


def _astra_ubown_bind_v1(L, k, n, fused):
    """The existing B-side view layout, rooted in a fresh independent tree."""
    r, c = k // (1 << L), n // (1 << L)
    T = fnp.empty((7,) * L + (r, c), dtype=fnp.float32)
    data = slice(0, 4)
    chunks, levels = [None, None], {}
    for j in range(2, L + 1):
        nr = L - j
        Tj = T[(data,) * nr] if nr else T
        chunks.append(fnp.unstack(Tj, axis=nr))
        levels[j] = (Tj, nr)
    lv1, groups = _lvf_rs_level1(1, L, T, (r, c))
    chunks[1], slot = lv1[0]
    ops = _lvf_ops(1, L, levels, chunks, groups[0]) if fused else None
    return T, chunks, slot, ops, _rs_basis_views(1, L, T)


def _astra_ubown_forward_v1(b, L, k, n, side):
    # Exact body of the base B transform, with its supplied side entry.
    TB, Y1, B02, ops, basis = side
    h = _BH_HOLD.append
    _rs_load_b(b, Y1[1], B02, L, k, n)
    for lo, hi in basis:
        h(fnp.subtract(hi, lo, out=lo))
    if ops is not None:
        for f, x, y, o in ops:
            h(f(x, y, out=o))
        return TB
    for j in range(1, L + 1):
        Y = Y1[j]
        h(fnp.add(Y[1], Y[0], out=Y[5]))
        h(fnp.subtract(Y[5], Y[2], out=Y[6]))
        h(fnp.subtract(Y[3], Y[5], out=Y[4]))
    return TB


def _ipt_forward_b_ab(b, L, k, n):
    if not _ASTRA_UBOWN_ON:
        return _ASTRA_UBOWN_PARENT_B(b, L, k, n)
    run, context = _UBC_RUN[0], _UBC_CONTEXT[0]
    if (run is None or context is None or not context['usable']
            or _UBX_RC[0] is not None or _UBX_CTX[0] is not None
            or context['uv'] is not run['uv'] or context['generation'] != run['generation']
            or run['owner'] is None or int(run['owner'].shape[0]) != 320):
        return _ASTRA_UBOWN_PARENT_B(b, L, k, n)
    token = _ubc_find(context, b)
    if token is None:
        return _ASTRA_UBOWN_PARENT_B(b, L, k, n)
    offset, valid, stored, rank = token
    leg, width = context['leg'], int(run['uv'][0].shape[1])
    main = offset == 0 and valid == stored == n == 4096
    tail = (leg in ('u', 'v') and offset == 4096 and valid == width - 4096
            and stored == n == ((valid + 63) // 64) * 64)
    if not (leg in ('u', 'v') and L == 6 and k == rank == (256 if leg == 'u' else 320)
            and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32' and (main or tail)):
        return _ASTRA_UBOWN_PARENT_B(b, L, k, n)
    key = (run['generation'], leg, offset, valid, L, k, n, 'float32')
    count = 7 ** L * (k // (1 << L)) * (n // (1 << L))
    size = count * 4
    if key in run['cache'] or len(run['cache']) >= 4 or run['bytes'] + size > _UBC_LIMIT:
        return _ASTRA_UBOWN_PARENT_B(b, L, k, n)
    # Preserve the parent's L7 retirement before the first late B tree.
    if _UBX_RCACHE['cache'] and _UVP_CUR[0] is not None and _UVP_CUR[0] >= 10:
        _UBX_RCACHE.update(owner=None, run=None, cache={}, bytes=0)
    # The existing own-prebind decision determines fused vs component mode.
    reference = _ipt_side(1, L, k, n)
    if reference[2][1] is None:
        _l3_quad(reference[2], 1, L)  # the baseline would bind this on its first load
    side = _astra_ubown_bind_v1(L, k, n, reference[3] is not None)
    result = _astra_ubown_forward_v1(b, L, k, n, side)
    run['cache'][key] = result
    run['bytes'] += size
    stats = run['stats']
    stats['eligible'] += 1
    stats['misses'] += 1
    stats['stores'] += 1
    # No copy occurred: copies/copied_bytes/copy_elements stay unchanged.
    stats['peak_cache_bytes'] = max(stats['peak_cache_bytes'], run['bytes'])
    stats['max_resident_bytes'] = stats['peak_cache_bytes']
    _ASTRA_UBOWN_STATS['stores'] += 1
    _ASTRA_UBOWN_STATS['avoided_copy_elements'] += count
    binding = 7 * 4 ** (L - 1) * (k // (1 << L)) * (n // (1 << L))
    _ASTRA_UBOWN_STATS['binding_reshape_elements'] += binding
    _ASTRA_UBOWN_STATS['owned_quad_elements'] += k * n
    _ASTRA_UBOWN_STATS['net_bill_credit'] += count - binding - k * n
    _ASTRA_UBOWN_STATS['owned_bytes'] += size
    return result

# Additional Astra UBC/UBX/L7 exact-copy cluster. Experimental; do not promote.
# Uses each cache's existing cap and existing prediction/owner retirement.
_ASTRA_CLUSTER_ON = True
_ASTRA_CLUSTER_PARENT_SHARED = _shared_old384_mm
_ASTRA_CLUSTER_PARENT_B = _ipt_forward_b_ab
_ASTRA_CLUSTER_PARENT_VIEWS = _hz_views
_ASTRA_CLUSTER_PARENT_LEAF = _hz_leaf
_ASTRA_CLUSTER_CTX = [None]
_ASTRA_CLUSTER_STATS = dict(direct_products=0, direct_hits=0, direct_stores=0,
    direct_uncached=0, direct_avoided_forward_elements=0, direct_new_binding_elements=0,
    existing_ubx_owned=0, existing_ubx_copy_elements=0, existing_ubx_binding_elements=0,
    rc_owned=0, rc_copy_elements=0, rc_binding_elements=0)


def _astra_cluster_owned_b(b, L, k, n):
    # Preserve the original B-wrapper retirement even when called directly by UBX.
    if (_UBX_RC[0] is None and _UBX_RCACHE['cache']
            and _UVP_CUR[0] is not None and _UVP_CUR[0] >= 10):
        _UBX_RCACHE.update(owner=None, run=None, cache={}, bytes=0)
    # Both layouts perform the same lazy parent-quad binding. The owned copy
    # additionally pays its own level-1 and quad reshapes, accounted below.
    reference = _ipt_side(1, L, k, n)
    if reference[2][1] is None:
        _l3_quad(reference[2], 1, L)
    side = _astra_ubown_bind_v1(L, k, n, reference[3] is not None)
    return _astra_ubown_forward_v1(b, L, k, n, side)


def _astra_cluster_counts(L, k, n):
    rc = (k // (1 << L)) * (n // (1 << L))
    copy = 7 ** L * rc
    binding = (7 * 4 ** (L - 1) + 4 ** L) * rc
    # Warm parent forward: input reshape/copy, L basis changes, 3 ops/level.
    forward = (7 ** L + 4 ** L + L * 4 ** (L - 1)) * rc
    return copy, binding, forward


def _shared_old384_mm(left, right):
    ctx = _UBX_CTX[0]
    prior = _ASTRA_CLUSTER_CTX[0]
    if (not _ASTRA_CLUSTER_ON or ctx is None or prior is not None
            or int(left.shape[0]) > 1024 or int(right.shape[0]) % 64
            or int(right.shape[1]) % 64 or tuple(right.shape) != tuple(ctx['tag'][3:5])):
        return _ASTRA_CLUSTER_PARENT_SHARED(left, right)
    current = dict(ubx=ctx, offset=0, pending=None, enabled=True)
    _ASTRA_CLUSTER_CTX[0] = current
    _ASTRA_CLUSTER_STATS['direct_products'] += 1
    try:
        return _ASTRA_CLUSTER_PARENT_SHARED(left, right)
    finally:
        current['pending'] = None
        _ASTRA_CLUSTER_CTX[0] = prior


def _astra_cluster_leaf_view(v, PB, hzr, hzc):
    # Same basic-slice groups as _ubx_strip_views; retain all parent metadata,
    # particularly ASZ's accumulator identity. A and output views are untouched.
    full = slice(None)
    mats, _rowfill, _colfill = _hz_groups(hzr, False, hzc)
    cb = int(PB.shape[-1]) - int(hzc)
    bviews = []
    for s, _rf, _kf, cf in mats:
        second = slice(None, cb) if (hzc and not cf) else full
        bviews.append(PB[s] if second is full else PB[s, ..., full, second])
    out = dict(v)
    out['mats'] = [(x, y, o) for (x, _old, o), y in zip(v['mats'], bviews)]
    return out


def _hz_views(M, q, N, hzr, hzc):
    entry = _ASTRA_CLUSTER_PARENT_VIEWS(M, q, N, hzr, hzc)
    # Metadata only; no array operation and no change to ASZ's own marker.
    for v in entry:
        v['_astra_cluster_shape'] = (M, q, N, bool(hzr), bool(hzc))
    return entry


def _hz_leaf(v):
    current = _ASTRA_CLUSTER_CTX[0]
    pending = current['pending'] if current is not None else None
    shape = v.get('_astra_cluster_shape')
    if pending is not None and shape is not None:
        entry, q, N = pending
        M, vq, vN, hzr, hzc = shape
        if vq != q or vN != N or hzc:
            raise RuntimeError('Astra inner-cache leaf shape mismatch')
        vk = (id(v), M, q, N, hzr, hzc)
        got = entry['views'].get(vk)
        if got is None:
            got = (v, _astra_cluster_leaf_view(v, entry['PB'], hzr, hzc))
            entry['views'][vk] = got
        elif got[0] is not v:
            raise RuntimeError('Astra inner-cache leaf identity mismatch')
        current['pending'] = None
        return _ASTRA_CLUSTER_PARENT_LEAF(got[1])
    return _ASTRA_CLUSTER_PARENT_LEAF(v)


def _ipt_forward_b_ab(b, L, k, n):
    if not _ASTRA_CLUSTER_ON:
        return _ASTRA_CLUSTER_PARENT_B(b, L, k, n)
    # L7 reconstruction consumes the returned B tree. Its exact existing key,
    # sequence index, owner lifetime and cap are preserved.
    rc = _UBX_RC[0]
    if rc is not None:
        j = rc['n']
        key = rc['tag'] + (j, L, k, n, tuple(int(x) for x in b.shape), str(b.dtype))
        count, binding, _forward = _astra_cluster_counts(L, k, n)
        if (key not in _UBX_RCACHE['cache'] and str(b.dtype) == 'float32'
                and tuple(b.shape) == (k, n) and k % (1 << L) == n % (1 << L) == 0
                and _UBX_RCACHE['bytes'] + 4 * count <= _UBX_RC_LIMIT):
            rc['n'] = j + 1
            result = _astra_cluster_owned_b(b, L, k, n)
            _UBX_RCACHE['cache'][key] = result
            _UBX_RCACHE['bytes'] += 4 * count
            _UBX_RC_STATS['stores'] += 1
            _ASTRA_CLUSTER_STATS['rc_owned'] += 1
            _ASTRA_CLUSTER_STATS['rc_copy_elements'] += count
            _ASTRA_CLUSTER_STATS['rc_binding_elements'] += binding
            return result
        return _ASTRA_CLUSTER_PARENT_B(b, L, k, n)
    current = _ASTRA_CLUSTER_CTX[0]
    if (current is None or not current['enabled'] or current['ubx'] is not _UBX_CTX[0]
            or _UBC_CONTEXT[0] is not None):
        return _ASTRA_CLUSTER_PARENT_B(b, L, k, n)
    ctx = current['ubx']
    offset = current['offset']
    if not (L == 6 and k % 64 == n % 64 == 0 and n == ctx['tag'][4]
            and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32'
            and offset + k <= ctx['tag'][3]):
        current['enabled'] = False
        current['pending'] = None
        return _ASTRA_CLUSTER_PARENT_B(b, L, k, n)
    current['offset'] += k
    key = ('astra-direct-v1',) + ctx['tag'] + (offset, L, k, n)
    cache = _UBX_STATE['cache']
    entry = cache.get(key)
    count, binding, forward = _astra_cluster_counts(L, k, n)
    if entry is None:
        if _UBX_STATE['bytes'] + 4 * count > _UBX_LIMIT:
            _ASTRA_CLUSTER_STATS['direct_uncached'] += 1
            current['pending'] = None
            return _ASTRA_CLUSTER_PARENT_B(b, L, k, n)
        # Preserve the old B wrapper's early L7 retirement even on this path.
        if _UBX_RCACHE['cache'] and _UVP_CUR[0] is not None and _UVP_CUR[0] >= 10:
            _UBX_RCACHE.update(owner=None, run=None, cache={}, bytes=0)
        result = _astra_cluster_owned_b(b, L, k, n)
        entry = cache[key] = dict(PB=result, views={})
        _UBX_STATE['bytes'] += 4 * count
        _UBX_STATS['stores'] += 1
        _ASTRA_CLUSTER_STATS['direct_stores'] += 1
        _ASTRA_CLUSTER_STATS['direct_new_binding_elements'] += binding
    else:
        # The uncached baseline still touches/binds this arena entry on every
        # forward. Preserve its LRU position and any lazy quadrant bind; only
        # the numerical forward work disappears, so the counter stays exact
        # even if the arena view entry was evicted between inner reads.
        reference = _ipt_side(1, L, k, n)
        if reference[2][1] is None:
            _l3_quad(reference[2], 1, L)
        _UBX_STATS['hits'] += 1
        _ASTRA_CLUSTER_STATS['direct_hits'] += 1
        _ASTRA_CLUSTER_STATS['direct_avoided_forward_elements'] += forward
    current['pending'] = (entry, k, n)
    return entry['PB']


# Exact parent UBX loop except owned B misses and same-PB leaf views.
def _ubx_ks_product(a, b, out, m, k, n, plan):
    # This route owns its chunk keys. Hits/owned misses skip the B wrapper,
    # so it must never share the sequential offset of direct bypass routes.
    _ac_direct = _ASTRA_CLUSTER_CTX[0]
    if _ac_direct is not None:
        _ac_direct['enabled'] = False
        _ac_direct['pending'] = None
    # = the v8 _hz_ks_product (HZD's parent) with the cached B trees of a UBC-X product
    M, N, sizes, hzr, hzc, K = plan
    ctx = _UBX_CTX[0]
    cache = _UBX_STATE['cache']
    keys = [None] * len(sizes)
    if ctx is not None and not hzc:
        keys = [ctx['tag'] + (j, q, N) for j, q in enumerate(sizes)]   # the B tree depends on the chunk only; leaf views per (M, strips)
    found = [cache.get(x) if x is not None else None for x in keys]
    allhit = all(f is not None for f in found)
    hold = []
    al, bl = M // 64 - 1, N // 64 - 1
    me, ne = (64 * al if hzr else m), (64 * bl if hzc else n)
    if hzr:
        aa = fnp.zeros((64, al + 1, K), dtype=fnp.float32)
        am = fnp.reshape(a[:me], (64, al, k))
        hold.append((am, fnp.copyto(aa[:, :al, :k], am), fnp.copyto(aa[0::2, al, :k], a[me:])))
        _HZ_STATS['row'] += 1
    elif K != k:
        aa = fnp.zeros((M, K), dtype=fnp.float32)
        hold.append(fnp.copyto(aa[:, :k], a))
    else:
        aa = a
    if allhit:
        bb = None
    elif hzc:
        bbt = fnp.zeros((64, bl + 1, K), dtype=fnp.float32)
        bt = b.T
        bm = fnp.reshape(bt[:ne], (64, bl, k))
        hold.append((bt, bm, fnp.copyto(bbt[:, :bl, :k], bm), fnp.copyto(bbt[0::2, bl, :k], bt[ne:])))
        bb = fnp.transpose(bbt, (2, 0, 1))
        hold.append(bbt)
        _HZ_STATS['col'] += 1
    elif K != k:
        bb = fnp.zeros((K, N), dtype=fnp.float32)
        hold.append(fnp.copyto(bb[:k], b))
    else:
        bb = b
    TC, Z1, _pair, ops, basis = _ipt_side(2, 6, M, N)
    h = _BH_HOLD.append
    k0 = 0
    for j, q in enumerate(sizes):
        ak = aa[..., k0:k0 + q]
        h(ak)
        TA = _ipt_forward_a_ab(ak, 6, M, q)
        hit = found[j]
        if hit is not None:
            PB = hit['PB']
            _UBX_STATS['hits'] += 1
            if hzr or hzc:
                _UBX_STATS['strip_hits'] += 1
                vk = (M, hzr, hzc)
                views = hit['views'].get(vk)
                if views is None:
                    views = hit['views'][vk] = _ubx_strip_views(PB, M, q, N, hzr, hzc)
                _hz_leaf(views[1 if j else 0])
            else:
                h(fnp.matmul(TA, PB, out=(_ks_acc(M, N) if j else TC)))
        else:
            bk = bb[k0:k0 + q]
            h(bk)
            _ac_count, _ac_binding, _ac_forward = _astra_cluster_counts(6, q, N)
            _ac_owned = (_ASTRA_CLUSTER_ON and keys[j] is not None
                         and _UBX_STATE['bytes'] + 4 * _ac_count <= _UBX_LIMIT)
            TB = (_astra_cluster_owned_b(bk, 6, q, N) if _ac_owned
                  else _ipt_forward_b_ab(bk, 6, q, N))
            if hzr or hzc:
                _ac_views = _hz_views(M, q, N, hzr, hzc)[1 if j else 0]
                _hz_leaf(_astra_cluster_leaf_view(_ac_views, TB, hzr, hzc) if _ac_owned else _ac_views)
            else:
                h(fnp.matmul(TA, TB, out=(_ks_acc(M, N) if j else TC)))
            if keys[j] is not None:
                count = 1
                for x in TB.shape:
                    count *= int(x)
                if _UBX_STATE['bytes'] + 4 * count <= _UBX_LIMIT:
                    PB = TB if _ac_owned else fnp.copy(TB)
                    if _ac_owned:
                        _ASTRA_CLUSTER_STATS['existing_ubx_owned'] += 1
                        _ASTRA_CLUSTER_STATS['existing_ubx_copy_elements'] += _ac_count
                        _ASTRA_CLUSTER_STATS['existing_ubx_binding_elements'] += _ac_binding
                    cache[keys[j]] = dict(PB=PB, views={})
                    _UBX_STATE['bytes'] += 4 * count
                    _UBX_STATS['stores'] += 1
                else:
                    _UBX_STATS['uncached'] += 1
        if j:
            h(fnp.add(TC, _ks_acc(M, N), out=TC))
        k0 += q
    if M == m and N == n:
        _ks_unwind(TC, Z1, ops, basis, out, 6, M, N)
    else:
        o = fnp.empty((M, N), dtype=fnp.float32)
        _ks_unwind(TC, Z1, ops, basis, o, 6, M, N)
        if hzr and hzc:
            o4 = fnp.reshape(o, (64, al + 1, 64, bl + 1))
            p0 = fnp.reshape(o4[:, :al, :, :bl], (me, ne)); p1 = fnp.reshape(o4[:, :al, 0::2, bl], (me, 32))
            p2 = fnp.reshape(o4[0::2, al, :, :bl], (32, ne))
            hold.append((o4, p0, p1, p2, fnp.copyto(out[:me, :ne], p0), fnp.copyto(out[:me, ne:], p1),
                         fnp.copyto(out[me:, :ne], p2), fnp.copyto(out[me:, ne:], o4[0::2, al, 0::2, bl])))
        elif hzr:
            o3 = fnp.reshape(o, (64, al + 1, N))
            p0 = fnp.reshape(o3[:, :al, :n], (me, n))
            hold.append((o3, p0, fnp.copyto(out[:me], p0), fnp.copyto(out[me:], o3[0::2, al, :n])))
        else:
            o3 = fnp.reshape(o, (M, 64, bl + 1))
            p0 = fnp.reshape(o3[:m, :, :bl], (m, ne))
            hold.append((o3, p0, fnp.copyto(out[:, :ne], p0), fnp.copyto(out[:, ne:], o3[:m, 0::2, bl])))
    _BH_HOLD.clear()
    del hold
    _HZ_STATS['ks'] += 1
    return out
