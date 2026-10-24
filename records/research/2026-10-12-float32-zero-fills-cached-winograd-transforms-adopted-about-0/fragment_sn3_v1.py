

# BEGIN SCOUT-NATIVE F32 ZERO FILLS v1 (2026-10-12; byte-identical): the strip zero fills of the tree kernel (_hz_leaf's fill slices and
# _hzd_load's two zero strips) passed a Python 0.0, which flopscope bills at the float64 rate (2 flops an element, measured) although
# the targets are float32; a float32 0-d zero bills 1 (measured). The same zeros land in the same places. About -0.034% of bill
# (census of the S50 twin: _hz_leaf:10212 0.0637%, _hzd_load:13112/13114 0.0045%, all at the float64 rate).
_F32Z = [None]


def _f32z():
    z = _F32Z[0]
    if z is None:
        z = _F32Z[0] = fnp.zeros((), dtype=fnp.float32)
    return z


def _hz_leaf(v):
    h = _BH_HOLD.append
    for x, y, o in v['mats']:
        h(fnp.matmul(x, y, out=o))
    z = _f32z()
    for f in v['fills']:
        h(fnp.copyto(f, z))


def _hzd_load(quad, x, L, rm, cm, hr, hc):
    # = the parent's _hzd_load with float32 zero fills
    main, rs, cs, cr, rz, cz = _hzd_parts(quad, L, rm, cm, hr, hc)
    h = _BH_HOLD.append
    u, s = 1 << L, 1 << (L - 1)
    R, C = u * rm, u * cm
    xm = x[:R, :C]
    vm = fnp.reshape(xm, (2,) * L + (rm,) + (2,) * L + (cm,))
    h((xm, vm)); h(fnp.copyto(main, vm))
    if hr:
        xr = x[R:, :C]
        vr = fnp.reshape(xr, (2,) * (L - 1) + (2,) * L + (cm,))
        h((xr, vr)); h(fnp.copyto(rs, vr))
    if hc:
        xc = x[:R, C:]
        vc = fnp.reshape(xc, (2,) * L + (rm,) + (2,) * (L - 1))
        h((xc, vc)); h(fnp.copyto(cs, vc))
    if hr and hc:
        xx = x[R:, C:]
        vx = fnp.reshape(xx, (2,) * (L - 1) + (2,) * (L - 1))
        h((xx, vx)); h(fnp.copyto(cr, vx))
    z = _f32z()
    if hr:
        h(fnp.copyto(rz, z))
    if hc:
        h(fnp.copyto(cz, z))
# END SCOUT-NATIVE F32 ZERO FILLS v1


# BEGIN SCOUT-NATIVE UBC-X v2 (2026-10-12; meant byte-identical: the same transforms of the same values, reused). The old tier's
# inner-read legs in u/v form (_uvp_inner, layers 10-14: inner = Dv (F_v/2)^T and Su F_u^T) take lam on the LEFT operand, so the
# right operands F_v/2^T and F_u^T of the UBC-tracked prefix frame hold the same values at every layer from the L10 join to layer 14
# (the L13 join recreates the uv arrays from the same unmodified base F; UBC's owner/generation stay). Their k-split products (v8
# _hz_ks_product, K padded from 5856) recomputed the depth-6 B-transform of each chunk every layer (20 ops, 24-29 Mflop each). Here
# the first layer keeps each chunk's B tree in a counted copy, and later layers skip the B padding and transform: a product without
# strips multiplies the copy directly; a half-zero strip product (rows an odd multiple of 32) gets leaf views over the copy, built once
# (the parent's group slices of the B tree), with the parent's A and output views. Only the prefix frame is cached (the L13 recent
# group has one later layer: its copy would cost as much as the transform it saves).
_UBX_ON = True
_UBX_LIMIT = 600000000                      # bytes of kept B trees
_UBX_CTX = [None]
_UBX_STATE = {'run': None, 'cache': {}, 'bytes': 0}
_UBX_STATS = dict(products=0, hits=0, stores=0, uncached=0, skipped=0, strip_hits=0)


def _ubx_product(left, right, uv, leg):
    run = _UBC_RUN[0]
    if not _UBX_ON or run is None or _UBX_CTX[0] is not None or run.get('uv') is not uv or run.get('owner') is None:
        _UBX_STATS['skipped'] += 1
        return _shared_old384_mm(left, right)
    if _UBX_STATE['run'] is not run:      # prediction-local, as UBC
        _UBX_STATE['run'], _UBX_STATE['bytes'] = run, 0
        _UBX_STATE['cache'].clear()
    _UBX_CTX[0] = dict(tag=(leg, id(run['owner']), run['generation'], int(right.shape[0]), int(right.shape[1])))
    _UBX_STATS['products'] += 1
    try:
        return _shared_old384_mm(left, right)
    finally:
        _UBX_CTX[0] = None


def _ubx_strip_views(PB, M, q, N, hzr, hzc):
    # the parent's two leaf-view dicts (output tree, leaf-sum buffer) with the B side's group slices taken of PB instead of the arena
    full = slice(None)
    mats, _rowfill, _colfill = _hz_groups(hzr, False, hzc)
    cb = N // 64 - (1 if hzc else 0)
    bviews = []
    for s, _rf, _kf, cf in mats:
        second = slice(None, cb) if (hzc and not cf) else full
        bviews.append(PB[s] if second is full else PB[s, ..., full, second])
    out = []
    for v in _hz_views(M, q, N, hzr, hzc):
        out.append(dict(mats=[(x, y, o) for (x, _old, o), y in zip(v['mats'], bviews)], fills=v['fills']))
    return out


def _ubx_ks_product(a, b, out, m, k, n, plan):
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
            TB = _ipt_forward_b_ab(bk, 6, q, N)
            if hzr or hzc:
                _hz_leaf(_hz_views(M, q, N, hzr, hzc)[1 if j else 0])
            else:
                h(fnp.matmul(TA, TB, out=(_ks_acc(M, N) if j else TC)))
            if keys[j] is not None:
                count = 1
                for x in TB.shape:
                    count *= int(x)
                if _UBX_STATE['bytes'] + 4 * count <= _UBX_LIMIT:
                    PB = fnp.copy(TB)     # counted copy: one flop per element; the arena tree is reused by the next product
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


_UBX_PARENT_V8 = _HZD_PARENT_KS


def _ubx_v8_dispatch(a, b, out, m, k, n, plan):
    if _UBX_CTX[0] is None:
        return _UBX_PARENT_V8(a, b, out, m, k, n, plan)
    return _ubx_ks_product(a, b, out, m, k, n, plan)


_HZD_PARENT_KS = _ubx_v8_dispatch


def _uvp_inner(left_sq, lam_cols, uv):
    """(L^2 lam) F^T with F in u/v form: rows x rank. = the parent's arithmetic; the two products through _ubx_product."""
    Fu, Fv2, blocks, ru = uv
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
    inner = _ubx_product(Dv, Fv2.T, uv, 'vt')
    inner_u = _ubx_product(Su, Fu.T, uv, 'ut')
    _BH_HOLD.append(fnp.add(inner[:, :ru], inner_u, out=inner[:, :ru]))
    _BH_HOLD.clear()
    del Su, Dv, inner_u
    return inner


_UBX_PARENT_PREDICT = _predict_binary


def _predict_binary(*args, **kwargs):
    try:
        return _UBX_PARENT_PREDICT(*args, **kwargs)
    finally:
        _UBX_STATE['cache'].clear()
        _UBX_STATE['bytes'] = 0
        _UBX_STATE['run'] = None
        _UBX_CTX[0] = None
# END SCOUT-NATIVE UBC-X v2


# BEGIN SCOUT-NATIVE UBC-X v3 ADD-ON (2026-10-12; meant byte-identical; appended AFTER UBC-X v2): two more B-transforms that are the
# same values at later layers. (1) UBC's u leg tail (F_u columns 4096.., [256 x 1792] at depth 6, layers 10-14): UBC admitted the v tail
# only and stored at most 3 trees; here the u tail is admitted too and the store cap is 4 (UBC's own limit _UBC_LIMIT unchanged).
# (2) The L7 frame's reconstruction U[rows] F at layers 7-9 (materialized frame, rank 384; F = [384 x 6144], two column packets
# 4096 + 2048, leaf reads the returned tree): its packets' B trees are kept from layer 7 and returned at 8-9; dropped when the frame
# changes. UBC's statistics dict gains a 'u:4096' slot.
_UBX_RC = [None]
_UBX_RCACHE = {'owner': None, 'run': None, 'cache': {}, 'bytes': 0}
_UBX_RC_STATS = dict(products=0, hits=0, stores=0, uncached=0)
_UBX_RC_LIMIT = 300000000
_UBX3_PARENT_RECON = _UVP_ORIG_RECON


def _ubx_recon7(U, state):
    run = _UBC_RUN[0]
    if (not _UBX_ON or run is None or _UBX_RC[0] is not None or state['lazy'] is not None or _UVP_CUR[0] not in (7, 8, 9)
            or int(U.shape[0]) % 64 or int(state['F'].shape[0]) % 64 or int(state['F'].shape[1]) % 64):
        return _UBX3_PARENT_RECON(U, state)
    owner = state['F']
    if _UBX_RCACHE['owner'] is not owner or _UBX_RCACHE['run'] is not run:
        _UBX_RCACHE.update(owner=owner, run=run, cache={}, bytes=0)   # a new frame or prediction: the previous trees go
    _UBX_RC[0] = dict(tag=('recon7', id(owner), tuple(int(x) for x in owner.shape)), n=0)
    _UBX_RC_STATS['products'] += 1
    try:
        return _UBX3_PARENT_RECON(U, state)
    finally:
        _UBX_RC[0] = None


_UVP_ORIG_RECON = _ubx_recon7


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
                    and offset == 0 and valid == stored == n == (256 if leg == 'ut' else 320)
                    and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32'):
                key = (run['generation'], leg, offset, valid, L, k, n, 'float32')
            elif (L == 6 and k == rank == (256 if leg == 'u' else 320)
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


_UBX3_PARENT_PREDICT = _predict_binary


def _predict_binary(*args, **kwargs):
    try:
        return _UBX3_PARENT_PREDICT(*args, **kwargs)
    finally:
        _UBX_RCACHE.update(owner=None, run=None, cache={}, bytes=0)
        _UBX_RC[0] = None
# END SCOUT-NATIVE UBC-X v3 ADD-ON


# BEGIN SCOUT-NATIVE HZD-A REUSE v1 (2026-10-12; meant byte-identical): the transform-reuse rule of E-73 (the resident forward-A tree is
# returned when the SAME operand object comes back with the same shape in the very next forward-A) extended to the direct strip loads
# (_hzd_forward_a). The young push and the newborn-common push pass the same gathered left operand W'[kept, previous] to two
# products; when both run as half-zero strip products (layers 7, 14, 15 on the S50 twin's 7100136) the second recomputed the same
# tree. E-73's bookkeeping is used unchanged: every audited clearing function still clears the resident key, an ordinary forward-A
# replaces it, and the parent _hzd_forward_a itself starts by computing (it clears the key), so a reuse needs nothing else to have
# touched the A region in between.
_HZDA_PARENT = _hzd_forward_a
_HZDA_STATS = dict(reused=0, computed=0)


def _hzd_forward_a(x, M, K, hr, hc):
    key = ('hzd', M, K, hr, hc)
    last = _TR_LAST[0]
    if last is not None and last[0] is x and last[1] == key:
        _HZDA_STATS['reused'] += 1
        return _ipt_side(0, 6, M, K)[0]
    TA = _HZDA_PARENT(x, M, K, hr, hc)
    _TR_LAST[0] = (x, key)
    _HZDA_STATS['computed'] += 1
    return TA
# END SCOUT-NATIVE HZD-A REUSE v1
