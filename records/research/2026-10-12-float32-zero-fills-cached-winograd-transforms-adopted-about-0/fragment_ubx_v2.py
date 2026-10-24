

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
