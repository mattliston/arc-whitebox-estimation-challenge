# Source-owned, complete-product HK dispatch. Appended after setup_integration.
# No frame inspection, array-storage introspection, input copies or value tests.
_HK_SCOPE = 0
_HK_BLOCK = 0
_HK_MM_DEPTH = 0
_HK_OWNER = None
_HK_STATS = {}


def _hk_granted():
    return _W5_PORT_SCOPE is not None and _W5_PORT_SCOPE.get('grant') is not None


def _hk_cat(parts):
    if any(p is None for p in parts):
        return None
    result = tuple(k for part in parts for k in part)
    return result if len(result) <= 64 else None


def _hk_leaf_plan(L, m, k, n, ppc=True):
    # D6X precedes the final P6 wrapper only on the cached-product path.
    if (ppc and L == 5 and _D6X_ON and _d6x_gain(m, k, n) >= _D6X_MIN
            and _ipt_fits(6, m, k, n)):
        L = 6
    if not _ipt_fits(L, m, k, n):
        return None
    if L == 5 and _P6_ON:
        plan = _p6_plan(m, k, n)
        if plan is not None:
            return ((6,) + tuple(plan),)
    return ((L, m, k, n),)


def _hk_ks_keys(plan):
    M, N, sizes = plan
    return tuple((6, M, q, N) for q in sizes)


def _hk_push6_keys(m, k, n):
    if not (64 <= m <= 1024 and m % 64 == 0 and 0 < k <= 1024
            and k % 64 == n % 64 == 0 and 1024 < n <= 8192):
        return None
    widths = (4096, n - 4096) if n > 4096 else (n,)
    return _hk_cat([_hk_leaf_plan(6, m, k, w) for w in widths])


def _hk_schedule(m, k, n, level=0):
    """Source-pinned scalar routing only; original functions execute the work.

    Unknown ordinary/old-read/D7 paths reject the complete owner. Scalar plan
    helpers are the parent's unchanged shape-only rules and keep their caches.
    """
    if level > 8 or min(m, k, n) <= 0 or m > 1024:
        return None
    if _D7SQ_ON and (m, k, n) == (1024, 1024, 1024):
        return None
    if (_D7_ON and m in _D7_DIMS and (k in _D7_DIMS or k in _D7_KPAD)
            and n > 1024):
        return None
    if _KS_ON and k > 1024 and n <= 1024:
        plan = _ks_plan(m, k, n)
        if plan is not None and len(plan[2]) >= 2:
            return _hk_ks_keys(plan)
    # Original read6 is the distinct two-pass family, never the HK leaf family.
    if _read6_take(m, k, n) and _rule_depth('read', m, k, n) == 5:
        return None
    if n > _PACKED_TILE:
        return _hk_cat([_hk_schedule(m, k, min(_PACKED_TILE, n-j), level+1)
                        for j in range(0, n, _PACKED_TILE)])
    if k > _PACKED_TILE:
        return _hk_cat([_hk_schedule(m, min(_PACKED_TILE, k-j), n, level+1)
                        for j in range(0, k, _PACKED_TILE)])
    if m % 32 or k % 32 or n % 32:
        return None
    if max(m, k, n) <= 1024:
        L, dims = _square_schedule(m, k, n)
        return _hk_leaf_plan(L, *dims) if L in (3, 4, 5, 6) else None
    rd = _rule_depth('push' if n > 1024 else 'read', m, k, n)
    if rd == 0:
        return None
    if n > 1024:
        if rd == 6 and m % 64 == 0 and (k % 64 or n % 64):
            return _hk_push6_keys(m, (k+63)//64*64, (n+63)//64*64)
        if rd == 6:
            return _hk_push6_keys(m, k, n)
        if rd == 5 and m <= 1024 and k <= 1024:
            widths = (4096, n-4096) if n > 4096 else (n,)
            return _hk_cat([_hk_leaf_plan(5, m, k, w) for w in widths])
        return _hk_leaf_plan(4, m, k, n)
    if rd == 5:
        plan = _ks_plan(m, k, n) if _KS_ON else None
        if plan is not None and _ipt_fits(5, m, k, n):
            return _hk_ks_keys(plan)
        return _hk_leaf_plan(5, m, k, n, ppc=False)
    return _hk_leaf_plan(4, m, k, n)


_HK_PARENT_LEAF = _P6_OLD_IPAB
_HK_PARENT_CP = _cp_d6_shared
_HK_PARENT_KS = _ks_product
_HK_PARENT_MM = _mm


def _hk_leaf(a, b, out, L):
    owner = _HK_OWNER
    if owner is None:
        return _HK_PARENT_LEAF(a, b, out, L)
    key = (int(L), int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))
    pos = owner['seen']
    if (pos >= len(owner['keys']) or key != owner['keys'][pos]
            or tuple(b.shape) != (key[2], key[3])
            or tuple(out.shape) != (key[1], key[3])
            or str(a.dtype) != str(b.dtype) or str(a.dtype) != str(out.dtype)
            or str(a.dtype) != 'float32'):
        raise RuntimeError('HK complete-owner resolved schedule changed')
    owner['seen'] = pos + 1
    product = _hk_get(key)
    _TR_LAST[0] = None
    try:
        if product is None:
            _HK_STATS['independent_binary_products'] += 1
            return _HK_PARENT_LEAF(a, b, out, L)
        _HK_STATS['HKproducts'] += 1
        return product(a, b, out)
    finally:
        _TR_LAST[0] = None


# The existing P6 wrapper still owns its native padding/copy/crop operations.
# Its resolved alias is the sole independent alt-basis leaf implementation.
_P6_OLD_IPAB = _hk_leaf


def _cp_d6_shared(a, b, out):
    if _HK_OWNER is None:
        return _HK_PARENT_CP(a, b, out)
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    if not (n > 4096 and _ipt_fits(6, m, k, 4096)
            and _ipt_fits(6, m, k, n-4096)):
        raise RuntimeError('HK admitted shared packet no longer fits')
    # Complete replacement: no local TA survives an arena write. Preserve the
    # exact UBC source-owned slice lineage for any unselected original B tree.
    for j0, j1 in ((0, 4096), (4096, n)):
        bj, oj = _ubc_slice(b, j0, j1), out[:, j0:j1]
        _BH_HOLD.append((bj, oj))
        _hk_leaf(a, bj, oj, 6)
    _BH_HOLD.clear()
    return out


def _ks_product(a, b, out, m, k, n, plan):
    if _HK_OWNER is None:
        return _HK_PARENT_KS(a, b, out, m, k, n, plan)
    M, N, sizes = plan
    hold = []
    try:
        if M != m:
            aa = fnp.zeros((M, k), dtype=fnp.float32)
            hold.append(fnp.copyto(aa[:m, :], a))
        else:
            aa = a
        if N != n:
            bb = fnp.zeros((k, N), dtype=fnp.float32)
            hold.append(fnp.copyto(bb[:, :n], b))
        else:
            bb = b
        total = out if (M, N) == (m, n) else fnp.empty((M, N), dtype=fnp.float32)
        part = fnp.empty((M, N), dtype=fnp.float32) if len(sizes) > 1 else None
        k0 = 0
        for j, q in enumerate(sizes):
            aj, bj = aa[:, k0:k0+q], bb[k0:k0+q, :]
            _hk_leaf(aj, bj, total if j == 0 else part, 6)
            if j:
                hold.append(fnp.add(total, part, out=total))
            k0 += q
        if k0 != k:
            raise RuntimeError('HK complete K split width changed')
        if total is not out:
            hold.append(fnp.copyto(out, total[:m, :n]))
        return out
    finally:
        hold.clear()
        _BH_HOLD.clear()


def _mm(a, b):
    global _HK_MM_DEPTH, _HK_OWNER
    root = _HK_MM_DEPTH == 0
    admitted = False
    if root and _HK_SCOPE and not _HK_BLOCK and not _hk_granted():
        if (_HK_READY[0] and _WINOGRAD and _AB_ON and _SCRATCH is not None
                and len(a.shape) == len(b.shape) == 2
                and str(a.dtype) == str(b.dtype) == 'float32'
                and int(a.shape[1]) == int(b.shape[0])):
            keys = _hk_schedule(int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))
            if keys is not None and any(_hk_get(key) is not None for key in keys):
                _HK_OWNER = {'keys': keys, 'seen': 0}
                admitted = True
                _TR_LAST[0] = None
                _HK_STATS['selectedowners'] += 1
            else:
                _HK_STATS['schedulefallback'] += 1
        else:
            _HK_STATS['setupfallback'] += 1
    elif root:
        _HK_STATS['scopefallback'] += 1
    _HK_MM_DEPTH += 1
    try:
        result = _HK_PARENT_MM(a, b)
        if admitted and _HK_OWNER['seen'] != len(_HK_OWNER['keys']):
            raise RuntimeError('HK complete-owner sequence did not finish')
        return result
    finally:
        _HK_MM_DEPTH -= 1
        if admitted:
            _HK_OWNER = None
            _TR_LAST[0] = None
            _BH_HOLD.clear()


def _hk_scope_wrap(function):
    def scoped(*args, **kwargs):
        global _HK_SCOPE
        _HK_SCOPE += 1
        try:
            return function(*args, **kwargs)
        finally:
            _HK_SCOPE -= 1
    return scoped


# Source induction: both inputs to every _mm here come from weight/state
# owners, fresh gathers/pointwise/QR allocations, or earlier fresh _mm outputs.
# Neither function calls the pair-map arena producer. Predictor callsites pass
# the physical state, not kernel scratch. This grant is not a shape inference.
_shared_old384_push = _hk_scope_wrap(_shared_old384_push)
_cp_read_young_sq = _hk_scope_wrap(_cp_read_young_sq)


def _hk_block_wrap(function):
    def blocked(*args, **kwargs):
        global _HK_BLOCK
        _HK_BLOCK += 1
        try:
            return function(*args, **kwargs)
        finally:
            _HK_BLOCK -= 1
    return blocked


# No nested matrix call inside an original live multi-product owner may start
# an HK owner, even if a future source path reaches it from an admitted scope.
for _hk_block_name in ('_mm_batch', '_bb_product', '_w5_port_product',
                       '_w5_port_product_body', '_ipt_leaf_unwind_ab',
                       '_d7_product', '_d7_product_kpad'):
    globals()[_hk_block_name] = _hk_block_wrap(globals()[_hk_block_name])
del _hk_block_name


_HK_PARENT_PREDICT = _predict_binary


def _predict_binary(*args, **kwargs):
    global _HK_STATS, _HK_SCOPE, _HK_BLOCK, _HK_MM_DEPTH, _HK_OWNER
    if _HK_SCOPE or _HK_BLOCK or _HK_MM_DEPTH or _HK_OWNER is not None:
        raise RuntimeError('HK nested prediction or unretired scope')
    _HK_STATS = dict(selectedowners=0, HKproducts=0,
                     independent_binary_products=0, scopefallback=0,
                     schedulefallback=0, setupfallback=0, retired=0)
    try:
        return _HK_PARENT_PREDICT(*args, **kwargs)
    finally:
        _TR_LAST[0] = None
        _HK_SCOPE = _HK_BLOCK = _HK_MM_DEPTH = 0
        _HK_OWNER = None
        _HK_STATS['retired'] = 1
