# Fixed deeper leaf-sum reads; separate cache/layout in the existing arena.
_D7R_CACHES = (_CapLRU(), _CapLRU(), _CapLRU())
for _d7r_cache in _D7R_CACHES:
    _CAP_BY_ID[id(_d7r_cache)] = 24
_D7R_REGION = (7**7 * 128,) * 3
_D7R_STATS = {'calls': 0, 'chunks': 0}
_D7R_PLANS = {}
_D7R_OLD_KS = _ks_product


def _d7r_side(kind, L, rows, cols, region):
    cache = _D7R_CACHES[kind]
    key = (L, rows, cols, tuple(region), id(_SCRATCH[0]))
    found = cache.get(key)
    if found is not None:
        return found
    u = 1 << L
    r, c = rows // u, cols // u
    assert rows % u == cols % u == 0
    start = (0, region[0], region[0] + region[1])[kind]
    assert 7**L * r * c <= region[kind]
    T = _ipt_view(_SCRATCH[0], start, (7,)*L + (r, c))
    data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
    chunks, levels = [None, None], {}
    for j in range(2, L+1):
        nr = L-j
        Tj = T[(data,)*nr] if nr else T
        chunks.append(fnp.unstack(Tj, axis=nr))
        levels[j] = (Tj, nr)
    lv1, groups = _lvf_rs_level1(kind, L, T, (r, c))
    chunks[1], pair = lv1[0]
    # This cache always binds the same fused arithmetic, including cold calls.
    ops = _lvf_ops(kind, L, levels, chunks, groups[0])
    basis = _rs_basis_views(kind, L, T)
    acc = None
    if kind == 2:
        count = 7**L*r*c
        assert 2*count <= region[2]
        acc = _ipt_view(_SCRATCH[0], start+count, (7,)*L + (r, c))
    return _cache_put(cache, key, (T, chunks, pair, ops, basis, acc))


def _d7r_forward(a, kind, L, rows, cols, region):
    T, chunks, pair, ops, basis, _ = _d7r_side(kind, L, rows, cols, region)
    h = _BH_HOLD.append
    if kind == 0:
        _rs_load_a(a, chunks[1], pair, L, rows, cols)
        for lo, hi in basis:
            h(fnp.add(lo, hi, out=lo))
    else:
        _rs_load_b(a, chunks[1], pair, L, rows, cols)
        for lo, hi in basis:
            h(fnp.subtract(hi, lo, out=lo))
    for f, x, y, o in ops:
        h(f(x, y, out=o))
    return T


def _d7r_product(a, b, out, plan, L=7, region=_D7R_REGION):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    M, K, N, sizes = plan
    assert int(b.shape[0]) == k and tuple(out.shape) == (m, n)
    assert sum(sizes) == K and sum(region) <= int(_SCRATCH[0].shape[0])
    assert M >= m and K >= k and N >= n
    _TR_LAST[0] = None
    hold = []
    try:
        if (M, K) == (m, k):
            aa = a
        else:
            aa = fnp.zeros((M, K), dtype=fnp.float32)
            hold.append(fnp.copyto(aa[:m, :k], a))
        if (K, N) == (k, n):
            bb = b
        else:
            bb = fnp.zeros((K, N), dtype=fnp.float32)
            hold.append(fnp.copyto(bb[:k, :n], b))
        TC, Z1, _, ops, basis, acc = _d7r_side(2, L, M, N, region)
        h = _BH_HOLD.append
        k0 = 0
        for j, q in enumerate(sizes):
            TA = _d7r_forward(aa[:, k0:k0+q], 0, L, M, q, region)
            TB = _d7r_forward(bb[k0:k0+q, :], 1, L, q, N, region)
            if j == 0:
                h(fnp.matmul(TA, TB, out=TC))
            else:
                h(fnp.matmul(TA, TB, out=acc))
                h(fnp.add(TC, acc, out=TC))
            k0 += q
        if (M, N) == (m, n):
            _ks_unwind(TC, Z1, ops, basis, out, L, M, N)
        else:
            padded = fnp.empty((M, N), dtype=fnp.float32)
            _ks_unwind(TC, Z1, ops, basis, padded, L, M, N)
            hold.append(fnp.copyto(out, padded[:m, :n]))
        return out
    finally:
        _BH_HOLD.clear()
        hold.clear()
        _TR_LAST[0] = None


def _d7r_bill(L, M, K, N):
    # Parent fitted scalar op model, with one final unwind. The -MN leaf
    # terms and additions of subsequent leaves cancel down to one -MN term.
    S = ((7/4)**L-1)/.75
    return (2*(7/8)**L*M*K*N - (7/4)**L*M*N
            + .7669*M*K*S + .7668*K*N*S + 1.5213*M*N*S
            + 2.6307*(M*K+K*N+M*N))


def _d7r_plan(m, k, n, parent_plan):
    key = (m, k, n)
    if key in _D7R_PLANS:
        return _D7R_PLANS[key]
    plan = None
    if (k >= 4096 and k % 64 == 0 and 0 < m <= 1024 and 0 < n <= 1024
            and m % 32 == n % 32 == 0):
        M, K, N = (m+127)//128*128, (k+127)//128*128, (n+127)//128*128
        if (M == (m+63)//64*64 and N == (n+63)//64*64
                and min(M, N) >= 640 and (M, N) == tuple(parent_plan[:2])):
            pad = (M*K if (M,K)!=(m,k) else 0) + (K*N if (K,N)!=(k,n) else 0)
            basepad = (M*k if M!=m else 0) + (k*N if N!=n else 0)
            b7 = _d7r_bill(7,M,K,N) + 1.35*pad
            b6 = _d7r_bill(6,M,k,N) + 1.35*basepad
            if b7 <= .98*b6:
                blocks = K//128
                count = (blocks+15)//16
                q, r = divmod(blocks, count)
                sizes = tuple([(q+1)*128]*r + [q*128]*(count-r))
                assert max(sizes) <= 2048
                plan = (M, K, N, sizes)
    _D7R_PLANS[key] = plan
    return plan


def _ks_product(a, b, out, m, k, n, plan):
    p7 = _d7r_plan(m, k, n, plan)
    if p7 is not None:
        _D7R_STATS['calls'] += 1
        _D7R_STATS['chunks'] += len(p7[3])
        return _d7r_product(a, b, out, p7)
    return _D7R_OLD_KS(a, b, out, m, k, n, plan)
