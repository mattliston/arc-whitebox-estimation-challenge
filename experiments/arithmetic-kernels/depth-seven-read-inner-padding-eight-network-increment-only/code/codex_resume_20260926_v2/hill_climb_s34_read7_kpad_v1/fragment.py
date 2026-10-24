# One fixed cost-based extension; retained READ7's three routes remain unchanged.
_R7X_PARENT_MM = _mm
_R7X_PARENT_BIND = _pb_bind
_R7X_PLANS = {}
_R7X_A_KEYS = ((640, 1792), (640, 1920), (768, 1664), (768, 1792), (768, 1920))
_R7X_B_KEYS = ((1664, 896), (1792, 896), (1920, 896))
_R7X_C_KEYS = ((640, 896), (768, 896))


def _r7x_bill7(M, k, N):
    # Frozen depth7_inner_reads_v1 fitted scalar cost, one leaf-sum unwind.
    S = ((7 / 4) ** 7 - 1) / .75
    return (2 * (7 / 8) ** 7 * M * k * N - (7 / 4) ** 7 * M * N
            + .7669 * M * k * S + .7668 * k * N * S + 1.5213 * M * N * S
            + 2.6307 * (M * k + k * N + M * N))


def _r7x_plan(m, k, n):
    key = (m, k, n)
    if key in _R7X_PLANS:
        return _R7X_PLANS[key]
    plan = None
    if k >= 4096 and k % 64 == 0 and 0 < m <= 1024 and 0 < n <= 1024:
        parent = _ks_plan(m, k, n)
        if parent is not None:
            M, N, sizes6 = parent
            if min(M, N) >= 640 and M == (m + 127) // 128 * 128 and N == (n + 127) // 128 * 128:
                K = (k + 127) // 128 * 128
                basepad = (M * k if M != m else 0) + (k * N if N != n else 0)
                pad = (M * K if (M, K) != (m, k) else 0) + (K * N if (K, N) != (k, n) else 0)
                nch = len(sizes6)
                b6 = (sum(2 * (7 / 8) ** 6 * M * q * N + _KS_C6 * (M * q + q * N + M * N) for q in sizes6)
                      + (nch - 1) * m * n + _KS_PAD * basepad)
                b6 += (m * n if M != m or N != n else 0) - (nch - 1) * (58.88 * M * N + m * n - 7 ** 6 * (M // 64) * (N // 64))
                b7 = _r7x_bill7(M, K, N) + _KS_PAD * pad + (m * n if M != m or N != n else 0)
                if b7 <= .98 * b6:
                    blocks = K // 128
                    count = (blocks + 15) // 16
                    q, r = divmod(blocks, count)
                    sizes7 = tuple([(q + 1) * 128] * r + [q * 128] * (count - r))
                    assert max(sizes7) <= 2048 and sum(sizes7) == K
                    plan = (M, K, N, sizes7)
    _R7X_PLANS[key] = plan
    return plan


def _r7x_acc(M, N):
    found = _R7_ACC.get((M, N))
    if found is None:
        count = 7 ** 7 * (M // 128) * (N // 128)
        assert 2 * count <= _R7_REGION[2]
        start = _R7_REGION[0] + _R7_REGION[1] + count
        found = _ipt_view(_SCRATCH[0], start, (7,) * 7 + (M // 128, N // 128))
        _R7_ACC[(M, N)] = found
    return found


def _r7x_product(a, b, m, k, n, plan):
    M, K, N, sizes = plan
    saved = _r7_enter()
    try:
        hold = []
        if (M, K) != (m, k):
            aa = fnp.zeros((M, K), dtype=fnp.float32)
            hold.append(fnp.copyto(aa[:m, :k], a))
        else:
            aa = a
        if (K, N) != (k, n):
            bb = fnp.zeros((K, N), dtype=fnp.float32)
            hold.append(fnp.copyto(bb[:k, :n], b))
        else:
            bb = b
        TC, Z1, pair, ops, basis = _ipt_side(2, 7, M, N)
        acc = _r7x_acc(M, N)
        h = _BH_HOLD.append
        k0 = 0
        for j, q in enumerate(sizes):
            TA = _ipt_forward_a_ab(aa[:, k0:k0 + q], 7, M, q)
            TB = _ipt_forward_b_ab(bb[k0:k0 + q, :], 7, q, N)
            if j == 0:
                h(fnp.matmul(TA, TB, out=TC))
            else:
                h(fnp.matmul(TA, TB, out=acc))
                h(fnp.add(TC, acc, out=TC))
            k0 += q
        out = fnp.empty((M, N), dtype=fnp.float32)
        _ks_unwind(TC, Z1, ops, basis, out, 7, M, N)
        if M != m or N != n:
            cropped = fnp.empty((m, n), dtype=fnp.float32)
            hold.append(fnp.copyto(cropped, out[:m, :n]))
            return cropped
        return out
    finally:
        _r7_leave(saved)


def _mm(a, b):
    if (_SCRATCH is not None and _KS_ON and _AB_ON and _WINOGRAD
            and len(a.shape) == len(b.shape) == 2
            and str(a.dtype) == str(b.dtype) == 'float32'
            and int(a.shape[1]) == int(b.shape[0])
            and int(_SCRATCH[0].shape[0]) == 391602176):
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
        if (m, k, n) not in _R7_SHAPES:
            plan = _r7x_plan(m, k, n)
            if plan is not None:
                return _r7x_product(a, b, m, k, n, plan)
    return _R7X_PARENT_MM(a, b)


def _pb_bind(t0):
    _R7X_PARENT_BIND(t0)
    before = len(_R7_A) + len(_R7_B) + len(_R7_C) + len(_R7_ACC)
    saved = _r7_enter()
    try:
        for M, q in _R7X_A_KEYS:
            _ipt_side(0, 7, M, q)
        for q, N in _R7X_B_KEYS:
            _ipt_side(1, 7, q, N)
        for M, N in _R7X_C_KEYS:
            _ipt_side(2, 7, M, N)
            _r7x_acc(M, N)
    finally:
        _r7_leave(saved)
    _PB_LOG.append(('read7_extended_fused', len(_R7_A) + len(_R7_B) + len(_R7_C) + len(_R7_ACC) - before,
                    round(_pb_time.monotonic() - t0, 3)))
