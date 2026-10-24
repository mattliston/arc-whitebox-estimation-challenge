

# BEGIN FABLE-ENGINEER PUSH64 (E-12, 2026-09-22): depth-enabling zero padding for wide pushes. A 1024-row push whose k or n is not a
# multiple of 64 cannot take the six-level kernel and runs at depth 4. Here k and n are zero-padded to multiples of 64 whenever the
# depth rule's own value (bill + 750000 per request) of the padded six-level product, plus the padding copies (m*k + k*n), beats the
# unpadded choice. The units and physics are unchanged; the product changes only by rounding (padded zeros enter the transforms).
def _push_rule_value(m, k, n, depth):
    if depth == 0:
        return 2 * m * k * n - m * n + 750000
    packets = (4096, n - 4096) if depth == 6 and n > 4096 else (n,)
    growth = (7 / 4) ** depth
    bill = sum(2 * m * k * nc * (7 / 8) ** depth - m * nc * growth + (7 / 3) * (m * k + k * nc + m * nc) * (growth - 1) for nc in packets)
    return bill + 750000 * (18 * depth + 14) * len(packets)


def _mm(a, b):
    if not _WINOGRAD or str(a.dtype) != 'float32' or str(b.dtype) != 'float32':
        return a @ b
    m, k = (int(a.shape[0]), int(a.shape[1]))
    n = int(b.shape[1])
    if m == 0 or k == 0 or n == 0:
        return fnp.zeros((m, n), dtype=fnp.float32)
    if n > _PACKED_TILE:
        return fnp.concatenate([_mm(a, b[:, j:min(j + _PACKED_TILE, n)]) for j in range(0, n, _PACKED_TILE)], axis=1)
    if k > _PACKED_TILE:
        out = _mm(a[:, :_PACKED_TILE], b[:_PACKED_TILE, :])
        for j in range(_PACKED_TILE, k, _PACKED_TILE):
            end = min(j + _PACKED_TILE, k)
            out = out + _mm(a[:, j:end], b[j:end, :])
        return out
    if m % 32 or k % 32 or n % 32:
        return a @ b
    if max(m,k,n)<=1024:
        return _square_product(a,b,m,k,n)
    rd = _rule_depth('push' if n > 1024 else 'read', m, k, n)
    kp, npad = k, n
    if n > 1024 and m == 1024 and rd != 6:
        k64, n64 = (k + 63) // 64 * 64, (n + 63) // 64 * 64
        if k64 <= 1024 and n64 <= 8192 and (_push_rule_value(m, k64, n64, 6) + m * k + k * n
                                             < _push_rule_value(m, k, n, rd)):
            rd, kp, npad = 6, k64, n64
    if rd == 0:
        return a @ b
    if kp != k:
        aa = fnp.zeros((m, kp), dtype=fnp.float32)
        fnp.copyto(aa[:, :k], a)
    else:
        aa = a
    if kp != k or npad != n:
        bb = fnp.zeros((kp, npad), dtype=fnp.float32)
        fnp.copyto(bb[:k, :n], b)
    else:
        bb = b
    out = fnp.empty((m, npad), dtype=fnp.float32)
    if n > 1024:
        out = _packed_product_4(aa, bb, out, _SCRATCH[1]) if rd == 6 else _packed_product_cached(aa, bb, out, _SCRATCH[1], 4)
    else:
        out = (_packed_read5(aa, bb, out, _SCRATCH[2]) if rd == 5
               else _packed_product_cached(aa, bb, out, _SCRATCH[2], 4))
    return out if npad == n else out[:, :n]
# END FABLE-ENGINEER PUSH64
