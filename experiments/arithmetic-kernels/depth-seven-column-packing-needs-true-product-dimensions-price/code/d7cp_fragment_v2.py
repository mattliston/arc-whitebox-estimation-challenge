

# BEGIN FABLE-ENGINEER2 DEPTH-7 COLUMN PACKETS (d7cp v2, 2026-09-25; v2 = v1 plus the k = 864 / 992 colpack products. The S28 probe
# (depth7/probe_d7cp_calls.py) showed layer 5 (every net) and layer 4 (net 37) at k = 992: the native path already zero-pads them to
# k = 1024 (_push6_padded), so depth 7 on the same padded product is pure gain; v1 skipped them (in-process parity v1: -0.11..-0.29%
# instead of ~-0.45%). v2 pads exactly as _push6_padded does (A to K columns, B to K x NP, NP = n rounded up to 128) and returns the
# leading n columns as a view. k = 928 / 960 stay at depth 6: depth 7 would pad them past the native product. v1 text follows. rounding-level, NOT bit-exact; lead's selective depth-7 frontier,
# best slice). A push product whose left factor is 896 or 1024 square-ish (m and k in {896, 1024}) and whose width n exceeds 1024 is the
# colpack push's inherited product (_push_bank_shared_newborn: _mm(wp, p[:, :old_columns]); layers 2-5 on every census net, n = 2048,
# 4096, 6144, 8128). At depth 6 it runs as <= 4096-wide packets sharing one depth-6 A tree (_cp_d6_shared). Here it runs at depth 7 as
# 2048-wide column packets sharing ONE depth-7 A tree (4096-wide depth-7 B/C trees do not fit the arena); a last packet whose width is
# not a multiple of 128 is zero-padded to one (fnp.zeros + fnp.copyto in, its leading columns copied out). Priced
# (fable_engineer2_v1/depth7/, bench_colpack.py on S28's own _mm): bill -126/-275/-425/-505 MF for n = 2048/4096/6144/8128 = -0.437% of
# the net's bill; backend about +1.6 s/net (bench_frontier.py rates) = about +3.4 s on the slowest MLP by WALL v3; about +336
# requests/net; rounding error x2.0-2.2 on these products (rel 2.8e-5 -> 6e-5). The depth-7 region (A 52.7M, B/C 105.4M floats) lies
# inside the existing 391.6M-float arena; it starts at 0 like every depth's region. Transform reuse stays exact: the forward-A goes
# through _ipt_forward_a_ab, so the resident key names depth 7, and the A tree's 52.7M floats are inside the depth-6 A range whose writers
# the reuse guard already covers. The three depth-7 keys lead the prebind list (bound and fused in setup, off budget).
_D7_ON = True
_D7_PK = 2048
_D7_DIMS = (896, 1024)
_D7_KPAD = {864: 896, 992: 1024}   # k whose native depth-6 padding (to 64) already reaches the depth-7 multiple of 128
_IPT_REGION[7] = (7 ** 7 * 64, 7 ** 7 * 128, 7 ** 7 * 128)
_PB_SINGLE = ((0, 7, 1024, 1024), (1, 7, 1024, 2048), (2, 7, 1024, 2048)) + tuple(_PB_SINGLE)
_D7_STATS = {'products': 0, 'packets': 0, 'kpad': 0}
_D7_OLD_MM = _mm


def _d7_product(a, b, m, k, n):
    if k in _D7_KPAD:
        return _d7_product_kpad(a, b, m, k, n, _D7_KPAD[k])
    out = fnp.empty((m, n), dtype=fnp.float32)
    TA = _ipt_forward_a_ab(a, 7, m, k)
    h = _BH_HOLD.append
    for j0 in range(0, n, _D7_PK):
        j1 = min(n, j0 + _D7_PK)
        w = j1 - j0
        if w % 128:
            wp = (w + 127) // 128 * 128
            bj = fnp.zeros((k, wp), dtype=fnp.float32)
            h(fnp.copyto(bj[:, :w], b[:, j0:j1]))
            oj = fnp.empty((m, wp), dtype=fnp.float32)
            TB = _ipt_forward_b_ab(bj, 7, k, wp)
            _ipt_leaf_unwind_ab(TA, TB, oj, 7, m, wp)
            h(fnp.copyto(out[:, j0:j1], oj[:, :w]))
            h((bj, oj))
        else:
            bj, oj = b[:, j0:j1], out[:, j0:j1]
            h((bj, oj))
            TB = _ipt_forward_b_ab(bj, 7, k, w)
            _ipt_leaf_unwind_ab(TA, TB, oj, 7, m, w)
        _D7_STATS['packets'] += 1
    _BH_HOLD.clear()
    _D7_STATS['products'] += 1
    return out


def _d7_product_kpad(a, b, m, k, n, K):
    NP = (n + 127) // 128 * 128
    hold = []
    aa = fnp.zeros((m, K), dtype=fnp.float32)
    hold.append(fnp.copyto(aa[:, :k], a))
    bb = fnp.zeros((K, NP), dtype=fnp.float32)
    hold.append(fnp.copyto(bb[:k, :n], b))
    out = fnp.empty((m, NP), dtype=fnp.float32)
    TA = _ipt_forward_a_ab(aa, 7, m, K)
    h = _BH_HOLD.append
    for j0 in range(0, NP, _D7_PK):
        j1 = min(NP, j0 + _D7_PK)
        bj, oj = bb[:, j0:j1], out[:, j0:j1]
        h((bj, oj))
        TB = _ipt_forward_b_ab(bj, 7, K, j1 - j0)
        _ipt_leaf_unwind_ab(TA, TB, oj, 7, m, j1 - j0)
        _D7_STATS['packets'] += 1
    _BH_HOLD.clear()
    del hold
    _D7_STATS['products'] += 1
    _D7_STATS['kpad'] += 1
    return out if NP == n else out[:, :n]


def _mm(a, b):
    if (_D7_ON and _AB_ON and _WINOGRAD and _SCRATCH is not None and len(a.shape) == 2 and len(b.shape) == 2
            and str(a.dtype) == 'float32' and str(b.dtype) == 'float32'):
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
        if m in _D7_DIMS and (k in _D7_DIMS or k in _D7_KPAD) and n > 1024 and int(b.shape[0]) == k:
            return _d7_product(a, b, m, k, n)
    return _D7_OLD_MM(a, b)
# END FABLE-ENGINEER2 DEPTH-7 COLUMN PACKETS
