# BEGIN OPUS-ENGINEER INNER-SIZE COVARIANCE SANDWICH (a60 psdm v1, 2026-10-11; rounding-level, NOT bit-exact; bill down; needs XRQS and the
# dead cut DCV in 'pad' mode). With the dead cut, the symmetric product W_a' (C_aa W_a) ran at inner size 1024: W's rows in roster order
# times the cut operand C_aa W_a padded with 1024 - m zero rows (the specialised product's fixed keys). Here it runs at its true inner
# size m (a multiple of 64 below 1024) with the same structure: the three outer 256-row tiles (_psd_sym_tile takes any inner size) and
# the two P4 diagonal groups at depth 5, their left / right sides built for inner m (the output sides are the parent's). The zero rows'
# share of the product (1 - m / 1024 of its leaf and inner-side transforms), the padding copy, and the gather of the 1024 - m cut rows
# of W go away. The same product in exact arithmetic; the tree splits the inner index differently, so rounding differs.
_PSDM_ON = True
_PSDM_SIDES = {}                            # (kind, rows, cols) -> _PSDBSide at depth 5 for an inner size m < 1024
_PSDM_STATS = dict(inner=0, padded=0)


def _psdm_side(kind, rows, cols):
    key = (kind, rows, cols)
    found = _PSDM_SIDES.get(key)
    if found is None or found._borrowed_arena is not _SCRATCH[0]:
        region = _IPT_REGION[6]
        start = (0, region[0], region[0] + region[1])[kind]
        found = _PSDM_SIDES[key] = _PSDBSide(_SCRATCH[0], start, kind, 4, 5, rows, cols)
    return found


def _psdm_fits(m):
    if not (_PSDM_ON and _AB_ON and _WINOGRAD and m % 64 == 0 and 64 <= m < 1024 and _psdb_fits()):
        return False
    region = _IPT_REGION[6]
    return (all(_ipt_fits(6, rows, m, cols) for rows, _k, cols in _PSD_SHAPES[:3])
            and all(4 * 7 ** 5 * (rows // 32) * (m // 32) <= region[0] and 4 * 7 ** 5 * (m // 32) * (cols // 32) <= region[1]
                    for _offset, rows, cols in _PSDB_GROUPS)
            and all(_PSD5_DEPTHS.get((2, rows, cols)) == 5 for _offset, rows, cols in _PSDB_GROUPS))


def _psdm_group(A4, B4, out, offset, rows, cols, m):
    # = the XRQS P4 group (_rqs_psdb_group) with the left / right sides for inner size m
    keep = []
    _TR_LAST[0] = None
    try:
        a = A4[:, offset:offset + rows, :]
        b = B4[:, :, offset:offset + cols]
        keep.extend((a, b))
        sa = _psdm_side(0, rows, m)
        sb = _psdm_side(1, m, cols)
        sc = _psdb_side(2, rows, cols)
        if not (sa.L == sb.L == sc.L and sa.P == sb.P == sc.P):
            raise RuntimeError('inner-size P4 group: side depths')
        _parts, views, lows, quads, _rest = _rqs_get('psdb', (rows, cols, sc.P, sc.L), lambda: _rqs_psdb_parts(rows, cols, sc.P, sc.L))
        sa.forward(a, keep)
        sb.forward(b, keep)
        keep.append(fnp.matmul(sa.T, sb.T, out=sc.T))
        for f, x, y, o in sc.ops: keep.append(f(x, y, out=o))
        for lo, hi in sc.basis: keep.append(fnp.subtract(hi, lo, out=lo))
        tl, tr, bl, br = quads
        Z = sc.chunks[1]
        keep.append(fnp.add(Z[0], Z[5], out=Z[5])); keep.append(fnp.add(Z[0], Z[2], out=tl))
        keep.append(fnp.add(Z[5], Z[4], out=Z[5])); keep.append(fnp.add(Z[5], Z[3], out=tr))
        keep.append(fnp.add(Z[5], Z[1], out=br)); keep.append(fnp.add(Z[4], Z[6], out=Z[4])); keep.append(fnp.subtract(br, Z[4], out=bl))
        for p, part in enumerate(views):
            start = p * 256 + offset
            upper = out[start:start + rows, start:start + cols]
            keep.extend((upper, fnp.copyto(upper, part)))
            if cols > rows:
                lower = out[start + rows:start + cols, start:start + rows]
                keep.extend((lower, fnp.copyto(lower, lows[p])))
    finally:
        _TR_LAST[0] = None
        keep.clear()


def _psdm_sym_upper(A, B, m):
    # A (1024, m) = W_a', B (m, 1024) = C_aa W_a: the upper blocks of A B as _psdb_sym_upper computes them for inner size 1024
    out = fnp.empty((1024, 1024), dtype=fnp.float32)
    for start in (0, 256, 512):
        end = start + 256
        _psd_sym_tile(A, B, out, start, 256, end, 1024 - end, False)
    A4 = fnp.reshape(A, (4, 256, m))
    Bv = fnp.reshape(B, (m, 4, 256))
    B4 = fnp.moveaxis(Bv, 1, 0)
    for offset, rows, cols in _PSDB_GROUPS:
        _psdm_group(A4, B4, out, offset, rows, cols, m)
    _PSDM_STATS['inner'] += 1
    return out


def _dc_cov_star(w, C, A, gate, floor, retain32=False):
    # = the dead-cut sandwich (a44 dcv v3) with the 'pad' branch at inner size m when it fits (_psdm_fits), the padded product otherwise
    roster = _dcv_roster(gate, floor)
    if roster is None:
        return _flms_cov_star(w, C, A, retain32)
    if w.ndim != 2 or C.shape != (w.shape[0], w.shape[0]) or A.shape != (w.shape[0], w.shape[0]):
        raise ValueError('FLMS covariance/star shapes')
    if str(w.dtype) != 'float32':
        raise TypeError('FLMS weights must be FP32')
    hold = _E2Hold()
    wt = w.T
    C32 = _ps32(C)
    A32 = A if str(A.dtype) == 'float32' else A.astype(fnp.float32)
    wt2 = wt * wt
    P1, T2 = _ss_factors(wt, wt2, A32, hold)
    m = _DCV_STATS['m']
    if _DCV_SYM == 'pad' and int(w.shape[0]) == 1024 and int(w.shape[1]) == 1024 and _psdm_fits(m):
        live = hold(roster[:m])
        wa = hold(w[live])
        first = _mm(hold(C32[fnp.ix_(live, live)]), wa)
        s2 = _tsr_mm(T2, P1.T)
        cov = _psdm_sym_upper(hold(wa.T), first, m)
    elif _DCV_SYM == 'pad':        # roster = the whole order; W's rows in that order, the cut operand padded with zero rows
        wp = hold(w[roster])
        wa = hold(wp[:m])
        live = hold(roster[:m])
        first = _mm(hold(C32[fnp.ix_(live, live)]), wa)
        s2 = _tsr_mm(T2, P1.T)
        cov = _sym_upper(hold(wp.T), hold(fnp.concatenate((first, hold(fnp.zeros((int(w.shape[0]) - m, int(w.shape[1])), dtype=fnp.float32))))))
        _PSDM_STATS['padded'] += 1
    else:
        wa = hold(w[roster])
        first = _mm(hold(C32[fnp.ix_(roster, roster)]), wa)
        s2 = _tsr_mm(T2, P1.T)
        cov = _dcv_tiles(hold(wa.T), first, m) if _DCV_SYM == 'tiles' else _sym_upper(hold(wa.T), first)
    Cpre = cov.astype(fnp.float64)
    sub = s2 / 3
    diagonal = fnp.sum(hold(P1 * T2), axis=1, dtype=fnp.float64)
    result = sub.astype(fnp.float64)
    fnp.fill_diagonal(result, diagonal)
    _SS_STATS['planes'] += 1
    if retain32:
        fnp.fill_diagonal(sub, diagonal)
        return Cpre, result, cov, sub
    return Cpre, result


_PB_CACHES = tuple(_PB_CACHES) + ('_PSDM_SIDES',)
# END OPUS-ENGINEER INNER-SIZE COVARIANCE SANDWICH
