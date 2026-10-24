# BEGIN OPUS-ENGINEER PERSISTENT SCRATCH (a56 rqs v1, 2026-10-11; lead 06:26Z: residual cuts at zero bill; predictions byte-identical,
# bill identical; compose AFTER XRQ2 and HZ7R). Scratch arrays that a kernel allocates on every call with the same shape are made once per
# shape and kept, together with the views the kernel takes of them: the symmetric sandwich's tile outputs (_psd_sym_tile) and P4 group
# outputs (_psdb_group's parts, their split and the lower-copy views, the unwind's quadrant views). Every element a call reads is written
# by that call first (each is the output of a tree product: the last unwind writes all four quadrants), so a kept buffer holds exactly what
# a fresh one would. A kept output also keeps the quadrant views the tree product's last unwind takes of it (_l3o_last). The arithmetic,
# its operands and its order are unchanged; requests per call fall by the allocation and those views.
_RQS = {}
_RQS_CAP = 6                       # entries per role; a miss allocates exactly as before
_RQS_OUTV = {}                     # id(kept output) -> (output, {(L, m, n): (C0, bl, br, Cr, C1)})
_RQS_STATS = dict(hits=0, misses=0, outv=0)


def _rqs_get(role, key, make):
    full = (role, key)
    got = _RQS.get(full)
    if got is not None:
        _RQS_STATS['hits'] += 1
        return got
    same = [k for k in _RQS if k[0] == role]
    if len(same) >= _RQS_CAP:
        old = _RQS.pop(same[0])
        _RQS_OUTV.pop(id(old[0]), None)
    got = _RQS[full] = make()
    _RQS_STATS['misses'] += 1
    return got


def _rqs_out(buf):
    # a kept output: _l3o_last keeps its quadrant views (the registry holds the object, so its id stays its own)
    _RQS_OUTV[id(buf)] = (buf, {})
    return buf


_RQS_L3O_PARENT = _l3o_last


def _l3o_last(slot, Z, out, L, m, n):
    # = the parent with the output's quadrant views made once for a kept output (the same three view requests at its first use)
    owned = _RQS_OUTV.get(id(out))
    if owned is None or owned[0] is not out:
        return _RQS_L3O_PARENT(slot, Z, out, L, m, n)
    views = owned[1].get((L, m, n))
    if views is None:
        u = 1 << L
        Cr = fnp.reshape(out, (2,) * L + (m // u,) + (2,) * L + (n // u,))
        C0, C1 = fnp.unstack(Cr, axis=0)
        bl, br = fnp.unstack(C1, axis=L)
        views = owned[1][(L, m, n)] = (C0, bl, br, Cr, C1)
    else:
        _RQS_STATS['outv'] += 1
    C0, bl, br = views[0], views[1], views[2]
    h = _BH_HOLD.append
    p05, p23 = slot[1] if slot[1] is not None else _l3o_pairs(slot, L)
    h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[5], Z[4], out=Z[5])); h(fnp.add(p05, p23, out=C0))
    h(fnp.add(Z[5], Z[1], out=br)); h(fnp.add(Z[4], Z[6], out=Z[4])); h(fnp.subtract(br, Z[4], out=bl))
    return out


def _rqs_tile(rows, cols, skip):
    part = _rqs_out(fnp.empty((rows, cols), dtype=fnp.float32))
    if cols > skip:
        off = part[:, skip:]
        return part, off, off.T
    return part, None, None


def _psd_sym_tile(A, B, out, row, rows, col, cols, diagonal):
    # = the parent's tile with its part buffer and the transposed view of its off-diagonal columns kept per (rows, cols, skip)
    skip = rows if diagonal else 0
    part, _off, off_t = _rqs_get('psdt', (rows, cols, skip), lambda: _rqs_tile(rows, cols, skip))
    keep = []
    a = A[row:row + rows, :]
    b = B[:, col:col + cols]
    keep.extend((a, b))
    _ipt_product_ab(a, b, part, 6)
    upper = out[row:row + rows, col:col + cols]
    keep.extend((upper, fnp.copyto(upper, part)))
    if cols > skip:
        lower = out[col + skip:col + cols, row:row + rows]
        keep.extend((lower, fnp.copyto(lower, off_t)))


def _rqs_psdb_parts(rows, cols, P, L):
    parts = fnp.empty((P, rows, cols), dtype=fnp.float32)
    views = fnp.unstack(parts, axis=0)
    offs = [part[:, rows:] for part in views] if cols > rows else []
    lows = [off.T for off in offs]
    u = 1 << L
    V = fnp.reshape(parts, (P,) + (2,) * L + (rows // u,) + (2,) * L + (cols // u,))
    C0, C1 = fnp.unstack(V, axis=1)
    tl, tr = fnp.unstack(C0, axis=L + 1)
    bl, br = fnp.unstack(C1, axis=L + 1)
    return parts, views, lows, (tl, tr, bl, br), (offs, V, C0, C1)


def _rqs_psdb_group(A4, B4, out, offset, rows, cols):
    # = the P4 group (the parent _PSD5_PARENT_GROUP) with the output parts, their split, the lower-copy views and the unwind's quadrant
    # views kept per (rows, cols, P, L); the unwind is _PSDBSide.unwind on those views
    keep = []
    _TR_LAST[0] = None
    try:
        a = A4[:, offset:offset + rows, :]
        b = B4[:, :, offset:offset + cols]
        keep.extend((a, b))
        sa = _psdb_side(0, rows, 1024)
        sb = _psdb_side(1, 1024, cols)
        sc = _psdb_side(2, rows, cols)
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


_PSD5_PARENT_GROUP = _rqs_psdb_group
_PB_CACHES = tuple(_PB_CACHES) + ('_RQS', '_RQS_OUTV')
# END OPUS-ENGINEER PERSISTENT SCRATCH
