# Generic binary Side only, extracted from pinned blocked kernel.
class _PSDBSide:
    """Exact dense tree storage; P is never mixed with recursive bit axes."""
    def __init__(self, arena, start, kind, P, L, rows, cols):
        self._borrowed_arena, self._borrowed_start = arena, int(start)
        self.kind, self.P, self.L = kind, P, L
        self.rows, self.cols = rows, cols
        u = 1 << L; r, c = rows // u, cols // u
        size = P * 7**L * r * c
        self.T = fnp.reshape(arena[start:start + size], (P,) + (7,) * L + (r, c))
        data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
        self.chunks = [None, None]; levels = {}
        for j in range(2, L + 1):
            nr = L - j
            V = self.T[(slice(None),) + (data,) * nr]
            self.chunks.append(fnp.unstack(V, axis=nr + 1))
            levels[j] = fnp.moveaxis(V, nr + 1, 0)
        nr = L - 1
        R = self.T[(slice(None),) + (data,) * nr]
        V = fnp.reshape(R, (P,) + (2, 2) * nr + (7, r, c))
        ch = 1 + 2 * nr
        ri = [1 + 2 * (L - j) for j in range(2, L + 1)]
        ci = [x + 1 for x in ri]
        if kind == 2: ri, ci = ci, ri
        G = fnp.transpose(V, [ch, 0] + ri + [ch + 1] + ci + [ch + 2])
        self.chunks[1] = fnp.unstack(G, axis=0)
        if kind == 0:
            self.pair = fnp.transpose(G[0:3:2], [1] + list(range(2, L + 2)) + [0] + list(range(L + 2, 2 * L + 2)))
        elif kind == 1:
            self.pair = fnp.moveaxis(G[0:3:2], 0, 1)
        else: self.pair = None
        self.basis = []
        for j in range(2, L + 1):
            ax = L - j
            idx = [data] * L
            if kind == 2: idx[-1] = slice(None)
            lo, hi = list(idx), list(idx)
            lo[ax], hi[ax] = (4, 6) if kind == 0 else (1, 3)
            self.basis.append((self.T[(slice(None),) + tuple(lo)], self.T[(slice(None),) + tuple(hi)]))
        self.ops = []
        if kind == 0:
            for j in range(1, L + 1):
                X = self.chunks[j]; V = G if j == 1 else levels[j]
                self.ops += [(fnp.subtract, X[4], X[0], X[5]),
                             (fnp.subtract, V[2:7:4], X[5], V[3:0:-2])]
        elif kind == 1:
            for j in range(1, L + 1):
                Y = self.chunks[j]; V = G if j == 1 else levels[j]
                self.ops += [(fnp.add, Y[1], Y[0], Y[5]),
                             (fnp.subtract, V[5:2:-2], V[2:6:3], V[6:3:-2])]
        else:
            for j in range(L, 1, -1):
                Z = self.chunks[j]; V = levels[j]
                self.ops += [(fnp.add, Z[0], V[5:1:-3], V[5::-5]),
                             (fnp.add, Z[5], Z[4], Z[5]),
                             (fnp.add, Z[5], V[3:0:-2], V[2:4]),
                             (fnp.add, Z[4], Z[6], Z[1])]

    def forward(self, packed, hold):
        P, L, rows, cols = self.P, self.L, self.rows, self.cols
        u = 1 << L
        V = fnp.reshape(packed, (P,) + (2,) * L + (rows // u,) + (2,) * L + (cols // u,))
        X = self.chunks[1]
        if self.kind == 0:
            A0, A1 = fnp.unstack(V, axis=1)
            a21, a22 = fnp.unstack(A1, axis=L + 1)
            hold.extend((V,A0,A1,a21,a22))
            hold.append(fnp.copyto(self.pair, A0)); hold.append(fnp.copyto(X[6], a22)); hold.append(fnp.add(a21, a22, out=X[4]))
            for lo, hi in self.basis: hold.append(fnp.add(lo, hi, out=lo))
        else:
            B0, B1 = fnp.unstack(V, axis=L + 2)
            b12, b22 = fnp.unstack(B1, axis=1)
            hold.extend((V,B0,B1,b12,b22))
            hold.append(fnp.copyto(self.pair, B0)); hold.append(fnp.copyto(X[3], b22)); hold.append(fnp.subtract(b22, b12, out=X[1]))
            for lo, hi in self.basis: hold.append(fnp.subtract(hi, lo, out=lo))
        for f, x, y, out in self.ops: hold.append(f(x, y, out=out))

    def unwind(self, packed, hold):
        for f, x, y, out in self.ops: hold.append(f(x, y, out=out))
        for lo, hi in self.basis: hold.append(fnp.subtract(hi, lo, out=lo))
        P, L, m, n = self.P, self.L, self.rows, self.cols; u = 1 << L
        V = fnp.reshape(packed, (P,) + (2,) * L + (m // u,) + (2,) * L + (n // u,))
        C0, C1 = fnp.unstack(V, axis=1)
        tl, tr = fnp.unstack(C0, axis=L + 1); bl, br = fnp.unstack(C1, axis=L + 1)
        hold.extend((V,C0,C1,tl,tr,bl,br))
        Z = self.chunks[1]
        hold.append(fnp.add(Z[0], Z[5], out=Z[5])); hold.append(fnp.add(Z[0], Z[2], out=tl))
        hold.append(fnp.add(Z[5], Z[4], out=Z[5])); hold.append(fnp.add(Z[5], Z[3], out=tr))
        hold.append(fnp.add(Z[5], Z[1], out=br)); hold.append(fnp.add(Z[4], Z[6], out=Z[4])); hold.append(fnp.subtract(br, Z[4], out=bl))

"""Fixed P4/D6 diagonal batching; the three outer rectangles are unchanged.

The preceding _PSDBSide is the pinned generic leading-P binary kernel, with
only its class name changed. No HK circuit, rank, routing, or map is imported.
"""
_PSDB_PARENT_SYM_UPPER = _sym_upper
_PSDB_SIDES = {}
_PSDB_KEYS = ((0, 64, 1024), (0, 128, 1024),
              (1, 1024, 64), (1, 1024, 128), (1, 1024, 256),
              (2, 64, 64), (2, 64, 128), (2, 128, 256))
_PSDB_GROUPS = ((0, 128, 256), (128, 64, 128), (192, 64, 64))
_PB_CACHES = tuple(_PB_CACHES) + ('_PSDB_SIDES',)


def _psdb_fits():
    if _SCRATCH is None or 6 not in _IPT_REGION:
        return False
    region = _IPT_REGION[6]
    return (int(_SCRATCH[0].shape[0]) == 391602176
            and all(4 * 7**6 * (rows // 64) * (cols // 64) <= region[kind]
                    for kind, rows, cols in _PSDB_KEYS))


def _psdb_side(kind, rows, cols):
    key = (kind, rows, cols)
    if key not in _PSDB_KEYS or not _psdb_fits():
        raise RuntimeError('unadmitted P4 symmetry side')
    arena = _SCRATCH[0]
    region = _IPT_REGION[6]
    start = (0, region[0], region[0] + region[1])[kind]
    found = _PSDB_SIDES.get(key)
    if found is None:
        found = _PSDBSide(arena, start, kind, 4, 6, rows, cols)
        _PSDB_SIDES[key] = found
    if (found._borrowed_arena is not arena or found._borrowed_start != start
            or (found.kind, found.P, found.L, found.rows, found.cols)
            != (kind, 4, 6, rows, cols)):
        raise RuntimeError('stale P4 symmetry side')
    return found


_PSDB_OLD_BIND = _pb_bind


def _pb_bind(t0):
    _PSDB_OLD_BIND(t0)
    for kind, rows, cols in _PSDB_KEYS:
        _psdb_side(kind, rows, cols)
    _PB_LOG.append(('psdb_fused', len(_PSDB_SIDES),
                    round(_pb_time.monotonic() - t0, 3)))


def _psdb_group(A4, B4, out, offset, rows, cols):
    # Inputs, packed results and final output are external to the tree arena.
    # No per-product reductions cross P. Complete all copies before next group.
    keep = []
    _TR_LAST[0] = None
    try:
        a = A4[:, offset:offset + rows, :]
        b = B4[:, :, offset:offset + cols]
        parts = fnp.empty((4, rows, cols), dtype=fnp.float32)
        keep.extend((a, b, parts))
        sa = _psdb_side(0, rows, 1024)
        sb = _psdb_side(1, 1024, cols)
        sc = _psdb_side(2, rows, cols)
        sa.forward(a, keep)
        sb.forward(b, keep)
        keep.append(fnp.matmul(sa.T, sb.T, out=sc.T))
        sc.unwind(parts, keep)
        views = fnp.unstack(parts, axis=0)
        keep.extend(views)
        for p, part in enumerate(views):
            start = p * 256 + offset
            upper = out[start:start + rows, start:start + cols]
            keep.extend((upper, fnp.copyto(upper, part)))
            if cols > rows:
                lower = out[start + rows:start + cols, start:start + rows]
                off = part[:, rows:]
                off_t = off.T
                keep.extend((lower, off, off_t, fnp.copyto(lower, off_t)))
    finally:
        _TR_LAST[0] = None
        keep.clear()


def _psdb_sym_upper(A, B):
    if not (_AB_ON and _WINOGRAD
            and len(A.shape) == len(B.shape) == 2
            and tuple(A.shape) == tuple(B.shape) == (1024, 1024)
            and str(A.dtype) == str(B.dtype) == 'float32'
            and all(_ipt_fits(6, m, k, n) for m, k, n in _PSD_SHAPES)
            and _psdb_fits()):
        return _PSDB_PARENT_SYM_UPPER(A, B)
    out = fnp.empty((1024, 1024), dtype=fnp.float32)
    for start in (0, 256, 512):
        end = start + 256
        _psd_sym_tile(A, B, out, start, 256, end, 1024 - end, False)
    # Split existing axes only; these are regular views even for transposed A.
    A4 = fnp.reshape(A, (4, 256, 1024))
    Bv = fnp.reshape(B, (1024, 4, 256))
    B4 = fnp.moveaxis(Bv, 1, 0)
    for offset, rows, cols in _PSDB_GROUPS:
        _psdb_group(A4, B4, out, offset, rows, cols)
    return out


_sym_upper = _psdb_sym_upper
