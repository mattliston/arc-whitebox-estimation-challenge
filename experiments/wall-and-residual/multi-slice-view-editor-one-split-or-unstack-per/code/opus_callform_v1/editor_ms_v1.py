"""[opus-ms] MULTI-SLICE editor v1 (byte-identical residual diet; apply(text) -> text). Every edit stays on its own line (line count
unchanged until the helper block appended LAST), anchored on text, applied at every occurrence (dead copies are harmless).
 K1 plain k-split chunk loop:   aa[:, k0:k0 + q] / bb[k0:k0 + q, :] per chunk  -> one fnp.split per operand at the chunk boundaries
 K2 hz k-split chunk loop:      ak, bk = aa[..., k0:k0 + q], bb[k0:k0 + q]   -> the same
 K3 hzd k-split chunk loop:     ak = a[..., k0:k0 + q] / bk = b[k0:k0 + q]    -> the same
 K4 ubx k-split chunk loop:     ak = aa[..., k0:k0 + q]                       -> the same (bk stays: it is sliced only on cache misses)
 U1 _uvp_split: per cohort block F[:ru, a:a+h], F[:ru, a+h:a+2h], F[:, a:a+h], F[:, a+h:a+2h], Fu[:, b:b+h], Fv2[:, b:b+h] (6 getitems)
    -> F[:ru] once + one split of F[:ru], F, Fu, Fv2 at the cohort boundaries (falls back to the original slices if a piece is not
    exactly the slice it replaces)
 R1 _rs_basis_views: per level the (lo, hi) pair T[lo], T[hi] (2 getitems) -> one unstack, along the level's axis, of the shared strided
    base T[idx] (taken once per call)
Every replacement view is the same memory with the same shape and strides as the getitem it replaces (split / unstack of a basic view),
and no array is written between: same operands, same arithmetic. One-chunk reads keep their single getitem."""

EDITS = [
 ('K1', "        TA = _ipt_forward_a_ab(aa[:, k0:k0 + q], 6, M, q)\n        TB = _ipt_forward_b_ab(bb[k0:k0 + q, :], 6, q, N)\n",
        "        _ms_a = _ms_a if j else _ms_split(aa, sizes, 1); _ms_b = _ms_b if j else _ms_split(bb, sizes, 0); TA = _ipt_forward_a_ab(_ms_a[j], 6, M, q)   # [opus-ms] K1\n        TB = _ipt_forward_b_ab(_ms_b[j], 6, q, N)\n"),
 ('K2', "        ak, bk = aa[..., k0:k0 + q], bb[k0:k0 + q]\n",
        "        _ms_a = _ms_a if j else _ms_split(aa, sizes, -1); _ms_b = _ms_b if j else _ms_split(bb, sizes, 0); ak, bk = _ms_a[j], _ms_b[j]   # [opus-ms] K2\n"),
 ('K3a', "        ak = a[..., k0:k0 + q]\n",
         "        _ms_a = _ms_a if j else _ms_split(a, sizes, -1); ak = _ms_a[j]   # [opus-ms] K3\n"),
 ('K3b', "        bk = b[k0:k0 + q]\n",
         "        _ms_b = _ms_b if j else _ms_split(b, sizes, 0); bk = _ms_b[j]   # [opus-ms] K3\n"),
 ('K4', "        ak = aa[..., k0:k0 + q]\n",
        "        _ms_a = _ms_a if j else _ms_split(aa, sizes, -1); ak = _ms_a[j]   # [opus-ms] K4\n"),
 ('U1', """    b = 0
    for a, h in blocks:
        h_(fnp.add(F[:ru, a:a + h], F[:ru, a + h:a + 2 * h], out=Fu[:, b:b + h]))
        h_(fnp.subtract(F[:, a:a + h], F[:, a + h:a + 2 * h], out=Fv2[:, b:b + h]))
""", """    b = 0; _ms_P = _ms_uvp_pieces(F, Fu, Fv2, blocks, ru)   # [opus-ms] U1: one split per array (None -> the original slices)
    for a, h in blocks:
        h_(fnp.add(_ms_P['t'][a], _ms_P['t'][a + h], out=_ms_P['u'][b]) if _ms_P else fnp.add(F[:ru, a:a + h], F[:ru, a + h:a + 2 * h], out=Fu[:, b:b + h]))
        h_(fnp.subtract(_ms_P['f'][a], _ms_P['f'][a + h], out=_ms_P['v'][b]) if _ms_P else fnp.subtract(F[:, a:a + h], F[:, a + h:a + 2 * h], out=Fv2[:, b:b + h]))
"""),
 ('R1', "        pairs.append((T[tuple(lo)], T[tuple(hi)]))\n",
        "        _ms_B = _ms_B if j > 2 else T[tuple(idx)]; _ms_U = fnp.unstack(_ms_B, axis=ax); pairs.append((_ms_U[lo[ax] // 2 if kind == 0 else lo[ax]], _ms_U[hi[ax] // 2 if kind == 0 else hi[ax]]))   # [opus-ms] R1\n"),
]

HELPER = '''

# BEGIN [opus-ms] MULTI-SLICE helpers v1 (append LAST)
def _ms_split(x, sizes, axis):
    """[the slice [k0, k0 + q) of x along axis for q in sizes] as views: ONE split request for >= 2 chunks (the same views a getitem
    per chunk makes); a one-chunk read keeps its single getitem."""
    nd = len(x.shape)
    ax = axis % nd
    if len(sizes) < 2:
        return [x[(slice(None),) * ax + (slice(0, int(sizes[0])),)]]
    cuts, at = [], 0
    for q in sizes[:-1]:
        at += int(q)
        cuts.append(at)
    end = at + int(sizes[-1])
    if end < int(x.shape[ax]):
        cuts.append(end)
    return list(fnp.split(x, cuts, axis=axis))[:len(sizes)]


def _ms_uvp_pieces(F, Fu, Fv2, blocks, ru):
    """_uvp_split's per-cohort slices from one split per array; None if any piece is not exactly the slice it replaces."""
    cols = int(F.shape[1])
    cuts = sorted({x for a, h in blocks for x in (a, a + h, a + 2 * h) if 0 < x < cols})
    ub, b = [], 0
    for _a, h in blocks[:-1]:
        b += h
        ub.append(b)
    starts, ustarts = [0] + cuts, [0] + ub
    t = dict(zip(starts, fnp.split(F[:ru], cuts, axis=1)))
    f = dict(zip(starts, fnp.split(F, cuts, axis=1)))
    u = dict(zip(ustarts, fnp.split(Fu, ub, axis=1)))
    v = dict(zip(ustarts, fnp.split(Fv2, ub, axis=1)))
    b = 0
    for a, h in blocks:
        for d, key in ((t, a), (t, a + h), (f, a), (f, a + h)):
            if key not in d or int(d[key].shape[1]) != h:
                return None
        if int(u[b].shape[1]) != h or int(v[b].shape[1]) != h:
            return None
        b += h
    return dict(t=t, f=f, u=u, v=v)
# END [opus-ms] MULTI-SLICE helpers v1
'''


def apply(text, need=('K1', 'K2', 'K3a', 'K3b', 'K4', 'U1', 'R1')):
    counts = {}
    for tag, old, new in EDITS:
        counts[tag] = text.count(old)
        text = text.replace(old, new)
    missing = [t for t in need if not counts.get(t)]
    assert not missing, ('anchors not found', missing, counts)
    return text + HELPER, counts
