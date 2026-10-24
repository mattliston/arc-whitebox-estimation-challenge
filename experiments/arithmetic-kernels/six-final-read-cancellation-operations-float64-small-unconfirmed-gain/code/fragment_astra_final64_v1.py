# ASTRA RLF final reconstruction precision v1. Append AFTER read-leaf-fuse.
# Promote only six final-level cancellation operations to genuine float64.
# Deep decode, basis conversion, leaf products and accumulation stay unchanged.
# This is a numerical rounding mitigation, never a byte-equivalence claim.
_ASTRA_RLFP_PARENT = _ASTRA_RLF_PARENT_UNWIND
_ASTRA_RLFP_ON = True
_ASTRA_RLFP_STATS = dict(calls=0, output_elements=0)


def _astra_rlfp_unwind(TC, Z1, ops, basis, out, L, m, n):
    context = _ASTRA_RLF_CONTEXT[0]
    if not (_ASTRA_RLFP_ON and context is not None and context['done']):
        return _ASTRA_RLFP_PARENT(TC, Z1, ops, basis, out, L, m, n)
    if L != 6 or (m, n) != (context['M'], context['N']):
        raise RuntimeError('ASTRA RLF final64: unexpected tree')
    h = _BH_HOLD.append
    # Literally the parent deep unwind and basis conversion, in its order.
    if ops is not None:
        for f, x, y, o in ops:
            h(f(x, y, out=o))
    else:
        for j in range(L, 1, -1):
            Z = Z1[j]
            h(fnp.add(Z[0], Z[5], out=Z[5]))
            h(fnp.add(Z[0], Z[2], out=Z[0]))
            h(fnp.add(Z[5], Z[4], out=Z[5]))
            h(fnp.add(Z[5], Z[3], out=Z[2]))
            h(fnp.add(Z[5], Z[1], out=Z[3]))
            h(fnp.add(Z[4], Z[6], out=Z[1]))
    for lo, hi in basis:
        h(fnp.subtract(hi, lo, out=lo))
    tl, tr, bl, br = _rs_out_quadrants(out, L, m, n)
    Z = Z1[1]
    # Explicit dtype= selects AND bills the float64 ufunc loop, including
    # operations writing directly to float32 output quadrants. The installed
    # flopscope _pointwise._counted_binary honors this before consulting out.
    # No out-dtype billing shortcut is assumed. Six quarter-array operations
    # cost at the float64 rate; copyto(br, br64) uses both source/dest dtypes.
    t5 = fnp.add(Z[0], Z[5], dtype=fnp.float64)
    h(t5)
    h(fnp.add(Z[0], Z[2], out=tl))
    h(fnp.add(t5, Z[4], out=t5, dtype=fnp.float64))
    h(fnp.add(t5, Z[3], out=tr, dtype=fnp.float64))
    br64 = fnp.add(t5, Z[1], dtype=fnp.float64)
    t4 = fnp.add(Z[4], Z[6], dtype=fnp.float64)
    h((br64, t4))
    h(fnp.subtract(br64, t4, out=bl, dtype=fnp.float64))
    h(fnp.copyto(br, br64))
    _ASTRA_RLFP_STATS['calls'] += 1
    _ASTRA_RLFP_STATS['output_elements'] += m*n
    return out


_ASTRA_RLF_PARENT_UNWIND = _astra_rlfp_unwind
