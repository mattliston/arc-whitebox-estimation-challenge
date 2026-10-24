# ASTRA RLF canonical decode at every level v1; append after RLF.
# Pure float32 real-algebra rewrite, including final BL cancellation.
# Every deep output axis is canonicalized immediately; no deferred basis
# subtractions remain. No depth/layer/seed selection or precision changes.
# At L6: +27867*(M/64)*(N/64) ordinary flops, unchanged warm requests.
# No new numerical buffers. Alternative to all other RLF unwind fragments.
_ASTRA_RLFD_PARENT = _ASTRA_RLF_PARENT_UNWIND
_ASTRA_RLFD_ON = True
_ASTRA_RLFD_STATS = dict(calls=0, paired=0, output_elements=0, extra_bill=0)


def _astra_rlfd_unwind(TC, Z1, ops, basis, out, L, m, n):
    context = _ASTRA_RLF_CONTEXT[0]
    if not (_ASTRA_RLFD_ON and context is not None and context['done']):
        return _ASTRA_RLFD_PARENT(TC, Z1, ops, basis, out, L, m, n)
    if L != 6 or (m, n) != (context['M'], context['N']):
        raise RuntimeError('ASTRA RLF canonical decode: unexpected tree')
    h = _BH_HOLD.append
    # Each decoded axis is immediately put into canonical C order:
    # slots (0,1,2,3) = (TL,BL,TR,BR). Other-axis linear maps commute
    # in real arithmetic, so the deferred output basis passes disappear.
    if ops is not None:
        if len(ops) != 4*(L-1):
            raise RuntimeError('ASTRA RLF canonical: unknown fused op layout')
        for index, j in enumerate(range(L, 1, -1)):
            Z = Z1[j]
            f, x, y, o = ops[4*index]
            h(f(x, y, out=o))  # existing pair: (u,TL) -> (Z5,Z0)
            h(fnp.subtract(Z[1], Z[6], out=Z[6]))
            h(fnp.add(Z[5], Z[4], out=Z[4]))  # t5; preserve u in Z5
            f, _old_left, p31, p23 = ops[4*index+2]
            h(f(Z[4], p31, out=p23))  # same paired TR/BR views
            h(fnp.add(Z[5], Z[6], out=Z[1]))  # canonical BL
    else:
        for j in range(L, 1, -1):
            Z = Z1[j]
            h(fnp.add(Z[0], Z[5], out=Z[5]))
            h(fnp.subtract(Z[1], Z[6], out=Z[6]))
            h(fnp.add(Z[0], Z[2], out=Z[0]))
            h(fnp.add(Z[5], Z[4], out=Z[4]))
            h(fnp.add(Z[4], Z[3], out=Z[2]))
            h(fnp.add(Z[4], Z[1], out=Z[3]))
            h(fnp.add(Z[5], Z[6], out=Z[1]))
    # No basis pass: each deeper axis is already canonical.
    entry = _ipt_side(2, L, m, n)
    paired = entry[1] is Z1 and type(entry[2]) is list
    if paired:
        slot = entry[2]
        owned = _RQS_OUTV.get(id(out))
        owned = owned if owned is not None and owned[0] is out else None
        views = owned[1].get((L, m, n)) if owned is not None else None
        if views is None:
            u = 1 << L
            Cr = fnp.reshape(out, (2,) * L + (m // u,) + (2,) * L + (n // u,))
            C0, C1 = fnp.unstack(Cr, axis=0)
            bl, br = fnp.unstack(C1, axis=L)
            if owned is not None:
                owned[1][(L, m, n)] = (C0, bl, br, Cr, C1)
            else:
                h((Cr, C0, C1, bl, br))
        else:
            _RQS_STATS['outv'] += 1
            C0, bl, br = views[0], views[1], views[2]
        p05, p23 = slot[1] if slot[1] is not None else _l3o_pairs(slot, L)
    else:
        tl, tr, bl, br = _rs_out_quadrants(out, L, m, n)
    Z = Z1[1]
    h(fnp.add(Z[0], Z[5], out=Z[5]))  # u; the same first rounding
    # z6 is dead after BL. z1 remains intact for BR. The paired output
    # views refer only to z0,z5,z2,z3, so writing z6 cannot affect them.
    h(fnp.subtract(Z[1], Z[6], out=Z[6]))
    h(fnp.add(Z[5], Z[6], out=bl))
    if not paired:
        h(fnp.add(Z[0], Z[2], out=tl))
    h(fnp.add(Z[5], Z[4], out=Z[5]))
    if paired:
        h(fnp.add(p05, p23, out=C0))
    else:
        h(fnp.add(Z[5], Z[3], out=tr))
    h(fnp.add(Z[5], Z[1], out=br))
    _ASTRA_RLFD_STATS['calls'] += 1
    _ASTRA_RLFD_STATS['paired'] += int(paired)
    _ASTRA_RLFD_STATS['output_elements'] += m*n
    _ASTRA_RLFD_STATS['extra_bill'] += 27867*(m//64)*(n//64)
    return out


_ASTRA_RLF_PARENT_UNWIND = _astra_rlfd_unwind
