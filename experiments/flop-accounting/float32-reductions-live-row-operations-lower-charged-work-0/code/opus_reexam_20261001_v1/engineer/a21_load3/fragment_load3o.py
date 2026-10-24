# BEGIN OPUS-ENGINEER PAIRED OUTPUT QUADRANTS (load3o v1, a21, 2026-10-02; predictions bit-identical; bill unchanged; needs load3,
# whose two level-1 functions it extends to the output trees): the last unwind of a tree product writes its four output quadrants
# with four additions into four views that every call has to split off the output array (a reshape and three splits). Here the top
# two quadrants are written by ONE addition on pair views, (tl, tr) = (Z0, Z5) + (Z2, Z3), straight into the output's upper half,
# which carries the two quadrants along its column-half axis: one arithmetic request and one split fewer per product. The pair
# views of the tree are bound once per key (at its first use, by index and transpose views: no flops). Each element gets the same
# addition of the same two operands; the only change of order is that tl is written after Z5's second update, which it does not
# read. A pair op bills its elements, i.e. exactly the two additions it replaces.
_L3O_STATS = dict(pairs=0)


def _l3o_pairs(slot, L):
    G = slot[0]                                    # the output tree's natural-order level-1 view, chunk axis first
    order = list(range(1, L + 1)) + [0] + list(range(L + 1, 2 * L + 1))      # the pair axis where the output's upper half carries j1
    a, b = G[0:6:5], G[2:4]
    pairs = (fnp.transpose(a, order), fnp.transpose(b, order))
    _BH_HOLD.append((G, a, b))
    slot[0] = None; slot[1] = pairs
    _L3O_STATS['pairs'] += 1
    return pairs


def _rs_level1(kind, L, T, tail):
    # = load3's _rs_level1, with the natural-order view kept for the output trees as well (for _l3o_pairs)
    nr = L - 1
    data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
    R = T[(data,) * nr]
    V1 = fnp.reshape(R, (2, 2) * nr + (7,) + tail)
    swap = [x for p in range(nr) for x in (2 * p + 1, 2 * p)] if kind == 2 else list(range(2 * nr))   # C's pairs are (j, i)
    base = swap + list(range(2 * nr, 2 * nr + 1 + len(tail)))
    lead = len(tail) - 2
    Vn = fnp.transpose(V1, [base[x] for x in _rs_nat_perm(L, lead)])
    groups = [Vn] if lead == 0 else list(fnp.unstack(Vn, axis=1))
    return [(fnp.unstack(G, axis=0), [G, None]) for G in groups]


def _lvf_rs_level1(kind, L, T, tail):
    # = load3's level-fusion copy, likewise
    nr = L - 1
    data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
    R = T[(data,) * nr]
    V1 = fnp.reshape(R, (2, 2) * nr + (7,) + tail)
    swap = [x for p in range(nr) for x in (2 * p + 1, 2 * p)] if kind == 2 else list(range(2 * nr))
    base = swap + list(range(2 * nr, 2 * nr + 1 + len(tail)))
    lead = len(tail) - 2
    Vn = fnp.transpose(V1, [base[x] for x in _rs_nat_perm(L, lead)])
    groups = [Vn] if lead == 0 else list(fnp.unstack(Vn, axis=1))
    return [(fnp.unstack(G, axis=0), [G, None]) for G in groups], groups


def _l3o_last(slot, Z, out, L, m, n):
    # the last unwind: Z5 twice, the upper output half as one pair addition, then br, Z4 and bl as before
    h = _BH_HOLD.append
    u = 1 << L
    Cr = fnp.reshape(out, (2,) * L + (m // u,) + (2,) * L + (n // u,))
    C0, C1 = fnp.unstack(Cr, axis=0)
    bl, br = fnp.unstack(C1, axis=L)
    h((Cr, C0, C1, bl, br))
    p05, p23 = slot[1] if slot[1] is not None else _l3o_pairs(slot, L)
    h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[5], Z[4], out=Z[5])); h(fnp.add(p05, p23, out=C0))
    h(fnp.add(Z[5], Z[1], out=br)); h(fnp.add(Z[4], Z[6], out=Z[4])); h(fnp.subtract(br, Z[4], out=bl))
    return out


def _ipt_leaf_unwind_ab(TA, TB, out, L, m, n):
    # = the level-fusion _ipt_leaf_unwind_ab up to the last unwind
    TC, Z1, slot, ops, basis = _ipt_side(2, L, m, n)
    h = _BH_HOLD.append
    h(fnp.matmul(TA, TB, out=TC))
    if ops is not None:
        for f, x, y, o in ops:
            h(f(x, y, out=o))
    else:
        for j in range(L, 1, -1):
            Z = Z1[j]
            h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=Z[0])); h(fnp.add(Z[5], Z[4], out=Z[5]))
            h(fnp.add(Z[5], Z[3], out=Z[2])); h(fnp.add(Z[5], Z[1], out=Z[3])); h(fnp.add(Z[4], Z[6], out=Z[1]))
    for lo, hi in basis:
        h(fnp.subtract(hi, lo, out=lo))
    return _l3o_last(slot, Z1[1], out, L, m, n)


def _ks_unwind(TC, Z1, ops, basis, out, L, m, n):
    # = the k-split's _ks_unwind up to the last unwind; the key's entry (the caller holds the same one) gives the pair views
    h = _BH_HOLD.append
    if ops is not None:
        for f, x, y, o in ops:
            h(f(x, y, out=o))
    else:
        for j in range(L, 1, -1):
            Z = Z1[j]
            h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=Z[0])); h(fnp.add(Z[5], Z[4], out=Z[5]))
            h(fnp.add(Z[5], Z[3], out=Z[2])); h(fnp.add(Z[5], Z[1], out=Z[3])); h(fnp.add(Z[4], Z[6], out=Z[1]))
    for lo, hi in basis:
        h(fnp.subtract(hi, lo, out=lo))
    entry = _ipt_side(2, L, m, n)
    if entry[1] is Z1 and type(entry[2]) is list:
        return _l3o_last(entry[2], Z1[1], out, L, m, n)
    tl, tr, bl, br = _rs_out_quadrants(out, L, m, n)      # not the caller's entry (it was evicted in between): the four single writes
    Z = Z1[1]
    h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=tl)); h(fnp.add(Z[5], Z[4], out=Z[5]))
    h(fnp.add(Z[5], Z[3], out=tr)); h(fnp.add(Z[5], Z[1], out=br)); h(fnp.add(Z[4], Z[6], out=Z[4])); h(fnp.subtract(br, Z[4], out=bl))
    return out
# END OPUS-ENGINEER PAIRED OUTPUT QUADRANTS
