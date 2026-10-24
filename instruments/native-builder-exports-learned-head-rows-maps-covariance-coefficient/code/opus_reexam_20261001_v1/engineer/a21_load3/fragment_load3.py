# BEGIN OPUS-ENGINEER THREE-REQUEST OPERAND LOADS (load3 v1, a21, 2026-10-02; predictions bit-identical; bill + a quarter of every
# loaded operand's elements, about +0.06%): the alt-basis tree kernel loads each operand with six requests: a reshape, two splits,
# the copy of one half into the tree's pair view, the copy of one quarter, and the alt-basis sum (A: S1 = a21 + a22 in a21's slot;
# B: T3 = b22 - b12 in b12's slot). Here the key's FOUR level-1 data slots are bound as one view in the operand's own axis order
# (i1, i_2..i_L, row, j1, j_2..j_L, col), so a load is: reshape, ONE copy of the whole operand into that view, and the same sum or
# difference taken in place between the two slots (X4 <- X4 + X6, Y1 <- Y3 - Y1): three requests instead of six on every operand
# load (the release lane counted 284 A loads and 311 B loads per network on RRPI). Every tree element ends with the value it had
# before: the copy writes a21 (b12) into the slot the sum (difference) then overwrites with the same two operands in the same
# order. Cost: that one extra quarter copy per load.
# The four-slot view replaces the old pair view in slot 2 of the tree cache entry. It is built at the key's FIRST load, not at
# bind time, so the binding phase of setup makes no more requests than before (one or two fewer per key: the pair view is gone);
# its reshape bills the operand's element count once per key and process. Until then the slot holds the key's natural-order
# level-1 view. Nothing but the two loads reads slot 2 of an A or B entry.
_L3_STATS = dict(quads=0)


def _l3_quad(slot, kind, L):
    G = slot[0]                                    # axes (chunk, i_2..i_L, row, j_2..j_L, col); A's data chunks are 4 i1 + 2 j1, B's 2 i1 + j1
    data = G[0:7:2] if kind == 0 else G[0:4]
    Q = fnp.reshape(data, (2, 2) + tuple(int(x) for x in G.shape[1:]))
    quad = fnp.transpose(Q, [0] + list(range(2, L + 2)) + [1] + list(range(L + 2, 2 * L + 2)))
    _BH_HOLD.append((G, data, Q))
    slot[0] = None; slot[1] = quad
    _L3_STATS['quads'] += 1
    return quad


def _rs_level1(kind, L, T, tail):
    # = the resid3 _rs_level1 without the pair views: an A or B entry gets [natural-order level-1 view, None] for _l3_quad
    nr = L - 1
    data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
    R = T[(data,) * nr]
    V1 = fnp.reshape(R, (2, 2) * nr + (7,) + tail)
    swap = [x for p in range(nr) for x in (2 * p + 1, 2 * p)] if kind == 2 else list(range(2 * nr))   # C's pairs are (j, i)
    base = swap + list(range(2 * nr, 2 * nr + 1 + len(tail)))
    lead = len(tail) - 2
    Vn = fnp.transpose(V1, [base[x] for x in _rs_nat_perm(L, lead)])
    groups = [Vn] if lead == 0 else list(fnp.unstack(Vn, axis=1))
    return [(fnp.unstack(G, axis=0), None if kind == 2 else [G, None]) for G in groups]


def _lvf_rs_level1(kind, L, T, tail):
    # = the level-fusion copy of _rs_level1 (it also returns the natural-order views), without the pair views
    nr = L - 1
    data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
    R = T[(data,) * nr]
    V1 = fnp.reshape(R, (2, 2) * nr + (7,) + tail)
    swap = [x for p in range(nr) for x in (2 * p + 1, 2 * p)] if kind == 2 else list(range(2 * nr))
    base = swap + list(range(2 * nr, 2 * nr + 1 + len(tail)))
    lead = len(tail) - 2
    Vn = fnp.transpose(V1, [base[x] for x in _rs_nat_perm(L, lead)])
    groups = [Vn] if lead == 0 else list(fnp.unstack(Vn, axis=1))
    return [(fnp.unstack(G, axis=0), None if kind == 2 else [G, None]) for G in groups], groups


def _rs_load_a(a, X, slot, L, m, k):
    # A's level-1 load: the whole operand into the four data slots, then the alt-basis S1 = a21 + a22 in a21's slot
    h = _BH_HOLD.append
    u = 1 << L
    Ar = fnp.reshape(a, (2,) * L + (m // u,) + (2,) * L + (k // u,))
    h(Ar); h(fnp.copyto(slot[1] if slot[1] is not None else _l3_quad(slot, 0, L), Ar)); h(fnp.add(X[4], X[6], out=X[4]))


def _rs_load_b(b, Y, slot, L, k, n):
    # B's level-1 load: the whole operand into the four data slots, then the alt-basis T3 = b22 - b12 in b12's slot
    h = _BH_HOLD.append
    u = 1 << L
    Br = fnp.reshape(b, (2,) * L + (k // u,) + (2,) * L + (n // u,))
    h(Br); h(fnp.copyto(slot[1] if slot[1] is not None else _l3_quad(slot, 1, L), Br)); h(fnp.subtract(Y[3], Y[1], out=Y[1]))
# END OPUS-ENGINEER THREE-REQUEST OPERAND LOADS
