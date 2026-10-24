# Exact operand loading through the public flat-scatter API.
# The existing quad has the original operand's C logical order. put flattens
# its input values by definition, so the separately paid source reshape is
# unnecessary. The quad owner, following add/sub and all tree layouts remain.
_O20_FLATLOAD_ON = True
_O20_FLATLOAD_PARENT_A = _rs_load_a
_O20_FLATLOAD_PARENT_B = _rs_load_b
_O20_FLATLOAD_INDEX = [None, 0, {}]
_O20_FLATLOAD_STATS = dict(a=0, b=0, avoided_reshape_elements=0,
                          index_elements_created=0, index_capacity=0)
_O20_FLATLOAD_CAP = 1 << 23


def _o20_flatload_indices(size):
    owner, capacity, views = _O20_FLATLOAD_INDEX
    if capacity < size:
        capacity = 1 << (size - 1).bit_length()
        owner = fnp.arange(capacity, dtype=fnp.int32)
        views = {}
        _O20_FLATLOAD_INDEX[:] = [owner, capacity, views]
        _O20_FLATLOAD_STATS['index_elements_created'] += capacity
        _O20_FLATLOAD_STATS['index_capacity'] = capacity
    answer = views.get(size)
    if answer is None:
        answer = owner if size == capacity else owner[:size]
        # Retain only a bounded number of zero-copy prefix views.
        if len(views) >= 256:
            views.clear()
        views[size] = answer
    return answer


def _rs_load_a(a, X, slot, L, m, k):
    size = int(m) * int(k)
    if (not _O20_FLATLOAD_ON or _ASTRA_QL_ON or size <= 0
            or size > _O20_FLATLOAD_CAP or a.dtype != 'float32'):
        return _O20_FLATLOAD_PARENT_A(a, X, slot, L, m, k)
    assert _rc_math.prod(a.shape) == size
    quad = slot[1] if slot[1] is not None else _l3_quad(slot, 0, L)
    assert _rc_math.prod(quad.shape) == size
    h = _BH_HOLD.append
    h(fnp.put(quad, _o20_flatload_indices(size), a))
    h(fnp.add(X[4], X[6], out=X[4]))
    _O20_FLATLOAD_STATS['a'] += 1
    _O20_FLATLOAD_STATS['avoided_reshape_elements'] += size


def _rs_load_b(b, Y, slot, L, k, n):
    size = int(k) * int(n)
    if (not _O20_FLATLOAD_ON or _ASTRA_QL_ON or size <= 0
            or size > _O20_FLATLOAD_CAP or b.dtype != 'float32'):
        return _O20_FLATLOAD_PARENT_B(b, Y, slot, L, k, n)
    assert _rc_math.prod(b.shape) == size
    quad = slot[1] if slot[1] is not None else _l3_quad(slot, 1, L)
    assert _rc_math.prod(quad.shape) == size
    h = _BH_HOLD.append
    h(fnp.put(quad, _o20_flatload_indices(size), b))
    h(fnp.subtract(Y[3], Y[1], out=Y[1]))
    _O20_FLATLOAD_STATS['b'] += 1
    _O20_FLATLOAD_STATS['avoided_reshape_elements'] += size
