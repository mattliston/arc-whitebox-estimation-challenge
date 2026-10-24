# Fixed paid nearest-grid lookup, owned by one prediction and one shape.
_EDGE_LOOKUP_STATE = {}


def _edge_tanh_lookup(h):
    if str(h.dtype) != 'float32':
        raise TypeError('edge lookup requires float32')
    shape = tuple(h.shape)
    st = _EDGE_LOOKUP_STATE.get(shape)
    hold = []
    if st is None:
        _EDGE_LOOKUP_STATE.clear()
        constants = fnp.asarray((8.0, 1048576.0, 0.5, 0.00000095367431640625), dtype=fnp.float32)
        hi, scale, half, step = fnp.unstack(constants)
        table = fnp.arange(8388609, dtype=fnp.float32)
        hold.append(fnp.multiply(table, step, out=table))
        hold.append(fnp.tanh(table, out=table))
        st = dict(constants=constants, c=(hi, scale, half), table=table,
                  work=fnp.empty(shape, dtype=fnp.float32))
        _EDGE_LOOKUP_STATE[shape] = st
    hi, scale, half = st['c']
    work = st['work']
    hold.append(fnp.absolute(h, out=work))
    hold.append(fnp.minimum(work, hi, out=work))
    hold.append(fnp.multiply(work, scale, out=work))
    hold.append(fnp.add(work, half, out=work))
    indices = work.astype(fnp.int32)
    hold.append(fnp.take(st['table'], indices, out=work, mode='clip'))
    hold.append(fnp.copysign(work, h, out=h))
    return h
