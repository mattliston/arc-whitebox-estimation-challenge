# Fixed R5 on the seven learned covariance-edge activations only.
_R5_EDGE_STATE = {}


def _r5_edge_tanh(h):
    if str(h.dtype) != 'float32':
        raise TypeError('R5 edge activation requires float32')
    shape = tuple(h.shape)
    st = _R5_EDGE_STATE.get(shape)
    if st is None:
        _R5_EDGE_STATE.clear()  # one shape only; no cross-shape accumulation
        constants = fnp.asarray((-3.6467385292053223, 3.6467385292053223,
                                 105.0, 945.0, 15.0, 420.0), dtype=fnp.float32)
        st = dict(constants=constants, c=fnp.unstack(constants),
                  z=fnp.empty(shape, dtype=fnp.float32),
                  work=fnp.empty(shape, dtype=fnp.float32))
        _R5_EDGE_STATE[shape] = st
    lo, hi, c105, c945, c15, c420 = st['c']
    z, work = st['z'], st['work']
    hold = []
    hold.append(fnp.clip(h, lo, hi, out=h))
    hold.append(fnp.multiply(h, h, out=z))
    hold.append(fnp.add(z, c105, out=work))
    hold.append(fnp.multiply(z, work, out=work))
    hold.append(fnp.add(work, c945, out=work))
    hold.append(fnp.multiply(h, work, out=h))
    hold.append(fnp.multiply(z, c15, out=work))
    hold.append(fnp.add(work, c420, out=work))
    hold.append(fnp.multiply(z, work, out=work))
    hold.append(fnp.add(work, c945, out=work))
    hold.append(fnp.divide(h, work, out=h))
    return h
