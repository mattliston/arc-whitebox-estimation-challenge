# Keep the newborn contrast before separate FP32 plus/minus leg rounding.
def _sp1_make(common, directions, t, s):
    h = _E2Hold()
    midpoint = h(h(t + s) * 0.5)
    half_difference = h(h(t - s) * 0.5)
    return (common + h(directions * midpoint), directions * half_difference)


def _sp1_read(left, young_right, lam, live_indices, state, response_indices, layer):
    packet = _E2_LAYER.pop('stable_pair1', None)
    if layer != 1 or packet is None:
        return _cp_read_young_sq(left, young_right, lam, live_indices, state)
    h = _E2Hold()
    u, v = packet
    births = int(u.shape[1])
    if (state['columns'] != 0 or tuple(u.shape) != tuple(v.shape)
            or tuple(lam.shape) != (2 * births,)
            or tuple(left.shape) != (int(response_indices.shape[0]), 2 * births)
            or tuple(young_right.shape) != (int(live_indices.shape[0]), 2 * births)
            or not all(str(x.dtype) == 'float32' for x in (u, v, lam, left, young_right))):
        raise RuntimeError('stable L1 newborn ownership')
    coefficient = h(lam[:births])
    if not bool(fnp.array_equal(h(lam[births:]), h(-coefficient))):
        raise RuntimeError('stable L1 signed pair coefficients')
    ur = h(fnp.take(u, response_indices, axis=0))
    vr = h(fnp.take(v, response_indices, axis=0))
    ul = h(fnp.take(u, live_indices, axis=0))
    vl = h(fnp.take(v, live_indices, axis=0))
    first = h(h(ur * ur) + h(vr * vr))
    second = h(h(ur * vr) * 2)
    two_coefficient = h(coefficient * 2)
    packed_left = h(fnp.concatenate((first, second), axis=1))
    packed_right = h(fnp.concatenate((h(vl * two_coefficient), h(ul * two_coefficient)), axis=1))
    return _mm(packed_left, h(packed_right.T))
