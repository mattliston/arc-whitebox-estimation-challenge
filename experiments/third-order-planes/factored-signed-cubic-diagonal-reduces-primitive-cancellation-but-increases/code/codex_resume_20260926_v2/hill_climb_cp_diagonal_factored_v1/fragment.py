# Algebraically identical signed-cohort cubic diagonal in stored FP32 legs.
def _cp_factored_diagonal(pp, lam, state):
    hold = _E2Hold()
    plus, minus = [], []
    cursor = 0
    for start, width in state['cohorts']:
        start, width = int(start), int(width)
        assert start == cursor and width >= 0 and width % 2 == 0
        half = width // 2
        plus.extend(range(start, start + half))
        minus.extend(range(start + half, start + width))
        cursor += width
    assert cursor == int(pp.shape[1]) == int(lam.shape[0])
    ip = hold(fnp.asarray(plus, dtype=fnp.int64))
    im = hold(fnp.asarray(minus, dtype=fnp.int64))
    t = hold(fnp.take(pp, ip, axis=1))
    s = hold(fnp.take(pp, im, axis=1))
    delta = hold(t - s)
    square = hold(t * t)
    hold(fnp.add(square, hold(t * s), out=square))
    hold(fnp.add(square, hold(s * s), out=square))
    hold(fnp.multiply(square, delta, out=square))
    hold(fnp.multiply(square, hold(fnp.take(lam, ip)), out=square))
    return hold(fnp.sum(square, axis=1)).astype(fnp.float64)
