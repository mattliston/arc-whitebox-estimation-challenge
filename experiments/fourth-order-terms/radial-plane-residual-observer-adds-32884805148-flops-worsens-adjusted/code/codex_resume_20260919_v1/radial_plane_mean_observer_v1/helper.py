"""A paid fourth-plane residual observer; no writes to the parent state."""
_R22_ENABLED = True


def _r22_source(covariance, fourth_plane):
    if covariance.shape != fourth_plane.shape or covariance.ndim != 2:
        raise ValueError('Fourth-plane observer source shape')
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError('Fourth-plane observer source must be square')
    c = (covariance + covariance.T) / 2
    q = (fourth_plane + fourth_plane.T) / 2
    v = fnp.diag(c)
    cc = c*c
    radial = 3*fnp.sum(q)/(fnp.sum(v)**2 + 2*fnp.sum(cc))
    return (q - (radial/3)*(fnp.outer(v, v) + 2*cc)).astype(fnp.float32)


def _r22_ray(plane, weight):
    if plane.shape != (weight.shape[0], weight.shape[0]):
        raise ValueError('Fourth-plane observer ray shape')
    if str(plane.dtype) != 'float32' or str(weight.dtype) != 'float32':
        raise TypeError('Fourth-plane observer ray operands must be FP32')
    square = weight*weight
    product = fnp.matmul(plane, square)
    e = square.astype(fnp.float64)
    diagonal = fnp.diag(plane).astype(fnp.float64)
    return (3*fnp.sum(e*product.astype(fnp.float64), axis=0)
            - 2*fnp.sum(diagonal[:, None]*(e*e), axis=0))


def _r22_advance(state, weight32, weight64, aux):
    if state['plane'] is None:
        return
    ray = _r22_ray(state['plane'], weight32)
    a, p, s = aux['alpha'], aux['pdf'], aux['sigma']
    source = ((a*a-1)*p/(24*s*s*s))*ray
    if state['mean'] is not None:
        source = source + aux['gate']*(weight64.T @ state['mean'])
    state['mean'] = source


def _r22_read(mean, state):
    if state['mean'] is None:
        return mean.copy()
    return (mean + state['mean']).copy()

