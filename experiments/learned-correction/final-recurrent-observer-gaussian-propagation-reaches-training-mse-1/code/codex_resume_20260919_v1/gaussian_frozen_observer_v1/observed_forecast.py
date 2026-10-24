"""Exact old physical forecast with a reference-only callback hook."""
_OBSERVER = None
def forecast(weights, with_births):
    """One fixed causal Gaussian forecast, identical in both acquisition roles."""
    depth, n = len(weights), int(weights[0].shape[0])
    assert depth >= 2
    ledger, means, births, gates = [], [], [], []
    m = fnp.zeros(n, dtype=fnp.float64)
    covariance = None
    floors = 0
    for layer, w in enumerate(weights):
        assert isinstance(w, fnp.ndarray) and w.dtype == fnp.float32
        assert w.shape == (n, n)
        w64 = w.astype(fnp.float64)
        mu = w64.T @ m
        if layer == 0:
            sigma = _mm(w.T, w, ledger)
        else:
            cw = _mm(covariance, w, ledger)
            if layer != depth - 1:
                sigma = _mm(w.T, cw, ledger)
        if layer == depth - 1:
            variance = fnp.sum(w64 * cw.astype(fnp.float64), axis=0)
        else:
            sigma = (sigma + sigma.T) * .5
            _finite(sigma, 'pre-covariance')
            variance = fnp.diag(sigma).astype(fnp.float64)
        _finite(variance, 'pre-variance')
        floors += int(fnp.sum(variance < 1e-10))
        scalar = gaussian_raw(mu, variance)
        for name, value in scalar.items():
            _finite(value, name)
        if not bool(fnp.all(scalar['post_v'] >= 0)):
            raise ValueError('Negative Gaussian marginal variance; no repair')
        m = scalar['m']
        means.append(m)
        if layer != depth - 1:
            covariance = gaussian_covariance(sigma, scalar)
            if with_births:
                births.append(make_birth(sigma, scalar))
                gates.append(scalar['g'].astype(fnp.float32))
        if _OBSERVER is not None:
            _OBSERVER(layer,w,scalar,sigma if layer != depth-1 else None,covariance if layer != depth-1 else None)
    assert len(ledger) == 2 * depth - 2
    return dict(means=means, births=births, gates=gates, terminal=scalar,
                ledger=ledger, pre_variance_floor_count=floors)
