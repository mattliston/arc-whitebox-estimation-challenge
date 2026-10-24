# Same fixed analytic gate33 correction, fused into the incumbent polynomial call.
_GQ33_PARENT = _gq
_GQ33_CONST = {}
_GQ33_PREDICT_PARENT = Estimator.predict
_GQ33_STATS = dict(calls=0)

def _gq(g, d, aux, bpre):
    hold = _E2Hold()
    s = aux['sigma']
    S = _GQ33_CONST
    n = int(s.shape[0])
    if S.get('n') != n:
        o, G, A, P, I, K3, K4, D, B = range(9)
        terms = [((G,), 1.0), ((K3, I, I, I, A, A, P), _GSK / 6.0),
                 ((K3, I, I, I, P), -_GSK / 6.0),
                 ((K4, I, I, I, I, A, A, A, P), -_GK4 / 24.0),
                 ((K4, I, I, I, I, A, P), 3.0 * _GK4 / 24.0),
                 ((D, B), float(_GBK)),
                 ((K3, K3, I, I, I, I, I, I, A, A, A, A, A, P), -1.0 / 72.0),
                 ((K3, K3, I, I, I, I, I, I, A, A, A, P), 10.0 / 72.0),
                 ((K3, K3, I, I, I, I, I, I, A, P), -15.0 / 72.0)]
        degree = max(len(k) for k, _ in terms)
        S.update(n=n, one=fnp.ones(n, dtype=fnp.float64),
                 index=fnp.asarray([[(k[j] if j < len(k) else o) for k, _ in terms] for j in range(degree)]),
                 coefficient=fnp.asarray([float(c) for _, c in terms], dtype=fnp.float64))
    rows = hold(fnp.stack([S['one'], g, aux['alpha'], aux['pdf'],
                          hold(fnp.power(s, -1.0)), aux['pre_k3'], aux['pre_k4'], d, bpre]))
    _GQ33_STATS['calls'] += 1
    return S['coefficient'] @ hold(fnp.prod(hold(rows[S['index']]), axis=0))

def _gq33_predict(self, *args, **kwargs):
    _GQ33_STATS['calls'] = 0
    try:
        return _GQ33_PREDICT_PARENT(self, *args, **kwargs)
    finally:
        _ASTRA_COMPUTE_STATS['gate33'] = dict(_GQ33_STATS)

Estimator.predict = _gq33_predict
