"""First-order input-skew innovation to Gaussian K4L birth, L1..14.

For Z of variance one and h=(a+Z)+, S_j=cum(Z[4-j],h[j]).
Differentiate under density phi(Z)*(1+eta*H3(Z)/6), eta=k3/sigma**3.
Subtract the derivative already induced by the parent's corrected host mean.
Input k4 is zero to this order, so no gated input-fourth transport is added.
All existing birth geometry, cohorts, ranks, radial removal and learned weights
remain in place. Fixed full theoretical strength, no clipping or dial sweep.
"""
_KSB_ACTIVE = [False]
_KSB_STATS = dict(layers=[], hubs=0, sum_abs_eta=0.0, max_abs_eta=0.0, retired=True)
_KSB_PARENT_BINARY = _predict_binary

def _ksb_birth_increment(l, A, aux, selected):
    if not _KSB_ACTIVE[0] or not 1 <= l <= 14:
        return A
    n = int(selected.shape[0])
    _KSB_STATS['layers'].append(int(l)); _KSB_STATS['hubs'] += n
    if not n:
        return A
    hold = _E2Hold()
    a, g, p, sigma, k3 = [hold(aux[k][selected]) for k in ('alpha','gate','pdf','sigma','pre_k3')]
    a2 = hold(a*a); a4 = hold(a2*a2)
    p2 = hold(p*p); p3 = hold(p2*p)
    tail = hold(g*hold(1.0-g)); eta = hold(k3/hold(hold(sigma*sigma)*sigma))
    # Exact polynomial derivatives after subtracting the host-mean derivative.
    d1 = hold(p*(a4+3*a2+12)/6)
    d2 = hold((p*(g*a4+(g+1)*a2+11-10*g)+p2*a*(a2+3))/3+2*a*tail)
    d3 = hold((p*(a4*g*(2*g-1)+a2*g*(4*g-3)+8-10*g)
               +p2*a*(a2*(4*g-1)+8*g-2)+p3*(2*a2+4)+6*a*tail)/2)
    d4 = hold(2*(p*(a2*g*(2*g-1)+2-3*g)+p2*a*(4*g-1)+2*p3+2*a*tail))
    delta = hold(hold(fnp.stack([d1,d2,d3,d4]))*eta)
    _KSB_STATS['sum_abs_eta'] += float(fnp.sum(fnp.abs(eta)))
    _KSB_STATS['max_abs_eta'] = max(_KSB_STATS['max_abs_eta'], float(fnp.max(fnp.abs(eta))))
    return fnp.concatenate([A[:4], hold(A[4:]+delta)])

def _predict_binary(mlp, *args, **kwargs):
    if int(mlp.width) != 1024 or int(mlp.depth) != 16 or kwargs.get('final_hook') is None:
        return _KSB_PARENT_BINARY(mlp, *args, **kwargs)
    if _KSB_ACTIVE[0]:
        raise RuntimeError('K4L skew-birth scope leaked')
    _KSB_STATS.clear()
    _KSB_STATS.update(layers=[],hubs=0,sum_abs_eta=0.0,max_abs_eta=0.0,retired=False)
    _KSB_ACTIVE[0] = True
    try:
        return _KSB_PARENT_BINARY(mlp, *args, **kwargs)
    finally:
        _KSB_ACTIVE[0] = False; _KSB_STATS['retired'] = True
