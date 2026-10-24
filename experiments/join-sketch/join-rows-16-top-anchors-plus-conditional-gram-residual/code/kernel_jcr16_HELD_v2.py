# Fixed16 energy anchors, remaining landmarks by conditional residual energy.
_JCR_STATS = dict(joins=[], retired=True)
_JCR_ACTIVE = False


def _jcr_rows(layer, J, weight, Qp, core, diagonal, rank):
    order = fnp.argsort(-diagonal.astype(fnp.float64), kind='stable')
    original = fnp.sort(order[:rank])
    if layer == 7:
        _JCR_STATS['joins'].clear()
    if layer not in (10, 13):
        _JCR_STATS['joins'].append(dict(layer=int(layer), rank=int(rank), changed=False, fallback=False))
        return original
    assert _JCR_ACTIVE and rank > 16 and int(J.shape[0]) >= rank
    anchors = order[:16]
    # Independent charged products: no parent transform/value caches touched.
    Y = J @ (weight[:, None] * J[anchors, :].T)
    if Qp is not None:
        Y = Y + Qp @ (core @ Qp[anchors, :].T)
    Y64 = Y.astype(fnp.float64)
    K = Y64[anchors, :]
    K = (K + K.T) * 0.5
    fallback = False
    try:
        if not bool(fnp.all(fnp.isfinite(K))) or not bool(fnp.all(fnp.isfinite(Y64))):
            raise ValueError('nonfinite anchor Gram')
        L = fnp.linalg.cholesky(K)
        dl = fnp.diagonal(L)
        if not bool(fnp.all(dl > (16 * 2.220446049250313e-16) * fnp.max(dl))):
            raise ValueError('numerically deficient anchor Gram')
        Z = fnp.linalg.solve(L, Y64.T)
        residual = diagonal.astype(fnp.float64) - fnp.sum(Z * Z, axis=0)
        if not bool(fnp.all(fnp.isfinite(residual))):
            raise ValueError('nonfinite conditional residual')
        residual = fnp.maximum(residual, 0.0)
        # The anchors must remain in the final set, even when residuals tie.
        fnp.put(residual, anchors, -fnp.inf)
        other = fnp.argsort(-residual, kind='stable')[:rank-16]
        answer = fnp.sort(fnp.concatenate([anchors, other]))
    except Exception as error:
        if type(error).__name__ not in ('ValueError', 'LinAlgError'):
            raise
        answer = original
        fallback = True
    _JCR_STATS['joins'].append(dict(layer=int(layer), rank=int(rank), anchors=16,
        n=int(J.shape[0]), width=int(J.shape[1]), old_rank=0 if Qp is None else int(Qp.shape[1]),
        changed=True, fallback=fallback))
    return answer
