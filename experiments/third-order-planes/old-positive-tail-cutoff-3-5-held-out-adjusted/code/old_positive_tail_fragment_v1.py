# Research only: omit old-tier local reconstruction on strongly positive rows.
# The carried Q/F tensor, the full pre-basis and every young column stay intact.
# This changes the old contribution to local diagonal/radial consumers; it is
# an approximation, distinct from retaining only the |alpha|<2.326 response set.
_OPT_Z = 3.5
_OPT_CONTEXT = [None]
_OPT_STATS = {'calls': 0, 'rows_omitted': 0}
_OPT_ORIG_PUSH = _shared_old384_push
_OPT_ORIG_RECON = _slr_reconstruct


def _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):
    assert _OPT_CONTEXT[0] is None
    if 7 <= layer <= 14:
        a = _E2_LAYER['a']
        _OPT_CONTEXT[0] = a if rows is None else a[rows]
    try:
        return _OPT_ORIG_PUSH(layer, w32, p, previous, newborn, lam, state, rows=rows)
    finally:
        _OPT_CONTEXT[0] = None


def _slr_reconstruct(U, state):
    a = _OPT_CONTEXT[0]
    if a is None:
        return _OPT_ORIG_RECON(U, state)
    m = int(U.shape[0])
    assert tuple(a.shape) == (m,)
    score = _OPT_Z - a
    kept = min(m, (int(fnp.nonzero(score >= 0.0)[0].shape[0]) + 63) // 64 * 64)
    if kept == m:
        return _OPT_ORIG_RECON(U, state)
    _OPT_STATS['calls'] += 1
    _OPT_STATS['rows_omitted'] += m - kept
    rows = fnp.argsort(-score, kind='stable')[:kept]
    old = _OPT_ORIG_RECON(U[rows], state)
    inverse = fnp.full(m, kept, dtype=fnp.int64)
    fnp.put(inverse, rows, fnp.arange(kept, dtype=fnp.int64))
    return fnp.concatenate([old, fnp.zeros((1, int(old.shape[1])), dtype=old.dtype)], axis=0)[inverse]
