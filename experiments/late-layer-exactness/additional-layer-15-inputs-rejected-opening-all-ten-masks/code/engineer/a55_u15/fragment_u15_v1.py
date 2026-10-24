# BEGIN OPUS-ENGINEER LAYER-15 UNMASK (a55 u15 v1, 2026-10-10; lead 20:49Z, the physicist's arm unmask15; OFF with _U15_POS = () =
# bit-identical). The final layer's learned correction (gnn_step at layer 15) masks 10 of its 30 inputs because the final layer forms
# only the diagonal of C_pre15. Here features 13-15 (mean / rms / mean |rho| of the unit's pre-correlation row) and 26 (rho @ alpha / n)
# are formed as at layers 1-14 (features(): rho = C_pre / (s s'), float32) from the full C_pre15 = W' C14 W on the covariance dead-cut
# roster, and let through (the layer-15 mask is 1 at _U15_POS). The final layer already pays C_aa W_a for its diagonal; the full
# matrix adds only the symmetric product W' (C_aa W_a) (+0.708 Gflop on 651) and about 9 n^2 flops of row statistics.
# Needs the builder's two edits: the zero columns of final_scalar_features and the mask literal of _e2_gnn_consts.
_U15_POS = (13, 14, 15, 26)
_U15_STATE = dict(cov=None)
_U15_STATS = dict(full=0, used=0)
_U15_PARENT_DIAG = _dc_cov_diagonal


def _dc_cov_diagonal(w, C, gate, floor):
    """The parent's diagonal at the final layer, bit for bit, and the full C_pre15 (float32) kept for the layer's features."""
    if not _U15_POS:
        return _U15_PARENT_DIAG(w, C, gate, floor)
    roster = _dcv_roster(gate, floor)
    hold = _E2Hold()
    n = int(w.shape[0])
    if roster is None:
        first = _mm(_ps32(C), w)
        v = fnp.sum(hold(fnp.multiply(w, first, dtype=fnp.float64)), axis=0)
        cov = _sym_upper(hold(w.T), first)
    else:
        m = _DCV_STATS['m']
        wp = hold(w[roster]) if _DCV_SYM == 'pad' else None
        live = hold(roster[:m]) if _DCV_SYM == 'pad' else roster
        wa = hold(wp[:m]) if _DCV_SYM == 'pad' else hold(w[live])
        first = _mm(hold(hold(_ps32(C))[fnp.ix_(live, live)]), wa)
        v = fnp.sum(hold(fnp.multiply(wa, first, dtype=fnp.float64)), axis=0)
        if _DCV_SYM == 'pad':
            cov = _sym_upper(hold(wp.T), hold(fnp.concatenate((first, hold(fnp.zeros((n - m, int(w.shape[1])), dtype=fnp.float32))))))
        else:
            cov = _dcv_tiles(hold(wa.T), first, m) if _DCV_SYM == 'tiles' else _sym_upper(hold(wa.T), first)
    _U15_STATE['cov'] = cov
    _U15_STATS['full'] += 1
    return v, v


def _u15_columns(n, aux, zero):
    """{position: column} for the final layer's features; zero where the position is masked or no C_pre15 was formed."""
    out = {13: zero, 14: zero, 15: zero, 26: zero}
    cov = _U15_STATE['cov']
    if not _U15_POS or cov is None:
        return out
    _U15_STATE['cov'] = None
    hold = _E2Hold()
    s32 = _ps32(aux['sigma'])
    rho = hold(cov / hold(fnp.outer(s32, s32)))
    got = {13: lambda: fnp.mean(rho, axis=1), 14: lambda: fnp.sqrt(hold(fnp.mean(hold(rho * rho), axis=1))),
           15: lambda: fnp.mean(hold(fnp.abs(rho)), axis=1), 26: lambda: hold(rho @ aux['alpha']) / n}
    for pos in _U15_POS:
        out[pos] = got[pos]()
    _U15_STATS['used'] += 1
    return out
# END OPUS-ENGINEER LAYER-15 UNMASK
