# Structural old==0 prefix calibration: recover the omitted rowwise cross term.
# All old-tier reads retain the original calibrated energy unchanged.
_O18_RE22_ENERGY_PARENT = _re22_energy

def _re22_energy(prefix, Q, rows, prefix_cols, tail_cols, variance, hold):
    context = _RTT_CONTEXT[0]
    assert context is not None and context['state'] is not None
    if int(context['state']['columns']):
        return _O18_RE22_ENERGY_PARENT(prefix, Q, rows, prefix_cols,
                                       tail_cols, variance, hold)
    m, p = map(int, prefix.shape)
    assert int(Q.shape[0]) == m and tuple(rows.shape) == (m,)
    assert tuple(prefix_cols.shape) == (p,) and int(tail_cols.shape[0]) > 0
    assert prefix.dtype == Q.dtype == 'float32'
    v = hold(variance.astype(fnp.float32))
    residual = hold(prefix - hold(Q @ hold(Q.T @ prefix)))
    values, ir, ic = fnp.intersect1d(rows, prefix_cols, assume_unique=True,
                                    return_indices=True)
    values, ir, ic = hold(values), hold(ir), hold(ic)
    if int(ir.shape[0]):
        fnp.put(residual, hold(hold(ir * p) + ic), _rqs2_z32())
    vp = hold(v[prefix_cols])
    scale = hold(fnp.sqrt(vp))
    denominator = hold(hold(v[rows, None]) * hold(scale[None, :]))
    normalized = hold(residual / denominator)
    normalized_prefix = hold(prefix / denominator)
    # R is diagonal-free. Therefore 2<P,R>-<R,R> is exactly the
    # diagonal-free row energy lost by P -> P-R, in the current normalization.
    cross_difference = hold(hold(2.0 * normalized_prefix) - normalized)
    energy = hold(fnp.einsum('ij,ij->i', normalized, cross_difference,
                            optimize=False))
    ratio = hold(hold(fnp.sum(hold(1 / hold(v[tail_cols])))) /
                 hold(fnp.sum(hold(1 / vp))))
    _RE22_STATS.setdefault('cross_old0_layers', []).append(int(_RS_LAYER[0]))
    return energy * ratio
