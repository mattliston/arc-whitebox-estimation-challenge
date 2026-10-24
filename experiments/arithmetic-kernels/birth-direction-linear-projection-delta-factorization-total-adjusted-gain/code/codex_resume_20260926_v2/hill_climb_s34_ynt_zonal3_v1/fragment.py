def _ynt_trace(w, p, previous, newborn, coefficient, tail):
    q, selected, base, t, s = newborn
    q, births = int(q), int(selected.shape[0])
    weights = coefficient.astype(fnp.float64)
    B = base.astype(fnp.float64)
    t64, s64 = t.astype(fnp.float64), s.astype(fnp.float64)
    normB = fnp.sum(B * B, axis=0)
    ct = weights[q:q + births] * (normB + t64 * t64)
    cs = weights[q + births:] * (normB + s64 * s64)
    tau = B @ (ct + cs)
    if q:
        P = p[:, :q].astype(fnp.float64)
        carried = P @ (weights[:q] * fnp.sum(P * P, axis=0))
        fnp.put(tau, previous, tau[previous] + carried)
    fnp.put(tau, selected, tau[selected] + t64 * ct + s64 * cs)
    wt = w[:, tail].astype(fnp.float64)
    normw = fnp.sum(wt * wt, axis=0)
    n = int(w.shape[0])
    linear = (3 * normw / (n + 2)) * (wt.T @ tau)
    # Conditional mean of the missing degree-three spherical harmonic.
    mean = _E2_LAYER['ynt_previous_mean']
    u = mean / fnp.sqrt(fnp.maximum(fnp.sum(mean * mean), 1e-30))
    hb = u @ B
    ht = hb + u[selected] * t64
    hs = hb + u[selected] * s64
    axial = fnp.sum(weights[q:q + births] * ht * ht * ht + weights[q + births:] * hs * hs * hs)
    if q:
        hp = u[previous] @ P
        axial = axial + fnp.sum(weights[:q] * hp * hp * hp)
    trace_axis = fnp.sum(u * tau)
    z = wt.T @ u
    zonal = ((n + 2) * axial - 3 * trace_axis) / (n - 1)
    return linear + zonal * (z * z * z - 3 * normw * z / (n + 2))
