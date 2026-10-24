# Fixed source-only experiment, appended to matched C (S51 + gk).
# No fitted parameters, calibration data, setup, or prediction at import.
_TCH_ON = True
_TCH_CTX = [None]
_TCH_STATS = dict(calls=0, rows=0, kept=0, omitted=0, trace_blocks=0,
                  hcf_proxy_calls=0, fallback_small=0)
_TCH_PARENT_L15 = _l15_rowcut_k3
_TCH_PARENT_RECON = _slr_reconstruct
_TCH_PARENT_HCF = _hcf_rows15


def _tch_keep_count(m):
    # ceil(m / 2) rounded upward to 64. With m <= 64 there is no arm.
    return min(m, max(64, ((m + 127) // 128) * 64))


def _tch_trace(U, uv, lam, tail):
    """First Gaussian chaos of the EXACT effective truncated u/v CP tensor.

    Unlike a raw-F trace, this respects the incumbent common-u truncation.
    No assumption that the two signed leg weights are exact opposites.
    The isotropic row covariance rho I is an explicit approximation.
    """
    Fu, Fv, blocks, ru = uv
    plus, minus = [], []
    for start, width in blocks:
        plus.append(lam[start:start + width])
        minus.append(lam[start + width:start + 2 * width])
    lp = plus[0] if len(plus) == 1 else fnp.concatenate(plus)
    lm = minus[0] if len(minus) == 1 else fnp.concatenate(minus)
    nu = fnp.sum(Fu * Fu, axis=0)
    nv = fnp.sum(Fv * Fv, axis=0)
    cross = fnp.sum(Fu * Fv[:ru], axis=0)
    ap = lp * (nu + nv + 2.0 * cross)
    am = lm * (nu + nv - 2.0 * cross)
    vector = Fv @ (ap - am)
    fnp.add(vector[:ru], Fu @ (ap + am), out=vector[:ru])
    rho = fnp.sum(U * U) / (int(U.shape[0]) * int(U.shape[1]))
    _TCH_STATS['trace_blocks'] += 1
    return ((U[tail] @ vector) * (3.0 * rho)).astype(fnp.float64)


def _tch_hcf_proxy(diagonal, mu, v, keep, tail):
    # Predict total B_ii from (v, 1, mu^2), fitting retained rows only.
    # The downstream incumbent residualization and normalization are unchanged.
    X = fnp.stack([v, fnp.ones(int(v.shape[0]), dtype=fnp.float64), mu * mu])
    train = X[:, keep]
    scale = fnp.sqrt(fnp.maximum(fnp.sum(train * train, axis=1)
                                / int(keep.shape[0]), 1e-24))
    train = train / scale[:, None]
    gram = (train @ train.T) / int(keep.shape[0])
    rhs = (train @ diagonal[keep]) / int(keep.shape[0])
    beta = fnp.linalg.solve(gram + 1e-12 * fnp.eye(3, dtype=fnp.float64), rhs)
    predicted = beta @ (X[:, tail] / scale[:, None])
    fnp.put(diagonal, tail, predicted)
    return diagonal


def _slr_reconstruct(U, state):
    ctx = _TCH_CTX[0]
    if not _TCH_ON or ctx is None:
        return _TCH_PARENT_RECON(U, state)
    if _UVP_CUR[0] != 15 or ctx['used']:
        raise RuntimeError('TCH expects exactly one layer-15 old reconstruction')
    m = int(U.shape[0])
    if m != int(ctx['rows'].shape[0]):
        raise RuntimeError('TCH old reconstruction row universe')
    columns = int(state['columns'])
    # Every active K4L cohort must belong entirely to the untouched young bank.
    age = max(int(_K4L_AGE), len(_K4L_C) + len(_K4LI))
    for j in range(max(15 - age, 0), 15):
        start, width = state['cohorts'][j]
        if int(width) and int(start) < columns:
            raise RuntimeError('TCH old bank overlaps an active K4L cohort')
    keep, tail = ctx['keep'], ctx['tail']
    uv = _uvp_state(state)
    lam = ctx['lam'][:columns]
    group = state['lazy']
    if group is None:
        if uv[0] != 'm':
            raise RuntimeError('TCH materialized UV layout')
        compact = _uvp_recon(U[keep], uv[1])
        delta = _tch_trace(U, uv[1], lam, tail)
    else:
        if uv[0] != 'l':
            raise RuntimeError('TCH lazy UV layout')
        mapped = _shared_old384_mm(U, group['map'])
        _shared_old384_finite(mapped, 'TCH mapped full basis')
        split = int(group['prefix'])
        prefix = _uvp_recon(mapped[keep], uv[1])
        recent = _uvp_recon(U[keep], uv[2])
        compact = fnp.concatenate([prefix, recent], axis=1)
        delta = (_tch_trace(mapped, uv[1], lam[:split], tail)
                 + _tch_trace(U, uv[2], lam[split:], tail))
    # Own expansion avoids the baseline DCR bookkeeping hooks.
    answer = fnp.zeros((m, columns), dtype=fnp.float32)
    fnp.put_along_axis(answer, keep[:, None], compact, axis=0)
    ctx['repair'], ctx['used'] = delta, True
    _TCH_STATS['calls'] += 1
    _TCH_STATS['rows'] += m
    _TCH_STATS['kept'] += int(keep.shape[0])
    _TCH_STATS['omitted'] += int(tail.shape[0])
    return answer


def _hcf_rows15(pp_rows, lam):
    out = _TCH_PARENT_HCF(pp_rows, lam)
    ctx = _TCH_CTX[0]
    if not _TCH_ON or ctx is None:
        return out
    if not ctx['used']:
        raise RuntimeError('TCH HCF before old reconstruction')
    if 'f15_diag' in out:
        out['f15_diag'] = _tch_hcf_proxy(out['f15_diag'], ctx['mu'], ctx['v'],
                                        ctx['keep'], ctx['tail'])
        _TCH_STATS['hcf_proxy_calls'] += 1
    return out


def _l15_rowcut_k3(layer, mu, v, w32, p, previous, newborn, lam, state, star):
    if not _TCH_ON or layer != 15:
        return _TCH_PARENT_L15(layer, mu, v, w32, p, previous, newborn,
                               lam, state, star)
    if _TCH_CTX[0] is not None:
        raise RuntimeError('TCH nested call')
    if tuple(_HCF_COLUMNS) != ('f15_diag',):
        raise RuntimeError('TCH only audited for the active f15_diag consumer')
    z = fnp.abs(mu / fnp.sqrt(fnp.maximum(v, 1e-10)))
    rows, unused_mask = select(_L15_ROWCUT_Z - z, 0.0)
    m = int(rows.shape[0])
    count = _tch_keep_count(m)
    if count == m:
        _TCH_STATS['fallback_small'] += 1
        return _TCH_PARENT_L15(layer, mu, v, w32, p, previous, newborn,
                               lam, state, star)
    # Stable sort breaks |alpha| ties by the incumbent response-row order.
    ordered = fnp.argsort(-z[rows], kind='stable')
    ctx = dict(rows=rows, keep=ordered[:count], tail=ordered[count:],
               mu=mu[rows], v=v[rows], lam=lam, used=False, repair=None)
    _TCH_CTX[0] = ctx
    try:
        answer = _TCH_PARENT_L15(layer, mu, v, w32, p, previous, newborn,
                                 lam, state, star)
        if not ctx['used'] or ctx['repair'] is None:
            raise RuntimeError('TCH reconstruction hook was not reached')
        full_tail = rows[ctx['tail']]
        fnp.put(answer, full_tail, answer[full_tail] + ctx['repair'])
        return answer
    finally:
        _TCH_CTX[0] = None
