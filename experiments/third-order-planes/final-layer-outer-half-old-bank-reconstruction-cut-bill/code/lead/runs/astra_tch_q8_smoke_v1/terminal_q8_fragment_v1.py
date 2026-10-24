# Fixed rank-eight third-Hermite repair, appended to preserved TCH v2.
# Capture eight retained row values already reconstructed by TCH; no truth fit.
_TQ8_RECON = _slr_reconstruct
_TQ8_UVP = _uvp_recon
_TQ8_CAPTURE = [None]
_TQ8_STATS = dict(calls=0, rank=8, captured_groups=0)


def _uvp_recon(U, uv):
    answer = _TQ8_UVP(U, uv)
    capture = _TQ8_CAPTURE[0]
    if capture is not None:
        count = int(U.shape[0])
        # Includes both endpoints; integer floor, no value-dependent pivoting.
        indices = fnp.asarray([(j * (count - 1)) // 7 for j in range(8)],
                              dtype=fnp.int64)
        capture.append((uv, answer[indices]))
    return answer


def _tq8_legs(uv, lam):
    plus, minus = [], []
    for start, width in uv[2]:
        plus.append(lam[start:start + width])
        minus.append(lam[start + width:start + 2 * width])
    return (plus[0] if len(plus) == 1 else fnp.concatenate(plus),
            minus[0] if len(minus) == 1 else fnp.concatenate(minus))


def _tq8_gradient(uv, sample, lam):
    Fu, Fv, blocks, ru = uv
    plus, minus = [], []
    for start, width in blocks:
        plus.append((sample[:, start:start + width] ** 2)
                    * lam[start:start + width])
        minus.append((sample[:, start + width:start + 2 * width] ** 2)
                     * lam[start + width:start + 2 * width])
    gp = plus[0] if len(plus) == 1 else fnp.concatenate(plus, axis=1)
    gm = minus[0] if len(minus) == 1 else fnp.concatenate(minus, axis=1)
    result = (gp - gm) @ Fv.T
    fnp.add(result[:, :ru], (gp + gm) @ Fu.T, out=result[:, :ru])
    return result


def _tq8_core(uv, lam, Q, pairs):
    Fu, Fv, blocks, ru = uv
    bu, bv = Q[:ru].T @ Fu, Q.T @ Fv
    lp, lm = _tq8_legs(uv, lam)
    bp, bm = bu + bv, bu - bv
    hp = fnp.stack([bp[j] * bp[k] for j, k in pairs])
    hm = fnp.stack([bm[j] * bm[k] for j, k in pairs])
    return ((hp * lp) @ bp.T + (hm * lm) @ bm.T).astype(fnp.float64)


def _slr_reconstruct(U, state):
    ctx = _TCH_CTX[0]
    if not _TCH_ON or ctx is None:
        return _TQ8_RECON(U, state)
    if _TQ8_CAPTURE[0] is not None or int(U.shape[1]) < 8:
        raise RuntimeError('q8 terminal scope or basis size')
    capture = []
    _TQ8_CAPTURE[0] = capture
    try:
        answer = _TQ8_RECON(U, state)
    finally:
        _TQ8_CAPTURE[0] = None
    group, columns = state['lazy'], int(state['columns'])
    lam = ctx['lam'][:columns]
    pairs = [(j, k) for j in range(8) for k in range(j, 8)]
    if group is None:
        if len(capture) != 1:
            raise RuntimeError('q8 materialized capture')
        uv, sample = capture[0]
        gradient = _tq8_gradient(uv, sample, lam)
        Q, unused_R = fnp.linalg.qr(gradient.T, mode='reduced')
        core = _tq8_core(uv, lam, Q, pairs)
    else:
        if len(capture) != 2:
            raise RuntimeError('q8 lazy capture')
        split = int(group['prefix'])
        (uv0, sample0), (uv1, sample1) = capture
        gradient = (_tq8_gradient(uv0, sample0, lam[:split]) @ group['map'].T
                    + _tq8_gradient(uv1, sample1, lam[split:]))
        Q, unused_R = fnp.linalg.qr(gradient.T, mode='reduced')
        mapped_Q = group['map'].T @ Q
        core = (_tq8_core(uv0, lam[:split], mapped_Q, pairs)
                + _tq8_core(uv1, lam[split:], Q, pairs))
    z = (U[ctx['tail']] @ Q).astype(fnp.float64)
    quadratic = fnp.stack([(1.0 if j == k else 2.0) * z[:, j] * z[:, k]
                           for j, k in pairs], axis=1)
    cubic = fnp.sum((quadratic @ core) * z, axis=1)
    trace = fnp.sum(fnp.stack([core[i] for i, (j, k) in enumerate(pairs)
                              if j == k]), axis=0)
    rho = fnp.sum(U * U) / (int(U.shape[0]) * int(U.shape[1]))
    ctx['repair'] = ctx['repair'] + cubic - (3.0 * rho) * (z @ trace)
    _TQ8_STATS['calls'] += 1
    _TQ8_STATS['captured_groups'] += len(capture)
    return answer
