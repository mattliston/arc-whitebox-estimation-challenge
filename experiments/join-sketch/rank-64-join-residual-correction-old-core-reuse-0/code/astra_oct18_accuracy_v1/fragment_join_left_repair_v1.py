# HELD ASTRA L13 ephemeral left-factor repair. No future-bank or K3 changes.
_JLR_MODE = 'exact256'
_JLR_CONTEXT = [None]
_JLR_STATS = dict(prepared=0, consumed=0, rows=0, columns=0)


def _jlr_prepare(w32, previous, J, Qn, Fj, start, stop, weight):
    if _JLR_CONTEXT[0] is not None:
        raise RuntimeError('JLR context was not consumed')
    response = _E2_LAYER['ecut_rows']
    if _JLR_MODE == 'exact256':
        a = _E2_LAYER['a'][response]
        score = fnp.exp(-0.5 * a * a) / _E2_LAYER['s'][response]
        positions = fnp.argsort(-score, kind='stable')[:min(256, int(response.shape[0]))]
        chosen = response[positions]
        actual = _shared_old384_mm(w32[previous, :][:, chosen].T, J).copy()
        relative = False
    elif _JLR_MODE == 'residual64':
        positions = None
        chosen = response
        score = fnp.maximum(fnp.sum(J * J, axis=0) - fnp.sum(Fj * Fj, axis=0), 0.0) * weight
        cols = fnp.argsort(-score.astype(fnp.float64), kind='stable')[:64]
        residual = J[:, cols] - _shared_old384_mm(Qn, Fj[:, cols])
        basis, unused = fnp.linalg.qr(residual, mode='reduced')
        del unused, residual
        coefficient = _shared_old384_mm(basis.T, J)
        pushed = _shared_old384_mm(w32[previous, :][:, chosen].T, basis)
        actual = _shared_old384_mm(pushed, coefficient).copy()
        relative = True
    else:
        raise RuntimeError('Unregistered JLR mode')
    if int(actual.shape[1]) != stop - start:
        raise RuntimeError('JLR joining cohort range')
    _JLR_CONTEXT[0] = (positions, chosen, int(start), int(stop), actual, relative)
    _JLR_STATS['prepared'] += 1
    _JLR_STATS['rows'] = int(actual.shape[0])
    _JLR_STATS['columns'] = int(actual.shape[1])


_JLR_PARENT_LEFT = _np_left


def _np_left(l, pp2, pp, response_indices, response_mask, alive, lam, mu, vc, vv, cov):
    # Parent computes and retains all NP/diagnostic terms from unchanged pp/pp2.
    left = _JLR_PARENT_LEFT(l, pp2, pp, response_indices, response_mask, alive, lam, mu, vc, vv, cov)
    context = _JLR_CONTEXT[0]
    if context is None:
        return left
    if l != 13:
        raise RuntimeError('JLR consumed outside L13')
    positions, chosen, start, stop, actual, relative = context
    _JLR_CONTEXT[0] = None
    # left is the parent's fresh owned matrix. Only raw CP squared-left content
    # changes, after NP subtraction. The subsequent read already multiplies lam.
    if relative:
        repaired = pp[chosen, start:stop] + actual
    else:
        repaired = actual
    delta = repaired * repaired - pp2[chosen, start:stop]
    block = left[:, start:stop]
    if positions is None:
        fnp.add(block, delta, out=block)
    else:
        updated = block[positions, :] + delta
        fnp.put_along_axis(block, positions[:, None], updated, axis=0)
    _JLR_STATS['consumed'] += 1
    return left


_JLR_PARENT_PREDICT = Estimator.predict


def _jlr_predict(self, mlp, budget):
    _JLR_CONTEXT[0] = None
    try:
        return _JLR_PARENT_PREDICT(self, mlp, budget)
    finally:
        _JLR_CONTEXT[0] = None


Estimator.predict = _jlr_predict
