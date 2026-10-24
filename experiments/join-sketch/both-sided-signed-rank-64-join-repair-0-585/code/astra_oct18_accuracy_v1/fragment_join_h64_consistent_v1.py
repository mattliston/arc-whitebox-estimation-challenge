# HELD ASTRA L13 transient signed-h restoration, fixed residual rank64.
# Corrected pp feeds the existing NP fold, scalar K3 and other current-layer
# consumers. The existing old right read gets its matching low-rank delta.
# Future Q/F/core bank factors remain unchanged; this is a one-layer repair.
_JHR_CONTEXT = [None]
_JHR_STATS = dict(prepared=0, applied=0, consumed=0)


def _jhr_prepare(w32, previous, J, Qn, Fj, start, stop, weight, state):
    if _JHR_CONTEXT[0] is not None:
        raise RuntimeError('JHR overlapping context')
    ip, im = [], []
    expected = 0
    first, last = _JOIN_SCHEDULE[13]
    for birth in range(first, last + 1):
        absolute, width = state['cohorts'][birth]
        at, width = int(absolute) - int(start), int(width)
        if at != expected or width <= 0 or width % 2:
            raise RuntimeError('JHR complete signed cohort ownership')
        half = width // 2
        ip.extend(range(at, at + half))
        im.extend(range(at + half, at + width))
        expected += width
    if expected != stop - start or expected != int(J.shape[1]):
        raise RuntimeError('JHR joining range')
    ip, im = fnp.asarray(ip, dtype=fnp.int64), fnp.asarray(im, dtype=fnp.int64)
    h = (J[:, ip] - J[:, im]) * 0.5
    fh = (Fj[:, ip] - Fj[:, im]) * 0.5
    energy = fnp.maximum(fnp.sum(h * h, axis=0) - fnp.sum(fh * fh, axis=0), 0.0) * weight[ip]
    cols = fnp.argsort(-energy.astype(fnp.float64), kind='stable')[:64]
    residual = h[:, cols] - _shared_old384_mm(Qn, fh[:, cols])
    # A second projection prevents tiny residuals from amplifying leakage into
    # the already-retained common space under float32 product rounding.
    residual = residual - _shared_old384_mm(Qn, _shared_old384_mm(Qn.T, residual))
    basis, unused = fnp.linalg.qr(residual, mode='reduced')
    del unused, residual, fh, energy, cols
    coefficient = _shared_old384_mm(basis.T, h).copy()
    pushed = _shared_old384_mm(w32[previous, :].T, basis).copy()
    _JHR_CONTEXT[0] = dict(start=int(start), stop=int(stop), ip=ip, im=im,
                           coefficient=coefficient, pushed=pushed, applied=False)
    _JHR_STATS['prepared'] += 1


def _jhr_apply(pp):
    context = _JHR_CONTEXT[0]
    if context is None:
        return pp
    if context['applied']:
        raise RuntimeError('JHR applied twice')
    ctx = _ASTRA_BCC_CTX[0]
    if ctx is None or int(pp.shape[0]) != 1024:
        raise RuntimeError('JHR needs owned full-row bank-copy output')
    # Keep the existing deadcut and omitted-tail rows unchanged. Their original
    # K3 trace repair still describes the same omitted original bank.
    outer = ctx['outer_rows']
    mask = fnp.ones(1024, dtype=fnp.float32) if outer is None else fnp.zeros(1024, dtype=fnp.float32)
    if outer is not None:
        fnp.put(mask, outer, 1.0)
    tail = _OTT_DATA.get('full_tail')
    if tail is not None:
        fnp.put(mask, tail, 0.0)
    pushed = context['pushed'] * mask[:, None]
    context['pushed'] = pushed
    delta = _shared_old384_mm(pushed, context['coefficient'])
    block = pp[:, context['start']:context['stop']]
    ip, im = context['ip'], context['im']
    plus, minus = block[:, ip] + delta, block[:, im] - delta
    fnp.put_along_axis(block, ip[None, :], plus, axis=1)
    fnp.put_along_axis(block, im[None, :], minus, axis=1)
    context['applied'] = True
    _JHR_STATS['applied'] += 1
    return pp


_JHR_PARENT_READ = _cp_read_young_sq


def _cp_read_young_sq(left, young_right, lam, live_indices, state):
    out = _JHR_PARENT_READ(left, young_right, lam, live_indices, state)
    context = _JHR_CONTEXT[0]
    if context is None:
        return out
    if not context['applied'] or _RS_LAYER[0] != 13:
        raise RuntimeError('JHR unmatched current-layer read')
    if _RS.get(13, (1.0, 1.0))[0] != 1.0:
        raise RuntimeError('JHR pinned old-read scale')
    start, stop = context['start'], context['stop']
    ip, im = context['ip'], context['im']
    block = left[:, start:stop]
    # lambda_minus = -lambda_plus; delta_right_minus = -delta_right_plus.
    # This includes the parent's corrected NP left factor, not just raw pp^2.
    combined = (block[:, ip] + block[:, im]) * lam[start:stop][ip][None, :]
    inner = _shared_old384_mm(combined, context['coefficient'].T)
    right_delta = _shared_old384_mm(inner, context['pushed'][live_indices, :].T)
    out = out + right_delta
    _JHR_CONTEXT[0] = None
    _JHR_STATS['consumed'] += 1
    return out


_JHR_PARENT_PREDICT = Estimator.predict


def _jhr_predict(self, mlp, budget):
    _JHR_CONTEXT[0] = None
    try:
        return _JHR_PARENT_PREDICT(self, mlp, budget)
    finally:
        _JHR_CONTEXT[0] = None


Estimator.predict = _jhr_predict
