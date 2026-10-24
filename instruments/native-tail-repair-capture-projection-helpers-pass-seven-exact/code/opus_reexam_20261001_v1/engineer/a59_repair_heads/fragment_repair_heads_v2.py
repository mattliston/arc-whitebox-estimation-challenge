# Separate old/young trace-repair heads. Builder emits this only for nonzero
# explicitly schema-bound weights. All numerical operations use flopscope fnp.
_RPH_U = ()
_RPH_STATE = {}
_RPH_YOUNG = [None]
_RPH_CONST = {}


def _rph_begin(layer):
    _RPH_STATE.clear()
    _RPH_STATE['layer'] = layer
    _RPH_YOUNG[0] = None


def _rph_const(n):
    K = _RPH_CONST
    if K.get('n') != n:
        K.clear()
        K.update(n=n, u=fnp.asarray(_RPH_U, dtype=fnp.float32),
                 zeros=fnp.zeros(n, dtype=fnp.float32))
    return K


def _rph_standardize(repair, rows, scale, hold):
    """All-unit sample moments, then retain only the actual component roster."""
    n = int(scale.shape[0])
    K = _rph_const(n)
    if repair is None:
        if rows is not None:
            raise RuntimeError('repair-head roster without component')
        return K['zeros']
    if rows is None or tuple(repair.shape) != (n,):
        raise RuntimeError('repair-head component ownership')
    raw = hold(repair / scale)
    mean = hold(fnp.sum(raw) / n)
    centred = hold(raw - mean)
    variance = hold(fnp.sum(hold(centred * centred)) / (n - 1))
    sd = hold(fnp.sqrt(variance))
    valid = hold(hold(fnp.isfinite(sd)) & hold(sd > 0.0))
    safe = hold(fnp.where(valid, sd, 1.0))
    selected = hold(fnp.clip(hold(hold(centred[rows]) / safe), -8.0, 8.0))
    selected = hold(fnp.where(valid, selected, 0.0))
    answer = fnp.zeros(n, dtype=fnp.float32)
    hold(fnp.put(answer, rows, hold(selected.astype(fnp.float32))))
    return answer


def _rph_diagonal(layer, diagonal, lam, vc, sigma):
    """Snapshot references before the original young wrapper clears its repair."""
    old = _OTT_DATA.get('repair')
    old_rows = _OTT_DATA.get('full_tail') if old is not None else None
    young, young_rows = _YNT_REPAIR[0], _RPH_YOUNG[0]
    try:
        result = _ott_diagonal(diagonal, lam)
        if 7 <= layer <= 14:
            if _RPH_STATE.get('layer') != layer:
                raise RuntimeError('repair-head layer ownership')
            hold = _E2Hold()
            scale = hold(vc * sigma)
            _RPH_STATE['columns'] = (
                _rph_standardize(old, old_rows, scale, hold),
                _rph_standardize(young, young_rows, scale, hold))
        return result
    except BaseException:
        _RPH_STATE.clear()
        raise
    finally:
        _RPH_YOUNG[0] = None


_RPH_PARENT_HC_BIAS = _hc_bias


def _hc_bias(layer, bias):
    try:
        out = _RPH_PARENT_HC_BIAS(layer, bias)
        if not 7 <= layer <= 14:
            return out
        if _RPH_STATE.get('layer') != layer:
            raise RuntimeError('repair-head bias layer mismatch')
        columns = _RPH_STATE.get('columns')
        if columns is None:
            raise RuntimeError('repair-head columns missing')
        hold = _E2Hold()
        Z = hold(fnp.stack(columns))  # exactly (2,1024), float32
        U = _rph_const(int(Z.shape[1]))['u']  # exactly (2,128), float32
        delta = hold(fnp.einsum('kn,kc->nc', Z, U, optimize=False))  # (1024,128)
        return out + delta  # incumbent HC5/HCF arithmetic completes first
    finally:
        _RPH_STATE.clear()
        _RPH_YOUNG[0] = None


_RPH_PARENT_PREDICT = _predict_binary


def _predict_binary(*args, **kwargs):
    _RPH_STATE.clear()
    _RPH_YOUNG[0] = None
    try:
        return _RPH_PARENT_PREDICT(*args, **kwargs)
    finally:
        _RPH_STATE.clear()
        _RPH_YOUNG[0] = None


_RPH_PARENT_CS_BUILD = _cs_build


def _cs_build(n=1024, depth=16):
    result = _RPH_PARENT_CS_BUILD(n, depth)
    _rph_const(n)
    return result


_CS_DICTS = tuple(_CS_DICTS) + ('_RPH_CONST', '_RPH_STATE')
