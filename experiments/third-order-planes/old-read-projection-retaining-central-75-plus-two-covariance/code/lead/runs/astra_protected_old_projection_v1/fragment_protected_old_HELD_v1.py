# Fixed protected old-only row contraction. Young reads are unchanged.
_POP_PARENT_INNER = _RTT_PARENT_INNER
_POP_PARENT_PREDICT = Estimator.predict
_POP_STATS = dict(reads=[], retired=True)

def _pop_inner(old_left, lam_old, state):
    layer = int(_RS_LAYER[0])
    if not _RTT_ACTIVE or layer not in range(7, 15):
        return _POP_PARENT_INNER(old_left, lam_old, state)
    m = int(old_left.shape[0])
    center = ((3 * m + 255) // 256) * 64 - 2
    record = dict(layer=layer, m=m, center=center, basis_rank=None, changed=False)
    if center < 1 or center + 2 >= m:
        _POP_STATS['reads'].append(record)
        return _POP_PARENT_INNER(old_left, lam_old, state)
    hold = _E2Hold()
    rows = _E2_LAYER['ecut_rows']
    assert tuple(rows.shape) == (m,) and _ASTRA_RLF_CONTEXT[0] is None
    Q = hold(_rtp_qr_basis(hold(_E2_LAYER['vc'][rows]), hold(_E2_LAYER['mu'][rows]), hold))
    record['basis_rank'] = int(Q.shape[1])
    if int(Q.shape[1]) != 2:
        _POP_STATS['reads'].append(record)
        return _POP_PARENT_INNER(old_left, lam_old, state)
    projected = hold(Q.T @ old_left)
    packed = hold(fnp.concatenate((hold(old_left[:center]), projected), axis=0))
    assert int(packed.shape[0]) == center + 2 and (center + 2) % 64 == 0
    inner = hold(_POP_PARENT_INNER(packed, lam_old, state))
    assert int(inner.shape[0]) == center + 2
    outer = hold(hold(Q[center:]) @ hold(inner[center:]))
    result = fnp.concatenate((hold(inner[:center]), outer), axis=0)
    assert int(result.shape[0]) == m
    record['changed'] = True
    _POP_STATS['reads'].append(record)
    return result

_RTT_PARENT_INNER = _pop_inner

def _pop_predict(self, *args, **kwargs):
    _POP_STATS['reads'].clear()
    _POP_STATS['retired'] = False
    try:
        return _POP_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _POP_STATS['retired'] = True

Estimator.predict = _pop_predict
