# Diagnostic reference-only observer. Appended after exact QR4d; never deploy.
# Observer is installed by the native judge only for TRAIN16 and must not mutate
# any array. It streams host copies before returning, so no layer payload remains.
_RCO_OBSERVER = None
_RCO_CONTEXT = [None]
_RCO_PARENT_READ = _RC_PARENT_READ
_RCO_PARENT_REFINE = _rtp_refine
assert _RCO_PARENT_READ is _rtt_read

def _rco_refine(prefix, Q, inner, old_right, hold):
    answer = _RCO_PARENT_REFINE(prefix, Q, inner, old_right, hold)
    context = _RCO_CONTEXT[0]
    if context is not None:
        assert context['refine'] is None
        context['refine'] = dict(Q=answer[0], prefix=prefix,
                                 old_inner=inner, old_right=old_right)
    return answer

def _rco_read(left, young_right, lam, live_indices, state):
    observer = _RCO_OBSERVER
    if observer is None:
        return _RCO_PARENT_READ(left, young_right, lam, live_indices, state)
    assert _RCO_CONTEXT[0] is None
    context = dict(refine=None)
    _RCO_CONTEXT[0] = context
    try:
        result = _RCO_PARENT_READ(left, young_right, lam, live_indices, state)
        record = dict(_RTT_STATS['reads'][-1])
        if record['changed']:
            assert context['refine'] is not None
            observer(record, dict(left=left, right=young_right, lam=lam,
                     old_columns=int(state['columns']), cols=live_indices,
                     rows=_E2_LAYER['ecut_rows'], mu=_E2_LAYER['mu'],
                     vc=_E2_LAYER['vc'], gate=_E2_LAYER['g'],
                     refine=context['refine']))
        return result
    finally:
        _RCO_CONTEXT[0] = None

_rtp_refine = _rco_refine
_RC_PARENT_READ = _rco_read
