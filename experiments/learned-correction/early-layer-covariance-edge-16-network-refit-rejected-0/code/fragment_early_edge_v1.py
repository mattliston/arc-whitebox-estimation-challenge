# BEGIN OPUS-ENGINEER a62 EARLY EDGE (2026-10-11 21:2xZ; the trainer's RAD-S48 arm 'edx', otr_m_v5.py comment EARLY EDGE). The learned covariance
# edge (trained for layers 8-14, weights frozen) ALSO applied at layers 1-7: C <- C + _EDX[l] x the BASE edge source called with layer 8 (the
# G5 base through _gn_message_edge_source_base: no GN edge factor, as the torch card's source, whose edge knob is a separate hook), added right
# after the GN covariance knobs (the torch card chains it after its 'cov_pm' hook, before the omitted-radial source; at layer 1, where the
# knobs do not run, it is added alone). The source's inputs are the regular edge's: the pair map's mean and aux, the layer's K21 and W[l],
# the head's state handed on by the previous layer, the head's parameters and the edge context (all through the head's own
# covariance_hook); the messages are shared with that layer's gnn_step by _e2_messages' (state, W) cache. _EDX = {} (no early edge) =
# the parent bit for bit: the wrappers only record references.
_EDX = {}
_EE_SIDE = {}
_EE_EARLY = [False]
_EE_STATS = dict(calls=0)
_EE_PARENT_PREDICT = _predict_binary
_EE_PARENT_NONLINEAR = nonlinear
_EE_PARENT_SOURCE = _message_edge_source
_EE_PARENT_DCOV = _gn_dcov
_EE_PARENT_COV_LAYERS = _GN_COV_LAYERS


def _predict_binary(*args, **kwargs):
    _EE_SIDE.clear()
    _EE_SIDE.update(hook=kwargs.get('covariance_hook'), W=(args[0] if args else kwargs['mlp']).weights)
    try:
        return _EE_PARENT_PREDICT(*args, **kwargs)
    finally:
        _EE_SIDE.clear()


def nonlinear(*args, **kwargs):
    out = _EE_PARENT_NONLINEAR(*args, **kwargs)
    _EE_SIDE['pair'] = (out[0], out[4])   # the pair map's mean and aux, for the early edge of this layer
    return out


def _message_edge_source(layer, m, K21, aux, w, state, P, context):
    if _EE_EARLY[0]:
        return _gn_message_edge_source_base(8, m, K21, aux, w, state, P, context)
    return _EE_PARENT_SOURCE(layer, m, K21, aux, w, state, P, context)


class _EECovLayers:
    def __contains__(self, l):
        return l in _EE_PARENT_COV_LAYERS or bool(_EDX.get(l))


_GN_COV_LAYERS = _EECovLayers()


def _gn_dcov(l, mu, Cpre, K21, k4, aux):
    scale = float(_EDX.get(l, 0.0))
    out = _EE_PARENT_DCOV(l, mu, Cpre, K21, k4, aux) if l in _EE_PARENT_COV_LAYERS else None
    if scale and 1 <= l <= 7:
        hook = _EE_SIDE.get('hook')
        m, pair_aux = _EE_SIDE['pair']
        if hook is None or pair_aux is not aux:
            raise RuntimeError('early edge: no covariance hook or a stale pair map')
        _EE_EARLY[0] = True
        try:
            edge = hook(l, m, K21, aux, _EE_SIDE['W'][l])
        finally:
            _EE_EARLY[0] = False
        _EE_STATS['calls'] = _EE_STATS['calls'] + 1
        term = edge * scale
        out = term if out is None else out + term
    if out is None:
        raise RuntimeError('early edge: layer %d asked for the knobs without a term' % l)
    return out
# END OPUS-ENGINEER a62 EARLY EDGE
