# BEGIN OPUS-ENGINEER STATE READOUTS (a43 ro v1, 2026-10-10; changes of what a layer hands on, OFF unless weights are given). The
# trainer's readouts (o2) and (o3) of batch 4 (O/trainer/otr_m.py, comment READOUTS; applied to the step's outputs at its end):
#   (o2) the carried b:  b_i <- b_i + sigma_i r_l (state_i . B[l]),   sigma_i = sqrt of the post variance the layer hands on,
#        r_l = the layer's rms of b / sigma over all units, state = the learned correction's 64 numbers per unit of this layer;
#   (o3) the radial:     radial <- radial (1 + mean_i(state_i) . RAD[l]);
# layers 1-14, float64 as in torch. The state is taken from gnn_step (wrapped: it records the state it returns); the two lines run
# at the end of the layer, after every use of b and the radial inside it.
_RO_B = None             # 16 x 64 weights of (o2) (a tuple of 16 tuples; rows 1-14 used) or None
_RO_RAD = None           # 16 x 64 weights of (o3) or None
_RO_STATE = dict(l=None, state=None)
_RO_CONST = {}
_RO_STATS = dict(b=0, radial=0)
_RO_PARENT_GNN = gnn_step


def gnn_step(l, x, sigma, W, state, P):
    delta, new_state = _RO_PARENT_GNN(l, x, sigma, W, state, P)
    _RO_STATE['l'], _RO_STATE['state'] = l, new_state
    return delta, new_state


def _ro_state64(l):
    K = _RO_CONST
    if K.get('l') != l or K.get('of') is not _RO_STATE['state']:
        K.update(l=l, of=_RO_STATE['state'], s64=_RO_STATE['state'].astype(fnp.float64))
    return K['s64']


def _ro_b(l, b, C):
    if _RO_B is None or not (1 <= l <= 14) or _RO_STATE['l'] != l:
        return b
    hold = _E2Hold()
    K = _RO_CONST
    if 'B' not in K:
        K['B'] = fnp.asarray(_RO_B, dtype=fnp.float64)
    sigma = hold(fnp.sqrt(hold(fnp.maximum(hold(fnp.diag(C)), 1e-12))))
    ratio = hold(b / sigma)
    r = hold(fnp.sqrt(hold(fnp.mean(hold(ratio * ratio)))))
    _RO_STATS['b'] += 1
    return b + hold(hold(sigma * r) * hold(_ro_state64(l) @ hold(K['B'][l])))


def _ro_radial(l, radial):
    if _RO_RAD is None or not (1 <= l <= 14) or _RO_STATE['l'] != l:
        return radial
    hold = _E2Hold()
    K = _RO_CONST
    if 'RAD' not in K:
        K['RAD'] = fnp.asarray(_RO_RAD, dtype=fnp.float64)
    _RO_STATS['radial'] += 1
    return radial * hold(1.0 + hold(hold(fnp.mean(_ro_state64(l), axis=0)) @ hold(K['RAD'][l])))
# END OPUS-ENGINEER STATE READOUTS
