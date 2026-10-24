# BEGIN OPUS-ENGINEER CONSTANTS AT SETUP (cs v1, a40, 2026-10-02; predictions bit-identical: the same objects, built by the same
# functions with the same arguments, only earlier; the release lane's request of 16:19 UTC). What a worker's FIRST network builds
# once per process is built at the head of the binding phase instead: the decoded literals of the learned parts, the learned
# correction's buffers and its parameter views per layer, the feature columns, the birth head's rows, the pair map's persistent
# buffers and views, the edge's planes, the birth update's stack, the weight-power stack, and the tables of the pair law, of the
# exact-term mixture and of K4L. About 465 requests leave every first network (release census of the S41 stack, sha 8d110449:
# O/release/runs/c_s41stack_sites_100first/first_only_sites.txt), on any worker, because they come before the first tree key and
# the wall guard only ever cuts tree keys. The arena is created here exactly as the binding phase creates it (the pair map's
# views live in it); the binding phase then finds it.
# Never fails setup: any failure clears the caches this fragment fills, and they bind lazily in predict as before. The reset path
# (binding failed, or the arena is gone at the first predict) clears them too: everything here holds handles made in setup.
_CS_PARENT_BIND = _pb_bind
_CS_PARENT_RESET = _pb_reset
_CS_DICTS = ('_E2_GNN_STATE', '_E2_FEAT', '_E2_BDNH', '_E2_EDGE_STATE', '_E2_PAIR_STATE', '_E2_PAIR_ARENA', '_E2_WPOW', '_BD32_STATE',
             '_NP_CACHE', '_F4C_CONST', '_K4L_CONST')
_CS_NONES = ('_GNN_PARAMS', '_MESSAGE_EDGE_PARAMETERS', '_EDGE_PARAMETERS', '_BDNH_PARAMETERS', '_BD_COEFFICIENTS')


def _cs_clear():
    g = globals()
    for name in _CS_DICTS:
        if name in g:
            g[name].clear()
    for name in _CS_NONES:
        if name in g:
            g[name] = None


def _cs_build(n=1024, depth=16):
    global _GNN_PARAMS, _EDGE_PARAMETERS
    g = globals()
    hold = _E2Hold()   # temporaries of this call, released together
    if _GNN_PARAMS is None:                               # as _predict_learned_compact
        _GNN_PARAMS = _unpack_static_fp32(_GNN_LITERAL)
    P = _GNN_PARAMS
    if _MESSAGE_EDGE_ENABLED:
        _message_edge_parameters()
        if _EDGE_PARAMETERS is None:                      # as _message_edge_source
            _EDGE_PARAMETERS = {k: fnp.asarray(v, dtype=fnp.float32) for k, v in _EDGE_LITERAL.items()}
        _e2_edge_consts(_EDGE_PARAMETERS, n)
    _bdnh_parameters()
    cstar_all = _bd_coefficients()
    for l in range(1, depth - 1):                         # the birth head's constant rows, layers 1..14, as _e2_bdnh_coefficients
        c = _E2_BDNH.get(l)
        if c is None or c[0] is not cstar_all:
            cstar = cstar_all[l]
            _E2_BDNH[l] = (cstar_all, cstar, 1 - hold(fnp.abs(cstar)))
    k = _e2_gnn_consts(P, n)
    for l in range(1, depth):                             # gnn_step and features run at layers 1..15
        _e2_gnn_layer(k, l)
        _e2_feature_columns(n, l)
    st = _e2_pair_state(n)
    _e2_pair_arena(n)
    _psp_views(st, n)
    _e1gn_state(st, n)
    if '_bd32_state' in g:
        _bd32_state(n)
    _e2_wpow(hold(fnp.zeros((n, n), dtype=fnp.float32)))['W'] = None   # the stack and its views; the first layer's weights fill it
    if '_np_const' in g:
        _np_const(n)
    if '_f4c_const' in g:
        C = _f4c_const()
        if 'rowi' in C and ('one', n) not in C:           # F4c v2's row of ones
            C[('one', n)] = fnp.ones(n)
    if '_k4l_const' in g:
        C = _k4l_const()
        if 'idxA' in C and 'one' not in C:                # K4L v5's row of ones, as _k4l_birth
            C['one'] = fnp.ones(n, dtype=fnp.float64)


def _pb_bind(t0):
    global _SCRATCH
    if _SCRATCH is None:   # exactly the first predict's arena setup (_predict_binary), as the binding phase does it
        arena = fnp.empty((_PACKED_ELEMENTS,), dtype=fnp.float32)
        _SCRATCH = [arena, _cp3_bind_arena_views(arena, _PACKED_PUSH), _cp3_bind_arena_views(arena, _PACKED_READ), None, None, None, None]
        _bind_push6(_SCRATCH[0])
    try:
        _cs_build()
        _PB_LOG.append(('constants', round(_pb_time.monotonic() - t0, 3)))
    except Exception as exc:
        _cs_clear()
        _PB_LOG.append(('constants left to predict: ' + type(exc).__name__, round(_pb_time.monotonic() - t0, 3)))
    _CS_PARENT_BIND(t0)


def _pb_reset(saved):
    try:
        return _CS_PARENT_RESET(saved)
    finally:
        _cs_clear()
# END OPUS-ENGINEER CONSTANTS AT SETUP
