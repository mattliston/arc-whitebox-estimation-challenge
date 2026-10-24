# Fixed base-difference read-only224. Full reconstruction and diagonal untouched.
_PRD_ACTIVE = False
_PRD_FRAME = [None]
_PRD_CAPTURE = [None]
_PRD_STATS = dict(builds=[], reads=[], retired=True)
_PRD_PARENT_RECON = _slr_reconstruct
_PRD_PARENT_INNER = _uvp_inner
_PRD_PARENT_PREDICT = Estimator.predict

def _slr_reconstruct(U, state):
    answer = _PRD_PARENT_RECON(U, state)
    if _PRD_ACTIVE and _UVP_CUR[0] == 10 and state['lazy'] is None:
        assert int(state['rank']) == 320 and int(U.shape[1]) == 320
        _PRD_CAPTURE[0] = (U, state['F'])
    return answer

def _prd_difference(Dv, Fv, uv, lam_cols):
    run = _UBC_RUN[0]
    if (not _PRD_ACTIVE or _UVP_CUR[0] not in (10,11,12,13,14)
            or int(Fv.shape[0]) != 320 or run is None or run['uv'] is not uv):
        return _ubx_product(Dv, Fv.T, uv, 'vt')
    frame = _PRD_FRAME[0]
    if frame is None:
        assert _UVP_CUR[0] == 10 and _PRD_CAPTURE[0] is not None
        U, owner = _PRD_CAPTURE[0]
        assert run['owner'] is owner and 0 < int(U.shape[0]) <= 1024
        h = _E2Hold(); P = int(Fv.shape[1])
        weight = h(fnp.empty((P,), dtype=fnp.float32)); b = 0
        for a,k in uv[2]:
            plus = h(lam_cols[a:a+k]); minus = h(lam_cols[a+k:a+2*k])
            assert bool(fnp.all(minus == h(-plus)))
            h(fnp.multiply(plus, plus, out=weight[b:b+k])); b += k
        assert b == P
        power = h(fnp.sum(h(h(Fv*Fv)*weight[None,:]), axis=1, dtype=fnp.float64))
        norm = h(fnp.sum(h(U*U), axis=0, dtype=fnp.float64))
        score = h(power*norm)
        assert bool(fnp.all(fnp.isfinite(score))) and bool(fnp.all(score >= 0))
        order = h(fnp.argsort(h(-score), kind='stable'))
        selected = fnp.sort(order[:224]); compact = Fv[selected]
        frame = _PRD_FRAME[0] = (owner, run['generation'], selected, compact)
        _PRD_CAPTURE[0] = None
        _PRD_STATS['builds'].append([10,int(U.shape[0]),320,P,224])
    owner, generation, selected, compact = frame
    assert run['owner'] is owner and run['generation'] == generation
    assert tuple(compact.shape) == (224,int(Dv.shape[1]))
    assert _UBX_CTX[0] is None
    if _UBX_STATE['run'] is not run:
        _UBX_STATE['run'], _UBX_STATE['bytes'] = run, 0
        _UBX_STATE['cache'].clear()
    _UBX_CTX[0] = dict(tag=('prd224',id(owner),generation,int(compact.shape[1]),224))
    _UBX_STATS['products'] += 1
    try:
        rows,P = map(int,Dv.shape)
        plan = _hz_plan(rows,P,224,True)
        assert plan is not None
        reduced = fnp.empty((rows,224),dtype=fnp.float32)
        _prd224_ks_product(Dv,compact.T,reduced,rows,P,224,plan)
    finally:
        _UBX_CTX[0] = None
    inner = fnp.zeros((rows,320),dtype=fnp.float32)
    fnp.put_along_axis(inner,selected[None,:],reduced,axis=1)
    _PRD_STATS['reads'].append([int(_UVP_CUR[0]),rows,P,224])
    return inner

def _uvp_inner(left_sq, lam_cols, uv):
    if not _PRD_ACTIVE:
        return _PRD_PARENT_INNER(left_sq,lam_cols,uv)
    return _prd224_inner(left_sq,lam_cols,uv)

def _prd_predict(self,mlp,flop_budget):
    global _PRD_ACTIVE
    assert not _PRD_ACTIVE and _PRD_FRAME[0] is None and _PRD_CAPTURE[0] is None
    _PRD_STATS.update(builds=[],reads=[],retired=False)
    _PRD_ACTIVE = True
    try:
        return _PRD_PARENT_PREDICT(self,mlp,flop_budget)
    finally:
        _PRD_ACTIVE = False
        _PRD_FRAME[0] = None
        _PRD_CAPTURE[0] = None
        _PRD_STATS['retired'] = True

Estimator.predict = _prd_predict
