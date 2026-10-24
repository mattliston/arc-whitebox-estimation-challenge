# Fixed all-eight-layer post-clamp residual ridge; filled only after fixed CV gate.
_LMR_LITERAL = __LMR_COEFFICIENTS__
_LMR_MAPS = {}
_LMR_PENDING = [None]
_LMR_ACTIVE = [False]
_LMR_STATS = dict(gnn=[], applied=[], retired=True, state_identity=True)
_LMR_PARENT_SETUP = Estimator.setup
_LMR_PARENT_RESET = _pb_reset
_LMR_PARENT_PREDICT = Estimator.predict
_LMR_PARENT_GNN = gnn_step
_LMR_PARENT_EFFECTIVE = _astra_fhm_effective

def _lmr_maps():
    if not _LMR_MAPS:
        matrix = fnp.asarray(_LMR_LITERAL, dtype=fnp.float64)
        assert tuple(matrix.shape) == (8, 95)
        _LMR_MAPS['owner'] = matrix
        for i in range(8):
            row = matrix[i]
            _LMR_MAPS[i+7] = (row[:30], row[30:94], row[94])
    return _LMR_MAPS

def _lmr_setup(self, context):
    _LMR_PARENT_SETUP(self, context)
    if int(getattr(context,'width',0)) != 1024 or int(getattr(context,'depth',0)) != 16:
        return
    try:
        _lmr_maps()
    except Exception as exc:
        if type(exc).__name__ != 'NoBudgetContextError':raise
        with flops.BudgetContext(flop_budget=2**40,quiet=True,namespace='local_mean_const'):
            _lmr_maps()

def _pb_reset(saved):
    _LMR_MAPS.clear()
    return _LMR_PARENT_RESET(saved)

def gnn_step(l,x,sigma,W,state,P):
    answer = _LMR_PARENT_GNN(l,x,sigma,W,state,P)
    if not _LMR_ACTIVE[0] or int(l) not in range(7,15):return answer
    assert _LMR_PENDING[0] is None and tuple(x.shape)==(int(sigma.shape[0]),30)
    assert tuple(answer[1].shape)==(int(sigma.shape[0]),64)
    hold=_E2Hold();a,b,c=_lmr_maps()[int(l)]
    physical=hold(fnp.asarray(x,dtype=fnp.float64));hidden=hold(fnp.asarray(answer[1],dtype=fnp.float64))
    normalized=hold(hold(physical@a)+hold(hidden@b))+c
    correction=hold(hold(.5*hold(fnp.asarray(sigma,dtype=fnp.float64)))*normalized)
    _LMR_PENDING[0]=(int(l),correction)
    _LMR_STATS['gnn'].append(int(l))
    return answer

def _astra_fhm_effective(layer,mean,delta):
    original = _LMR_PARENT_EFFECTIVE(layer,mean,delta)
    if not _LMR_ACTIVE[0] or int(layer) not in range(7,15):return original
    pending=_LMR_PENDING[0];_LMR_PENDING[0]=None
    assert pending is not None and pending[0]==int(layer)
    hold=_E2Hold()
    result=fnp.maximum(hold(original+pending[1]),hold(-mean))
    _LMR_STATS['applied'].append(int(layer))
    return result

def _lmr_predict(self,mlp,budget):
    assert not _LMR_ACTIVE[0] and _LMR_PENDING[0] is None
    _LMR_STATS.update(gnn=[],applied=[],retired=False,state_identity=True)
    _LMR_ACTIVE[0]=int(mlp.width)==1024 and int(mlp.depth)==16
    try:
        answer=_LMR_PARENT_PREDICT(self,mlp,budget)
        if _LMR_ACTIVE[0]:assert _LMR_STATS['gnn']==_LMR_STATS['applied']==list(range(7,15))
        return answer
    finally:
        _LMR_ACTIVE[0]=False;_LMR_PENDING[0]=None;_LMR_STATS['retired']=True

Estimator.setup=_lmr_setup
Estimator.predict=_lmr_predict
