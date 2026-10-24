# Fixed negative-only a<-2 response-row rank2 complement; all positive rows remain exact.
_NR2_ON = True
_NR2_PARENT_READ = _RC_PARENT_READ
_NR2_PARENT_INNER = _cp_slr_inner_sq
_NR2_PARENT_PREDICT = Estimator.predict
assert _NR2_PARENT_READ is _rtt_read and _RC_COLZ == 2.0
_NR2_CONTEXT = [None]
_NR2_ACTIVE = [False]
_NR2_STATS = dict(reads=[], retired=True)

def _cp_slr_inner_sq(old_left, lam_old, state):
    context = _NR2_CONTEXT[0]
    if context is None:
        return _NR2_PARENT_INNER(old_left, lam_old, state)
    assert state is context['state'] and context['inner'] is None
    assert tuple(old_left.shape) == (context['m2'], context['old'])
    # Original full-row old inner, computed once; the existing central read receives a view.
    inner = _RTT_PARENT_INNER(context['full_old_left'], lam_old, state)
    assert tuple(inner.shape) == (context['m'], context['rank'])
    context['inner'] = inner
    context['inner_calls'] += 1
    return fnp.take(inner, context['keep'], axis=0)

def _nr2_read(left, young_right, lam, live_indices, state):
    if not (_NR2_ON and _NR2_ACTIVE[0] and _RTT_ACTIVE):
        return _NR2_PARENT_READ(left, young_right, lam, live_indices, state)
    layer = int(_RS_LAYER[0])
    assert 1 <= layer <= 14 and _NR2_CONTEXT[0] is None and _RTT_CONTEXT[0] is None
    hold = _E2Hold()
    m, k, c2 = int(left.shape[0]), int(young_right.shape[1]), int(young_right.shape[0])
    old = int(state['columns']); rank = int(state['pre_basis'].shape[1]) if old else 0
    assert tuple(left.shape) == (m, old+k) and tuple(lam.shape) == (old+k,)
    rows = _E2_LAYER['ecut_rows']; assert tuple(rows.shape) == (m,)
    alpha = hold(_E2_LAYER['a'][rows])
    negative = int(fnp.nonzero(hold(alpha < -2.0))[0].shape[0])
    omitted_count = 32*(negative//32)
    m2 = m-omitted_count
    info = dict(layer=layer,m=m,m2=m2,negative_tail=negative,omitted_count=omitted_count,young_k=k,c2=c2,old_rank=rank,row_changed=m2<m,delegated=m2==m,basis_rank=None,old_inner_calls=0)
    if m2 == m:
        out = _NR2_PARENT_READ(left,young_right,lam,live_indices,state)
        _NR2_STATS['reads'].append(info)
        return out
    order = hold(fnp.argsort(alpha,kind='stable'))
    omitted_rows = hold(order[:omitted_count])
    keep_rows = hold(fnp.sort(hold(order[omitted_count:])))
    inverse_order = hold(fnp.argsort(hold(fnp.concatenate((keep_rows,omitted_rows))),kind='stable'))
    floor = .5*_rc_math.erfc(1.5/_rc_math.sqrt(2.0))
    c15 = min(c2, ((int(fnp.nonzero(_E2_LAYER['g'] >= floor)[0].shape[0])+31)//32)*32)
    assert 0 < c15 <= c2 and int(live_indices.shape[0]) == c2
    context = dict(state=state,m=m,m2=m2,old=old,rank=rank,keep=keep_rows,full_old_left=hold(left[:,:old]) if old else None,inner=None,inner_calls=0)
    _NR2_CONTEXT[0] = context
    try:
        prefix_indices = hold(live_indices[:c15])
        prefix_right = hold(young_right[:c15])
        if m2:
            central_out = hold(_RTT_PARENT_READ(hold(fnp.take(left,keep_rows,axis=0)),prefix_right,lam,prefix_indices,state))
            assert tuple(central_out.shape) == (m2,c15)
        else:
            central_out = hold(fnp.zeros((0,c15),dtype=fnp.float32))
            if old:
                context['inner'] = hold(_RTT_PARENT_INNER(context['full_old_left'],hold(lam[:old]),state))
                context['inner_calls'] = 1
        Q = hold(_rtt_basis(hold(_E2_LAYER['vc'][rows]),hold(_E2_LAYER['mu'][rows]),hold))
        qrank = int(Q.shape[1]); info['basis_rank'] = qrank
        young_lam = hold(lam[old:])
        if qrank:
            scores = hold(hold(Q.T) @ hold(left[:,old:]))
            coefficients = hold(scores @ hold(hold(young_right*young_lam).T))
            prefix_fit = hold(hold(fnp.take(Q,omitted_rows,axis=0)) @ hold(coefficients[:,:c15]))
        else:
            scores = None
            prefix_fit = hold(fnp.zeros((m-m2,c15),dtype=fnp.float32))
        if old:
            assert context['inner_calls'] == 1 and context['inner'] is not None
            inner = context['inner']; assert tuple(inner.shape) == (m,rank)
            old_prefix_right = hold(_xsg_rows(5119,state['pre_basis'],prefix_indices))
            prefix_fit = hold(prefix_fit + hold(hold(fnp.take(inner,omitted_rows,axis=0)) @ hold(old_prefix_right.T)))
        else:
            assert context['inner_calls'] == 0
        prefix = hold(fnp.take(hold(fnp.concatenate((central_out,prefix_fit),axis=0)),inverse_order,axis=0))
        if c15 < c2:
            if qrank:
                tail = hold(Q @ hold(coefficients[:,c15:c2]))
            else:
                tail = hold(fnp.zeros((m,c2-c15),dtype=fnp.float32))
            if old:
                old_tail_right = hold(_xsg_rows(5119,state['pre_basis'],hold(live_indices[c15:c2])))
                tail = hold(tail + hold(inner @ hold(old_tail_right.T)))
            answer = fnp.concatenate((prefix,tail),axis=1)
        else:
            answer = prefix
        info['c15'] = c15; info['old_inner_calls'] = context['inner_calls']
        _NR2_STATS['reads'].append(info)
        # Truthful RTT record: row-only cases have a basis/inner despite no column cut.
        _RTT_STATS['reads'].append(dict(layer=layer,m=m,k=k,c15=c15,c2=c2,old_rank=rank,changed=c15<c2,basis_rank=qrank,old_inner_calls=context['inner_calls'],forced_d7=False,row_changed=True,m2=m2))
        return answer
    finally:
        _NR2_CONTEXT[0] = None

_RC_PARENT_READ = _nr2_read

def _nr2_predict(self,*args,**kwargs):
    assert not _NR2_ACTIVE[0] and _NR2_CONTEXT[0] is None
    _NR2_ACTIVE[0] = bool(_NR2_ON)
    _NR2_STATS['reads'].clear(); _NR2_STATS['retired'] = False
    try:
        return _NR2_PARENT_PREDICT(self,*args,**kwargs)
    finally:
        _NR2_CONTEXT[0] = None; _NR2_ACTIVE[0] = False; _NR2_STATS['retired'] = True

Estimator.predict = _nr2_predict
