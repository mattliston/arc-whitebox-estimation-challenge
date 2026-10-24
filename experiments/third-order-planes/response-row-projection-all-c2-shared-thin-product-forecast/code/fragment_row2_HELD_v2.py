# Fixed z2 response-row rank2 complement on the submitted e406 column-tail scheme.
_RR2_ON = True
_RR2_PARENT_READ = _RC_PARENT_READ
_RR2_PARENT_INNER = _cp_slr_inner_sq
_RR2_PARENT_PREDICT = Estimator.predict
assert _RR2_PARENT_READ is _rtt_read and _RC_COLZ == 2.0
_RR2_CONTEXT = [None]
_RR2_ACTIVE = [False]
_RR2_STATS = dict(reads=[], retired=True)

def _cp_slr_inner_sq(old_left, lam_old, state):
    context = _RR2_CONTEXT[0]
    if context is None:
        return _RR2_PARENT_INNER(old_left, lam_old, state)
    assert state is context['state'] and context['inner'] is None
    assert tuple(old_left.shape) == (context['m2'], context['old'])
    # Original full-row old inner, computed once; the existing central read receives a view.
    inner = _RTT_PARENT_INNER(context['full_old_left'], lam_old, state)
    assert tuple(inner.shape) == (context['m'], context['rank'])
    context['inner'] = inner
    context['inner_calls'] += 1
    return inner[:context['m2']]

def _rr2_read(left, young_right, lam, live_indices, state):
    if not (_RR2_ON and _RR2_ACTIVE[0] and _RTT_ACTIVE):
        return _RR2_PARENT_READ(left, young_right, lam, live_indices, state)
    layer = int(_RS_LAYER[0])
    assert 1 <= layer <= 14 and _RR2_CONTEXT[0] is None and _RTT_CONTEXT[0] is None
    hold = _E2Hold()
    m, k, c2 = int(left.shape[0]), int(young_right.shape[1]), int(young_right.shape[0])
    old = int(state['columns']); rank = int(state['pre_basis'].shape[1]) if old else 0
    assert tuple(left.shape) == (m, old+k) and tuple(lam.shape) == (old+k,)
    rows = _E2_LAYER['ecut_rows']; assert tuple(rows.shape) == (m,)
    alpha = hold(_E2_LAYER['a'][rows])
    central = int(fnp.nonzero(hold(fnp.absolute(alpha) <= 2.0))[0].shape[0])
    m2 = min(m, ((central+31)//32)*32)
    info = dict(layer=layer,m=m,m2=m2,young_k=k,c2=c2,old_rank=rank,row_changed=m2<m,delegated=m2==m,basis_rank=None,old_inner_calls=0)
    if m2 == m:
        out = _RR2_PARENT_READ(left,young_right,lam,live_indices,state)
        _RR2_STATS['reads'].append(info)
        return out
    floor = .5*_rc_math.erfc(1.5/_rc_math.sqrt(2.0))
    c15 = min(c2, ((int(fnp.nonzero(_E2_LAYER['g'] >= floor)[0].shape[0])+31)//32)*32)
    assert 0 < c15 <= c2 and int(live_indices.shape[0]) == c2
    context = dict(state=state,m=m,m2=m2,old=old,rank=rank,full_old_left=hold(left[:,:old]) if old else None,inner=None,inner_calls=0)
    _RR2_CONTEXT[0] = context
    try:
        prefix_indices = hold(live_indices[:c15])
        prefix_right = hold(young_right[:c15])
        if m2:
            central_out = hold(_RTT_PARENT_READ(hold(left[:m2]),prefix_right,lam,prefix_indices,state))
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
            prefix_fit = hold(hold(Q[m2:]) @ hold(coefficients[:,:c15]))
        else:
            scores = None
            prefix_fit = hold(fnp.zeros((m-m2,c15),dtype=fnp.float32))
        if old:
            assert context['inner_calls'] == 1 and context['inner'] is not None
            inner = context['inner']; assert tuple(inner.shape) == (m,rank)
            old_prefix_right = hold(_xsg_rows(5119,state['pre_basis'],prefix_indices))
            prefix_fit = hold(prefix_fit + hold(hold(inner[m2:]) @ hold(old_prefix_right.T)))
        else:
            assert context['inner_calls'] == 0
        prefix = hold(fnp.concatenate((central_out,prefix_fit),axis=0))
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
        _RR2_STATS['reads'].append(info)
        # Truthful RTT record: row-only cases have a basis/inner despite no column cut.
        _RTT_STATS['reads'].append(dict(layer=layer,m=m,k=k,c15=c15,c2=c2,old_rank=rank,changed=c15<c2,basis_rank=qrank,old_inner_calls=context['inner_calls'],forced_d7=False,row_changed=True,m2=m2))
        return answer
    finally:
        _RR2_CONTEXT[0] = None

_RC_PARENT_READ = _rr2_read

def _rr2_predict(self,*args,**kwargs):
    assert not _RR2_ACTIVE[0] and _RR2_CONTEXT[0] is None
    _RR2_ACTIVE[0] = bool(_RR2_ON)
    _RR2_STATS['reads'].clear(); _RR2_STATS['retired'] = False
    try:
        return _RR2_PARENT_PREDICT(self,*args,**kwargs)
    finally:
        _RR2_CONTEXT[0] = None; _RR2_ACTIVE[0] = False; _RR2_STATS['retired'] = True

Estimator.predict = _rr2_predict
