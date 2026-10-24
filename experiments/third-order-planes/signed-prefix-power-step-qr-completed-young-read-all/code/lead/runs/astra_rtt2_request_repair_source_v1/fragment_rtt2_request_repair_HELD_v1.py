# Exact RTT2 basis reuse: no in-place writes; q is x before the first projection.
def _rtt_basis(vc, mu, hold):
    m = int(vc.shape[0]);assert tuple(mu.shape)==(m,)
    basis=[]
    for x in (vc,mu):
        x=x if str(x.dtype)=='float64' else hold(x.astype(fnp.float64));q=x
        original=float(hold(fnp.sum(hold(x*x))))**0.5
        for _ in range(2):
            for b in basis:q=hold(q-hold(b*hold(fnp.sum(hold(b*q)))))
        norm=original if not basis else float(hold(fnp.sum(hold(q*q))))**0.5
        assert original<float('inf') and norm<float('inf')
        if norm>2.220446049250313e-16*max(m,2)*original:
            basis.append(hold(q/norm))
    if not basis:return hold(fnp.zeros((m,0),dtype=fnp.float32))
    return hold(hold(fnp.stack(basis,axis=1)).astype(fnp.float32))


def _rtt_read(left, young_right, lam, live_indices, state):
    global _YR7_SHAPES
    if not _RTT_ACTIVE:return _RTT_PARENT_READ(left,young_right,lam,live_indices,state)
    layer=int(_RS_LAYER[0]);assert 1<=layer<=14
    m,k,c2=int(left.shape[0]),int(young_right.shape[1]),int(young_right.shape[0])
    assert int(live_indices.shape[0])==c2
    old=int(state['columns']);rank=int(state['pre_basis'].shape[1])if old else 0
    assert int(left.shape[1])==old+k and tuple(lam.shape)==(old+k,)
    floor=.5*_rc_math.erfc(1.5/_rc_math.sqrt(2.0))
    c15=min(c2,(int(fnp.nonzero(_E2_LAYER['g']>=floor)[0].shape[0])+31)//32*32)
    record=dict(layer=layer,m=m,k=k,c15=c15,c2=c2,old_rank=rank,changed=c15<c2,basis_rank=None,old_inner_calls=0,forced_d7=False)
    if c15==c2:
        _RTT_STATS['reads'].append(record)
        return _RTT_PARENT_READ(left,young_right,lam,live_indices,state)
    assert 0<c15<c2 and _RTT_CONTEXT[0]is None
    key=(m,k,c2,rank);prefix_key=(m,k,c15,rank)
    preserve_d7=False  # Existing parent dispatch; no new narrowed-prefix D7 keys.
    saved_shapes=_YR7_SHAPES
    context=dict(state=state,m=m,inner=None,inner_calls=0)
    hold=_E2Hold();_RTT_CONTEXT[0]=context
    try:
        if preserve_d7:
            assert _YR7_ON and _YR7_ACTIVE[0] and _WINOGRAD and _AB_ON
            assert _SCRATCH is not None and int(_SCRATCH[0].shape[0])==391602176
            assert _ASTRA_RLF_CONTEXT[0]is None and _UBX_CTX[0]is None and _UBC_CONTEXT[0]is None and _UBX_RC[0]is None
            assert str(left.dtype)==str(young_right.dtype)==str(lam.dtype)=='float32'
            _YR7_SHAPES=saved_shapes|frozenset((prefix_key,))
        prefix=hold(_RTT_PARENT_READ(left,hold(young_right[:c15]),lam,hold(live_indices[:c15]),state))
        _YR7_SHAPES=saved_shapes
        assert tuple(prefix.shape)==(m,c15)
        rows=_E2_LAYER['ecut_rows'];assert tuple(rows.shape)==(m,)
        Q=hold(_rtt_basis(hold(_E2_LAYER['vc'][rows]),hold(_E2_LAYER['mu'][rows]),hold))
        record['basis_rank']=int(Q.shape[1])
        if int(Q.shape[1]):
            scores=hold(Q.T@hold(left[:,old:]))
            weighted=hold(hold(young_right[c15:c2])*hold(lam[old:]))
            tail=hold(Q@hold(scores@hold(weighted.T)))
        else:tail=hold(fnp.zeros((m,c2-c15),dtype=fnp.float32))
        if old:
            assert context['inner_calls']==1 and context['inner']is not None
            inner=context['inner'];assert tuple(inner.shape)==(m,rank)
            old_right=hold(_xsg_rows(5119,state['pre_basis'],hold(live_indices[c15:c2])))
            # Paid thin product only; reuses the exact original old inner once.
            tail=hold(tail+hold(inner@hold(old_right.T)))
        else:assert context['inner_calls']==0 and context['inner']is None
        record['old_inner_calls']=context['inner_calls'];record['forced_d7']=bool(preserve_d7)
        _RTT_STATS['reads'].append(record)
        return fnp.concatenate((prefix,tail),axis=1)
    finally:
        _YR7_SHAPES=saved_shapes
        _RTT_CONTEXT[0]=None

_RC_PARENT_READ=_rtt_read
