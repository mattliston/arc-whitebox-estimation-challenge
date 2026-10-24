# ONE fixed signed-prefix power step on exact e406 request-repaired RTT2.
# Frozen source-only: no model/outcome-dependent coefficients or bands.
# Reduced QR changes reduction order/signs. Deficient/ambiguous input delegates
# the inherited twice-reorthogonalized basis; refined rank loss retains original Q.
_RTP_ON = True
_RTP_REPAIRED_READ = _rtt_read
_RTP_REPAIRED_BASIS = _rtt_basis
assert _RC_PARENT_READ is _RTP_REPAIRED_READ

def _rtp_qr_basis(vc, mu, hold):
    m=int(vc.shape[0]);assert tuple(mu.shape)==(m,) and m>=2
    a=hold(fnp.stack((vc,mu),axis=1,dtype=fnp.float64))
    norms=hold(fnp.linalg.norm(a,axis=0))
    n0,n1=float(norms[0]),float(norms[1])
    assert 0.0<=n0<float('inf') and 0.0<=n1<float('inf')
    if n0==0.0 or n1==0.0:return _RTP_REPAIRED_BASIS(vc,mu,hold)
    q,r=fnp.linalg.qr(a,mode='reduced');q=hold(q);r=hold(r)
    r0,r1=abs(float(r[0,0])),abs(float(r[1,1]))
    assert r0<float('inf') and r1<float('inf')
    # Conservative ambiguity band is fixed from floating-point precision only.
    # It never invents a direction when QR's unpivoted diagonal loses rank.
    eps=32.0*2.220446049250313e-16*max(m,2)
    if r0<=eps*n0 or r1<=eps*n1:return _RTP_REPAIRED_BASIS(vc,mu,hold)
    return hold(q.astype(fnp.float32))

def _rtp_refine(prefix, Q, inner, old_right, hold):
    rank=int(Q.shape[1]);assert rank in (1,2)
    V=hold(prefix.T@Q)
    if inner is not None:
        assert old_right is not None and tuple(inner.shape)==(int(prefix.shape[0]),int(old_right.shape[1]))
        V=hold(V-hold(old_right@hold(inner.T@Q)))
    else:assert old_right is None
    U=hold(prefix@V)
    if inner is not None:U=hold(U-hold(inner@hold(old_right.T@V)))
    second=hold(U[:,1]) if rank==2 else hold(fnp.zeros(int(U.shape[0]),dtype=fnp.float32))
    fresh=hold(_rtp_qr_basis(hold(U[:,0]),second,hold))
    qr_rank=int(fresh.shape[1]);fallback=qr_rank<rank
    return (Q if fallback else fresh),qr_rank,fallback

def _rtt_read(left, young_right, lam, live_indices, state):
    global _YR7_SHAPES
    if not _RTP_ON:return _RTP_REPAIRED_READ(left,young_right,lam,live_indices,state)
    if not _RTT_ACTIVE:return _RTT_PARENT_READ(left,young_right,lam,live_indices,state)
    layer=int(_RS_LAYER[0]);assert 1<=layer<=14
    m,k,c2=int(left.shape[0]),int(young_right.shape[1]),int(young_right.shape[0])
    assert int(live_indices.shape[0])==c2
    old=int(state['columns']);rank=int(state['pre_basis'].shape[1])if old else 0
    assert int(left.shape[1])==old+k and tuple(lam.shape)==(old+k,)
    floor=.5*_rc_math.erfc(1.5/_rc_math.sqrt(2.0))
    c15=min(c2,(int(fnp.nonzero(_E2_LAYER['g']>=floor)[0].shape[0])+31)//32*32)
    record=dict(layer=layer,m=m,k=k,c15=c15,c2=c2,old_rank=rank,changed=c15<c2,basis_rank=None,old_inner_calls=0,forced_d7=False,refine_rank_in=None,refine_rank_out=None,refine_qr_rank=None,refine_fallback=None,refine_old_rank=None)
    if c15==c2:
        _RTT_STATS['reads'].append(record)
        return _RTT_PARENT_READ(left,young_right,lam,live_indices,state)
    assert 0<c15<c2 and _RTT_CONTEXT[0]is None
    key=(m,k,c2,rank);prefix_key=(m,k,c15,rank)
    preserve_d7=False  # Preserve exact e406 request-repair dispatch.
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
        Q=hold(_rtp_qr_basis(hold(_E2_LAYER['vc'][rows]),hold(_E2_LAYER['mu'][rows]),hold))
        record['basis_rank']=int(Q.shape[1])
        if _RTP_ON:
            record['refine_rank_in']=int(Q.shape[1])
            record['refine_old_rank']=rank
            if int(Q.shape[1]):
                if old:
                    assert context['inner_calls']==1 and context['inner']is not None
                    prefix_old=hold(_xsg_rows(5119,state['pre_basis'],hold(live_indices[:c15])))
                    assert tuple(prefix_old.shape)==(c15,rank)
                    Q,qr_rank,fallback=_rtp_refine(prefix,Q,context['inner'],prefix_old,hold)
                else:Q,qr_rank,fallback=_rtp_refine(prefix,Q,None,None,hold)
            else:qr_rank,fallback=0,False
            record['refine_rank_out']=int(Q.shape[1])
            record['refine_qr_rank']=qr_rank;record['refine_fallback']=bool(fallback)
            assert int(Q.shape[1])==record['refine_rank_in']
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
