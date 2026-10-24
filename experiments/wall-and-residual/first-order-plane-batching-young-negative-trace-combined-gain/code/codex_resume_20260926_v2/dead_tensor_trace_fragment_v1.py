"""Skip only nonpropagating tensor rows; restore their Gaussian CP diagonals.

Every actual padded live row is retained. Full lower moments are untouched.
All weight-dependent trace calculations use metered fnp operations.
"""
_DCT_CONTEXT = [None]
_DCT_REPAIR = [None]
_DCT_RECON = _slr_reconstruct
_DCT_YOUNG = _push_bank_shared_newborn
_DCT_DIAGONAL = _ott_diagonal

def _dct_old_trace(U, F, lam, dead):
    U64, F64 = U.astype(fnp.float64), F.astype(fnp.float64)
    rho = fnp.sum(U64*U64)/(int(U.shape[0])*int(U.shape[1]))
    norm2 = fnp.sum(F64*F64,axis=0)
    return (U64[dead,:] @ (F64 @ (lam.astype(fnp.float64)*norm2))) * (3*rho)

def _slr_reconstruct(U, state):
    result = _DCT_RECON(U,state)
    ctx = _DCT_CONTEXT[0]
    if ctx is not None and 'Ufull' in ctx:
        full = ctx['Ufull']
        columns = int(state['columns'])
        lam = ctx['lam'][:columns]
        group = state['lazy']
        if group is None:
            delta = _dct_old_trace(full,state['F'],lam,ctx['dead'])
        else:
            mapped = _shared_old384_mm(full,group['map'])
            split = int(group['prefix'])
            delta = (_dct_old_trace(mapped,group['base'],lam[:split],ctx['dead'])
                     + _dct_old_trace(full,group['join'],lam[split:],ctx['dead']))
        fnp.put(_DCT_REPAIR[0],ctx['dead'],delta)
    return result

def _dct_young_trace(wfull, p, previous, newborn, lam, dead):
    inherited, selected, base, t, s = newborn
    inherited, births = int(inherited), int(selected.shape[0])
    coefficient = lam[-(inherited+2*births):].astype(fnp.float64)
    wp = wfull[previous,:].astype(fnp.float64)
    rho = fnp.sum(wp*wp)/(int(wp.shape[0])*int(wp.shape[1]))
    B = base[previous,:].astype(fnp.float64)
    diagonal = fnp.diag(base[selected,:]).astype(fnp.float64)
    t64, s64 = t.astype(fnp.float64), s.astype(fnp.float64)
    normB = fnp.sum(B*B,axis=0)
    norm_t = normB + 2*t64*diagonal + t64*t64
    norm_s = normB + 2*s64*diagonal + s64*s64
    ct = coefficient[inherited:inherited+births]*norm_t
    cs = coefficient[inherited+births:]*norm_s
    vector = B @ (ct+cs)
    if inherited:
        P = p[:,:inherited].astype(fnp.float64)
        vector = vector + P @ (coefficient[:inherited]*fnp.sum(P*P,axis=0))
    direction = wfull[selected,:].astype(fnp.float64)
    return ((wp[:,dead].T @ vector) + (direction[:,dead].T @ (t64*ct+s64*cs))) * (3*rho)

def _push_bank_shared_newborn(w,p,previous,newborn):
    result = _DCT_YOUNG(w,p,previous,newborn)
    ctx = _DCT_CONTEXT[0]
    if ctx is not None:
        if newborn is None:
            raise RuntimeError('Dead tensor trace requires the inherited newborn layout')
        delta = _dct_young_trace(ctx['wfull'],p,previous,newborn,ctx['lam'],ctx['dead'])
        fnp.put(_DCT_REPAIR[0],ctx['dead'],_DCT_REPAIR[0][ctx['dead']]+delta)
    return result

def _deadcut_push(layer,w32,p,previous,newborn,lam,state,a):
    global _W5_PORT_SCOPE
    _DCT_REPAIR[0] = None
    if not 1 <= layer <= 14:
        return _shared_old384_push(layer,w32,p,previous,newborn,lam,state)
    alive = _E2_LAYER['ott_live_mask']
    n = int(a.shape[0])
    count = min(n,(int(fnp.nonzero(alive)[0].shape[0])+63)//64*64)
    if count == n:
        return _shared_old384_push(layer,w32,p,previous,newborn,lam,state)
    ordered = fnp.argsort(-a,kind='stable')
    rows, dead = ordered[:count],ordered[count:]
    if bool(fnp.any(alive[dead])):
        raise RuntimeError('Dead tensor cut attempted to remove a padded live row')
    _DCT_REPAIR[0] = fnp.zeros(n,dtype=fnp.float64)
    assert _DCT_CONTEXT[0] is None
    _DCT_CONTEXT[0] = dict(wfull=w32,lam=lam,dead=dead,rows=rows,n=n)
    saved_scope = _W5_PORT_SCOPE
    _W5_PORT_SCOPE = None
    try:
        pushed = _shared_old384_push(layer,w32,p,previous,newborn,lam,state,rows=rows)
    finally:
        _W5_PORT_SCOPE = saved_scope
        _DCT_CONTEXT[0] = None
    inverse = fnp.full(n,count,dtype=fnp.int64)
    fnp.put(inverse,rows,fnp.arange(count,dtype=fnp.int64))
    answer = fnp.concatenate([pushed,fnp.zeros((1,int(pushed.shape[1])),dtype=pushed.dtype)],axis=0)[inverse]
    basis = state.get('pre_basis')
    if basis is not None:
        state['pre_basis'] = fnp.concatenate([basis,fnp.zeros((1,int(basis.shape[1])),dtype=basis.dtype)],axis=0)[inverse]
    return answer

def _ott_diagonal(diagonal,lam):
    result = _DCT_DIAGONAL(diagonal,lam)
    if _DCT_REPAIR[0] is not None:
        result = result + _DCT_REPAIR[0]
    return result
