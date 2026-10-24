# Fixed recent-tier difference prefix320. Approximation, not an identity claim.
_RV320_PARENT_STATE = _uvp_state
_RV320_PARENT_RECON = _uvp_recon
_RV320_PARENT_BCC = _astra_bcc_uvrecon
_RV320_PARENT_INNER = _uvp_inner
_RV320_PARENT_PREDICT = Estimator.predict
_RV320_RECENT = [None]
_RV320_STATS = dict(reconstruction=[], read=[], frames=0, retired=True)

def _uvp_state(state):
    uv = _RV320_PARENT_STATE(state)
    group = state.get('lazy')
    if (_UVP_CUR[0] in (13,14,15) and uv[0]=='l' and group is not None
            and group['layer']==13 and group['base_rank']==320 and int(state['rank'])==384):
        recent=uv[2]
        if (recent[3]==256 and int(recent[1].shape[0])==384
                and 0<int(recent[1].shape[1])<=3072 and int(recent[1].shape[1])%32==0):
            if _RV320_RECENT[0] is not recent:_RV320_STATS['frames']+=1
            _RV320_RECENT[0]=recent
        else:_RV320_RECENT[0]=None
    else:_RV320_RECENT[0]=None
    return uv

def _rv320_active(uv):
    return uv is _RV320_RECENT[0] and _UVP_CUR[0] in (13,14,15)

def _astra_bcc_uvrecon(U, uv, out):
    """U (rows x rank) -> the t and s columns of every pair, roster order: U[:, :r_u] F_u +- U (F_v/2)."""
    if not _rv320_active(uv):return _RV320_PARENT_BCC(U,uv,out)
    Fu, Fv2, blocks, ru = uv
    rows = int(U.shape[0])
    Pu = _ubc_product(U[:, :ru], Fu, uv, 'u')
    Pv = _ubc_product(U[:, :320], Fv2[:320], uv, 'v')
    cols = 2 * int(Fu.shape[1])
    if tuple(out.shape) != (rows, cols) or str(out.dtype) != 'float32':
        raise RuntimeError('bank-copy UV output ownership')
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(Pu[:, b:b + h], Pv[:, b:b + h], out=out[:, a:a + h]))
        h_(fnp.subtract(Pu[:, b:b + h], Pv[:, b:b + h], out=out[:, a + h:a + 2 * h]))
        b += h
    _BH_HOLD.clear()
    del Pu, Pv
    _RV320_STATS['reconstruction'].append([int(_UVP_CUR[0]),rows,int(Fv2.shape[1])])
    return out

def _uvp_recon(U,uv):
    if not _rv320_active(uv):return _RV320_PARENT_RECON(U,uv)
    out=fnp.empty((int(U.shape[0]),2*int(uv[0].shape[1])),dtype=fnp.float32)
    return _astra_bcc_uvrecon(U,uv,out)

def _uvp_inner(left_sq, lam_cols, uv):
    """(L^2 lam) F^T with F in u/v form: rows x rank. = the parent's arithmetic; the two products through _ubx_product."""
    if not _rv320_active(uv):return _RV320_PARENT_INNER(left_sq,lam_cols,uv)
    Fu, Fv2, blocks, ru = uv
    rows = int(left_sq.shape[0])
    P = int(Fu.shape[1])
    a_ = left_sq * lam_cols[None, :]
    Su = fnp.empty((rows, P), dtype=fnp.float32)
    Dv = fnp.empty((rows, P), dtype=fnp.float32)
    h_ = _BH_HOLD.append
    b = 0
    for a, h in blocks:
        h_(fnp.add(a_[:, a:a + h], a_[:, a + h:a + 2 * h], out=Su[:, b:b + h]))
        h_(fnp.subtract(a_[:, a:a + h], a_[:, a + h:a + 2 * h], out=Dv[:, b:b + h]))
        b += h
    _BH_HOLD.clear()
    del a_
    reduced = _ubx_product(Dv, Fv2[:320].T, uv, 'vt')
    inner = fnp.zeros((rows,384),dtype=fnp.float32)
    _BH_HOLD.append(fnp.copyto(inner[:,:320],reduced))
    del reduced
    inner_u = _ubx_product(Su, Fu.T, uv, 'ut')
    _BH_HOLD.append(fnp.add(inner[:, :ru], inner_u, out=inner[:, :ru]))
    _BH_HOLD.clear()
    del Su, Dv, inner_u
    _RV320_STATS['read'].append([int(_UVP_CUR[0]),rows,P])
    return inner

def _rv320_predict(self,mlp,flop_budget):
    if _RV320_RECENT[0] is not None:raise RuntimeError('RV320 stale frame')
    _RV320_STATS.update(reconstruction=[],read=[],frames=0,retired=False)
    try:return _RV320_PARENT_PREDICT(self,mlp,flop_budget)
    finally:
        _RV320_RECENT[0]=None
        _RV320_STATS['retired']=True
Estimator.predict=_rv320_predict
