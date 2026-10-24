# DCT2: add the Gaussian cubic projection containing at least two mean legs.
_DCM_OLD_TRACE = _dct_old_trace
_DCM_YOUNG_TRACE = _dct_young_trace


def _dcm_combine(linear, cubic, ctx):
    mu = ctx['mean_mu'][ctx['dead']]
    return 3*(mu*mu-ctx['mean_variance'])*linear - 2*mu*mu*mu*cubic


def _dct_old_trace(U, F, lam, dead, mean_basis):
    original = _DCM_OLD_TRACE(U, F, lam, dead)
    F64 = F.astype(fnp.float64)
    coefficient = lam.astype(fnp.float64)
    h = mean_basis @ F64
    weight = coefficient*h*h
    linear = U[dead,:].astype(fnp.float64) @ (F64 @ weight)
    cubic = fnp.sum(weight*h)
    return original + _dcm_combine(linear,cubic,_DCT_CONTEXT[0])


def _slr_reconstruct(U, state):
    result = _DCT_RECON(U,state)
    ctx = _DCT_CONTEXT[0]
    if ctx is not None and 'Ufull' in ctx:
        full = ctx['Ufull']
        columns = int(state['columns'])
        lam = ctx['lam'][:columns]
        group = state['lazy']
        mean_basis = ctx['mean_basis']
        if group is None:
            delta = _dct_old_trace(full,state['F'],lam,ctx['dead'],mean_basis)
        else:
            mapped = _shared_old384_mm(full,group['map'])
            split = int(group['prefix'])
            mapped_mean = mean_basis @ group['map'].astype(fnp.float64)
            delta = (_dct_old_trace(mapped,group['base'],lam[:split],ctx['dead'],mapped_mean)
                     + _dct_old_trace(full,group['join'],lam[split:],ctx['dead'],mean_basis))
        fnp.put(_DCT_REPAIR[0],ctx['dead'],delta)
    return result


def _dct_young_trace(wfull,p,previous,newborn,lam,dead):
    original = _DCM_YOUNG_TRACE(wfull,p,previous,newborn,lam,dead)
    inherited,selected,base,t,s = newborn
    inherited,births = int(inherited),int(selected.shape[0])
    ctx = _DCT_CONTEXT[0]
    coefficient = lam[-(inherited+2*births):].astype(fnp.float64)
    m = ctx['mean_previous']
    norm = ctx['mean_norm']
    mp = m[previous]
    B = base[previous,:].astype(fnp.float64)
    hb = (mp @ B)/norm
    ht = hb + (m[selected]*t.astype(fnp.float64))/norm
    hs = hb + (m[selected]*s.astype(fnp.float64))/norm
    ct = coefficient[inherited:inherited+births]*ht*ht
    cs = coefficient[inherited+births:]*hs*hs
    vector = B @ (ct+cs)
    cubic = fnp.sum(ct*ht+cs*hs)
    if inherited:
        P = p[:,:inherited].astype(fnp.float64)
        hp = (mp @ P)/norm
        cp = coefficient[:inherited]*hp*hp
        vector = vector + P @ cp
        cubic = cubic + fnp.sum(cp*hp)
    linear = (wfull[previous,:][:,dead].astype(fnp.float64).T @ vector
              + wfull[selected,:][:,dead].astype(fnp.float64).T
                @ (t.astype(fnp.float64)*ct+s.astype(fnp.float64)*cs))
    return original + _dcm_combine(linear,cubic,ctx)
