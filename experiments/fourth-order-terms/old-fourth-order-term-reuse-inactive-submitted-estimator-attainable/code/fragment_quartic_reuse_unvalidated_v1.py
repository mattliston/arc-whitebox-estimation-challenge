# UNVALIDATED source-only draft. No numerical qualification or launch.
# Mode0 is exact parent delegation; mode1 reuses powers and avoids concatenates;
# mode2 additionally reuses identical current-call quartic cohort intermediates.
_QO_MODE = 2
_QO_CONTEXT = [None]
_QO_STATS = dict(captured=0, optimized=0, base_power_elements=0,
                 packed_copy_elements=0, coefficient_copy_elements=0,
                 overlap_elements=0, final_reduction_elements=0,
                 contexts_cleared=0, peak_context_bytes=0)
_QO_PARENT_B6 = _b6_old
_QO_PARENT_OUTER = _k4l_local
_QO_PARENT_INNER = _HCF_PARENT_K4L_LOCAL
_QO_PARENT_PREDICT = _predict_binary

def _qo_match(c,l,pp,pp2,pp3,state):
    return (c is not None and c['layer']==l and c['pp'] is pp and c['pp2'] is pp2
            and c['pp3'] is pp3 and c['state'] is state)

def _qo_buffers(c):
    if c['H'] is None:
        shape=(int(c['pp'].shape[0]),c['width'])
        c['H']=fnp.empty(shape,dtype=c['pp'].dtype)
        c['D']=fnp.empty(shape,dtype=c['pp'].dtype)
        _QO_STATS['peak_context_bytes']=max(_QO_STATS['peak_context_bytes'],8*shape[0]*shape[1])
    return c['H'],c['D']

def _qo_slot(c,l,pp,pp2,pp3,state,j):
    if _QO_MODE!=2 or not _qo_match(c,l,pp,pp2,pp3,state) or j not in c['slices']:
        return None
    H,D=_qo_buffers(c);a,b=c['slices'][j]
    return D[:,a:b],H[:,a:b]


def _qo_inner_local(l, pp, pp2, pp3, state):
    if _QO_MODE != 2:
        return _QO_PARENT_INNER(l,pp,pp2,pp3,state)
    """Sum over the pairs born at most _K4L_AGE layers before layer l of their hub terms, per row of pp (float32).
    pp: pushed bank columns (rows, all columns); pp2, pp3: its squares and cubes, or None."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    parts = []
    cohorts = []
    for j in range(max(l - _K4L_AGE, 0), l):
        start, width = state['cohorts'][j]
        start, K = int(start), int(width) // 2
        if K:
            cohorts.append((start, K, j))
    if not cohorts:
        return None
    cuts = sorted({c for start, K, _j in cohorts for c in (start, start + K, start + 2 * K)})
    at = {c: i + 1 for i, c in enumerate(cuts)}      # the piece that starts at a cut
    b1, b2, b3 = (None if x is None else _e2_H(fnp.split(x, cuts, axis=1)) for x in (pp, pp2, pp3))   # views, one request per array
    for start, K, j in cohorts:
        e1, e2, e3, e4 = _K4L_B[j]
        p = b1[at[start]]
        p2 = _e2_H(p * p) if b2 is None else b2[at[start]]
        p3 = _e2_H(p2 * p) if b3 is None else b3[at[start]]
        _qo_c = _QO_CONTEXT[0]
        _qo_span = _qo_slot(_qo_c,l,pp,pp2,pp3,state,j)
        if _qo_span is None:
            D = p - b1[at[start + K]]
            h = D * e4
        else:
            D,h = _qo_span
            _e2_H(fnp.subtract(p,b1[at[start + K]],out=D))
            _e2_H(fnp.multiply(D,e4,out=h))
        x = p * e3
        _e2_H(fnp.add(h, x, out=h))
        _e2_H(fnp.multiply(h, D, out=h))
        _e2_H(fnp.multiply(p2, e2, out=x))
        _e2_H(fnp.add(h, x, out=h))
        _e2_H(fnp.multiply(h, D, out=h))
        _e2_H(fnp.multiply(p3, e1, out=x))
        _e2_H(fnp.add(h, x, out=h))
        if _qo_span is not None:
            _qo_c['seen'].add(j)
        part = fnp.einsum('ij,ij->i', h, D, optimize=False)
        if isinstance(_K4L_C, tuple):   # the cohort born at j is l - j layers old
            part = _e2_H(part) * _K4L_C[l - j - 1]
        parts.append(part)
        _e2_H((p, p2, p3, D, h, x))                   # this cohort's arrays, released with the rest
        _K4L_STATS.update(blocks=_K4L_STATS['blocks'] + 1, pairs=_K4L_STATS['pairs'] + K,
                          elements=_K4L_STATS['elements'] + int(pp.shape[0]) * K)
    if not parts:
        return None
    loc = parts[0]
    for part in parts[1:]:
        loc = _e2_H(loc) + part
    return loc

def _qo_b6_core(l, pp, pp2, lam, state, vc, response_mask, nrows, ctx):
    want_e, want_q = 'eold' in _HC_COLUMNS, 'qold' in _HC_COLUMNS
    if _HC_U is None or not (want_e or want_q) or not (1 <= l <= 14):
        return
    cohorts = []
    for j in range(0, l - 2):                        # born at j <= l - 3: aged 3 and more
        start, width = state['cohorts'][j]
        start, K = int(start), int(width) // 2
        if K:
            cohorts.append((start, K, j))
    if not cohorts:
        return
    hold = _E2Hold()
    if want_e:
        spans = [(start, start + 2 * K) for start, K, _j in cohorts]
        if all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1)):
            a0, a1 = spans[0][0], spans[-1][1]
            e = hold(pp2[:, a0:a1] @ hold(fnp.abs(lam[a0:a1])))
        else:
            e = hold(fnp.concatenate([pp2[:, x0:x1] for x0, x1 in spans], axis=1) @ hold(fnp.abs(hold(fnp.concatenate([lam[x0:x1] for x0, x1 in spans])))))
        _hc_put(l, 'eold', hold(hold(e.astype(fnp.float64)) / vc), response_mask, nrows)
    if want_q:
        assert [(a,k,j) for a,k,j in cohorts]==ctx['cohorts']
        H,D=_qo_buffers(ctx)
        x=fnp.empty((int(pp.shape[0]),max(k for _,k,_ in cohorts)),dtype=pp.dtype)
        for start,K,j in cohorts:
            if j in ctx['seen']:
                _QO_STATS['overlap_elements']+=int(pp.shape[0])*K
                continue
            a,b=ctx['slices'][j]
            h,dd=H[:,a:b],D[:,a:b]
            P,Q=pp[:,start:start+K],pp[:,start+K:start+2*K]
            p2,p3=pp2[:,start:start+K],ctx['pp3'][:,start:start+K]
            work=x[:,:K]
            e1,e2,e3,e4=fnp.unstack(_K4L_RAW[j])
            hold(fnp.subtract(P,Q,out=dd))
            hold(fnp.multiply(dd,e4,out=h))
            hold(fnp.multiply(P,e3,out=work))
            hold(fnp.add(h,work,out=h))
            hold(fnp.multiply(h,dd,out=h))
            hold(fnp.multiply(p2,e2,out=work))
            hold(fnp.add(h,work,out=h))
            hold(fnp.multiply(h,dd,out=h))
            hold(fnp.multiply(p3,e1,out=work))
            hold(fnp.add(h,work,out=h))
        # Exactly the original full cohort-column order, shape, dtype and axis.
        q=hold(hold(fnp.einsum('ij,ij->i',H,D,optimize=False)).astype(fnp.float64))
        _hc_put(l,'qold',hold(q/hold(vc*vc)),response_mask,nrows)
        elements=int(pp.shape[0])*ctx['width']
        _QO_STATS['optimized']+=1
        _QO_STATS['base_power_elements']+=2*elements
        _QO_STATS['final_reduction_elements']+=elements
        if len(cohorts)>1:
            _QO_STATS['packed_copy_elements']+=2*elements
            _QO_STATS['coefficient_copy_elements']+=4*ctx['width']



def _k4l_local(l,pp,pp2,pp3,state):
    _QO_CONTEXT[0]=None
    if (_QO_MODE and 1<=l<=14 and _HC_U is not None and 'qold' in _HC_COLUMNS
            and pp2 is not None and pp3 is not None and str(pp.dtype)=='float32'
            and pp.shape==pp2.shape==pp3.shape and str(pp2.dtype)==str(pp3.dtype)=='float32'):
        cohorts=[];slices={};at=0
        for j in range(0,l-2):
            start,width=state['cohorts'][j];start,K=int(start),int(width)//2
            if K:
                cohorts.append((start,K,j));slices[j]=(at,at+K);at+=K
        if cohorts:
            _QO_CONTEXT[0]=dict(layer=l,pp=pp,pp2=pp2,pp3=pp3,state=state,
                cohorts=cohorts,slices=slices,width=at,H=None,D=None,seen=set())
            _QO_STATS['captured']+=1
    try:
        return _QO_PARENT_OUTER(l,pp,pp2,pp3,state)
    except BaseException:
        _QO_CONTEXT[0]=None
        raise

def _b6_old(l,pp,pp2,lam,state,vc,response_mask,nrows):
    c=_QO_CONTEXT[0]
    _QO_CONTEXT[0]=None
    try:
        if not _QO_MODE or c is None or not _qo_match(c,l,pp,pp2,c['pp3'],state):
            return _QO_PARENT_B6(l,pp,pp2,lam,state,vc,response_mask,nrows)
        return _qo_b6_core(l,pp,pp2,lam,state,vc,response_mask,nrows,c)
    finally:
        if c is not None:_QO_STATS['contexts_cleared']+=1
        _QO_CONTEXT[0]=None

def _predict_binary(*args,**kwargs):
    _QO_CONTEXT[0]=None
    for key in _QO_STATS:_QO_STATS[key]=0
    try:
        return _QO_PARENT_PREDICT(*args,**kwargs)
    finally:
        _QO_CONTEXT[0]=None

_HCF_PARENT_K4L_LOCAL = _qo_inner_local
