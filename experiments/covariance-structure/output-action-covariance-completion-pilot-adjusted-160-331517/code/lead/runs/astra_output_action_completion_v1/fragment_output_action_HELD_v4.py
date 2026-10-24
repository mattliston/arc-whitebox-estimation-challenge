# Fixed output-side .05 pruning with marginal and feature compensation.
_OAC_ON = True
_OAC_STATE = {}
_OAC_SIDES = {}
_OAC_STATS = dict(calls=[], feature_layers=[], np_layers=[], retired=True)
_OAC_ACTIVE = [False]

def _oac_side(kind, P, rows, cols):
    key = (kind, P, rows, cols)
    region = _IPT_REGION[6]
    start = (0, region[0], region[0] + region[1])[kind]
    assert P * 7**5 * (rows//32) * (cols//32) <= region[kind]
    got = _OAC_SIDES.get(key)
    if got is None or got._borrowed_arena is not _SCRATCH[0]:
        got = _PSDBSide(_SCRATCH[0], start, kind, P, 5, rows, cols)
        _OAC_SIDES[key] = got
    return got

def _oac_symmetric(A, B):
    n, k = int(A.shape[0]), int(A.shape[1])
    assert tuple(B.shape) == (k,n) and n%64 == k%64 == 0
    out=fnp.empty((n,n),dtype=fnp.float32)
    full, rem=divmod(n,256)
    for block in range(full):
        end=(block+1)*256
        if end<n: _psd_sym_tile(A,B,out,block*256,256,end,n-end,False)
    if full:
        A4=A[:full*256].reshape(full,256,k)
        B4=fnp.moveaxis(B[:,:full*256].reshape(k,full,256),1,0)
        for offset,r,c in _PSDB_GROUPS:
            sa,sb,sc=_oac_side(0,full,r,k),_oac_side(1,full,k,c),_oac_side(2,full,r,c)
            keep=[]
            part=fnp.empty((full,r,c),dtype=fnp.float32)
            sa.forward(A4[:,offset:offset+r],keep)
            sb.forward(B4[:,:,offset:offset+c],keep)
            keep.append(fnp.matmul(sa.T,sb.T,out=sc.T))
            sc.unwind(part,keep)
            for j in range(full):
                start=j*256+offset
                keep.append(fnp.copyto(out[start:start+r,start:start+c],part[j]))
                if c>r: keep.append(fnp.copyto(out[start+r:start+c,start:start+r],part[j,:,r:].T))
    if rem:
        start=full*256
        for r in ((128,64) if rem==192 else (128,) if rem==128 else (64,)):
            _psd_sym_tile(A,B,out,start,r,start,n-start,True)
            start+=r
    return out

def _oac_complete(wa, first, current_mu, input_mu, active, tail, exact_block, var, input_rows, n):
    h=_E2Hold()
    s=h(fnp.sqrt(h(fnp.maximum(var,1e-10))))
    alpha=h(current_mu/s)
    # Exact full pre-covariance actions used by feature13/26 and the NP law.
    V=h(fnp.stack((h(1/s),h(alpha/s),current_mu),axis=1,dtype=fnp.float32))
    actions=h(wa.T@h(first@V))
    Vb=h(V[tail].astype(fnp.float64))
    Q,unused=h(fnp.linalg.qr(Vb,mode='reduced'))
    Q=h(Q.astype(fnp.float32))
    # Cross and tail action completion, with symmetric tail and exact diagonal.
    fb=h(first[:,tail]);wb=h(wa[:,tail]);aa=h(wa[:,active])
    Y=h(first[:,tail]@Q)
    L=h(wa.T@Y)
    cross=h(h(L[active])@Q.T)
    R=h(L[tail]);H=h(Q.T@R)
    T=h(R@Q.T)
    tailblock=h(h(T+T.T)-h(h(Q@H)@Q.T))
    out=fnp.empty((n,n),dtype=fnp.float32)
    fnp.put(out,h(h(active*n)[:,None]+active),exact_block)
    fnp.put(out,h(h(active*n)[:,None]+tail),cross)
    fnp.put(out,h(h(tail*n)[:,None]+active),cross.T)
    fnp.put(out,h(h(tail*n)[:,None]+tail),tailblock)
    fnp.fill_diagonal(out,var)
    # Gaussian-weight conditional moments. The full input mean conditions the
    # weight column, even though the original covariance input cut is retained.
    mi=h(input_mu if input_rows is None else input_mu[input_rows])
    norm=h(fnp.sum(h(input_mu*input_mu)))
    if float(norm)>0:
        coef=h(h(first.T@h(mi.astype(fnp.float32)))/norm)
        residual=h(h(fnp.sum(h(first*first),axis=0))-h(h(coef*coef)*norm))
    else:
        coef=h(fnp.zeros(n,dtype=fnp.float32))
        residual=h(fnp.sum(h(first*first),axis=0))
    sd=h(fnp.sqrt(h(fnp.maximum(h(residual*(2.0/n)),1e-30))).astype(fnp.float32))
    s32=h(s.astype(fnp.float32));mu32=h(current_mu.astype(fnp.float32));coef32=h(coef.astype(fnp.float32))
    na,nb=int(active.shape[0]),int(tail.shape[0])
    sum2=fnp.empty(n,dtype=fnp.float32);sumabs=fnp.empty(n,dtype=fnp.float32)
    kept=h(exact_block/h(fnp.outer(h(s32[active]),h(s32[active]))))
    fnp.fill_diagonal(kept,1.0)
    kept2=h(fnp.sum(h(kept*kept),axis=1));keptabs=h(fnp.sum(h(fnp.abs(kept)),axis=1))
    def moments(which,columns,is_tail):
        ci=h(coef32[which]);sdi=h(sd[which]);si=h(s32[which])
        mj=mu32 if columns is None else h(mu32[columns])
        sj=s32 if columns is None else h(s32[columns])
        loc=h(fnp.outer(ci,mj));sdcol=h(sdi[:,None]);z=h(loc/sdcol)
        scale=h(fnp.outer(si,sj))
        rr=h(h(h(loc*loc)+h(sdcol*sdcol))/h(scale*scale))
        folded=h(h(h(sdcol*0.7978845608028654)*h(fnp.exp(h(-.5*h(z*z)))))+h(loc*h(2*flops.stats.norm.cdf(z)-1)))
        ar=h(folded/scale)
        if is_tail:
            diagonal=h(h(fnp.arange(nb)*n)+tail)
            fnp.put(rr,diagonal,1.0);fnp.put(ar,diagonal,1.0)
        return h(fnp.sum(rr,axis=1)),h(fnp.sum(ar,axis=1))
    e2,ea=moments(active,tail,False)
    fnp.put(sum2,active,h(kept2+e2));fnp.put(sumabs,active,h(keptabs+ea))
    e2,ea=moments(tail,None,True)
    fnp.put(sum2,tail,e2);fnp.put(sumabs,tail,ea)
    features={13:h(h(actions[:,0]/s)/n),14:h(fnp.sqrt(h(sum2/n))),15:h(sumabs/n),26:h(h(actions[:,1]/s)/n)}
    return out,features,h(actions[:,2].astype(fnp.float64))

def _oac_cov_star(w,C,A,gate,floor,current_mu,input_mu,retain32=False,layer=None):
    _OAC_STATE.clear()
    if not _OAC_ON:return _dc_cov_star(w,C,A,gate,floor,retain32)
    h=_E2Hold();roster=_dcv_roster(gate,floor);n=int(w.shape[1]);wt=h(w.T)
    C32=h(_ps32(C));A32=A if str(A.dtype)=='float32' else h(A.astype(fnp.float32))
    P1,T2=_ss_factors(wt,h(wt*wt),A32,h)
    if roster is None:
        rows=None;wa=w;ca=C32;k=int(w.shape[0])
    else:
        k=int(_DCV_STATS['m']);rows=h(roster[:k]);wa=h(_xsg_rows(13378,w,rows));ca=h(C32[fnp.ix_(rows,rows)])
    first=h(_mm(ca,wa))
    sub=h(_astra_star_scaled_product(T2,P1,h))
    var=h(fnp.sum(h(fnp.multiply(wa,first,dtype=fnp.float64)),axis=0))
    sigma=h(fnp.sqrt(h(fnp.maximum(var,1e-10))));g=h(flops.stats.norm.cdf(h(current_mu/sigma)))
    kept=min(n,(int(fnp.nonzero(g>=.05)[0].shape[0])+63)//64*64)
    # An empty retained roster still has a mathematically defined completion,
    # but the fixed existing symmetric engine requires at least64 outputs.
    kept=max(64,kept)
    if kept==n:
        cov=h(_psdm_sym_upper(h(wa.T),first,k) if k<n and _psdm_fits(k) else _sym_upper(h(wa.T),first))
        features=cmu=None
    else:
        order=h(fnp.argsort(-g,kind='stable'));active=h(order[:kept]);tail=h(order[kept:])
        block=h(_oac_symmetric(h(wa[:,active].T),h(first[:,active])))
        cov,features,cmu=_oac_complete(wa,first,current_mu,input_mu,active,tail,block,var,rows,n)
    Cpre=cov.astype(fnp.float64)
    diagonal=h(fnp.sum(h(P1*T2),axis=1,dtype=fnp.float64))
    result=sub.astype(fnp.float64);fnp.fill_diagonal(result,diagonal)
    _SS_STATS['planes']+=1
    _OAC_STATS['calls'].append(dict(layer=layer,input=k,output=kept,changed=kept<n,threshold=.05,preserved_diagonal=kept<n))
    if features is not None:_OAC_STATE.update(Cpre=Cpre,features=features,cmu=cmu,mu=current_mu,layer=layer)
    if retain32:
        fnp.fill_diagonal(sub,diagonal)
        return Cpre,result,cov,sub
    return Cpre,result

def _oac_cmu(cov,mu):
    if _OAC_STATE.get('Cpre') is cov and _OAC_STATE.get('mu') is mu:
        _OAC_STATS['np_layers'].append(_OAC_STATE['layer'])
        return _OAC_STATE['cmu']
    return cov@mu

_OAC_PARENT_PREDICT=Estimator.predict
def _oac_predict(self,*args,**kwargs):
    assert not _OAC_ACTIVE[0]
    _OAC_ACTIVE[0]=True
    _OAC_STATE.clear();_OAC_STATS.clear();_OAC_STATS.update(calls=[],feature_layers=[],np_layers=[],retired=False)
    try:return _OAC_PARENT_PREDICT(self,*args,**kwargs)
    finally:_OAC_STATE.clear();_OAC_ACTIVE[0]=False;_OAC_STATS['retired']=True
Estimator.predict=_oac_predict


def _oac_features(l,aux,vals):
    if _OAC_STATE.get('Cpre') is aux['pre_cov']:_OAC_STATS['feature_layers'].append(l)
    return fnp.stack(vals,axis=1,dtype=fnp.float32)
