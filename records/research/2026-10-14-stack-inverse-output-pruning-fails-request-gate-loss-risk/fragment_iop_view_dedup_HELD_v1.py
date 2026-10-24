# HELD exact request optimization: no numerical choice or array-value cache.
_ASTRA_IVD_ON = True
_ASTRA_IVD_PARENT_PLAN = _astra_iop_plan
_ASTRA_IVD_STATS = dict(plan_calls=0,raw_view_requests=0,view_created=0,view_reused=0,peak_local_entries=0)

def _astra_ivd_plan(fnp, Z1, T, out, L, m, n, hr, hc, rs_out, final=True):
    u=1<<L;r=m//u;c=n//u;ops=[]
    views={}
    _ASTRA_IVD_STATS['plan_calls'] += 1
    def view(owner,idx):
        # All indices in the pinned parent plan are tuples of basic slices.
        # Hold the owner in this local map, preventing id reuse during assembly.
        assert isinstance(idx,tuple) and all(isinstance(v,slice) for v in idx)
        key=(id(owner),tuple((v.start,v.stop,v.step) for v in idx))
        _ASTRA_IVD_STATS['raw_view_requests'] += 1
        found=views.get(key)
        if found is not None:
            assert found[0] is owner
            _ASTRA_IVD_STATS['view_reused'] += 1
            return found[1]
        answer=owner[idx]
        views[key]=(owner,answer)
        _ASTRA_IVD_STATS['view_created'] += 1
        _ASTRA_IVD_STATS['peak_local_entries']=max(_ASTRA_IVD_STATS['peak_local_entries'],len(views))
        return answer
    def emit(f,x,y,o,idx=None):
        if idx is not None:
            x,y,o=view(x,idx),view(y,idx),view(o,idx)
        if all(int(k)>0 for k in o.shape):ops.append((f,x,y,o))
    def regions(f,x,y,o,natural=False):
        # Region partitions are disjoint. Before the final first-axis basis
        # conversion, component3 still feeds component1 on a column strip.
        ndim=len(o.shape);ra=L-1 if natural else ndim-2;ca=ndim-1
        def e(rv,cv,branch=None,rb=False,cb=False):
            idx=[slice(None)]*ndim;idx[ra]=rv;idx[ca]=cv
            if natural:
                if rb:idx[L-2]=slice(0,1)
                if cb:idx[ndim-2]=slice(0,1)
            elif branch is not None:idx[0]=branch
            emit(f,x,y,o,tuple(idx))
        full=slice(None);core_r=slice(None,-1) if hr else full;core_c=slice(None,-1) if hc else full;last=slice(-1,None)
        e(core_r,core_c)
        if hr:e(last,core_c,slice(0,4,2),rb=True)
        if hc:
            if natural:e(core_r,last,cb=True)
            else:
                e(core_r,last,slice(0,2));e(core_r,last,slice(3,4))
        if hr and hc:e(last,last,slice(0,1),rb=True,cb=True)
    for j in range(L,1,-1):
        Z=Z1[j]
        raw=((Z[0],Z[5],Z[5]),(Z[0],Z[2],Z[0]),(Z[5],Z[4],Z[5]),(Z[5],Z[3],Z[2]),(Z[5],Z[1],Z[3]),(Z[4],Z[6],Z[1]))
        if j<L:
            for x,y,o in raw:regions(fnp.add,x,y,o)
        else:
            for a,(x,y,o) in enumerate(raw):
                if a in (4,5) and hr:
                    emit(fnp.add,x,y,o,(slice(None),)*(len(o.shape)-2)+(slice(None,-1),slice(None)))
                elif a==3 and hc:
                    emit(fnp.add,x,y,o,(slice(None),)*(len(o.shape)-1)+(slice(None,-1),))
                elif a in (0,2) and hr and hc:
                    # Only the intersection has no remaining consumer of Z5.
                    emit(fnp.add,x,y,o,(slice(None),)*(len(o.shape)-2)+(slice(None,-1),slice(None)))
                    emit(fnp.add,x,y,o,(slice(None),)*(len(o.shape)-2)+(slice(-1,None),slice(None,-1)))
                else:emit(fnp.add,x,y,o)
    for j in range(2,L+1):
        ax=L-j;idx=[slice(0,4)]*L;idx[-1]=slice(None)
        lo=list(idx);lo[ax]=1;hi=list(idx);hi[ax]=3;x,y=T[tuple(hi)],T[tuple(lo)]
        if ax==0:
            ix=(slice(None),)*(len(y.shape)-2)+(slice(None,-1),slice(None)) if hr else None
            emit(fnp.subtract,x,y,y,ix)
        else:regions(fnp.subtract,x,y,y)
    if not final:return tuple(ops)
    tl,tr,bl,br=rs_out(out,L,m,n);Z=Z1[1]
    raw=((fnp.add,Z[0],Z[5],Z[5]),(fnp.add,Z[0],Z[2],tl),(fnp.add,Z[5],Z[4],Z[5]),(fnp.add,Z[5],Z[3],tr),(fnp.add,Z[5],Z[1],br),(fnp.add,Z[4],Z[6],Z[4]),(fnp.subtract,br,Z[4],bl))
    for f,x,y,o in raw:regions(f,x,y,o,natural=True)
    return tuple(ops)

def _astra_iop_plan(fnp,Z1,T,out,L,m,n,hr,hc,rs_out,final=True):
    if not _ASTRA_IVD_ON:
        return _ASTRA_IVD_PARENT_PLAN(fnp,Z1,T,out,L,m,n,hr,hc,rs_out,final=final)
    return _astra_ivd_plan(fnp,Z1,T,out,L,m,n,hr,hc,rs_out,final=final)

_ASTRA_IVD_PARENT_PREDICT = Estimator.predict
def _astra_ivd_predict(self,mlp,flop_budget):
    if _ASTRA_IVD_ON:
        for name in _ASTRA_IVD_STATS:
            _ASTRA_IVD_STATS[name]=0
    return _ASTRA_IVD_PARENT_PREDICT(self,mlp,flop_budget)
Estimator.predict = _astra_ivd_predict
