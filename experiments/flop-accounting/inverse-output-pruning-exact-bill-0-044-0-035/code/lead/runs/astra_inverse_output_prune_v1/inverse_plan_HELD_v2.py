"""HELD basic-view inverse plan; source-isolated primitive, not installed.
The operation ancestry is the existing alternative-basis inverse. Only output
coordinates discarded by a half strip are omitted. No product/reduction changes.
"""
def inverse_plan(fnp, Z1, T, out, L, m, n, hr, hc, rs_out, final=True):
    u=1<<L;r=m//u;c=n//u;ops=[]
    def emit(f,x,y,o,idx=None):
        if idx is not None:
            x,y,o=x[idx],y[idx],o[idx]
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
