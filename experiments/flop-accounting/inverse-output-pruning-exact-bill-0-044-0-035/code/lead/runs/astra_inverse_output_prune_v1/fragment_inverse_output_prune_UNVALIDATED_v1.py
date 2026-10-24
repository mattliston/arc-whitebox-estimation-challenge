# ASTRA IOP v1 HELD: retain exact arithmetic ancestry of non-padding outputs.
_ASTRA_IOP_ON = True
_ASTRA_IOP_CTX = [None]
_ASTRA_IOP_CACHE = {}
_ASTRA_IOP_LIMIT = 32
_ASTRA_IOP_STATS = dict(calls=0, hzg=0, ks=0, ubx=0, hz7=0, saved_elements=0,
                        prefix_operations_delta=0, cache_hits=0, cache_stores=0,
                        cache_evictions=0, peak_entries=0, live_entries=0, fallback=0)
_ASTRA_IOP_PARENT_UNWIND = _ASTRA_RLF_PARENT_UNWIND
_ASTRA_IOP_PARENT_HZG = _hzg_product
_ASTRA_IOP_PARENT_KS = _hz_ks_product
_ASTRA_IOP_PARENT_UBX = _ubx_ks_product
_ASTRA_IOP_PARENT_HZ7 = _hz7_product
_ASTRA_IOP_PARENT_PREDICT = Estimator.predict
assert _ASTRA_IOP_PARENT_UNWIND.__name__ == '_ks_unwind'

def _astra_iop_plan(fnp, Z1, T, out, L, m, n, hr, hc, rs_out, final=True):
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

def _astra_iop_count(x):
    count = 1
    for k in x.shape:
        count *= int(k)
    return count


def _astra_iop_unwind(TC, Z1, ops, basis, out, L, m, n):
    context = _ASTRA_IOP_CTX[0]
    if not _ASTRA_IOP_ON or context is None or L != context['L'] or m != context['M'] or n not in context['Ns']:
        return _ASTRA_IOP_PARENT_UNWIND(TC, Z1, ops, basis, out, L, m, n)
    if _ASTRA_RLF_CONTEXT[0] is not None and not _ASTRA_RLF_CONTEXT[0]['done']:
        raise RuntimeError('IOP must follow the original read-fusion leaf addition')
    entry = _ipt_side(2, L, m, n)
    if entry[0] is not TC or entry[1] is not Z1:
        _ASTRA_IOP_STATS['fallback'] += 1
        return _ASTRA_IOP_PARENT_UNWIND(TC, Z1, ops, basis, out, L, m, n)
    hr, hc = context['hr'], context['hc']
    key = (L, m, n, hr, hc)
    got = _ASTRA_IOP_CACHE.get(key)
    if got is not None and got['TC'] is TC and got['Z1'] is Z1:
        plan = got['plan']
        _ASTRA_IOP_STATS['cache_hits'] += 1
    else:
        plan = _astra_iop_plan(fnp, Z1, TC, None, L, m, n, hr, hc, None, final=False)
        if key not in _ASTRA_IOP_CACHE and len(_ASTRA_IOP_CACHE) >= _ASTRA_IOP_LIMIT:
            _ASTRA_IOP_CACHE.pop(next(iter(_ASTRA_IOP_CACHE)))
            _ASTRA_IOP_STATS['cache_evictions'] += 1
        _ASTRA_IOP_CACHE[key] = dict(TC=TC, Z1=Z1, plan=plan)
        _ASTRA_IOP_STATS['cache_stores'] += 1
        _ASTRA_IOP_STATS['peak_entries'] = max(_ASTRA_IOP_STATS['peak_entries'], len(_ASTRA_IOP_CACHE))
    _ASTRA_IOP_STATS['live_entries'] = len(_ASTRA_IOP_CACHE)
    full = sum(_astra_iop_count(o) for _, _, _, o in ops) if ops is not None else sum(6 * _astra_iop_count(Z1[j][0]) for j in range(L, 1, -1))
    full += sum(_astra_iop_count(lo) for lo, hi in basis)
    kept = sum(_astra_iop_count(o) for _, _, _, o in plan)
    if kept >= full:
        raise RuntimeError('IOP admitted no positive arithmetic saving')
    _ASTRA_IOP_STATS['saved_elements'] += full - kept
    original_ops = (len(ops) if ops is not None else 6 * (L - 1)) + len(basis)
    _ASTRA_IOP_STATS['prefix_operations_delta'] += len(plan) - original_ops
    _ASTRA_IOP_STATS['calls'] += 1
    _ASTRA_IOP_STATS[context['family']] += 1
    h = _BH_HOLD.append
    for f, x, y, o in plan:
        h(f(x, y, out=o))
    # Leave the original final output write unchanged; all omitted coordinates
    # above feed discarded padding only. This avoids new output-owned caches.
    if type(entry[2]) is list:
        return _l3o_last(entry[2], Z1[1], out, L, m, n)
    tl, tr, bl, br = _rs_out_quadrants(out, L, m, n)
    Z = Z1[1]
    h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[2], out=tl)); h(fnp.add(Z[5], Z[4], out=Z[5]))
    h(fnp.add(Z[5], Z[3], out=tr)); h(fnp.add(Z[5], Z[1], out=br)); h(fnp.add(Z[4], Z[6], out=Z[4])); h(fnp.subtract(br, Z[4], out=bl))
    return out


_ASTRA_RLF_PARENT_UNWIND = _astra_iop_unwind


def _hzg_product(a, b, m, k, n):
    if not _ASTRA_IOP_ON:
        return _ASTRA_IOP_PARENT_HZG(a, b, m, k, n)
    hr, hc = m % 64 == 32, n % 64 == 32
    prior = _ASTRA_IOP_CTX[0]
    # The current wide column-strip path is excluded until separately proved.
    active = _HZD_ON and (hr or hc) and (not hc or n <= 4096)
    _ASTRA_IOP_CTX[0] = dict(L=6, M=(m + 63) // 64 * 64,
        Ns=tuple(min(4096, (n + 63) // 64 * 64 - j) for j in range(0, (n + 63) // 64 * 64, 4096)),
        hr=hr, hc=hc, family='hzg') if active else None
    try:
        return _ASTRA_IOP_PARENT_HZG(a, b, m, k, n)
    finally:
        _ASTRA_IOP_CTX[0] = prior


def _astra_iop_ks(parent, family, a, b, out, m, k, n, plan):
    if not _ASTRA_IOP_ON:
        return parent(a, b, out, m, k, n, plan)
    M, N, sizes, hr, hc, K = plan
    prior = _ASTRA_IOP_CTX[0]
    _ASTRA_IOP_CTX[0] = dict(L=6, M=M, Ns=(N,), hr=bool(hr), hc=bool(hc), family=family) if _HZD_ON and (hr or hc) else None
    try:
        return parent(a, b, out, m, k, n, plan)
    finally:
        _ASTRA_IOP_CTX[0] = prior


def _hz_ks_product(a, b, out, m, k, n, plan):
    return _astra_iop_ks(_ASTRA_IOP_PARENT_KS, 'ks', a, b, out, m, k, n, plan)


def _ubx_ks_product(a, b, out, m, k, n, plan):
    return _astra_iop_ks(_ASTRA_IOP_PARENT_UBX, 'ubx', a, b, out, m, k, n, plan)


def _hz7_product(a, b, m, k, n, dest=None):
    if not _ASTRA_IOP_ON:
        return _ASTRA_IOP_PARENT_HZ7(a, b, m, k, n, dest=dest)
    prior = _ASTRA_IOP_CTX[0]
    active = m % 128 == 64 and n > 0 and n % _HZ7_PK == 0
    _ASTRA_IOP_CTX[0] = dict(L=7, M=(m + 127) // 128 * 128, Ns=(_HZ7_PK,), hr=True, hc=False, family='hz7') if active else None
    try:
        return _ASTRA_IOP_PARENT_HZ7(a, b, m, k, n, dest=dest)
    finally:
        _ASTRA_IOP_CTX[0] = prior


def _astra_iop_predict(self, mlp, flop_budget):
    if not _ASTRA_IOP_ON:
        return _ASTRA_IOP_PARENT_PREDICT(self, mlp, flop_budget)
    if _ASTRA_IOP_CTX[0] is not None:
        raise RuntimeError('IOP stale context at prediction entry')
    _ASTRA_IOP_CACHE.clear()
    for key in _ASTRA_IOP_STATS:
        _ASTRA_IOP_STATS[key] = 0
    try:
        return _ASTRA_IOP_PARENT_PREDICT(self, mlp, flop_budget)
    finally:
        _ASTRA_IOP_CTX[0] = None
        _ASTRA_IOP_CACHE.clear()
        _ASTRA_IOP_STATS['live_entries'] = 0


Estimator.predict = _astra_iop_predict
_PB_CACHES = tuple(_PB_CACHES) + ('_ASTRA_IOP_CACHE',)
