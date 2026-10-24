
# Fixed runtime conversion: two additional eligible symmetric strips, a single
# publicly priced rank320 fusion shape, and class-preserving own aliases.
_O18_FUSION320_RULES = {(896,832,896,9504): (0,77)}
# Literal wrapper price saves4.082MF/80requests; reserve3gather calls and all
# bill credit. Existing20MF spend budget and every reuse/peel exclusion remain.
_O18_FUSION320_PREPARE_PARENT = _o18_fusion_prepare
def _o18_fusion_prepare320(layer, w, p, previous, newborn, basis):
    m, k = int(w.shape[1]), int(previous.shape[0])
    n, selected, base, t, ss = newborn
    n, q = int(n), int(base.shape[1])
    plan = dict(w=w, p=p, previous=previous, newborn=newborn, basis=basis,
                total=m, young_rows=m, negative=None, kept_count=m,
                ynt_context=None, rows=None, tail=None, U=None, eligible=False)
    ctx = _OTT_CONTEXT[0]
    if ctx is not None and 7 <= layer <= 14:
        # These are exactly the existing YNT eligibility checks/count operations.
        # Cache their result for its later invocation; no duplicate remote work.
        births = int(selected.shape[0])
        if (int(w.shape[0]) != 1024 or ctx['n'] != 1024
                or tuple(ctx['a'].shape) != (m,) or tuple(ctx['live'].shape) != (m,)
                or tuple(p.shape) != (k, n) or tuple(base.shape) != (1024, births)
                or tuple(t.shape) != (births,) or tuple(ss.shape) != (births,)
                or n + 2 * births > int(ctx['lam'].shape[0])):
            return None  # Unmodified parent retains its validation/error behavior.
        negative = (ctx['a'] < -2.5) & (~ctx['live'])
        kept = min(m, (int(fnp.nonzero(~negative)[0].shape[0]) + 63) // 64 * 64)
        plan.update(negative=negative, kept_count=kept, ynt_context=ctx,
                    young_rows=kept if m - kept >= 64 else m)
    my = plan['young_rows']
    key = (m, my, k, n + q)
    price = _O18_FUSION320_RULES.get(key)
    if (price is None or tuple(basis.shape) != (k, 320)
            or _W5_PORT_SCOPE is not None or _FAST_RESCUE_ON
            or _UBC_CONTEXT[0] is not None or _UBX_CTX[0] is not None
            or _ASTRA_CLUSTER_CTX[0] is not None or _UBX_RC[0] is not None
            or _ASTRA_RLF_CONTEXT[0] is not None
            or (m, k, 320) in _SM12_RULE or (m, k, 320) in _O18_SHARED_RULES):
        return plan
    # Preserve every currently affordable compact10 route and its debit.
    if any(cost <= _O18_PEEL_STATE['left']
           for gain, cost, route in _O18_UNION_RULES.get((my, k, n + q), ())):
        return plan
    loss = max(0, -int(price[0]))
    if loss <= _O18_FUSION_STATE['left']:
        plan.update(eligible=True, key=key, price=price, loss=loss)
    return plan

def _o18_fusion_prepare(layer,w,p,previous,newborn,basis):
    if len(basis.shape)==2 and int(basis.shape[1])==320:
        return _o18_fusion_prepare320(layer,w,p,previous,newborn,basis)
    return _O18_FUSION320_PREPARE_PARENT(layer,w,p,previous,newborn,basis)

def _astra_bcc_newborn_impl(w, p, previous, newborn):
    ctx = _O18_FUSION_CONTEXT[0]
    if (ctx is None or not ctx['eligible'] or p is not ctx['p']
            or previous is not ctx['previous'] or newborn is not ctx['newborn']):
        return _O18_FUSION_PARENT_NEWBORN(w, p, previous, newborn)
    n, selected, base, t, ss = newborn
    m, k, n, q = int(w.shape[1]), int(previous.shape[0]), int(n), int(base.shape[1])
    if (m != ctx['young_rows'] or tuple(ctx['basis'].shape) != (k, int(ctx['basis'].shape[1]))
            or ctx['U'] is not None or int(p.shape[0]) != k
            or not _AB_ON or not _WINOGRAD or _SCRATCH is None
            or w.dtype != 'float32' or p.dtype != 'float32' or base.dtype != 'float32'):
        raise RuntimeError('U/young fusion ownership or kernel contract')
    if m != ctx['total'] and (ctx['rows'] is None or ctx['tail'] is None
                             or int(ctx['tail'].shape[0]) != 64):
        raise RuntimeError('U/young fusion requires exact captured YNT tail')
    hold = _E2Hold()
    wp = _xsg_rows(5154, w, previous).T
    prior = p[:, :n]
    common_rhs = _xsg_rows(5159, base, previous)
    rhs = fnp.concatenate((prior, common_rhs, ctx['basis']), axis=1)
    product = _mm(wp, rhs)
    # Copy before another kernel can touch scratch. No basis row is omitted.
    basis_kept = fnp.copy(product[:, n + q:])
    inherited = product[:, :n]
    common = fnp.copy(product[:, n:n + q])
    _PHYS_HANDLES['newborn_common'] = common
    directions = _xsg_rows(5161, w, selected).T
    answer = fnp.empty((m, n + 2 * q), dtype=fnp.float32)
    hold.append(fnp.copyto(answer[:, :n], inherited))
    hold.append(fnp.add(common, directions * t, out=answer[:, n:n + q]))
    hold.append(fnp.add(common, directions * ss, out=answer[:, n + q:]))
    if m == ctx['total']:
        U = basis_kept
    else:
        wt = _xsg_cols(36501, ctx['w'], ctx['tail'])
        tail_left = _xsg_rows(36502, wt, previous).T
        basis_tail = _shared_old384_mm(tail_left, ctx['basis'])
        U = fnp.empty((ctx['total'], int(ctx['basis'].shape[1])), dtype=fnp.float32)
        hold.append(fnp.put_along_axis(U, ctx['rows'][:, None], basis_kept, axis=0))
        hold.append(fnp.put_along_axis(U, ctx['tail'][:, None], basis_tail, axis=0))
    ctx['U'] = U
    _ASTRA_BCC_STATS['newborn_elements'] += 2 * m * q
    _JUN_STATS['calls'] += 1
    _JUN_STATS['shapes'].append((m, k, n, q))
    _O18_FUSION_STATE['left'] -= ctx['loss']
    _O18_FUSION_STATE['accepted'] += 1
    _O18_FUSION_STATE['selected'].append((ctx['key'], ctx['price']))
    return answer

_O18_MARGIN_STATE.update(left=7)
def _o18_margin_predict(self, *args, **kwargs):
    _O18_MARGIN_STATE.update(left=7, calls=0, inners=[], retired=False)
    try:
        return _O18_MARGIN_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _O18_MARGIN_STATE['retired'] = True
Estimator.predict = _o18_margin_predict

_O18_L7_OWN_KEYS = ((0, 3, 384, 832), (0, 3, 832, 320), (0, 6, 320, 960), (0, 6, 384, 320), (0, 6, 704, 320), (0, 6, 768, 320), (0, 6, 832, 320), (1, 3, 320, 320), (1, 3, 320, 384), (1, 3, 832, 320), (1, 6, 3072, 320), (1, 6, 320, 2048), (1, 6, 320, 4096), (1, 6, 320, 704), (1, 6, 320, 768), (1, 6, 896, 320), (1, 6, 960, 320), (2, 3, 384, 320), (2, 3, 832, 320), (2, 3, 832, 384), (2, 6, 320, 2048), (2, 6, 320, 4096), (2, 6, 704, 320), (2, 6, 768, 320), (2, 6, 896, 320), (2, 6, 960, 320))
assert len(_O18_L7_OWN_KEYS)==26 and not set(_O18_L7_OWN_KEYS).intersection(_PB_SINGLE)
_O18_L7_OWN_PARENT_BIND = _ASH_PARENT_BIND
def _o18_l7_own_bind(t0):
    result=_O18_L7_OWN_PARENT_BIND(t0)
    if _SCRATCH is None:
        return result
    saved_lb,saved_eager=_LB_SETUP[0],_LVF_EAGER[0]
    _LB_SETUP[0],_LVF_EAGER[0]=False,False
    try:
        for kind,level,rows,cols in _O18_L7_OWN_KEYS:
            entry=_ipt_side(kind,level,rows,cols)
            assert entry[3] is None, 'Own rank320 alias changed its normal circuit class'
            slot=entry[2]
            if type(slot)is list and slot[1]is None:
                _l3o_pairs(slot,level) if kind==2 else _l3_quad(slot,kind,level)
    finally:
        _LB_SETUP[0],_LVF_EAGER[0]=saved_lb,saved_eager
    return result
# Bind inside existing hold facade; outer setup rollback/collect/freeze retained.
_ASH_PARENT_BIND = _o18_l7_own_bind
