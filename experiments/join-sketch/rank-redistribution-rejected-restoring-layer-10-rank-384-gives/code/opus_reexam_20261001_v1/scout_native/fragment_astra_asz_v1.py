# ASTRA accumulator zero strips once per HZD K-split product v1.
# Exact dead-store removal. TC's first fill and acc's first fill remain.
# Later acc matmuls cannot write its zero strips; TC += acc reads acc.
# A/B encoders and all view binds are disjoint from / do not write acc.
# The final RLF old-product callback uses the same acc geometry and scope.
# No reuse across independent products, no new views or numerical buffers.
_ASZ_ON = True
_ASZ_CONTEXT = [None]
_ASZ_PARENT_VIEWS = _hz_views
_ASZ_PARENT_LEAF = _hz_leaf
_ASZ_PARENT_PRODUCT = _hz_ks_product
_ASZ_STATS = dict(products=0, initialized=0, skipped_products=0,
                  skipped_fills=0, saved_elements=0, nested=0)


def _hz_views(M, q, N, hzr, hzc):
    entry = _ASZ_PARENT_VIEWS(M, q, N, hzr, hzc)
    # The parent's entry1 is precisely _ks_acc(M,N), regardless of q.
    # Metadata only; no extra indexed views or array operations.
    entry[1]['_asz_acc'] = (id(_SCRATCH[0]), M, N, bool(hzr), bool(hzc))
    return entry


def _hz_leaf(v):
    context = _ASZ_CONTEXT[0]
    matched = context is not None and v.get('_asz_acc') == context['key']
    if _ASZ_ON and matched and context['initialized']:
        h = _BH_HOLD.append
        for x, y, out in v['mats']:
            h(fnp.matmul(x, y, out=out))
        _ASZ_STATS['skipped_products'] += 1
        _ASZ_STATS['skipped_fills'] += len(v['fills'])
        _ASZ_STATS['saved_elements'] += context['fill_elements']
        return
    result = _ASZ_PARENT_LEAF(v)
    if matched:
        context['initialized'] = True
        _ASZ_STATS['initialized'] += 1
    return result


def _hz_ks_product(a, b, out, m, k, n, plan):
    prior = _ASZ_CONTEXT[0]
    if prior is not None:
        # No current route nests this kernel. If a future route does, it
        # cannot inherit the outer proof or leave a stale initialized flag.
        _ASZ_STATS['nested'] += 1
        _ASZ_CONTEXT[0] = None
        try:
            return _ASZ_PARENT_PRODUCT(a, b, out, m, k, n, plan)
        finally:
            prior['initialized'] = False
            _ASZ_CONTEXT[0] = prior
    M, N, sizes, hzr, hzc, K = plan
    if not (_ASZ_ON and _HZD_ON and (hzr or hzc) and K == k and len(sizes) >= 2):
        return _ASZ_PARENT_PRODUCT(a, b, out, m, k, n, plan)
    _ASZ_CONTEXT[0] = dict(
        key=(id(_SCRATCH[0]), M, N, bool(hzr), bool(hzc)), initialized=False,
        fill_elements=3*7**5*(int(hzr)*(N//64)+int(hzc)*(M//64)))
    _ASZ_STATS['products'] += 1
    try:
        return _ASZ_PARENT_PRODUCT(a, b, out, m, k, n, plan)
    finally:
        _ASZ_CONTEXT[0] = None
