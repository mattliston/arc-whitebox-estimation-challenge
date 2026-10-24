# Frozen public-shape padding reserve; actual0.12.3 prices in SCREEN_v1.json.
# Only local temporary arrays are introduced. Existing dispatcher cache policy stays parent-owned.
_O18PAD_RULES = {('mm', (672, 800, 6752)): (704, 832, 6784), ('mm', (736, 768, 6944)): (768, 768, 6976), ('mm', (736, 800, 6880)): (768, 832, 6912), ('mm', (768, 800, 6688)): (768, 832, 6720), ('mm', (832, 864, 6144)): (832, 896, 6144), ('mm', (832, 864, 7008)): (832, 896, 7040), ('mm', (832, 864, 8256)): (832, 896, 8256), ('mm', (896, 928, 5728)): (896, 1024, 5760), ('mm', (896, 928, 6304)): (896, 960, 6336), ('mm', (896, 928, 7936)): (896, 960, 7936), ('mm', (896, 928, 7968)): (896, 960, 8000), ('mm', (896, 928, 8000)): (896, 960, 8000), ('mm', (896, 928, 8032)): (896, 960, 8064), ('mm', (896, 928, 9888)): (896, 960, 9920), ('mm', (960, 928, 96)): (960, 960, 128), ('mm', (960, 928, 9824)): (960, 960, 9856), ('mm', (1024, 4096, 992)): (1024, 4096, 1024), ('shared_old384', (544, 384, 384)): (576, 384, 384)}
_O18PAD_PARENT_MM = _mm
_O18PAD_PARENT_SHARED = _shared_old384_mm
_O18PAD_PARENT_PREDICT = Estimator.predict
_O18PAD_BUSY = [False]
_O18PAD_STATS = dict(mm=0, shared=0, pad_elements=0, crop_elements=0)

def _o18pad_product(entry, parent, a, b):
    scope = _W5_PORT_SCOPE
    if (_O18PAD_BUSY[0] or not _WINOGRAD or not _AB_ON or _SCRATCH is None
            or _FAST_RESCUE_ON or _UBC_CONTEXT[0] is not None or _UBX_CTX[0] is not None
            or _ASTRA_CLUSTER_CTX[0] is not None or _UBX_RC[0] is not None
            or _ASTRA_RLF_CONTEXT[0] is not None
            or (scope is not None and scope.get('grant') is not None)
            or len(a.shape) != 2 or len(b.shape) != 2
            or a.dtype != 'float32' or b.dtype != 'float32'
            or int(a.shape[1]) != int(b.shape[0])):
        return parent(a, b)
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    dims = _O18PAD_RULES.get((entry, (m, k, n)))
    if dims is None:
        return parent(a, b)
    M, K, N = dims
    hold = _E2Hold()
    _O18PAD_BUSY[0] = True
    try:
        aa, bb = a, b
        if (M, K) != (m, k):
            aa = hold(fnp.zeros((M, K), dtype=fnp.float32))
            hold(fnp.copyto(hold(aa[:m, :k]), a))
            _O18PAD_STATS['pad_elements'] += M*K
        if (K, N) != (k, n):
            bb = hold(fnp.zeros((K, N), dtype=fnp.float32))
            hold(fnp.copyto(hold(bb[:k, :n]), b))
            _O18PAD_STATS['pad_elements'] += K*N
        answer = parent(aa, bb); hold.clear(); del aa, bb  # Owned product no longer needs padded inputs.
        if (M, N) != (m, n):
            hold(answer)
            answer = fnp.copy(hold(answer[:m, :n]))
            _O18PAD_STATS['crop_elements'] += m*n
        _O18PAD_STATS['mm' if entry == 'mm' else 'shared'] += 1
        return answer
    finally:
        _O18PAD_BUSY[0] = False

def _mm(a, b):
    return _o18pad_product('mm', _O18PAD_PARENT_MM, a, b)

def _shared_old384_mm(a, b):
    return _o18pad_product('shared_old384', _O18PAD_PARENT_SHARED, a, b)

def _o18pad_predict(self, *args, **kwargs):
    _O18PAD_STATS.update(mm=0, shared=0, pad_elements=0, crop_elements=0)
    try:
        return _O18PAD_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _O18PAD_BUSY[0] = False
        _ASTRA_COMPUTE_STATS['padding18'] = dict(_O18PAD_STATS)

Estimator.predict = _o18pad_predict
