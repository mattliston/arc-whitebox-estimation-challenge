# BEGIN OPUS-ENGINEER BARE TREE VIEWS (bvp v1, 2026-10-01; bit-identical: every arithmetic op, operand and order is the parent's): a tree
# cache entry that is not prebound is bound inside predict, and its first step, the reshape of an arena slice into the tree shape
# (7,)*L + (r, c), is billed at the tree's element count (30-120 Mflop per key; 60-115 keys per network = 0.5-0.6% of the bill;
# E-75 measured 0.8% per network over a 20-network worker). The parent prebinds whole entries for 687 recurring keys in setup
# (about 65 views and 8 ms each), so the network-specific keys, most of them, cannot be covered that way. Here setup also makes
# ONLY the bare tree view (2 requests, no sub-views) for every (kind, depth, r, c) the kernels can ask for, most likely families
# first, inside its own slice of the setup wall guard; _ipt_view then answers from that store, so a lazy bind (first use, or
# after an eviction) reshapes nothing and bills nothing. The sub-views of an entry are free slices and are still built at first
# use, exactly as before. Trees are the same compact exact-shape views of the same arena memory: no gaps, same CPU.
# Every tree view the parent itself binds inside setup goes into the store as well (an evicted prebound entry re-binds for nothing).
_BVP_ON = True
_BVP_WALL_S = 0.45                          # seconds of the setup guard given to this phase (it runs first, like the leaf buffers)
_BVP_T = {}                                 # (start, shape) -> the tree view; filled in setup only, never evicted
_BVP_STATS = {'bound': 0, 'hits': 0, 'misses': 0}
_BVP_SETUP = [False]
_BVP_OLD_VIEW = _ipt_view
_BVP_OLD_BIND = _pb_bind
_BVP_FAMILIES = ()                          # ((kind, L, r range, c range), ...), most valuable first; set by the builder


def _ipt_view(arena, start, shape):
    if _BVP_ON and _SCRATCH is not None and arena is _SCRATCH[0]:
        key = (int(start), tuple(shape))
        found = _BVP_T.get(key)
        if found is not None:
            _BVP_STATS['hits'] += 1
            return found
        if _BVP_SETUP[0]:
            view = _BVP_T[key] = _BVP_OLD_VIEW(arena, start, shape)
            return view
        _BVP_STATS['misses'] += 1
    return _BVP_OLD_VIEW(arena, start, shape)


def _bvp_start(kind, L, r, c):
    # arena offset of a tree: kinds 0, 1, 2 = the A, B, C regions; kind 3 = the k-split's second leaf buffer, behind the C tree
    region = _IPT_REGION[L]
    return (0, region[0], region[0] + region[1], region[0] + region[1] + 7 ** L * r * c)[kind]


def _bvp_keys():
    seen = set()
    for kind, L, rs, cs in _BVP_FAMILIES:
        region = _IPT_REGION.get(L)
        if region is None:
            continue
        cap = region[min(kind, 2)] // 7 ** L // (2 if kind == 3 else 1)
        for c in cs:
            for r in rs:
                if r * c <= cap and (kind, L, r, c) not in seen:
                    seen.add((kind, L, r, c))
                    yield kind, L, r, c


def _pb_bind(t0):
    global _SCRATCH
    if _SCRATCH is None:   # exactly the first predict's arena setup (_predict_binary), as the leaf-buffer phase has it
        arena = fnp.empty((_PACKED_ELEMENTS,), dtype=fnp.float32)
        _SCRATCH = [arena, _cp3_bind_arena_views(arena, _PACKED_PUSH), _cp3_bind_arena_views(arena, _PACKED_READ), None, None, None, None]
        _bind_push6(_SCRATCH[0])
    _BVP_SETUP[0] = True
    try:
        arena = _SCRATCH[0]
        deadline = _pb_time.monotonic() + _BVP_WALL_S
        for kind, L, r, c in (_bvp_keys() if _BVP_ON else ()):
            if _pb_time.monotonic() > deadline:
                break
            _ipt_view(arena, _bvp_start(kind, L, r, c), (7,) * L + (r, c))
            _BVP_STATS['bound'] += 1
        _PB_LOG.append(('bare', _BVP_STATS['bound'], round(_pb_time.monotonic() - t0, 3)))
        _BH_HOLD.clear()
        _BVP_OLD_BIND(t0)
    finally:
        _BVP_SETUP[0] = False


_PB_CACHES = tuple(_PB_CACHES) + ('_BVP_T',)
# END OPUS-ENGINEER BARE TREE VIEWS
