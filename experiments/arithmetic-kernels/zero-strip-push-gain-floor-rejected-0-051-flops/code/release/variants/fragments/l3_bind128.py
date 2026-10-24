# FIRST-NETWORK DIET (release lane 2026-10-02; needs the load3 and load3o fragments (L3, L3o) and the level-fusion prebind): the
# first-load views of the first _FNV_KEYS listed tree keys are built in setup, right after each key is bound, instead of at the
# key's first load inside a prediction. load3 / load3o build them at the first load so that the binding phase makes no more
# requests; the price is paid by the first network a worker runs after a setup, and that network is the one nearest the 0.4 s
# residual limit (#333528: 0.313-0.347 s against at most 0.322 s for the other 93 networks; #333559: 0.3652 s on the first
# network of a worker set up a second time at 1.124x pace). A network uses about 102 of the first 128 listed keys (20 fresh
# networks, the engineer's key censuses), 3 requests per left- or right-operand key (_l3_quad), 4 per output key (_l3o_pairs).
# Same views, built by the same two functions, only earlier: predictions bit-identical; the reshape of _l3_quad is billed to the
# setup instead of the first prediction that loads the key (the bill of that prediction falls by the operand's element count).
# Setup: +3 or +4 requests per listed key inside the binding phase, under the same wall guard (128 keys: about +430 requests of
# the 19,000 a normal platform worker gets; the deferred fusion gets that much less of the slack).
_FNV_KEYS = 128
_FNV_STATS = dict(views=0)
_FNV_BASE_BIND = _LVF_OLD_PB_BIND


def _LVF_OLD_PB_BIND(t0):
    # the binding phase exactly as before (the level-fusion _pb_bind calls it by this name), with _ipt_side wrapped for its
    # duration only: a key of the first _FNV_KEYS gets its four-slot view (A, B) or its output pair views (C) as it is bound
    global _ipt_side
    plain = _ipt_side
    first = set(_PB_SINGLE[:_FNV_KEYS])

    def bind(kind, L, rows, cols):
        entry = plain(kind, L, rows, cols)
        slot = entry[2]
        if (kind, L, rows, cols) in first and type(slot) is list and slot[1] is None:
            if kind == 2:
                _l3o_pairs(slot, L)
            else:
                _l3_quad(slot, kind, L)
            _FNV_STATS['views'] += 1
        return entry

    _ipt_side = bind
    try:
        _FNV_BASE_BIND(t0)
    finally:
        _ipt_side = plain
        _PB_LOG.append(('first_load_views', _FNV_STATS['views'], round(_pb_time.monotonic() - t0, 3)))
