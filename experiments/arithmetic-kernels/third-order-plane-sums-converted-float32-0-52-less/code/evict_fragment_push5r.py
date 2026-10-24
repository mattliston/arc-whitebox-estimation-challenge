

# BEGIN FABLE-ENGINEER PER-NET EVICTION OF SINGLE-USE FRAMES (E-31 v2, 2026-09-23): the server keeps one handle per cached view and
# caps its store at 100k per worker connection (about 20 networks). Two frame families serve one network only: the five-level
# push frames (keyed by per-layer active counts) and every frame bound inside the physicist's L15 row cut (its row count |R|
# changes per network). When a new prediction starts (_w5_port_new, the first call of _predict_binary), the frames of those two
# families that the previous prediction did not use elsewhere are dropped. Frames are views of the arena: predictions are
# unchanged; only a dropped frame that a later network needs again is re-bound (billed). _predict_binary is not redefined.
_EVR_USED = set()      # keys used outside the row cut in the current prediction
_EVR_ROWCUT = set()    # keys used inside the row cut in the current prediction
_EVR_IN_ROWCUT = [False]
_EVR_CACHES = ('_A_FRAME_CACHE', '_B_FRAME_CACHE', '_O_FRAME_CACHE', '_PACKED_FRAME_CACHE', '_BH_O', '_BH_F', '_BH_X',
               '_READ5_FRAME_CACHE', '_READ5_CHILD_CACHE')
_EVR_BH_FWD = _bh_fwd
_EVR_BH_O_FRAME = _bh_o_frame
_EVR_BH_LEAFX = _bh_leafx
_EVR_PACKED_FRAME = _packed_frame
_EVR_READ5_FRAMES = _read5_packet_frames
_EVR_W5_NEW = _w5_port_new


def _evr_mark(*keys):
    (_EVR_ROWCUT if _EVR_IN_ROWCUT[0] else _EVR_USED).update(keys)


def _bh_fwd(views, L, m, k, n):
    v = id(views)
    _evr_mark((0, v, L, m, k), (1, v, L, k, n), (5, v, L, m, k, n))
    return _EVR_BH_FWD(views, L, m, k, n)


def _bh_o_frame(views, L, m, n):
    _evr_mark((4, id(views), L, m, n))
    return _EVR_BH_O_FRAME(views, L, m, n)


def _bh_leafx(views, L, m, k, n, leaf_a, leaf_b):
    _evr_mark((6, id(views), L, m, k, n))
    return _EVR_BH_LEAFX(views, L, m, k, n, leaf_a, leaf_b)


def _packed_frame(levels, m, k, n, views):
    v = id(views)
    _evr_mark((0, v, levels, m, k), (1, v, levels, k, n), (2, v, levels, m, n), (3, v, levels, m, k, n))
    return _EVR_PACKED_FRAME(levels, m, k, n, views)


def _read5_packet_frames(m, k, n, views, leaf_a, leaf_b, leaf):
    v = id(views); arena = id(_VIEW_SPECS[v][0])
    _evr_mark((7, v, m, k, n), (8, arena, 2, m, k, n), (8, arena, 3, m, k, n))
    return _EVR_READ5_FRAMES(m, k, n, views, leaf_a, leaf_b, leaf)


if '_l15_rowcut_k3' in globals():
    _EVR_ROWCUT_FN = _l15_rowcut_k3

    def _l15_rowcut_k3(*args, **kwargs):
        _EVR_IN_ROWCUT[0] = True
        try:
            return _EVR_ROWCUT_FN(*args, **kwargs)
        finally:
            _EVR_IN_ROWCUT[0] = False


def _w5_port_new(network):
    if _EVR_USED or _EVR_ROWCUT:
        g = globals()
        push5 = id(_SCRATCH_PUSH5) if _SCRATCH_PUSH5 is not None else None
        drop = set()
        for index, name in enumerate(_EVR_CACHES):
            cache = g.get(name)
            if not isinstance(cache, dict) or name == '_BH_X':
                continue
            for key in cache:
                full = (index,) + tuple(key)
                if full in _EVR_USED:
                    continue
                if full in _EVR_ROWCUT or (push5 is not None and key[0] == push5):
                    drop.add((name, key))
        for name, key in drop:
            del g[name][key]
        live = set()
        for found in _READ5_CHILD_CACHE.values():
            live.update((id(found[0]), id(found[1]), id(found[2])))
        for key in list(_BH_X):
            if key[0] == 'r5':
                if key[1] not in live or key[2] not in live or key[3] not in live:
                    del _BH_X[key]
            else:
                full = (6,) + tuple(key)
                if full not in _EVR_USED and (full in _EVR_ROWCUT or (push5 is not None and key[0] == push5)):
                    del _BH_X[key]
    _EVR_USED.clear(); _EVR_ROWCUT.clear()
    return _EVR_W5_NEW(network)
# END FABLE-ENGINEER PER-NET EVICTION OF SINGLE-USE FRAMES
