

# BEGIN FABLE-ENGINEER PER-NET FRAME EVICTION, ALL LAYOUTS (E-31, 2026-09-23): the server keeps one handle per cached view and
# caps its store at 100k per worker connection (about 20 networks); 69% of cached frame arrays serve one network only (their
# shapes carry per-layer active counts). When a new prediction starts (_w5_port_new, the first call of _predict_binary), every
# frame-cache entry that the previous prediction did not use is dropped; entries every network uses stay. Frames are views of the
# arena, so a re-bound frame is identical: predictions are unchanged, and only a frame that skips a network is re-bound (billed).
_EVA_USED = set()
_EVA_CACHES = ('_A_FRAME_CACHE', '_B_FRAME_CACHE', '_O_FRAME_CACHE', '_PACKED_FRAME_CACHE', '_BH_O', '_BH_F', '_BH_X',
               '_READ5_FRAME_CACHE', '_READ5_CHILD_CACHE')
_EVA_BH_FWD = _bh_fwd
_EVA_BH_O_FRAME = _bh_o_frame
_EVA_BH_LEAFX = _bh_leafx
_EVA_PACKED_FRAME = _packed_frame
_EVA_READ5_FRAMES = _read5_packet_frames
_EVA_W5_NEW = _w5_port_new


def _bh_fwd(views, L, m, k, n):
    v = id(views)
    _EVA_USED.update(((0, v, L, m, k), (1, v, L, k, n), (5, v, L, m, k, n)))
    return _EVA_BH_FWD(views, L, m, k, n)


def _bh_o_frame(views, L, m, n):
    _EVA_USED.add((4, id(views), L, m, n))
    return _EVA_BH_O_FRAME(views, L, m, n)


def _bh_leafx(views, L, m, k, n, leaf_a, leaf_b):
    _EVA_USED.add((6, id(views), L, m, k, n))
    return _EVA_BH_LEAFX(views, L, m, k, n, leaf_a, leaf_b)


def _packed_frame(levels, m, k, n, views):
    v = id(views)
    _EVA_USED.update(((0, v, levels, m, k), (1, v, levels, k, n), (2, v, levels, m, n), (3, v, levels, m, k, n)))
    return _EVA_PACKED_FRAME(levels, m, k, n, views)


def _read5_packet_frames(m, k, n, views, leaf_a, leaf_b, leaf):
    v = id(views); arena = id(_VIEW_SPECS[v][0])
    _EVA_USED.update(((7, v, m, k, n), (8, arena, 2, m, k, n), (8, arena, 3, m, k, n)))
    return _EVA_READ5_FRAMES(m, k, n, views, leaf_a, leaf_b, leaf)


def _w5_port_new(network):
    if _EVA_USED:
        g = globals()
        for index, name in enumerate(_EVA_CACHES):
            cache = g.get(name)
            if not isinstance(cache, dict) or name == '_BH_X':
                continue
            for key in [key for key in cache if (index,) + tuple(key) not in _EVA_USED]:
                del cache[key]
        # read5 leaf extras are keyed by their child arrays: keep those whose child frame survived
        live = set()
        for found in _READ5_CHILD_CACHE.values():
            live.update((id(found[0]), id(found[1]), id(found[2])))
        for key in [key for key in _BH_X if (key[0] == 'r5' and (key[1] not in live or key[2] not in live or key[3] not in live))
                    or (key[0] != 'r5' and (6,) + tuple(key) not in _EVA_USED)]:
            del _BH_X[key]
    _EVA_USED.clear()
    return _EVA_W5_NEW(network)
# END FABLE-ENGINEER PER-NET FRAME EVICTION, ALL LAYOUTS
