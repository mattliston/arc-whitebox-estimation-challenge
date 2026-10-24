

# BEGIN FABLE-ENGINEER THRESHOLD LRU FOR THE FRAME CACHES (E-31 v3, 2026-09-23): the server keeps one handle per cached view and
# caps its store at 100k per worker connection (about 20 networks). Every frame-cache entry records the last prediction that used
# it. When a prediction starts (_w5_port_new, the first call of _predict_binary) and the caches hold more than _EVL_HIGH distinct
# arrays, the least recently used entries (whole predictions at a time, oldest first) are dropped until at most _EVL_LOW remain.
# Below the threshold nothing is evicted, so the bill is unchanged; above it, only frames that a later network needs again after a
# long idle stretch are re-bound. Frames are views of the arena, so predictions never change. _predict_binary is not redefined.
_EVL_HIGH = 60000
_EVL_LOW = 40000
_EVL_KEY_GATE = 1200    # the array walk runs only above this many tracked keys (about 34 arrays per key: roughly 40k arrays)
_EVL_NET = [0]
_EVL_LAST = {}
_EVL_CACHES = ('_A_FRAME_CACHE', '_B_FRAME_CACHE', '_O_FRAME_CACHE', '_PACKED_FRAME_CACHE', '_BH_O', '_BH_F', '_BH_X',
               '_READ5_FRAME_CACHE', '_READ5_CHILD_CACHE')
_EVL_BH_FWD = _bh_fwd
_EVL_BH_O_FRAME = _bh_o_frame
_EVL_BH_LEAFX = _bh_leafx
_EVL_PACKED_FRAME = _packed_frame
_EVL_READ5_FRAMES = _read5_packet_frames
_EVL_W5_NEW = _w5_port_new


def _evl_mark(*keys):
    net = _EVL_NET[0]
    for key in keys:
        _EVL_LAST[key] = net


def _bh_fwd(views, L, m, k, n):
    v = id(views)
    _evl_mark((0, v, L, m, k), (1, v, L, k, n), (5, v, L, m, k, n))
    return _EVL_BH_FWD(views, L, m, k, n)


def _bh_o_frame(views, L, m, n):
    _evl_mark((4, id(views), L, m, n))
    return _EVL_BH_O_FRAME(views, L, m, n)


def _bh_leafx(views, L, m, k, n, leaf_a, leaf_b):
    _evl_mark((6, id(views), L, m, k, n))
    return _EVL_BH_LEAFX(views, L, m, k, n, leaf_a, leaf_b)


def _packed_frame(levels, m, k, n, views):
    v = id(views)
    _evl_mark((0, v, levels, m, k), (1, v, levels, k, n), (2, v, levels, m, n), (3, v, levels, m, k, n))
    return _EVL_PACKED_FRAME(levels, m, k, n, views)


def _read5_packet_frames(m, k, n, views, leaf_a, leaf_b, leaf):
    v = id(views); arena = id(_VIEW_SPECS[v][0])
    _evl_mark((7, v, m, k, n), (8, arena, 2, m, k, n), (8, arena, 3, m, k, n))
    return _EVL_READ5_FRAMES(m, k, n, views, leaf_a, leaf_b, leaf)


def _evl_leaves(value, out):
    # arrays are the leaves of the cached tuples/lists (the client's array type is not assumed)
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (tuple, list)):
            stack.extend(item)
        elif item is not None and not isinstance(item, (int, float, str, bool)):
            out.append(id(item))


def _evl_full(index, key):
    return (index,) + tuple(key) if not (index == 6 and key[0] == 'r5') else None


def _w5_port_new(network):
    _EVL_NET[0] += 1
    if len(_EVL_LAST) < _EVL_KEY_GATE:
        return _EVL_W5_NEW(network)
    g = globals()
    newest = {}
    for index, name in enumerate(_EVL_CACHES):
        cache = g.get(name)
        if not isinstance(cache, dict):
            continue
        for key, value in cache.items():
            full = _evl_full(index, key)
            last = _EVL_LAST.get(full, 0) if full is not None else _EVL_NET[0]
            leaves = []
            _evl_leaves(value, leaves)
            for a in leaves:
                if newest.get(a, -1) < last:
                    newest[a] = last
    if len(newest) > _EVL_HIGH:
        per_net = {}
        for last in newest.values():
            per_net[last] = per_net.get(last, 0) + 1
        keep, cutoff = 0, -1
        for net in sorted(per_net, reverse=True):
            if keep + per_net[net] > _EVL_LOW:
                cutoff = net
                break
            keep += per_net[net]
        if cutoff >= 0:
            for index, name in enumerate(_EVL_CACHES):
                cache = g.get(name)
                if not isinstance(cache, dict) or name == '_BH_X':
                    continue
                for key in [key for key in cache if _EVL_LAST.get(_evl_full(index, key), 0) <= cutoff]:
                    del cache[key]
                    _EVL_LAST.pop(_evl_full(index, key), None)
            live = set()
            for found in _READ5_CHILD_CACHE.values():
                live.update((id(found[0]), id(found[1]), id(found[2])))
            for key in list(_BH_X):
                if key[0] == 'r5':
                    if key[1] not in live or key[2] not in live or key[3] not in live:
                        del _BH_X[key]
                elif _EVL_LAST.get((6,) + tuple(key), 0) <= cutoff:
                    del _BH_X[key]
                    _EVL_LAST.pop((6,) + tuple(key), None)
    return _EVL_W5_NEW(network)
# END FABLE-ENGINEER THRESHOLD LRU FOR THE FRAME CACHES
