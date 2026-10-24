

# BEGIN [fable-lead] BOX VIEWS v3ab47 (2026-10-19; append LAST). A/B sides only, depths 4 and 7 (the C-side depth-6 boxes broke the first network: see TEAM_LOG 05:1xZ). v2 boxed the depth-6 A and B trees that setup never binds. The lazy binds
# that remain charged (0.12.3-weighted census of the base a7ffd713, net 7103256: 21 tree reshapes = 0.18% of the bill, plus their
# level-1 and quadrant reshapes 0.05%) are the depth-6 C trees and the depth-4 / depth-7 trees of all three sides. Same mechanism as v2:
# setup reshapes ONE maximal tree per (side, depth) over the whole region, with its level-1 natural-order view and (A/B) quadrant view;
# a key bound lazily in predict gets that box's [..., :r, :c] slice, and its level-1 / quadrant views are the box's views sliced the same
# way (free views, same elements, a strided leaf). Same exclusions as v2: keys setup bound keep the packed layout, keys GCV prepared keep
# their prepared tree, u/v leg B trees (256 / 384 rows) stay packed, nothing is boxed while the r7 caches are swapped in. Values untouched.
_LB3_ON = True
_LB3_DEPTHS = (4, 7)
_LB3_BOX = {}                                               # (kind, L) -> dict(arena, start, R, C, T, V1, quad, keep)
_LB3_OF = {}                                                # id(view) -> (view, kind, L, r, c) for views cut from a box
_LB3_OF_CAP = 4096
_LB3_STATS = dict(boxes=0, tree=0, level1=0, quad=0)
_LB3_CACHES = (_IPT_A, _IPT_B, _IPT_C)
_LB3_PARENT_VIEW = _ipt_view
_LB3_PARENT_LV1 = _lvf_rs_level1
_LB3_PARENT_QUAD = _l3_quad
_LB3_PARENT_PB = _pb_bind
_PB_CACHES = tuple(_PB_CACHES) + ('_LB3_BOX', '_LB3_OF')


def _lb3_note(view, kind, L, r, c):
    if len(_LB3_OF) >= _LB3_OF_CAP:
        _LB3_OF.pop(next(iter(_LB3_OF)))
    _LB3_OF[id(view)] = (view, kind, L, r, c)


def _lb3_make_boxes():
    if not _LB3_ON or _SCRATCH is None or _LB3_BOX:
        return
    arena = _SCRATCH[0]
    for L in _LB3_DEPTHS:
        region = _IPT_REGION.get(L)
        if region is None:
            continue
        u = 1 << L
        nr = L - 1
        for kind in (0, 1, 2):
            if kind == 2 or L == 6:
                continue                                    # no C-side boxes (strided out= of the leaf matmul broke rank 35); depth 6 A/B are v2's
            R = 1024 // u
            C = (region[kind] // 7 ** L) // R
            if C < 1 or 7 ** L * R * C > region[kind]:
                continue
            start = (0, region[0], region[0] + region[1])[kind]
            T = _LB3_PARENT_VIEW(arena, start, (7,) * L + (R, C))
            data = slice(0, 7, 2) if kind == 0 else slice(0, 4)
            V1 = fnp.reshape(T[(data,) * nr], (2, 2) * nr + (7, R, C))
            swap = [x for p in range(nr) for x in (2 * p + 1, 2 * p)] if kind == 2 else list(range(2 * nr))
            base = swap + list(range(2 * nr, 2 * nr + 3))
            G = fnp.transpose(V1, [base[x] for x in _rs_nat_perm(L, 0)])
            quad, keep = None, (G,)
            if kind < 2:
                Dv = G[0:7:2] if kind == 0 else G[0:4]
                Q = fnp.reshape(Dv, (2, 2) + tuple(int(x) for x in G.shape[1:]))
                quad = fnp.transpose(Q, [0] + list(range(2, L + 2)) + [1] + list(range(L + 2, 2 * L + 2)))
                keep = (G, Dv, Q)
            _LB3_BOX[(kind, L)] = dict(arena=arena, start=start, R=R, C=C, T=T, V1=V1, quad=quad, keep=keep)
            _LB3_STATS['boxes'] += 1


def _pb_bind(t0):
    _LB3_PARENT_PB(t0)
    _lb3_make_boxes()


def _ipt_view(arena, start, shape):
    if _LB3_ON and _LB3_BOX and _SCRATCH is not None and arena is _SCRATCH[0] and not _LB_SETUP[0]:
        L = len(shape) - 2
        if L in _LB3_DEPTHS and tuple(shape[:L]) == (7,) * L and _GCV_TREES.get((start, tuple(shape))) is None:
            region = _IPT_REGION[L]
            kind = 0 if start == 0 else 1 if start == region[0] else 2 if start == region[0] + region[1] else None
            box = _LB3_BOX.get((kind, L)) if kind is not None else None
            r, c = int(shape[L]), int(shape[L + 1])
            u = 1 << L
            if (box is not None and box['arena'] is arena and box['start'] == start and r <= box['R'] and c <= box['C']
                    and _LB_IN_SIDE[0] == (kind, L, u * r, u * c)
                    and (kind, L, u * r, u * c) not in _LB_SETUP_KEYS
                    and (_IPT_A, _IPT_B, _IPT_C)[kind] is _LB3_CACHES[kind]
                    and not (kind == 1 and u * r in (256, 384))):
                view = box['T'][..., :r, :c]
                _BH_HOLD.append(view)
                _lb3_note(view, kind, L, r, c)
                _LB3_STATS['tree'] += 1
                return view
    return _LB3_PARENT_VIEW(arena, start, shape)


def _lvf_rs_level1(kind, L, T, tail):
    got = _LB3_OF.get(id(T))
    if got is None or got[0] is not T or got[1] != kind or got[2] != L or len(tail) != 2:
        return _LB3_PARENT_LV1(kind, L, T, tail)
    _view, _kind, _L, r, c = got
    if (r, c) != (int(tail[0]), int(tail[1])):
        return _LB3_PARENT_LV1(kind, L, T, tail)
    V1 = _LB3_BOX[(kind, L)]['V1'][..., :r, :c]
    nr = L - 1
    swap = [x for p in range(nr) for x in (2 * p + 1, 2 * p)] if kind == 2 else list(range(2 * nr))
    base = swap + list(range(2 * nr, 2 * nr + 3))
    Vn = fnp.transpose(V1, [base[x] for x in _rs_nat_perm(L, 0)])
    _lb3_note(Vn, kind, L, r, c)
    _LB3_STATS['level1'] += 1
    return [(fnp.unstack(Vn, axis=0), [Vn, None])], [Vn]


def _l3_quad(slot, kind, L):
    G = slot[0]
    got = _LB3_OF.get(id(G)) if G is not None else None
    if got is None or got[0] is not G or got[1] != kind or got[2] != L or kind > 1 or (kind, L) not in _LB3_BOX:
        return _LB3_PARENT_QUAD(slot, kind, L)
    _view, _kind, _L, r, c = got
    f = slice(None)
    quad = _LB3_BOX[(kind, L)]['quad'][(f,) * L + (slice(0, r),) + (f,) * L + (slice(0, c),)]
    _BH_HOLD.append(G)
    slot[0] = None; slot[1] = quad
    _L3_STATS['quads'] += 1
    _LB3_STATS['quad'] += 1
    return quad


_LB3_PREDICT_PARENT = Estimator.predict
def _lb3_predict(self, *args, **kwargs):
    try:
        return _LB3_PREDICT_PARENT(self, *args, **kwargs)
    finally:
        _ASTRA_COMPUTE_STATS['box_views3'] = dict(_LB3_STATS)
Estimator.predict = _lb3_predict
# END [fable-lead] BOX VIEWS v3
