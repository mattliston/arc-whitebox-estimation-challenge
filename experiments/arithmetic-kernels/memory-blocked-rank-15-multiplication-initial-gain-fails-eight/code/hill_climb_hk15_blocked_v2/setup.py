# Appended after the frozen constants and blocked kernel. All numerical work
# executes under the parent's ordinary setup BudgetContext; no module arrays.
_HK_PRODUCTS = {}
_HK_SIDES = {}
_HK_ARENA = [None]
_HK_READY = [False]
_HK_BINDING = [False]
_HK_GENERATION = [0]
_HK_STATS = {}
_HK_SETUP_LOG = {}
_HK_SIDE_KEY_SET = frozenset(_HK_EXPECTED_SIDE_KEYS)

_PACKED_ELEMENTS = _HK_ARENA_ELEMENTS
_PB_SINGLE = tuple(_HK_OLD_PREBIND)
for _hk_kind, _hk_cap in enumerate(_HK_OLD_SIDE_CAPS):
    _CAP_BY_ID[id((_IPT_A, _IPT_B, _IPT_C)[_hk_kind])] = _hk_cap
# Parent bound-side count includes HK sides; the84 product descriptors are
# cleared explicitly and are not mislabeled as side bindings.
_PB_CACHES = tuple(_PB_CACHES) + ('_HK_SIDES',)


def _hk_drop():
    _HK_READY[0] = False
    _HK_BINDING[0] = False
    _HK_PRODUCTS.clear()
    _HK_SIDES.clear()
    _HK_ARENA[0] = None


_HK_PARENT_RESET = _pb_reset


def _pb_reset(saved):
    # Drop descriptor references before the parent discards/replaces its arena.
    _hk_drop()
    return _HK_PARENT_RESET(saved)


def _hk_side(arena, start, kind, P, L, rows, cols):
    key = (kind, L, P, rows, cols, start)
    if (not _HK_BINDING[0] or arena is not _HK_ARENA[0]
            or _SCRATCH is None or arena is not _SCRATCH[0]
            or key not in _HK_SIDE_KEY_SET):
        raise RuntimeError('HK side binding outside fixed arena generation')
    found = _HK_SIDES.get(key)
    if found is not None:
        return found
    if len(_HK_SIDES) >= _HK_POOL_COUNT:
        raise RuntimeError('HK static side cap')
    found = Side(arena, start, kind, P, L, rows, cols)
    _HK_SIDES[key] = found
    return found


def _hk_get(key):
    # Runtime lookup is metadata only. Unknown shapes and lost setup arenas
    # retain the original route; no lazy allocation expands the fixed pool.
    if (not _HK_READY[0] or _SCRATCH is None
            or _HK_ARENA[0] is not _SCRATCH[0]):
        return None
    return _HK_PRODUCTS.get(tuple(key))


_HK_PARENT_PB_BIND = _pb_bind


def _pb_bind(t0):
    _HK_READY[0] = False
    _HK_PARENT_PB_BIND(t0)
    if _SCRATCH is None or int(_SCRATCH[0].shape[0]) != _HK_ARENA_ELEMENTS:
        raise RuntimeError('HK original-capacity parent arena missing')
    _hk_drop()
    _HK_ARENA[0] = _SCRATCH[0]
    _HK_GENERATION[0] += 1
    _HK_BINDING[0] = True
    descriptor_flops = 0
    try:
        for key, name, depth, shape in _HK_FIXED_DESCRIPTORS:
            if _pb_time.monotonic() - t0 > _PB_WALL_S:
                raise RuntimeError('HK setup incomplete at unchanged wall guard')
            if key in _HK_PRODUCTS or len(_HK_PRODUCTS) >= 84:
                raise RuntimeError('HK static descriptor cap')
            item = BorrowedProduct(_HK_RECORDS[name], depth, shape, _SCRATCH[0], _hk_side)
            expected = _HK_DESCRIPTOR_EXPECTATIONS[key]
            if (int(item.required_arena_elements) != expected['required_arena_elements']
                    or int(item.descriptor_bind_flops) != expected['descriptor_bind_flops']
                    or int(item.fresh_padding_bytes) != expected['fresh_padding_bytes']
                    or int(item.batch_size) != expected['batch_size']
                    or int(item.batch_count) != expected['batch_count']
                    or int(item.batch_view_count) != expected['batch_view_count']
                    or int(item.batch_view_bind_flops) != 0
                    or int(item.required_arena_elements) + int(item.fresh_padding_bytes)//4 != expected['complete_workspace_elements']):
                raise RuntimeError('HK descriptor accounting mismatch')
            descriptor_flops += int(item.descriptor_bind_flops)
            _HK_PRODUCTS[key] = item
        old_bound = 0; old_fused = 0
        for kind, L, rows, cols in _PB_SINGLE:
            entry = dict.get((_IPT_A, _IPT_B, _IPT_C)[kind], (L, rows, cols))
            old_bound += int(entry is not None)
            old_fused += int(entry is not None and entry[3] is not None)
        elapsed = _pb_time.monotonic() - t0
        _HK_SETUP_LOG.update(dict(completed=False, generation=_HK_GENERATION[0],
            descriptors=len(_HK_PRODUCTS), HK_sides=len(_HK_SIDES),
            old_prebound=old_bound, old_fused=old_fused, KS_bound=len(_KS_ACC),
            side_bind_flops=_HK_SIDE_BIND_FLOPS,
            descriptor_bind_flops=descriptor_flops,
            descriptor_bind_views=dict(_HK_DESCRIPTOR_BIND_VIEWS),
            arena_elements=int(_SCRATCH[0].shape[0]), elapsed_s=float(elapsed),
            old_side_caps=list(_HK_OLD_SIDE_CAPS), HK_reserved=list(_HK_SIDE_COUNTS),
            batch_counts=dict(_HK_BATCH_COUNTS),max_shared_workspace_elements=_HK_MAX_REQUIRED_ELEMENTS,
            max_complete_workspace_elements=_HK_MAX_COMPLETE_ELEMENTS))
        if (len(_HK_PRODUCTS) != 84 or set(_HK_SIDES) != _HK_SIDE_KEY_SET
                or old_bound != 430 or old_fused != 430 or len(_KS_ACC) != 24
                or descriptor_flops != _HK_DESCRIPTOR_BIND_FLOPS
                or elapsed > _PB_WALL_S):
            raise RuntimeError('HK setup incomplete or wrong fixed binding totals')
        _HK_READY[0] = True
        _HK_SETUP_LOG['completed'] = True
        _HK_SETUP_LOG['side_bindings'] = _HK_SIDE_BINDING_COUNT
        _HK_SETUP_LOG['total_side_bindings_with_KS'] = _HK_TOTAL_BINDING_COUNT
        _HK_SETUP_LOG['fused_side_bindings'] = _HK_SIDE_BINDING_COUNT
        _PB_LOG.append(('hk_blocked', 84, _HK_POOL_COUNT, 430, round(elapsed, 3)))
    except Exception as exc:
        _HK_SETUP_LOG.update(dict(completed=False, error=type(exc).__name__+': '+str(exc),
            descriptors=len(_HK_PRODUCTS), HK_sides=len(_HK_SIDES),
            descriptor_bind_flops=descriptor_flops,
            elapsed_s=float(_pb_time.monotonic()-t0)))
        _hk_drop()
        raise
    finally:
        _HK_BINDING[0] = False
        _BH_HOLD.clear()


_HK_PARENT_SETUP = Estimator.setup


def _hk_setup(self, context):
    active = int(getattr(context, 'width', 0)) == 1024 and int(getattr(context, 'depth', 0)) == 16
    if active:
        _HK_SETUP_LOG.clear()
    _HK_PARENT_SETUP(self, context)
    if active and not _HK_READY[0]:
        # The parent intentionally catches binding failures; this candidate's
        # fixed prebind qualification must not silently become a lazy route.
        raise RuntimeError('HK fixed setup did not complete: '+str(_HK_SETUP_LOG))


Estimator.setup = _hk_setup
