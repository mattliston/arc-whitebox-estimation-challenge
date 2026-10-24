# Pair routing repair, appended after candidate_bini_pair_combined_e8_v2.py.
# _pushsm_mm receives ONLY the inherited young bank. The newest signed cohort
# is in state['cohorts'] but its two legs are assembled from common/directions,
# so its width MUST NOT be included in the inherited-product permutation.
_BIN_PAIR_ROUTING_VERSION = 1
_BIN_STATS['pair_rosters'] = []
_BIN_ROUTE_PARENT_SHARED_PUSH = _shared_old384_push

def _shared_old384_push(*args, **kwargs):
    prior = _BIN_STATE[0]
    _BIN_STATE[0] = kwargs['state'] if 'state' in kwargs else args[6]
    try:
        return _BIN_ROUTE_PARENT_SHARED_PUSH(*args, **kwargs)
    finally:
        _BIN_STATE[0] = prior


def _bin_inherited_cohort_widths(state, n):
    if state is None:
        return None
    # A join updates columns before calling the young push. Its input p has
    # already been sliced at that updated absolute old-bank boundary.
    old = int(state['columns'])
    stop = old + int(n)
    position = old
    widths = []
    for start, width in state['cohorts']:
        start, width = int(start), int(width)
        if start + width <= old:
            continue
        if start >= stop:
            break
        if start != position or width <= 0 or width % 2 or start + width > stop:
            return None
        widths.append(width)
        position += width
    return tuple(widths) if widths and position == stop else None


def _pushsm_mm(a, b):
    key = (int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))
    plan = _BINI_PUSH_PLANS.get(key)
    if plan is None:
        return _BIN_PARENT_PUSH(a, b)
    _BIN_STATS['push'] += 1
    _BIN_STATS['push_shapes'].append(key)
    widths = None
    if plan[0] == '223':
        state = _BIN_STATE[0]
        widths = _bin_inherited_cohort_widths(state, key[2])
        _BIN_STATS['pair_rosters'].append(dict(
            shape=key,
            old=None if state is None else int(state['columns']),
            cohorts=None if state is None else tuple((int(s), int(w)) for s,w in state['cohorts']),
            inherited_width=key[2],
            packed_widths=widths))
        if widths is None:
            _BIN_STATS['pair_fallback'] += 1
        else:
            _BIN_STATS['pair_layout'] += 1
    return _bin_product(a, b, plan, widths)
