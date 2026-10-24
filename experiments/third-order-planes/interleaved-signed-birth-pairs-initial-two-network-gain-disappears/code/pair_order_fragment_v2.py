# Research only: place equal-and-opposite atom partners next to each other
# inside old-tier dot products. The complete contraction is unchanged.
# This is ordinary index permutation, with every arithmetic operation metered.
_BPO_STATS = {'reads': 0, 'factor_gathers': 0}
_BPO_OLD_FINISH = _slr_finish_factors


def _slr_finish_factors(state, factors):
    _BPO_OLD_FINISH(state, factors)
    state['balanced_read_factors'] = None


def _bpo_factor(F, state, lo, hi):
    order = []
    for start, width in state['cohorts']:
        start, width = int(start), int(width)
        if width and start >= lo and start + width <= hi:
            _slr_require(width % 2 == 0, 'balanced read preserves complete pairs')
            half = width // 2
            for j in range(half):
                order.extend((start - lo + j, start - lo + half + j))
    _slr_require(len(order) == hi - lo and len(set(order)) == hi - lo,
                 'balanced read is a permutation of the complete roster')
    perm = fnp.asarray(tuple(order), dtype=fnp.int64)
    _BPO_STATS['factor_gathers'] += 1
    return perm, F[:, perm]


def _bpo_factors(state):
    cache = state.get('balanced_read_factors')
    if cache is None:
        group = state['lazy']
        cols = int(state['columns'])
        if group is None:
            cache = (_bpo_factor(state['F'], state, 0, cols),)
        else:
            split = int(group['prefix'])
            cache = (_bpo_factor(group['base'], state, 0, split),
                     _bpo_factor(group['join'], state, split, cols))
        state['balanced_read_factors'] = cache
    return cache


def _bpo_product(left_squared, coefficients, item):
    perm, factors = item
    weighted_factors = factors * coefficients[perm][None, :]
    return _shared_old384_mm(left_squared[:, perm], weighted_factors.T)


def _bpo_inner_sq(old_left, lam_old, state):
    columns = int(state['columns'])
    _slr_layout(state, columns)
    _slr_require(len(old_left.shape) == 2 and int(old_left.shape[1]) == columns
                 and tuple(lam_old.shape) == (columns,)
                 and str(old_left.dtype) == str(lam_old.dtype) == 'float32',
                 'complete original weighted repeated-index read')
    _BPO_STATS['reads'] += 1
    items = _bpo_factors(state)
    group = state['lazy']
    if group is None:
        return _bpo_product(old_left, lam_old, items[0])
    split = int(group['prefix'])
    first = _bpo_product(old_left[:, :split], lam_old[:split], items[0])
    second = _bpo_product(old_left[:, split:], lam_old[split:], items[1])
    mapped = _shared_old384_mm(first, group['map'].T)
    _shared_old384_finite(mapped, 'balanced lazy response inner')
    return mapped + second


# Keep the L15-only u/v approximation of the frozen candidate. This replaces
# its original-reader alias used at layers 7..14; factors and joins are intact.
_UVP_ORIG_READ = _bpo_inner_sq
