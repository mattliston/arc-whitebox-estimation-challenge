"""DRAFT numerical fragment: bounded full-rank lazy coefficient rotations.

No imports, allocations, model calls or source I/O at module entry. The exact
BOX parent supplies fnp and its existing shared matrix/finite helpers. This
fragment is appended by the inert composer, not imported by a model harness.
"""

_SLR_RANK = 384
_SLR_LAZY_LAYERS = (8, 10, 12, 14)
_SLR_CONSOLIDATE_LAYERS = (9, 11, 13)


def _slr_require(ok, message):
    if not ok:
        raise RuntimeError('Shared lazy rotation: ' + message)


def _slr_matrix(value, rows, columns, name):
    _slr_require(value is not None and tuple(value.shape) == (rows, columns)
                 and str(value.dtype) == 'float32', name + ' FP32 shape')


def _slr_layout(state, columns):
    """Exact ordered factor roster; no array arithmetic or numerical scan."""
    _slr_require(columns > 0 and columns == int(state['columns']),
                 'complete positive old column count')
    group = state['lazy']
    if group is None:
        _slr_matrix(state['F'], _SLR_RANK, columns, 'materialized factors')
        return
    _slr_require(state['F'] is None and group['layer'] in _SLR_LAZY_LAYERS,
                 'one lazy owner, never stale materialized factors')
    old, joined = int(group['prefix']), int(group['joined'])
    _slr_require(old > 0 and joined > 0 and old % 2 == joined % 2 == 0
                 and old + joined == columns, 'two intact ordered factor groups')
    _slr_matrix(group['base'], _SLR_RANK, old, 'unrotated prefix')
    _slr_matrix(group['join'], _SLR_RANK, joined, 'new cohort factors')
    _slr_matrix(group['map'], _SLR_RANK, _SLR_RANK, 'full-rank coordinate map')


def _slr_join_factors(layer, overlap, state, Fj):
    """Return materialized factors, or retain exactly one bounded lazy map.

    The caller owns the unchanged QR, Fj construction and canonical core update.
    Existing factors/maps are independent matrix outputs, never arena views.
    """
    old = int(state['columns'])
    _slr_layout(state, old)
    _slr_matrix(overlap, _SLR_RANK, _SLR_RANK, 'current overlap')
    _slr_require(len(Fj.shape) == 2 and int(Fj.shape[0]) == _SLR_RANK
                 and str(Fj.dtype) == 'float32' and int(Fj.shape[1]) > 0
                 and int(Fj.shape[1]) % 2 == 0, 'intact joining factor pair roster')
    group = state['lazy']
    if layer in _SLR_LAZY_LAYERS:
        _slr_require(group is None, 'no accumulation of lazy maps')
        state['lazy'] = dict(layer=layer, prefix=old, joined=int(Fj.shape[1]),
                             base=state['F'], join=Fj, map=overlap)
        return None
    if layer in _SLR_CONSOLIDATE_LAYERS:
        _slr_require(group is not None and group['layer'] == layer - 1,
                     'immediate next-join consolidation')
        combined = _shared_old384_mm(overlap, group['map'])
        _shared_old384_finite(combined, 'lazy composed coordinate map')
        prefix = _shared_old384_mm(combined, group['base'])
        recent = _shared_old384_mm(overlap, group['join'])
        factors = fnp.concatenate([prefix, recent, Fj], axis=1)
        state['lazy'] = None
        del combined, prefix, recent, group
        return factors
    _slr_require(layer in (6, 7) and group is None,
                 'all other old-factor joins retain their original schedule')
    rotated = _shared_old384_mm(overlap, state['F'])
    factors = fnp.concatenate([rotated, Fj], axis=1)
    del rotated
    return factors


def _slr_finish_factors(state, factors):
    """Preserve full finite checking of the actual stored representation."""
    group = state['lazy']
    if group is None:
        state['F'] = _shared_old384_finite(factors, 'old coefficient factors')
    else:
        _slr_require(factors is None, 'no hidden canonical-factor materialization')
        # Rescan BOTH coefficient blocks, including the retained prefix. This
        # deliberately preserves a full coefficient finite pass at every join.
        _shared_old384_finite(group['base'], 'lazy old coefficient prefix')
        _shared_old384_finite(group['join'], 'lazy joining coefficients')
        _shared_old384_finite(group['map'], 'lazy retained coordinate map')
        _slr_virtual_rotation_envelope(group['map'], group['base'])
        state['F'] = None
    _slr_layout(state, int(state['columns']))


def _slr_virtual_rotation_envelope(overlap, base):
    """Paid conservative overflow guard for the deliberately unformed L@F0.

    At depth d<=6, each input transform and output recombination grows maxabs
    by at most4**d. A leaf has r/2**d terms, so 2**(5*d) times r*max(L)*max(F0)
    bounds every scalar intermediate. Both maxima also individually must be
    <=2**80, so even a huge/tiny pair cannot overflow an input transform.
    The product gate leaves >=18 exponent bits for products/recombination.
    All scalar values remain typed one-element FP64 arrays on native and RPC.
    """
    largest_map = fnp.max(fnp.abs(overlap), keepdims=True).astype(fnp.float64)
    largest_base = fnp.max(fnp.abs(base), keepdims=True).astype(fnp.float64)
    width = fnp.full((1, 1), 384.0, dtype=fnp.float64)
    limit = fnp.full((1, 1), 1208925819614629174706176.0, dtype=fnp.float64)
    bound = (largest_map * largest_base) * width
    allowed = (largest_map <= limit) & (largest_base <= limit) & (bound <= limit)
    _slr_require(bool(fnp.all(allowed)), 'virtual rotation overflow envelope')
    del largest_map, largest_base, width, limit, bound, allowed


def _slr_reconstruct(U, state):
    """Full-row reconstruction; the caller retains its original finite guard."""
    _slr_layout(state, int(state['columns']))
    _slr_require(len(U.shape) == 2 and int(U.shape[0]) > 0
                 and int(U.shape[1]) == _SLR_RANK and str(U.dtype) == 'float32',
                 'current common basis')
    group = state['lazy']
    if group is None:
        return _shared_old384_mm(U, state['F'])
    mapped = _shared_old384_mm(U, group['map'])
    _shared_old384_finite(mapped, 'lazy mapped all-row basis')
    prefix = _shared_old384_mm(mapped, group['base'])
    recent = _shared_old384_mm(U, group['join'])
    old = fnp.concatenate([prefix, recent], axis=1)
    del mapped, prefix, recent
    return old


def _slr_inner(old_left, lam_old, state):
    """One common trailing read remains outside; Lambda is never approximated."""
    columns = int(state['columns'])
    _slr_layout(state, columns)
    _slr_require(len(old_left.shape) == 2 and int(old_left.shape[1]) == columns
                 and tuple(lam_old.shape) == (columns,)
                 and str(old_left.dtype) == str(lam_old.dtype) == 'float32',
                 'complete original weighted repeated-index read')
    group = state['lazy']
    if group is None:
        weighted_factors = state['F'] * lam_old[None, :]
        return _shared_old384_mm(old_left * old_left, weighted_factors.T)
    split = int(group['prefix'])
    left0, left1 = old_left[:, :split], old_left[:, split:]
    weights0 = group['base'] * lam_old[:split][None, :]
    inner0 = _shared_old384_mm(left0 * left0, weights0.T)
    del weights0
    weights1 = group['join'] * lam_old[split:][None, :]
    inner1 = _shared_old384_mm(left1 * left1, weights1.T)
    del weights1
    mapped = _shared_old384_mm(inner0, group['map'].T)
    _shared_old384_finite(mapped, 'lazy mapped response inner')
    inner = mapped + inner1
    del inner0, inner1, mapped
    return inner
