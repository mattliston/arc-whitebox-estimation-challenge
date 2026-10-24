# BEGIN OPUS-ENGINEER LIVE ROWS BY SLICE (dcr v1, 2026-10-01; bit-identical: the same rows in the same order; needs the builder's one
# loop edit, right = _dcr_right(...)): at layers where the dead-side row cut is active the push is computed on the kept rows, in
# descending order of mu/sigma, and then scattered back into an all-unit plane, from which the loop scatters the live rows out again
# (put_along_axis, one flop per element of the young columns of the full plane). The live roster is the leading part of the kept
# roster whenever both sorts agree, so the live rows are the leading rows of the pushed array itself: a slice. The rosters are
# compared element by element first; any difference (a tie in the gate that the mu/sigma sort breaks differently) keeps the scatter.
_DCR_ON = True
_DCR_OPEN = [False]
_DCR_CALLS = []
_DCR_STASH = [None]                          # (pushed kept rows, kept roster, the expanded plane the loop holds), this layer only
_DCR_STATS = {'slice': 0, 'scatter': 0}
_DCR_OLD_EXPAND = _exact_expand_rows
_DCR_OLD_DEADCUT = _deadcut_push


def _exact_expand_rows(values, rows, total_rows):
    answer = _DCR_OLD_EXPAND(values, rows, total_rows)
    if _DCR_OPEN[0]:
        _DCR_CALLS.append((values, rows, answer))
    return answer


def _deadcut_push(layer, w32, p, previous, newborn, lam, state, a):
    _DCR_STASH[0] = None
    _DCR_OPEN[0] = _DCR_ON
    try:
        pp = _DCR_OLD_DEADCUT(layer, w32, p, previous, newborn, lam, state, a)
        for values, rows, answer in _DCR_CALLS:
            if answer is pp:
                _DCR_STASH[0] = (values, rows, pp)
        return pp
    finally:
        _DCR_OPEN[0] = False
        _DCR_CALLS.clear()


def _dcr_right(pp, old_columns, live_indices):
    # = _unique_rows_via_scatter(pp[:, old_columns:], live_indices)
    stash, _DCR_STASH[0] = _DCR_STASH[0], None
    if stash is not None and stash[2] is pp:
        values, rows, _plane = stash
        k = int(live_indices.shape[0])
        if k <= int(rows.shape[0]) and bool(fnp.all(rows[:k] == live_indices)):
            _DCR_STATS['slice'] += 1
            return values[:k, old_columns:]
    _DCR_STATS['scatter'] += 1
    return _unique_rows_via_scatter(pp[:, old_columns:], live_indices)
# END OPUS-ENGINEER LIVE ROWS BY SLICE
