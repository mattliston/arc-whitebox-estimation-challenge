
# BEGIN OPUS-ENGINEER a62 READ COLUMN PREFIX (fragment_readcol_v1, 2026-10-11; a physics change): the bank read's plane is computed on
# the live roster's leading columns only, a >= -_RC_COLZ (the roster is sorted by gate, most active first: a prefix slice; count
# rounded up to 32 as select() rounds), and zero on the rest of the live columns, whose gate is below Phi(-_RC_COLZ). Rows, births,
# pushes, transport, the diagonal and the omitted rows are unchanged; the pair law's forms go back on the read's block only. Appended
# after the release fragments: it wraps the read and the forms (no loop edit); the loop's put writes the padded plane as before.
# Same arithmetic as make_rd_v1.py's loop-edit build cand_s47e_rd_c20.py plus a zero padding of the cut columns (one copy of
# response rows x cut columns a layer). Native, 16 fresh networks against the fired S47e: final-layer MSE +0.38% (se 0.40), bill -0.88%.
import math as _rc_math
_RC_COLZ = 2.0
_RC_STATE = [None]
_RC_STATS = dict(layers=0, cols_cut=0)
_RC_PARENT_READ = _cp_read_young_sq
_RC_PARENT_FORMS = _np_forms


def _cp_read_young_sq(left, young_right, lam, live_indices, state):
    _RC_STATE[0] = None
    L = int(live_indices.shape[0])
    if _RC_COLZ is None:
        return _RC_PARENT_READ(left, young_right, lam, live_indices, state)
    gate = _E2_LAYER['g']
    floor = 0.5 * _rc_math.erfc(_RC_COLZ / _rc_math.sqrt(2.0))          # Phi(-z)
    c = min(L, (int(fnp.nonzero(gate >= floor)[0].shape[0]) + 31) // 32 * 32)
    if c >= L:
        return _RC_PARENT_READ(left, young_right, lam, live_indices, state)
    plane = _RC_PARENT_READ(left, young_right[:c], lam, live_indices[:c], state)
    _RC_STATE[0] = (c, live_indices)
    _RC_STATS.update(layers=_RC_STATS['layers'] + 1, cols_cut=_RC_STATS['cols_cut'] + L - c)
    return fnp.concatenate([plane, fnp.zeros((int(plane.shape[0]), L - c), dtype=fnp.float32)], axis=1)


def _np_forms(l, plane, cov32):
    found, _RC_STATE[0] = _RC_STATE[0], None
    S = _NP_STATE
    if found is not None and S['fm'] is not None:
        c, live_indices = found
        keep = fnp.ones(int(plane.shape[0]), dtype=fnp.float32)
        fnp.put(keep, live_indices[c:], 0.0)
        S['aa'] = S['aa'] * keep
        if S['Rb'] is not None:
            S['Rb'] = S['Rb'] * keep
    return _RC_PARENT_FORMS(l, plane, cov32)
# END OPUS-ENGINEER a62 READ COLUMN PREFIX
