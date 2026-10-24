# BEGIN OPUS-ENGINEER a62 READ SCALES rso / rsy v1 (file written 2026-10-12 04:22Z; for a trainer point carrying the 'rso' / 'rsy' families, S51's
# likeliest band arm: rsy at layers 11-14 passed S48's band rule, t -3.01, halves -2.4 / -1.8).
# Torch (trainer/otr_m_v5.py read_scaled): the innermost wrapper of the bank read scales the column weights it reads, lam -> lam x (1 + rso_l)
# on the old tier's columns and x (1 + rsy_l) on the young ones, at the read ONLY (the carried weights, the plane's diagonal and every
# other consumer of lam keep theirs). Native: the outermost wrapper of the read (_cp_read_young_sq, called by name at the response plane;
# its parents, the read column cut and the original read, run unchanged on the scaled weights), the layer taken from _np_left, which the
# step calls right before the read. _RS = {layer: (1 + rso, 1 + rsy)}; layers not in _RS read exactly as before (the certificate: _RS {}).
_RS = {}
_RS_LAYER = [None]
_RS_STATS = dict(scaled=0)
_RS_PARENT_LEFT = _np_left
_RS_PARENT_READ = _cp_read_young_sq


def _np_left(l, *args, **kwargs):
    _RS_LAYER[0] = l
    return _RS_PARENT_LEFT(l, *args, **kwargs)


def _cp_read_young_sq(left, young_right, lam, live_indices, state):
    s = _RS.get(_RS_LAYER[0])
    if s is None:
        return _RS_PARENT_READ(left, young_right, lam, live_indices, state)
    old = int(state['columns'])
    parts = ([lam[:old] * float(s[0])] if old else []) + ([lam[old:] * float(s[1])] if int(lam.shape[0]) > old else [])
    _RS_STATS['scaled'] += 1
    return _RS_PARENT_READ(left, young_right, fnp.concatenate(parts) if len(parts) > 1 else parts[0], live_indices, state)
# END OPUS-ENGINEER a62 READ SCALES rso / rsy v1
