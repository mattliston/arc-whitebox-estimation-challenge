

# BEGIN SCOUT-NATIVE NB-MERGE v2 (2026-10-12; = v1 at every layer with both parts, the W5 scope no longer excludes layers 1-6; rounding-level, NOT bit-exact): the young push and the newborn-common push share their
# left operand W'[kept, previous] but ran as two tree products; the newborn product's odd widths (576-992 columns) run at a worse
# rate and each product transforms W' again. Here the inherited young columns and the newborn bases go through ONE product on
# [p | base[previous]], split after (views). Exact product bills through _mm (scout_native/price_nbmerge_v1.py, S50 twin shapes on
# 7100136): -0.160% of the bill over layers 2-15, minus the concatenation (about 0.03%); about -800 charged ops a network. Only when
# the W5 port is inactive (the dead cut or the young crop is on: layers 4-15 in practice) and both parts are non-empty; otherwise the
# parent runs unchanged. Every downstream value keeps its meaning; only the product's tree partition (rounding) changes.
_NBM_ON = True
_NBM_STATS = dict(merged=0, parent=0)
_NBM_PARENT_PUSH = _YNT_PARENT_PUSH


def _nbm_push(w, p, previous, newborn):
    if not _NBM_ON or newborn is None:
        _NBM_STATS['parent'] += 1
        return _NBM_PARENT_PUSH(w, p, previous, newborn)
    old_columns, selected, base, t, s = newborn
    q, births = int(old_columns), int(selected.shape[0])
    if q <= 0 or births <= 0 or int(p.shape[1]) != q:
        _NBM_STATS['parent'] += 1
        return _NBM_PARENT_PUSH(w, p, previous, newborn)
    _e2_H = _E2Hold()
    wp = _e2_H(_xsg_rows(5154, w, previous)).T
    both = _mm(wp, _e2_H(fnp.concatenate([p, _e2_H(_xsg_rows(5159, base, previous))], axis=1)))
    inherited = both[:, :q]
    common = both[:, q:]
    directions = _e2_H(_xsg_rows(5161, w, selected)).T
    _NBM_STATS['merged'] += 1
    return fnp.concatenate([inherited, _e2_H(common + _e2_H(directions * t)), _e2_H(common + _e2_H(directions * s))], axis=1)


_YNT_PARENT_PUSH = _nbm_push
# END SCOUT-NATIVE NB-MERGE v2
