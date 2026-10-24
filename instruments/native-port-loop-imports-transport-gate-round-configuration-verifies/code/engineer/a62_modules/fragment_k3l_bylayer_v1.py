# BEGIN OPUS-ENGINEER a62 K3L BY LAYER (2026-10-11 21:3xZ, ready for RAD-S48's 'k31l age 2 x 3 bands' arm): the young pairs' fourth-chaos
# plane strengths by cohort age may differ by READ layer (the torch k31l instances read their leaves nkl1 / nkl2 / nkl3 at the layer
# that reads, otr_m_v5.py NCAR_BASE / LEAVES['k31l']). _NP_K3L_BY_LAYER = {l: (age 1, age 2, age 3)} replaces _np_ages() inside
# _np_left(l, ...) only (the read: strength = ages[l - b - 1] per cohort b); the law's tables, its key and the K4L coefficient roster
# keep _NP_K3L (they use only whether ages exist and how many), so a per-layer tuple must keep _NP_K3L's age count (a 0 inside is
# allowed: that cohort is skipped at that layer). _NP_K3L_BY_LAYER = {} = the parent bit for bit.
_NP_K3L_BY_LAYER = {}
_K3LB_LAYER = [None]
_K3LB_PARENT_AGES = _np_ages
_K3LB_PARENT_LEFT = _np_left


def _np_ages():
    ages = _K3LB_PARENT_AGES()
    l = _K3LB_LAYER[0]
    if l is None or l not in _NP_K3L_BY_LAYER or not ages:
        return ages
    got = tuple(float(x) for x in _NP_K3L_BY_LAYER[l])
    if len(got) != len(ages):
        raise RuntimeError('K3L by layer: layer %d has %d ages, _NP_K3L %d' % (l, len(got), len(ages)))
    return got


def _np_left(l, *args, **kwargs):
    _K3LB_LAYER[0] = l
    try:
        return _K3LB_PARENT_LEFT(l, *args, **kwargs)
    finally:
        _K3LB_LAYER[0] = None
# END OPUS-ENGINEER a62 K3L BY LAYER
