

# BEGIN SCOUT-NATIVE K4L AGES 3-4 AT INNER LAYERS (fragment_k4li_v1, 2026-10-13, lead 09:5xZ, the physicist's T139 arms on top of k4lf34h:
# pl k4la3 = k4la4 = -0.5, i.e. strength 0.5, at the inner layers in _K4LI_LAYERS; the final layer stays with fragment_k4lf_v1, l = 15).
# Native: K4L v7's _k4l_local(l, ..) is called once a layer by the main loop (the young pairs' hub terms of layer l's per-unit fourth
# cumulant) and once by the final row-cut push (l = 15). It sums the cohorts born at most _K4L_AGE layers before l, each times
# _K4L_C[l - j - 1]. At l in _K4LI_LAYERS only, this wrapper extends _K4L_C by _K4LI (the strengths of ages 3 and 4) and _K4L_AGE to match,
# as fragment_k4lf_v1 does at l = 15. The per-layer dials wrapper inside replaces only the first strength (k4la1) and keeps the rest; a
# joined cohort has width 0 and adds nothing (joins 7 / 10 / 13 take births <= l - 5, so ages 3-4 are young at every inner layer).
# _K4LI_LAYERS = () gives the parent bit for bit.
_K4LI = (0.5, 0.5)
_K4LI_LAYERS = (8, 9, 10, 11, 12, 13, 14)
_K4LI_PARENT = _k4l_local


def _k4l_local(l, pp, pp2, pp3, state):
    global _K4L_AGE, _K4L_C
    if l not in _K4LI_LAYERS or not _K4LI or not isinstance(_K4L_C, tuple) or len(_K4L_C) != _K4L_AGE:
        return _K4LI_PARENT(l, pp, pp2, pp3, state)
    saved = (_K4L_AGE, _K4L_C)
    _K4L_C = tuple(_K4L_C) + tuple(float(x) for x in _K4LI)
    _K4L_AGE = len(_K4L_C)
    try:
        return _K4LI_PARENT(l, pp, pp2, pp3, state)
    finally:
        _K4L_AGE, _K4L_C = saved
# END SCOUT-NATIVE K4L AGES 3-4 AT INNER LAYERS
