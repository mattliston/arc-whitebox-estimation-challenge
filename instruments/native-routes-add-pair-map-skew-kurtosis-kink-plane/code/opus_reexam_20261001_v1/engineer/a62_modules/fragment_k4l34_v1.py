# BEGIN OPUS-ENGINEER2 a62 K4L AGES 3-4 BY LAYER v1 (file written 2026-10-13T10:45:52Z; lead 10:5xZ: from S56 the RAD card carries K4L ages 3-4,
# trainer otr_m_v7: K4L_AGE 2 -> 4 and the per-layer families k4la3 / k4la4, age strength = 1 + shift, -1 = off, read at the layer the term is
# APPLIED: the cohort born at j enters layer l's per-unit fourth cumulant at age l - j). _K4L34_BY_LAYER = {layer l: (strength of age 3,
# strength of age 4)} at the layers where either is on (layers 8-15 at 0.5 for the absorbed k4lf34h + K4L-inner); the final layer is l = 15,
# the row-cut push's call. Native, both diets:
#   K4L v7 (the twin; strength applied in _k4l_local): around the call at a listed layer _K4L_C gets the two strengths appended and _K4L_AGE
#     becomes 4 (scout-native's fragment_k4lf_v1 / fragment_k4li_v1 mechanics, here by table); the per-layer dials' age-1 wrapper and the
#     age-2-by-layer wrapper inside replace their own entry and keep the rest;
#   K4L v6 (the safe; strengths folded into the age tables at birth): around the call at a listed layer _K4L_AGE becomes 4 and the two
#     cohorts read at ages 3 and 4 (born l - 3, l - 4) get their age tables = the unscaled birth table (_K4L_RAW) x that layer's strength,
#     in copies of their table lists that are put back after the call (each (birth, age) is read at exactly one layer).
# A layer whose K4L state is not the plain two ages (_K4L_AGE != 2: ages 3-4 already applied by another fragment, e.g. fragment_k4lf_v1 /
# fragment_k4li_v1 in a stack built on this file) raises instead of applying them twice. _K4L34_BY_LAYER = {} = the parent bit for bit.
assert '_k4l_local' in globals(), 'fragment_k4l34_v1 needs the K4L lever'
_K4L34_BY_LAYER = {}
_K4L34_PARENT_LOCAL = _k4l_local


def _k4l_local(l, *args, **kwargs):
    global _K4L_AGE, _K4L_C
    v = _K4L34_BY_LAYER.get(l)
    if v is None:
        return _K4L34_PARENT_LOCAL(l, *args, **kwargs)
    if _K4L_AGE != 2 or not isinstance(_K4L_C, tuple) or len(_K4L_C) != 2:
        raise RuntimeError('fragment_k4l34_v1: K4L ages 3-4 at layer %r on a K4L state that is not the plain two ages (_K4L_AGE %r)' % (l, _K4L_AGE))
    s3, s4 = float(v[0]), float(v[1])
    E = globals().get('_K4L_E')
    if E is None:   # K4L v7
        saved = (_K4L_AGE, _K4L_C)
        _K4L_C = tuple(_K4L_C) + (s3, s4)
        _K4L_AGE = 4
        try:
            return _K4L34_PARENT_LOCAL(l, *args, **kwargs)
        finally:
            _K4L_AGE, _K4L_C = saved
    saved_lists = {}   # K4L v6
    for j, age, s in ((l - 3, 3, s3), (l - 4, 4, s4)):
        if j in E and j in _K4L_RAW:
            saved_lists[j] = E[j]
            parts = list(E[j][:2])
            raw = _K4L_RAW[j]
            if age == 4:
                parts.append(raw)   # the age-3 slot of this cohort: read at layer j + 3, already past
            parts.append(raw * s)
            E[j] = parts
    saved_age = _K4L_AGE
    _K4L_AGE = 4
    try:
        return _K4L34_PARENT_LOCAL(l, *args, **kwargs)
    finally:
        _K4L_AGE = saved_age
        E.update(saved_lists)
# END OPUS-ENGINEER2 a62 K4L AGES 3-4 BY LAYER v1
