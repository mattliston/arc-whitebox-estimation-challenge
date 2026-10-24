# BEGIN OPUS-ENGINEER a62 K4L AGE 2 BY LAYER v1 (file written 2026-10-12 07:33Z; trainer 07:4xZ: S52's arms fk4la2_1_7 / fk4la2_8_15; until now age 2
# had one literal, _K4L_C[1]). Torch (trainer/otr_m_v5.py line 432): loc += (K4L_C + eps['k4la%d' % (l - j)][l]) x part, the cohort born at j
# read at layer l: the age-2 strength belongs to the layer the term is APPLIED (as fragment_pld_v3's 'k4la1' for age 1). Native, both diets:
#   K4L v7 (the twin; no _K4L_E): the strength is _K4L_C[age - 1] inside _k4l_local at layer l: _K4L_C's age-2 entry is set to
#     _K4LA2_BY_LAYER[l] around that call (outside fragment_pld_v3's age-1 wrapper, which keeps the age-2 entry it finds);
#   K4L v6 (the safe file; _K4L_E): the age tables are scaled by _K4L_C at the cohort's birth: the cohort born at j has its age-2 table
#     rescaled by _K4LA2_BY_LAYER[j + 2] / _K4L_C[1] right after its birth (after fragment_pld_v3's age-1 rescale).
# _K4LA2_BY_LAYER = {} = the parent bit for bit.
assert '_k4l_birth' in globals() and '_k4l_local' in globals(), 'fragment_k4la2_v1 needs the K4L lever'
_K4LA2_BY_LAYER = {}
_K4LA2_PARENT_BIRTH = _k4l_birth


def _k4l_birth(l, aux, host, selected, t, s):
    out = _K4LA2_PARENT_BIRTH(l, aux, host, selected, t, s)
    E = globals().get('_K4L_E')
    if E is not None and (l + 2) in _K4LA2_BY_LAYER and isinstance(_K4L_C, tuple) and len(_K4L_C) >= 2 and l in E and len(E[l]) >= 2:
        factor = float(_K4LA2_BY_LAYER[l + 2]) / float(_K4L_C[1])
        if factor != 1.0:
            parts = list(E[l])
            parts[1] = parts[1] * factor
            E[l] = parts
    return out


if '_K4L_E' not in globals():   # K4L v7
    _K4LA2_PARENT_LOCAL = _k4l_local

    def _k4l_local(l, *args, **kwargs):
        global _K4L_C
        v = _K4LA2_BY_LAYER.get(l)
        if v is None or not isinstance(_K4L_C, tuple) or len(_K4L_C) < 2:
            return _K4LA2_PARENT_LOCAL(l, *args, **kwargs)
        saved = _K4L_C
        _K4L_C = (saved[0], float(v)) + tuple(saved[2:])
        try:
            return _K4LA2_PARENT_LOCAL(l, *args, **kwargs)
        finally:
            _K4L_C = saved
# END OPUS-ENGINEER a62 K4L AGE 2 BY LAYER v1
