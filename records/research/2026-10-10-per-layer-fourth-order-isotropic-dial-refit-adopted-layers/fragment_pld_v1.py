# BEGIN OPUS-ENGINEER PER-LAYER DIALS (a52 pld v1, 2026-10-10; values only: with _PLD empty the file is bit-identical). The trainer's late-band
# dials (jmI: layers 10-14 apart from 1-9): a value per layer for a global literal, picked in Python where the literal is used:
#   'bscale'  the carried b's scale (_S19_BSCALE) in the loop at layer l
#   'X1'      the birth-leg dial (_BD_J1_PHYS) in the birth update of layer l (the layer the loop publishes in _PHYS_LAYER)
#   'iso'     _GN_KNOBS['iso'] in the GN covariance correction of layer l
#   'k4la1'   the age-1 K4L strength at the layer the term is applied: the cohort born at l is rescaled at its birth when l + 1 has a value
# No new array operation except k4la1's one rescale per affected birth (4 x pairs elements).
_PLD = {}
_PLD_KNOBS = {}


def _pld(name, l, default):
    d = _PLD.get(name)
    if not d or l not in d:
        return default
    return d[l]


def _pld_knobs(l):
    d = _PLD.get('iso')
    if not d or l not in d:
        return _GN_KNOBS
    got = _PLD_KNOBS.get(l)
    if got is None or got[0] is not _GN_KNOBS:
        got = _PLD_KNOBS[l] = (_GN_KNOBS, dict(_GN_KNOBS, iso=d[l]))
    return got[1]


_PLD_PARENT_K4L_BIRTH = _k4l_birth


def _k4l_birth(l, aux, host, selected, t, s):
    out = _PLD_PARENT_K4L_BIRTH(l, aux, host, selected, t, s)
    d = _PLD.get('k4la1')
    if d and (l + 1) in d and isinstance(_K4L_C, tuple) and l in _K4L_E:
        factor = float(d[l + 1]) / float(_K4L_C[0])
        if factor != 1.0:
            parts = list(_K4L_E[l])
            parts[0] = parts[0] * factor
            _K4L_E[l] = parts
    return out
# END OPUS-ENGINEER PER-LAYER DIALS
