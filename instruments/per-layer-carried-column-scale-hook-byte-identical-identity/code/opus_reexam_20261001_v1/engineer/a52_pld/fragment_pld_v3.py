# BEGIN OPUS-ENGINEER PER-LAYER DIALS (a52 pld v3, 2026-10-11; v3 = v2 + 'scarry', below; v2 = v1 + K4L v7: there the strength by age is read in _k4l_local, so the
# age-1 value of layer l is set around that call; v1 assumed v6's _K4L_E and stopped with a NameError in a K4L v7 file (release, 22:04Z).
# Values only: with _PLD empty the file is bit-identical). The trainer's late-band
# dials (jmI: layers 10-14 apart from 1-9): a value per layer for a global literal, picked in Python where the literal is used:
#   'bscale'  the carried b's scale (_S19_BSCALE) in the loop at layer l
#   'X1'      the birth-leg dial (_BD_J1_PHYS) in the birth update of layer l (the layer the loop publishes in _PHYS_LAYER)
#   'iso'     _GN_KNOBS['iso'] in the GN covariance correction of layer l
#   'k4la1'   the age-1 K4L strength at the layer the term is applied: the cohort born at l is rescaled at its birth when l + 1 has a value
#   'scarry'  the carried columns' weight scale s_carry (_S19_LAM[0]) of layer l, as the trainer's card applies it (state_out: the old
#             columns' weights x s_carry(l) at the end of layer l): the loop's carried lam, F4c's step lam2 and the pair law's
#             per-layer constants take layer l's value (builder edits); F4c's birth mix, which divides by the dials a cohort's weight
#             collects (s_new, then s_carry per layer: s_carry ** (age + 1)), takes the product of the values of layers l + 1 .. l + age + 1
#             (the wrapper below, one scale per age, only where that product differs from s_carry ** (age + 1)).
# No new array operation except k4la1's one rescale per affected birth (4 x pairs elements).
_PLD = {}
_PLD_KNOBS = {}
if _PLD.get('scarry') and '_np_tables' in globals():   # 'scarry': the pair law's tables were built at import, before _PLD existed
    _NP_TABLES = _np_tables(1024)


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
    E = globals().get('_K4L_E')   # K4L v6: the age tables are scaled at birth; v7 has none (see below)
    if E is not None and d and (l + 1) in d and isinstance(_K4L_C, tuple) and l in E:
        factor = float(d[l + 1]) / float(_K4L_C[0])
        if factor != 1.0:
            parts = list(E[l])
            parts[0] = parts[0] * factor
            E[l] = parts
    return out


if '_K4L_E' not in globals():   # K4L v7: part * _K4L_C[age - 1] inside _k4l_local at the layer the term is applied
    _PLD_PARENT_K4L_LOCAL = _k4l_local

    def _k4l_local(l, pp, pp2, pp3, state):
        global _K4L_C
        d = _PLD.get('k4la1')
        if not d or l not in d or not isinstance(_K4L_C, tuple):
            return _PLD_PARENT_K4L_LOCAL(l, pp, pp2, pp3, state)
        saved = _K4L_C
        _K4L_C = (float(d[l]),) + tuple(saved[1:])
        try:
            return _PLD_PARENT_K4L_LOCAL(l, pp, pp2, pp3, state)
        finally:
            _K4L_C = saved


_PLD_F4C_MIXB = {}


def _pld_f4c_mixb(l, base):
    # F4c's birth mix for the cohort born at layer l: one scale per age, s_carry ** (age + 1) / (the values of layers l + 1 .. l + age + 1)
    carry = float(_S19_LAM[0])
    ratios = []
    for age in range(_F4C_AGE):
        prod = 1.0
        same = True
        for j in range(l + 1, l + age + 2):
            v = float(_pld('scarry', j, carry))
            same = same and v == carry
            prod = prod * v
        ratios.append(1.0 if same else carry ** (age + 1) / prod)
    if all(r == 1.0 for r in ratios):
        return base
    got = _PLD_F4C_MIXB.get(l)
    if got is None or got[0] is not base:
        scale = fnp.asarray(ratios, dtype=fnp.float64)
        got = _PLD_F4C_MIXB[l] = (base, base * scale[:, None, None, None, None])
    return got[1]


if '_f4c_birth' in globals() and _S19_LAM is not None:
    _PLD_PARENT_F4C_BIRTH = _f4c_birth

    def _f4c_birth(l, *args):
        if not _PLD.get('scarry'):
            return _PLD_PARENT_F4C_BIRTH(l, *args)
        C = _f4c_const()
        saved = C['mixb']
        C['mixb'] = _pld_f4c_mixb(l, saved)
        try:
            return _PLD_PARENT_F4C_BIRTH(l, *args)
        finally:
            C['mixb'] = saved
# END OPUS-ENGINEER PER-LAYER DIALS
