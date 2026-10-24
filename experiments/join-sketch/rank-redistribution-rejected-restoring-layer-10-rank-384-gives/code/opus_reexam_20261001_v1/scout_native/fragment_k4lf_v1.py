
# BEGIN SCOUT-NATIVE K4L AGES 3-4 AT THE FINAL LAYER (fragment_k4lf_v1, 2026-10-13, for scout-card's K13 arm k4lf34h = O/scout_card/make_k3_v7.py:
# pl k4la3 = k4la4 = -1 on layers 0-14 and c - 1 at layer 15, i.e. K4L's terms of the cohorts aged 3 and 4 at strength c in the final layer's
# k4 only; the pair map, the births and layers 1-14 are untouched). Native: K4L v7's _k4l_local(l, ..) sums the cohorts born at most
# _K4L_AGE layers before l, each times _K4L_C[l - j - 1], before the level is projected out (_k4l_dk); the final layer's call is the row-cut
# push's (l = 15). This wrapper, at l = 15 only, extends _K4L_C by _K4LF (the strengths of ages 3, 4) and _K4L_AGE to match; _K4L_B keeps every
# birth of the prediction, and cohorts 11-12 are young at layer 15 (joins 7 / 10 / 13 take cohorts 0-8). _K4LF = () = the parent bit for bit.
_K4LF = (0.5, 0.5)
_K4LF_PARENT = _k4l_local


def _k4l_local(l, pp, pp2, pp3, state):
    global _K4L_AGE, _K4L_C
    if l != 15 or not _K4LF or not isinstance(_K4L_C, tuple) or len(_K4L_C) != _K4L_AGE:
        return _K4LF_PARENT(l, pp, pp2, pp3, state)
    saved = (_K4L_AGE, _K4L_C)
    _K4L_C = tuple(_K4L_C) + tuple(float(x) for x in _K4LF)
    _K4L_AGE = len(_K4L_C)
    try:
        return _K4LF_PARENT(l, pp, pp2, pp3, state)
    finally:
        _K4L_AGE, _K4L_C = saved
# END SCOUT-NATIVE K4L AGES 3-4 AT THE FINAL LAYER
