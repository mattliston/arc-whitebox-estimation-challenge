"""Fixed active congruence; injected fnp, _mm and resident helper only."""


def _active_congruence512(wa, ca):
    cs = (ca + ca.T) * 0.5
    middle = _mm(cs, wa)
    return _resident_gram512(wa.T, middle)
