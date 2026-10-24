# BEGIN OPUS-ENGINEER E2 v1 (a29, 2026-10-02): the lead's second-order skew term of the mean formula (O/e16b_k4_closed_loop.py arm e2;
# the trainer's leaf e2 of O/trainer/otr_m.py). The card's mean formula stops at the first order in the third cumulant; the next
# term of the same series is (k3^2 / 72) x the sixth derivative of the Gaussian mean,
#     e2_l x k3^2 (alpha^4 - 6 alpha^2 + 3) pdf / (72 sigma^5),      k3 = the pre-activation third cumulant of the unit,
# added to the mean of layers 1-15 through the slot of the learned mean correction, so that the covariance is recentred for it
# like for any other mean shift. It is NOT scaled by the correction's own scale. An ERROR lever: predictions move by design.
# Ten operations on n-vectors per layer (about 20 kflop).
_E2_STRENGTH = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # one strength per layer; layer 0 is never used


def _e2_skew(l, aux):
    """The term of layer l (n,) float64. aux: the pair map's (layers 1-14) or the final layer's dict."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    a2 = aux['alpha'] * aux['alpha']
    he4 = _e2_H(_e2_H(a2 - 6.0) * a2) + 3.0                                 # alpha^4 - 6 alpha^2 + 3
    k3 = aux['pre_k3']
    q = _e2_H(aux['pdf'] * _e2_H(fnp.power(aux['sigma'], -5.0))) * (_E2_STRENGTH[l] / 72.0)
    return _e2_H(_e2_H(k3 * k3) * he4) * q
# END OPUS-ENGINEER E2 v1
