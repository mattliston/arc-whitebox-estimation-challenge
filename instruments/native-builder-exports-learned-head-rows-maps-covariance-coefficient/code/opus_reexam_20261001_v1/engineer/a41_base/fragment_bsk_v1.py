# BEGIN OPUS-ENGINEER BIRTH WEIGHT WITH THE SKEW-CORRECTED KINK DENSITY (a41 bsk v1, 2026-10-02; a physics switch, OFF at strength 0:
# the birth coefficient is then returned as it came). Native form of the physicist's arm bsk (O/physicist/e16b_f4b.py):
#     birth = d (1 + radial / 3)  ->  d (1 + radial / 3) (1 - x (t3 / 6) He3(alpha)),   t3 = k3 / sigma^3 of the layer's pre-activations,
# the density at the kink to first order in the unit's own skewness. One builder edit calls _bsk right after the card's own line.
# One polynomial evaluation on four per-unit rows (float64), 6 requests a layer.
_BSK = 0.0
_BSK_CONST = {}


def _bsk(birth, aux):
    if not _BSK:
        return birth
    hold = _E2Hold()   # temporaries of this call, released together
    s = aux['sigma']
    S = _BSK_CONST
    n = int(s.shape[0])
    if S.get('n') != n:   # rows (one, alpha, 1 / s, k3): 1 - x k3 a^3 / (6 s^3) + x k3 a / (2 s^3)
        S.update(n=n, one=fnp.ones(n, dtype=fnp.float64), index=fnp.asarray([[0, 3, 3], [0, 2, 2], [0, 2, 2], [0, 2, 2], [0, 1, 1], [0, 1, 0], [0, 1, 0]]),
                 coefficient=fnp.asarray([1.0, -float(_BSK) / 6.0, float(_BSK) / 2.0], dtype=fnp.float64))
    rows = hold(fnp.stack([S['one'], aux['alpha'], hold(fnp.power(s, -1.0)), aux['pre_k3']]))
    return birth * hold(S['coefficient'] @ hold(fnp.prod(hold(rows[S['index']]), axis=0)))
# END OPUS-ENGINEER BIRTH WEIGHT WITH THE SKEW-CORRECTED KINK DENSITY
