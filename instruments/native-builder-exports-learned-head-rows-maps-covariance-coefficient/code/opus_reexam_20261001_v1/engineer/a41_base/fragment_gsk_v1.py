# BEGIN OPUS-ENGINEER TRANSPORT GATE WITH THE FIRST-ORDER CORRECTIONS OF THE MEAN GATE (a41 gsk v1, 2026-10-02; a physics switch, OFF
# when the three strengths are 0: the card's gate is then returned as the same object and nothing changes). Native form of the
# physicist's arms gsk / gk4 / gbk (O/physicist/e16b_f4b.py): the gate that carries the third-order content forward,
#     gq = g + x_s (t3 / 6) He2(alpha) phi - x_k (t4 / 24) He3(alpha) phi + x_b d bpre,   t3 = k3 / sigma^3,  t4 = k4 / sigma^4,
# takes the place of g in the four statements the reference edits: the carried vector b = gq bpre + .., the inherited plane
# gq_i^2 gq_j cp_ij (and its diagonal), the gated young columns, and the old tier's gated basis. The exact-term mixture's anchor
# (F4c) is a function of the gated columns, so its row weights take gq too (the reference's module reads the step's gated
# columns). The birth plane's g Cpre, the births, K4L and the pair law keep the card's g, as in the reference.
# Six builder edits; one polynomial evaluation on nine per-unit rows (float64), 5 requests a layer.
_GSK = 0.0               # x_s: skew correction
_GK4 = 0.0               # x_k: kurtosis correction
_GBK = 0.0               # x_b: the feed of the cross form Sym(b (x) kappa3) into the carried third cumulant
_GSK_CONST = {}


def _gq(g, d, aux, bpre):
    if not (_GSK or _GK4 or _GBK):
        return g
    hold = _E2Hold()   # temporaries of this call, released together
    s = aux['sigma']
    S = _GSK_CONST
    n = int(s.shape[0])
    if S.get('n') != n:   # rows (one, g, alpha, pdf, 1 / s, k3, k4, d, bpre)
        o, G, A, P, I, K3, K4, D, B = range(9)
        terms = [((G,), 1.0), ((K3, I, I, I, A, A, P), _GSK / 6.0), ((K3, I, I, I, P), -_GSK / 6.0),
                 ((K4, I, I, I, I, A, A, A, P), -_GK4 / 24.0), ((K4, I, I, I, I, A, P), 3.0 * _GK4 / 24.0), ((D, B), float(_GBK))]
        degree = max(len(k) for k, _c in terms)
        S.update(n=n, one=fnp.ones(n, dtype=fnp.float64), index=fnp.asarray([[(k[j] if j < len(k) else o) for k, _c in terms] for j in range(degree)]),
                 coefficient=fnp.asarray([float(c) for _k, c in terms], dtype=fnp.float64))
    rows = hold(fnp.stack([S['one'], g, aux['alpha'], aux['pdf'], hold(fnp.power(s, -1.0)), aux['pre_k3'], aux['pre_k4'], d, bpre]))
    return S['coefficient'] @ hold(fnp.prod(hold(rows[S['index']]), axis=0))
# END OPUS-ENGINEER TRANSPORT GATE WITH THE FIRST-ORDER CORRECTIONS OF THE MEAN GATE
