# BEGIN OPUS-ENGINEER BATCH-7 HEAD COLUMNS (a50 b7 v2, 2026-10-10; v2 = v1's kmu kg qk e34 s1 in fewer requests (lead 10:4xZ: +560 a network
# did not fit), fold left out (+1.34% of bill, out). Each column standardised over ALL units, clipped at 8 (the T64 convention 'a'):
#   'kmu'   sum_j Ko_ij mu_j / (v_i s_i)            Ko = the K21 plane with its diagonal set to 0
#   'kg'    sum_j Ko_ij g_j / (v_i s_i)             g = the layer's gate Phi(mu / s)
#   'qk'    (sum_j Ko_ij c_j / (c.mu - c_i mu_i)) / v_i,  c = (g / s)^2 mu (qb's c and den, taken from qb when it is named: fragment b6 v2)
#   'e34'   -t3 t4 He5(a) phi(a) / 144,  t3 = k3 / (v s), t4 = k4 / v^2, He5 = a^5 - 10 a^3 + 15 a
#   's1'    sum_r lam_r pp_ir^2 over the columns of the cohort born one layer back (age 1), over v
# Requests a layer: kmu / kg / qk 9 together (11 with the product in float32, _B7_K21_F32) (one product of the K21 plane with up to three vectors, the diagonal out by one einsum, one
# division by a stacked denominator); e34 rides on m2 (b6): both from ONE polynomial table (base rows a, k3, k4, 1 / v, 1 / s, exp(-a^2 / 2),
# six monomials, one gather, one product, one matrix): 10 requests for both, against m2's 14 alone; s1 5.
_B7_K21_F32 = True   # the product in float32 (the plane the pair map received, K32): 2 requests a layer more, the float64 product's bill less
_B67_CONST = {}
_B67_SHARE = globals().get('_B67_SHARE', {})


def _b67_tables(n):
    K = _B67_CONST
    if K.get('n') != n:
        rt = 1.0 / (2.0 * 3.141592653589793) ** 0.5
        # rows of H: 0 a, 1 k3, 2 k4, 3 1/v, 4 1/s, 5 exp(-a^2/2), 6 one; each monomial a product of 12 rows (6 = 1 pads)
        mono = [[1, 1, 3, 3, 3, 0, 0, 0, 0, 5], [1, 1, 3, 3, 3, 0, 0, 5], [1, 1, 3, 3, 3, 5],
                [1, 2, 3, 3, 3, 4, 0, 0, 0, 0, 0, 5], [1, 2, 3, 3, 3, 4, 0, 0, 0, 5], [1, 2, 3, 3, 3, 4, 0, 5]]
        idx = [[m[f] if f < len(m) else 6 for m in mono] for f in range(12)]
        coef = [[rt / 72.0, -6.0 * rt / 72.0, 3.0 * rt / 72.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -rt / 144.0, 10.0 * rt / 144.0, -15.0 * rt / 144.0]]
        K.update(n=n, idx=fnp.asarray(idx), coef=fnp.asarray(coef), one=fnp.ones(n, dtype=fnp.float64))
    return K


def _b67_poly(l, a, vc, s, k3, k4, response_mask, nrows):
    """m2 (on the response rows, as fragment b6) and e34 (all units) from one polynomial table."""
    want_m2, want_e34 = 'm2' in _HC_COLUMNS, 'e34' in _HC_COLUMNS
    if _HC_U is None or not (want_m2 or want_e34) or not (1 <= l <= 14):
        return
    hold = _E2Hold()
    K = _b67_tables(int(a.shape[0]))
    a2 = hold(a * a)
    e = hold(fnp.exp(hold(a2 * -0.5)))
    H = hold(fnp.stack([a, k3, k4, hold(1.0 / vc), hold(1.0 / s), e, K['one']]))
    m2v, e34v = hold(fnp.unstack(hold(K['coef'] @ hold(fnp.prod(hold(H[K['idx']]), axis=0)))))
    if want_m2:
        _hc_put(l, 'm2', m2v, response_mask, nrows)
    if want_e34:
        _hc_put(l, 'e34', e34v)


def _b7_k21(l, K21, kdiag, mu, g, s, vc, K32=None):
    """kmu, kg, qk from the K21 plane the pair map receives (float64, its diagonal kdiag)."""
    names = [x for x in ('kmu', 'kg', 'qk') if _HC_U is not None and x in _HC_COLUMNS]
    if not names or not (1 <= l <= 14):
        return
    hold = _E2Hold()
    cols, dens = [], []
    vs = hold(vc * s)
    if 'kmu' in names:
        cols.append(mu); dens.append(vs)
    if 'kg' in names:
        cols.append(g); dens.append(vs)
    if 'qk' in names:
        share = _B67_SHARE.get('cden')
        if share is not None and share[0] == l:
            c, den = share[1], share[2]
        else:
            gs = hold(g / s)
            c = hold(hold(gs * gs) * mu)
            cm = hold(c * mu)
            den = hold(hold(fnp.sum(cm)) - cm)
        cols.append(c); dens.append(hold(den * vc))
    X = hold(fnp.stack(cols, axis=1))                                                  # (n, k) float64
    KX = hold(hold(K32 @ hold(X.astype(fnp.float32))).astype(fnp.float64)) if (_B7_K21_F32 and K32 is not None) else hold(K21 @ X)
    off = hold(KX - hold(fnp.einsum('i,ik->ik', kdiag, X, optimize=False)))         # the plane without its diagonal
    R = hold(off / hold(fnp.stack(dens, axis=1)))
    for name, x in zip(names, hold(fnp.unstack(R, axis=1))):
        _hc_put(l, name, x)


def _b7_bank(l, pp, pp2, lam, state, vc):
    """s1 from the pushed bank columns of layer l (all units)."""
    if _HC_U is None or 's1' not in _HC_COLUMNS or not (1 <= l <= 14):
        return
    hold = _E2Hold()
    n = int(pp.shape[0])
    start, width = state['cohorts'][l - 1] if l - 1 < len(state['cohorts']) else (0, 0)   # one entry per birth layer
    start, width = int(start), int(width)
    s1 = hold(pp2[:, start:start + width] @ hold(lam[start:start + width])) if width else hold(fnp.zeros(n, dtype=fnp.float32))
    _hc_put(l, 's1', hold(hold(s1.astype(fnp.float64)) / vc))
# END OPUS-ENGINEER BATCH-7 HEAD COLUMNS
