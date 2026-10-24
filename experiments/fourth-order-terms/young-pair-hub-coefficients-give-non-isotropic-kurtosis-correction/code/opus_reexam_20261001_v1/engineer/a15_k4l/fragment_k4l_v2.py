# BEGIN OPUS-ENGINEER K4L v2 (a15, 2026-10-02; v2 = v1's arithmetic with every temporary held to the end of the call and lower-case
# einsum subscripts): native port of the lead's K4L rule (O/K4L_SPEC.md, torch reference
# O/e16b_k4_closed_loop.py). An ERROR lever, not a rounding-level change: it moves the predictions by design.
# The card's per-unit fourth cumulant of the pre-activations is one number per layer, k4_i = radial v_i^2 (times the band dial).
# Every column pair of the bank was born at a hub h as p+- = s_h + t+- e_h, and the ReLU at the hub gives the post-activations a
# fourth cumulant with m legs on the hub and 4 - m on its covariance column, with exactly known coefficients S1..S4. At a later
# layer the pair's pushed columns p (+) and q (-) hold the two transported legs, Wt = (p - q) / (t+ - t-) and Y = p - t+ Wt, and the
# hub adds  4 S1 Y^3 Wt + 6 S2 Y^2 Wt^2 + 4 S3 Y Wt^3 + S4 Wt^4  to the fourth cumulant of every unit. Summed over the pairs born
# at most _K4L_AGE layers earlier this is loc_i; the level the card was trained with is kept:
#     beta = sum_i w_i loc_i v_i^2 / sum_i w_i v_i^4,   w_i = ((a_i^2 - 1) exp(-a_i^2 / 2))^2,  a = mu / sigma,
#     k4_i <- k4_i + c (loc_i - beta v_i^2)
# for the k4 the pair map receives at layers 1-14 and for the final layer's k4.
# Native form: the quartic is written on p and D = p - q,  D (e1 p^3 + D (e2 p^2 + D (e3 p + e4 D))), with four coefficients per
# pair fixed at its birth (the shift from Y to p and the powers of the gap are folded into them), and p^2, p^3 are the squares and
# cubes the live loop already forms: 12 flops per element of (rows x pairs), float32. Rows the native row cuts do not push have
# loc = 0 exactly; they get no correction and are left out of the beta sums (the torch reference has no row cuts).
_K4L_C = 1.0
_K4L_AGE = 2
_K4L_B = {}                                       # birth layer -> (e1, e2, e3, e4), float32, one entry per pair; this prediction
_K4L_FINAL = [None]                               # (rows, local terms on those rows) left by the final layer's row-cut push
_K4L_CONST = {}
_K4L_STATS = dict(blocks=0, pairs=0, elements=0)


def _k4l_const():
    C = _K4L_CONST
    if not C:
        f64 = fnp.float64
        C['expo'] = fnp.arange(5, dtype=f64)[:, None]
        # Gaussian moments of the rectified hub in units of sigma: rho_k = (row k) . powers of a  times the gate, plus
        # (row k + 3) . powers of a  times the pdf, k = 2, 3, 4
        C['moments'] = fnp.asarray([[1, 0, 1, 0, 0], [0, 3, 0, 1, 0], [3, 0, 6, 0, 1],
                                    [0, 1, 0, 0, 0], [2, 0, 1, 0, 0], [0, 5, 0, 1, 0]], dtype=f64)
        # S1..S4 in units of sigma as sums of products L_k R_k (see _k4l_birth for the two lists)
        C['hub'] = fnp.asarray([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 3, -6, -3, 6, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, -4, 12, -6, -3]], dtype=f64)
        # the shift Y = p - t+ Wt: coefficient of p^(4-M) Wt^M = binom(4, M) sum_m binom(M, m) (-t+)^(M-m) S_m
        binom = ((1, 0, 0, 0), (2, 1, 0, 0), (3, 3, 1, 0), (4, 6, 4, 1))
        shift = [[[0.0] * 4 for _m in range(4)] for _M in range(4)]
        for M in range(1, 5):
            for m in range(1, M + 1):
                shift[M - 1][m - 1][M - m] = float((4, 6, 4, 1)[M - 1] * binom[M - 1][m - 1])
        C['shift'] = fnp.asarray(shift, dtype=f64)
        C['one_to_four'] = fnp.arange(1, 5, dtype=f64)[:, None]
        C['expo4'] = C['expo'][:4]
    return C


def _k4l_birth(l, aux, host, selected, t, s):
    """The four coefficients of the pairs born at layer l. host: the pair map's mean (before the learned correction); t, s: the
    + and - hub entries of the newborn columns (n,)."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    C = _k4l_const()
    if l == 0:
        _K4L_B.clear()
        _K4L_STATS.update(blocks=0, pairs=0, elements=0)
    a, sg, g, ph, m, tp, tm = fnp.unstack(_e2_H(fnp.stack([aux['alpha'], aux['sigma'], aux['gate'], aux['pdf'], host, t, s]))[:, selected])
    G = C['moments'] @ _e2_H(fnp.power(a, C['expo']))
    Gp = _e2_H(G[3:]) * ph
    rho2, rho3, rho4 = fnp.unstack(_e2_H(_e2_H(G[:3]) * g) + Gp)
    one, mt, mt2, _unused, mt4 = fnp.unstack(fnp.power(_e2_H(m / sg), C['expo']))   # powers of the host mean in units of sigma
    L = fnp.stack([_e2_H(Gp[0]), g, g, mt, rho2, mt2, g, g, rho4, rho3, rho2, mt4, rho2])
    R = fnp.stack([one, one, g, ph, one, one, rho2, mt2, one, mt, mt2, one, rho2])
    S = fnp.einsum('qk,kj,kj->qj', C['hub'], L, R, optimize=False)                  # S_m / sigma^(2m-4), m = 1..4
    v = sg * sg
    T = fnp.power(_e2_H(-_e2_H(tp / v)), C['expo4'])
    E = fnp.einsum('amk,kj,mj->aj', C['shift'], T, S, optimize=False)
    scale = _e2_H(fnp.power(_e2_H(v / _e2_H(tp - tm)), C['one_to_four'])) / _e2_H(v * v)
    _K4L_B[l] = fnp.unstack(_e2_H(E * scale).astype(fnp.float32))


def _k4l_local(l, pp, pp2, pp3, state):
    """Sum over the pairs born at most _K4L_AGE layers before layer l of their hub terms, per row of pp (float32).
    pp: pushed bank columns (rows, all columns); pp2, pp3: its squares and cubes, or None."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    parts = []
    for j in range(max(l - _K4L_AGE, 0), l):
        start, width = state['cohorts'][j]
        start, K = int(start), int(width) // 2
        if K == 0:
            continue
        e1, e2, e3, e4 = _K4L_B[j]
        p = pp[:, start:start + K]
        p2 = _e2_H(p * p) if pp2 is None else pp2[:, start:start + K]
        p3 = _e2_H(p2 * p) if pp3 is None else pp3[:, start:start + K]
        D = p - _e2_H(pp[:, start + K:start + 2 * K])
        h = D * e4
        x = p * e3
        _e2_H(fnp.add(h, x, out=h))
        _e2_H(fnp.multiply(h, D, out=h))
        _e2_H(fnp.multiply(p2, e2, out=x))
        _e2_H(fnp.add(h, x, out=h))
        _e2_H(fnp.multiply(h, D, out=h))
        _e2_H(fnp.multiply(p3, e1, out=x))
        _e2_H(fnp.add(h, x, out=h))
        parts.append(fnp.einsum('ij,ij->i', h, D, optimize=False))
        _K4L_STATS.update(blocks=_K4L_STATS['blocks'] + 1, pairs=_K4L_STATS['pairs'] + K,
                          elements=_K4L_STATS['elements'] + int(pp.shape[0]) * K)
    if not parts:
        return None
    loc = parts[0]
    for part in parts[1:]:
        loc = _e2_H(loc) + part
    return loc


def _k4l_dk(loc, a, vv):
    """c (loc - beta vv^2) on the pushed rows (loc != 0), 0 elsewhere. a = mu / sigma, vv = pre-activation variances (n,)."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    loc = loc.astype(fnp.float64)
    a2 = a * a
    q = _e2_H(a2 - 1) * _e2_H(fnp.exp(_e2_H(a2 * -0.5)))
    pushed = fnp.not_equal(loc, 0.0)
    iso = vv * vv
    wi = _e2_H(_e2_H(_e2_H(q * q) + 1e-12) * pushed) * iso
    beta = _e2_H(fnp.sum(_e2_H(wi * loc))) / _e2_H(fnp.sum(_e2_H(wi * iso)))
    dk = _e2_H(loc - _e2_H(beta * iso)) * pushed
    return dk if _K4L_C == 1.0 else dk * _K4L_C


def _k4l_final(mu, v):
    """The final layer's correction, from the local terms the row-cut push left in _K4L_FINAL."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    found, _K4L_FINAL[0] = _K4L_FINAL[0], None
    if found is None or found[1] is None:
        return 0.0
    loc = fnp.zeros(int(mu.shape[0]), dtype=fnp.float32)
    fnp.put(loc, found[0], found[1])
    return _k4l_dk(loc, mu / _e2_H(fnp.sqrt(_e2_H(fnp.maximum(v, 1e-10)))), v)
# END OPUS-ENGINEER K4L v2
