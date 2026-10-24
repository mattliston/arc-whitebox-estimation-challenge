# BEGIN OPUS-ENGINEER F4 v2 (a14, 2026-10-02; v2 = v1 without the first cohort, whose birth scalars are exactly zero): native port of the physicist lane's F4 rule (O/physicist/f4.py, f4_native_ref.py and
# RESULT.md section T4). An ERROR lever, not a rounding-level change: it moves the predictions by design.
# Every column pair of the bank keeps one scalar from its birth, A = -alpha / (sigma (1 + radial/3)) of its hub unit (the third
# derivative coefficient of the ReLU at the hub over the birth weight). At the end of step l (0..14) the gated columns born at most
# _F4_AGE layers earlier are shifted in place,
#     u_r -> u_r + theta A_r w_r,      u_r = g o pp_r (the carried column),  w_r = (d/2) o pp_r o pp_r,  pp_r the pushed column,
# which makes the bank carry the Gaussian-born fourth-order star into the next births, and the plane this adds to the bank is taken
# out of the star the step leaves behind ("cheap anchor"): the star's trace term is built with b + beta in place of b (the carried
# b is unchanged) and its diagonal with the exact diagonal of the added plane,
#     beta_k = [ w @ (c o (v @ (u o u))) + 2 u @ (c o (v @ (u o w))) ]_k / sum(v^2),   c_r = theta lam_r A_r,   v = diag C (post),
#     diagonal_i = 3 sum_r c_r u_ri^2 w_ri.
# Native form: only the live rows (u and w vanish elsewhere), and factored through r = pp on the live rows so that no
# (rows x columns) array is formed twice:  v @ (u o u) = (v g^2) @ r^2,  v @ (u o w) = (v g d/2) @ r^3,  w @ x = (d/2) o (r^2 @ x),
# u @ x = g o (r @ x),  sum_r c_r u^2 w = g^2 (d/2) o (r^4 @ c).  All float32 (the reference sums in float64).
# Billed 16 flops per element of (live rows x shifted columns): r^2, r^2 A, r^4 A, r^3 (1 each), five matrix-vector passes (2 each),
# the row scaling of the shift and its addition (1 each).
# v2: the cohort born at layer 0 is never shifted. Its hubs have pre-activation mean exactly 0 (the input mean is 0), so its A is
# exactly 0 and v1's work on it at layers 1 and 2 added exact zeros (7% of the step's bill).
_F4_THETA = 0.4
_F4_AGE = 2
_F4_FIRST = 1                                      # the oldest cohort that is ever shifted (cohort 0 has A = 0 exactly)
_F4_A = [None]                                    # this prediction's birth scalars, one per bank column, in the order of lam
_F4_STATS = dict(layers=0, columns=0, elements=0)


def _f4_step(l, right, g, d, live_indices, v, lam, state, aux, selected, radial):
    """Called once per layer 0..14, after the births are selected and before the star's trace term is formed.
    right: the pushed young columns on the live rows (float32); g, d: gate and pdf/sigma (n,); v: diagonal of the carried post
    covariance (n,); lam: the carried third-order weights before this layer's dials; state: the old-tier state (cohort roster).
    Returns None when no column is shifted, else (shift, beta, diagonal): shift (live rows, k) float32 for the last k gated young
    columns, beta (n,) and diagonal (n,) float64 for the star. Appends this layer's birth scalars."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    f32 = fnp.float32
    if l == 0:
        _F4_STATS.update(layers=0, columns=0, elements=0)
    A = fnp.zeros(0, dtype=f32) if l == 0 else _F4_A[0]
    nc = int(lam.shape[0])
    if A is None or int(A.shape[0]) != nc:
        raise RuntimeError('F4: one birth scalar per carried column')
    out = None
    first = max(l - _F4_AGE, _F4_FIRST)           # the oldest cohort shifted at this layer; cohorts first .. l-1 are in the bank
    old = int(state['columns'])
    y0 = max(old, int(state['cohorts'][first][0])) if first < l else nc
    k = nc - y0
    if k > 0:
        n = int(g.shape[0])
        if int(right.shape[1]) != nc - old:
            raise RuntimeError('F4: young column roster')
        r = right[:, y0 - old:]
        hd = d * 0.5
        vg = v * g
        rows = _e2_H(_e2_H(fnp.stack([_e2_H(vg * g), _e2_H(vg * hd), hd, g]))[:, live_indices]).astype(f32)   # row weights on the live rows
        w1, w2, hl, gl = fnp.unstack(rows)
        tA = A[y0:] * _F4_THETA
        lamd = lam[y0:] if _S19_LAM is None else lam[y0:] * _S19_LAM[0]   # the weights these columns carry out of the step
        r2 = r * r
        s1 = w1 @ r2                                  # v . u_r^2
        wA = r2 * tA                                  # theta A_r pp_r^2
        m1 = wA @ _e2_H(lamd * s1)
        q = wA * r2
        de = q @ lamd
        q = fnp.multiply(r2, r, out=q)
        s2 = w2 @ q                                   # v . (u_r w_r)
        m2 = r @ _e2_H(_e2_H(lamd * tA) * s2)
        beta_l = _e2_H(_e2_H(hl * m1) + _e2_H(_e2_H(2 * gl) * m2)).astype(fnp.float64) / _e2_H(fnp.sum(_e2_H(v * v)))
        diag_l = _e2_H(_e2_H(_e2_H(_e2_H(3 * gl) * gl) * hl) * de).astype(fnp.float64)
        beta = fnp.zeros(n, dtype=fnp.float64)
        fnp.put(beta, live_indices, beta_l)
        diagonal = fnp.zeros(n, dtype=fnp.float64)
        fnp.put(diagonal, live_indices, diag_l)
        shift = fnp.multiply(wA, hl[:, None], out=wA)
        _F4_STATS.update(layers=_F4_STATS['layers'] + 1, columns=_F4_STATS['columns'] + k,
                         elements=_F4_STATS['elements'] + int(r.shape[0]) * k)
        out = (shift, beta, diagonal)
    An = _e2_H(-aux['alpha'][selected] / _e2_H(aux['sigma'][selected] * (1 + radial / 3))).astype(f32)
    _F4_A[0] = fnp.concatenate([A, An, An])
    return out
# END OPUS-ENGINEER F4 v2
