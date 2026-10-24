# BEGIN OPUS-ENGINEER EDGE CUT (a19 "ecut", 2026-10-02; an APPROXIMATION of one learned part, not bit-exact).
# The learned covariance edge (layers 8-14) is 1.20% of the bill: a 16-channel tanh layer on an (n, n, 16) tensor and one output
# vector give T_ij, and its raw output is T_ij pdf_i gate_j, symmetrised. |T_ij| is bounded by the output vector's 1-norm, so the
# entries whose row unit is far from its kink (pdf small) or whose column unit is nearly dead (gate small) are small.
# Kept: rows = the layer's response rows (alive and min(gate, 1 - gate) >= 0.01, i.e. |mu/sigma| <= 2.33, where pdf is at least
# 6.6% of its peak), columns = the alive columns (gate >= 0.01); both rosters are the live loop's own. The rest of raw is set to 0.
# The tensor shrinks to about half; the kept block is computed exactly as before (the bias is added to the (rows, 16) operand
# before the broadcast, one tensor pass fewer; rounding level). _ECUT_STANDIN replaces tanh by the 10-flop stand-in of a17
# (largest error 2.0e-4).
_ECUT_STANDIN = False
_ECUT_C1 = 0.165
_ECUT_C2 = 0.0098
_ECUT_STATS = dict(layers=0, kept=0, full=0)


def _ecut_raw(left, right, t, k, pf, gf, hold):
    """raw (n, n) float32 of the message edge: T_ij pdf_i gate_j on the kept rows x columns, 0 elsewhere.
    left, right: the two (n, 16) operands; t: edge parameters; k: the edge's persistent R, S planes and weight rows."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    L = _E2_LAYER
    I, J = L['ecut_rows'], L['ecut_cols']
    n, ni, nj = int(pf.shape[0]), int(I.shape[0]), int(J.shape[0])
    R, S = k['R'], k['S']
    Rij = _e2_H(R[I])[:, J]
    Sij = _e2_H(S[I])[:, J]
    Sji = _e2_H(S[J])[:, I]
    h = _e2_H(_e2_H(_e2_H(left[I]) + t['bias'])[:, None, :]) + _e2_H(right[J])
    hold.append(fnp.add(h, _e2_H(_e2_H(Rij[:, :, None]) * k['e0']), out=h))
    hold.append(fnp.add(h, _e2_H(_e2_H(Sij[:, :, None]) * k['e1']), out=h))
    hold.append(fnp.add(h, _e2_H(_e2_H(_e2_H(Sji.T)[:, :, None]) * k['e2']), out=h))
    if _ECUT_STANDIN:
        # tanh(x) ~ X / sqrt(X^2 + 1 / c2^2), X = x ((u + A)^2 + B), u = min(x^2, 81): sinh(x) ~ x (1 + c1 u + c2 u^2)
        A = _ECUT_C1 / (2 * _ECUT_C2)
        w = fnp.empty((ni, nj, int(h.shape[2])), dtype=fnp.float32)
        hold.append(fnp.multiply(h, h, out=w))
        hold.append(fnp.minimum(w, 81.0, out=w))
        hold.append(fnp.add(w, A, out=w))
        hold.append(fnp.multiply(w, w, out=w))
        hold.append(fnp.add(w, 1 / _ECUT_C2 - A * A, out=w))
        hold.append(fnp.multiply(h, w, out=h))
        hold.append(fnp.multiply(h, h, out=w))
        hold.append(fnp.add(w, 1 / _ECUT_C2 ** 2, out=w))
        hold.append(fnp.sqrt(w, out=w))
        hold.append(fnp.divide(h, w, out=h))
    else:
        hold.append(fnp.tanh(h, out=h))
    block = fnp.einsum('ij,i,j->ij', _e2_H(h @ t['output']), _e2_H(pf[I]), _e2_H(gf[J]), optimize=False)
    raw = fnp.zeros((n, n), dtype=fnp.float32)
    _e2_H(fnp.put(raw, _e2_H(_e2_H(_e2_H(I * n)[:, None]) + J), block))
    _ECUT_STATS.update(layers=_ECUT_STATS['layers'] + 1, kept=_ECUT_STATS['kept'] + ni * nj, full=_ECUT_STATS['full'] + n * n)
    return raw
# END OPUS-ENGINEER EDGE CUT
