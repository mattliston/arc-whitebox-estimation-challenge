# BEGIN OPUS-ENGINEER C2K v1 (a32, 2026-10-02): the lead's row-energy term of the per-unit fourth cumulant (O/e16b_k4_closed_loop.py arm
# c2k, E33): the pair map of layers 1-14 gets
#     k4_i += (6 b / n) (C^2)_ii - beta v_i^2,      (C^2)_ii = sum_j Cpre_ij^2,
# with the isotropic level beta taken out exactly as for the hub terms of K4L (weights ((a^2 - 1) exp(-a^2 / 2))^2 + 1e-12, a = mu / sigma).
# An ERROR lever. Bill: one row sum of the squared covariance the loop already holds (2 n^2 flops in float64) and ten n-vector passes.
_C2K_B = 0.25


def _c2k_dq(CC, a, vv):
    """The term for one layer (n,) float64. CC: Cpre o Cpre (n, n); a = mu / sigma (n,); vv = the pre-activation variances (n,)."""
    _e2_H = _E2Hold()   # temporaries of this call, released together
    q = _e2_H(fnp.sum(CC, axis=1)) * (6.0 * _C2K_B / int(vv.shape[0]))
    a2 = a * a
    w = _e2_H(a2 - 1) * _e2_H(fnp.exp(_e2_H(a2 * -0.5)))
    iso = vv * vv
    wi = _e2_H(_e2_H(w * w) + 1e-12) * iso
    beta = _e2_H(fnp.sum(_e2_H(wi * q))) / _e2_H(fnp.sum(_e2_H(wi * iso)))
    return q - _e2_H(beta * iso)
# END OPUS-ENGINEER C2K v1
