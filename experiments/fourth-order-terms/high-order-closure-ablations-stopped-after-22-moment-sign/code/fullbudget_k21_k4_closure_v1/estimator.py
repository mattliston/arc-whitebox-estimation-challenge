"""FB-EC4-32: target-free full-budget connected Edgeworth closure.

This deployed-source scaffold contains no reader, fit, target, score, retry,
submission, absolute-path import, or unmetered numerical library.  Every
array-valued numerical operation is expressed through FlopScope.  Literal
coefficient tables replace runtime factorial/binomial/square-root setup.

State after h_l (l < 16): fp64 mean/covariance, a full-birth fp32 CP3 state,
an fp32 cubic coordinate-star residual, a rank-2048 fp32 symmetric CP4
carrier, and an fp32 quartic coordinate-star plus one fp64 covariance-pairing
coefficient.
"""

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP


_CP3_CAP = 16384
_CP4_CAP = 2048
_Q32_TERMS = 33
_VAR_FLOOR = 1.0e-30
_PSD_EIGEN_FLOOR = 9.094947017729282e-13  # 2^-40
_INV_SQRT_2PI = 0.3989422804014327
_PI = 3.141592653589793
_INV_TWO_PI = 0.15915494309189535

# 1/t!, t=0..32.  These are frozen payload constants, never generated in
# setup/predict.  The last values intentionally use their binary64 spelling.
_INV_FACTORIAL = (
    1.0,
    1.0,
    0.5,
    0.16666666666666666,
    0.041666666666666664,
    0.008333333333333333,
    0.001388888888888889,
    0.0001984126984126984,
    2.48015873015873e-05,
    2.7557319223985893e-06,
    2.755731922398589e-07,
    2.505210838544172e-08,
    2.08767569878681e-09,
    1.6059043836821613e-10,
    1.1470745597729725e-11,
    7.647163731819816e-13,
    4.779477332387385e-14,
    2.8114572543455206e-15,
    1.5619206968586225e-16,
    8.22063524662433e-18,
    4.110317623312165e-19,
    1.9572941063391263e-20,
    8.896791392450574e-22,
    3.868170170630684e-23,
    1.6117375710961184e-24,
    6.446950284384474e-26,
    2.4795962632247976e-27,
    9.183689863795546e-29,
    3.279889237069838e-30,
    1.1309962886447718e-31,
    3.7699876288159054e-33,
    1.216125041553518e-34,
    3.8003907548547434e-36,
)

# He_(r+1)(x) = x He_r(x) - r He_(r-1)(x), r=1..35.  The
# recurrence coefficients are literal; its operands are metered arrays.
_HERMITE_RECURRENCE = (
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0,
)

# The 17 derivative pairs used by 1 + L3 + L4 + 1/2 L3^2.
_DERIVATIVE_CHANNELS = (
    (0, 0),
    (0, 3), (1, 2), (2, 1), (3, 0),
    (0, 4), (1, 3), (2, 2), (3, 1), (4, 0),
    (0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0),
)


def _off_diagonal(matrix):
    return matrix - fnp.diag(fnp.diag(matrix))


def _replace_diagonal(matrix, diagonal):
    return _off_diagonal(matrix) + fnp.diag(diagonal)


def _hermite_values(argument):
    """He_0..He_36 with frozen scalar coefficients and metered array math."""
    values = [fnp.ones_like(argument), argument]
    for coefficient in _HERMITE_RECURRENCE:
        values.append(argument * values[-1] - coefficient * values[-2])
    return values


def _gaussian_marginals(mean, variance):
    """Gaussian positive-part mass and raw moments zero through four."""
    variance = fnp.maximum(variance, _VAR_FLOOR)
    sigma = fnp.sqrt(variance)
    alpha = mean / sigma
    pdf = fnp.exp(-0.5 * alpha * alpha) * _INV_SQRT_2PI
    cdf = flops.stats.norm.cdf(alpha)
    mean2 = mean * mean
    variance2 = variance * variance
    raw1 = sigma * pdf + mean * cdf
    raw2 = (mean2 + variance) * cdf + mean * sigma * pdf
    raw3 = (
        mean * (mean2 + 3.0 * variance) * cdf
        + sigma * (mean2 + 2.0 * variance) * pdf
    )
    raw4 = (
        (mean2 * mean2 + 6.0 * mean2 * variance + 3.0 * variance2) * cdf
        + mean * sigma * (mean2 + 5.0 * variance) * pdf
    )
    return (cdf, raw1, raw2, raw3, raw4), sigma, alpha, pdf


def _q38_coefficients(mean, variance):
    """c[p,q]=E[X_+^p He_q(Z)] for p=1..3 and q=0..38."""
    raw, sigma, alpha, pdf = _gaussian_marginals(mean, variance)
    mass, raw1, raw2, raw3, raw4 = raw
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    minus_alpha = -alpha
    hermite = _hermite_values(minus_alpha)

    c1 = [raw1, sigma * mass]
    for order in range(2, 39):
        c1.append(sigma * hermite[order - 2] * pdf)

    c2 = [raw2, 2.0 * sigma * raw1, 2.0 * sigma2 * mass]
    for order in range(3, 39):
        c2.append(2.0 * sigma2 * hermite[order - 3] * pdf)

    c3 = [raw3, 3.0 * sigma * raw2, 6.0 * sigma2 * raw1, 6.0 * sigma3 * mass]
    for order in range(4, 39):
        c3.append(6.0 * sigma3 * hermite[order - 4] * pdf)

    return (c1, c2, c3), raw, sigma, alpha, pdf


def _univariate_edgeworth(coefficients, raw, sigma, alpha, pdf, k3, k4):
    """Exact frozen 1+L3+L4+1/2L3^2 univariate specialization."""
    mass, raw1, raw2, raw3, raw4 = raw
    inv_sigma = 1.0 / sigma
    inv_sigma2 = inv_sigma * inv_sigma
    inv_sigma3 = inv_sigma2 * inv_sigma
    inv_sigma4 = inv_sigma2 * inv_sigma2
    inv_sigma6 = inv_sigma3 * inv_sigma3
    k3_squared = k3 * k3

    c1, c2, c3 = coefficients
    corrected1 = (
        c1[0]
        + (k3 * inv_sigma3 / 6.0) * c1[3]
        + (k4 * inv_sigma4 / 24.0) * c1[4]
        + (k3_squared * inv_sigma6 / 72.0) * c1[6]
    )
    corrected2 = (
        c2[0]
        + (k3 * inv_sigma3 / 6.0) * c2[3]
        + (k4 * inv_sigma4 / 24.0) * c2[4]
        + (k3_squared * inv_sigma6 / 72.0) * c2[6]
    )
    corrected3 = (
        c3[0]
        + (k3 * inv_sigma3 / 6.0) * c3[3]
        + (k4 * inv_sigma4 / 24.0) * c3[4]
        + (k3_squared * inv_sigma6 / 72.0) * c3[6]
    )

    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    sigma4 = sigma2 * sigma2
    c4_3 = 24.0 * sigma3 * raw1
    c4_4 = 24.0 * sigma4 * mass
    c4_6 = -24.0 * sigma4 * alpha * pdf
    corrected4 = (
        raw4
        + (k3 * inv_sigma3 / 6.0) * c4_3
        + (k4 * inv_sigma4 / 24.0) * c4_4
        + (k3_squared * inv_sigma6 / 72.0) * c4_6
    )
    corrected1 = fnp.maximum(corrected1, 0.0)
    corrected2 = fnp.maximum(
        corrected2,
        corrected1 * corrected1 + _VAR_FLOOR,
    )
    return corrected1, corrected2, corrected3, corrected4


def _q32_context(coefficients, sigma, covariance):
    inv1 = 1.0 / sigma
    inv2 = inv1 * inv1
    inv3 = inv2 * inv1
    inv4 = inv2 * inv2
    inv5 = inv4 * inv1
    inv6 = inv3 * inv3
    inverse_sigma = (fnp.ones_like(sigma), inv1, inv2, inv3, inv4, inv5, inv6)

    correlation = fnp.clip(
        covariance * inv1[:, None] * inv1[None, :],
        -1.0,
        1.0,
    )
    powers = [fnp.ones_like(correlation)]
    for _index in range(1, _Q32_TERMS):
        powers.append(powers[-1] * correlation)
    return (
        coefficients,
        inverse_sigma,
        fnp.stack(powers, axis=0),
        fnp.asarray(_INV_FACTORIAL, dtype=fnp.float64),
    )


def _g_group(left_power, right_power, requests, context):
    """Evaluate at most three derivative kernels in one bounded temporary."""
    coefficients, inverse_sigma, rho_powers, inverse_factorial = context
    left_coefficients = coefficients[left_power - 1]
    right_coefficients = coefficients[right_power - 1]
    left_rows = []
    right_rows = []
    for left_derivative, right_derivative in requests:
        left_rows.append(
            fnp.stack(
                left_coefficients[
                    left_derivative:left_derivative + _Q32_TERMS
                ],
                axis=0,
            ) * inverse_sigma[left_derivative][None, :]
        )
        right_rows.append(
            fnp.stack(
                right_coefficients[
                    right_derivative:right_derivative + _Q32_TERMS
                ],
                axis=0,
            ) * inverse_sigma[right_derivative][None, :]
        )
    return fnp.einsum(
        "dti,dtj,tij,t->dij",
        fnp.stack(left_rows, axis=0),
        fnp.stack(right_rows, axis=0),
        rho_powers,
        inverse_factorial,
    )


def _edgeworth_weights(k3, k21, k4, k31, k22):
    """Literal bivariate weights including every L3^2 cross multiplicity."""
    w03 = k3[None, :] / 6.0
    w12 = fnp.transpose(k21) / 2.0
    w21 = k21 / 2.0
    w30 = k3[:, None] / 6.0

    v04 = k4[None, :] / 24.0
    v13 = fnp.transpose(k31) / 6.0
    v22 = k22 / 4.0
    v31 = k31 / 6.0
    v40 = k4[:, None] / 24.0

    # 0.5 times the convolution of (w03,w12,w21,w30) with itself.
    u06 = 0.5 * w03 * w03
    u15 = w03 * w12
    u24 = w03 * w21 + 0.5 * w12 * w12
    u33 = w03 * w30 + w12 * w21
    u42 = w12 * w30 + 0.5 * w21 * w21
    u51 = w21 * w30
    u60 = 0.5 * w30 * w30

    return (
        fnp.ones_like(k21),
        w03, w12, w21, w30,
        v04, v13, v22, v31, v40,
        u06, u15, u24, u33, u42, u51, u60,
    )


def _pair_edgeworth(
    left_power,
    right_power,
    context,
    k3,
    k21,
    k4,
    k31,
    k22,
):
    weights = _edgeworth_weights(k3, k21, k4, k31, k22)
    result = fnp.zeros_like(k21)
    for start in range(0, 17, 3):
        requests = _DERIVATIVE_CHANNELS[start:start + 3]
        kernels = _g_group(left_power, right_power, requests, context)
        for local_index in range(len(requests)):
            result = result + weights[start + local_index] * kernels[local_index]
    return result


def _gaussian_pair(left_power, right_power, context):
    return _g_group(left_power, right_power, ((0, 0),), context)[0]


def _connected_planes(mean, covariance, m11, m21, m31, m22, raw2, raw3, raw4):
    """Connected K21/K4 planes, all reset from the same repaired moments."""
    mean_i = mean[:, None]
    mean_j = mean[None, :]
    raw2_i = raw2[:, None]
    raw2_j = raw2[None, :]
    raw3_i = raw3[:, None]

    k21 = (
        m21
        - raw2_i * mean_j
        - 2.0 * mean_i * m11
        + 2.0 * mean_i * mean_i * mean_j
    )

    central31 = (
        m31
        - mean_j * raw3_i
        - 3.0 * mean_i * m21
        + 3.0 * mean_i * mean_j * raw2_i
        + 3.0 * mean_i * mean_i * m11
        - 3.0 * mean_i * mean_i * mean_i * mean_j
    )
    k31 = _off_diagonal(
        central31 - 3.0 * fnp.diag(covariance)[:, None] * covariance
    )

    m12 = fnp.transpose(m21)
    central22 = (
        m22
        - 2.0 * mean_j * m21
        - 2.0 * mean_i * m12
        + mean_j * mean_j * raw2_i
        + mean_i * mean_i * raw2_j
        + 4.0 * mean_i * mean_j * m11
        - 3.0 * mean_i * mean_i * mean_j * mean_j
    )
    variance = fnp.diag(covariance)
    k22 = _off_diagonal(
        central22
        - fnp.outer(variance, variance)
        - 2.0 * covariance * covariance
    )

    k4 = (
        raw4
        - 4.0 * mean * raw3
        - 3.0 * raw2 * raw2
        + 12.0 * mean * mean * raw2
        - 6.0 * mean * mean * mean * mean
    )
    k21 = _replace_diagonal(k21, fnp.diag(k21))
    return k21, k4, k31, k22


def _repair_covariance(raw_covariance, variance):
    """Diagonal-preserving spectral repair in the correlation gauge."""
    raw_covariance = _replace_diagonal(raw_covariance, variance)
    diagonal_scale = fnp.sqrt(variance)
    inverse_scale = 1.0 / diagonal_scale
    correlation = raw_covariance * inverse_scale[:, None] * inverse_scale[None, :]
    correlation = 0.5 * (correlation + fnp.transpose(correlation))
    eigenvalues, eigenvectors = fnp.linalg.eigh(correlation)
    repaired_eigenvalues = fnp.maximum(eigenvalues, _PSD_EIGEN_FLOOR)
    repaired0 = (
        eigenvectors * repaired_eigenvalues[None, :]
    ) @ fnp.transpose(eigenvectors)
    renormalizer = 1.0 / fnp.sqrt(fnp.diag(repaired0))
    repaired_correlation = repaired0 * renormalizer[:, None] * renormalizer[None, :]
    return repaired_correlation * diagonal_scale[:, None] * diagonal_scale[None, :]


def _cp_read(cp_lambda, cp_p, cp_q):
    if int(cp_lambda.shape[0]) == 0:
        width = int(cp_p.shape[0])
        return fnp.zeros((width, width), dtype=fnp.float64)
    return fnp.asarray(
        (cp_p * cp_p) @ fnp.transpose(cp_q * cp_lambda[None, :])
        + 2.0 * (cp_p * cp_q) @ fnp.transpose(cp_p * cp_lambda[None, :]),
        dtype=fnp.float64,
    )


def _cp_k3(cp_lambda, cp_p, cp_q):
    """Marginal-only final CP read; no n-by-n K21 plane is materialized."""
    return fnp.asarray(
        3.0 * ((cp_p * cp_p * cp_q) @ cp_lambda),
        dtype=fnp.float64,
    )


def _cp_push(cp_lambda, cp_p, cp_q, weight32):
    if int(cp_lambda.shape[0]) == 0:
        return cp_lambda, cp_p, cp_q
    return (
        cp_lambda,
        fnp.transpose(weight32) @ cp_p,
        fnp.transpose(weight32) @ cp_q,
    )


def _cp_post_read(pre_k21, gate, pre_covariance, birth_lambda):
    """Exact full-birth K21 read without a second pair of CP GEMMs."""
    inherited = gate[:, None] * gate[:, None] * gate[None, :] * pre_k21
    birth_direct = birth_lambda[None, :] * (
        gate[:, None] * pre_covariance
    ) ** 2
    birth_transpose = 2.0 * birth_lambda[:, None] * (
        gate * fnp.diag(pre_covariance)
    )[:, None] * (
        gate[None, :] * fnp.transpose(pre_covariance)
    )
    return inherited + birth_direct + birth_transpose


def _cp_grow(cp_lambda, pushed_p, pushed_q, gate, pre_covariance, birth_lambda, identity):
    gate32 = fnp.asarray(gate, dtype=fnp.float32)
    covariance32 = fnp.asarray(pre_covariance, dtype=fnp.float32)
    birth_lambda32 = fnp.asarray(birth_lambda, dtype=fnp.float32)
    post_p = fnp.concatenate(
        [gate32[:, None] * pushed_p, gate32[:, None] * covariance32],
        axis=1,
    )[:, :_CP3_CAP]
    post_q = fnp.concatenate(
        [gate32[:, None] * pushed_q, identity],
        axis=1,
    )[:, :_CP3_CAP]
    post_lambda = fnp.concatenate([cp_lambda, birth_lambda32])[:_CP3_CAP]
    return post_lambda, post_p, post_q


def _cp4_push(cp4_lambda, cp4_u, weight32):
    if int(cp4_lambda.shape[0]) == 0:
        return cp4_lambda, cp4_u
    return cp4_lambda, fnp.transpose(weight32) @ cp4_u


def _cp4_read_planes(cp4_lambda, cp4_u):
    """Read d/A/B from fp32 symmetric CP4, then cast completed planes."""
    width = int(cp4_u.shape[0])
    if int(cp4_lambda.shape[0]) == 0:
        zero_vector = fnp.zeros(width, dtype=fnp.float64)
        zero_plane = fnp.zeros((width, width), dtype=fnp.float64)
        return zero_vector, zero_plane, zero_plane
    u2 = cp4_u * cp4_u
    directed_full = (
        (u2 * cp4_u * cp4_lambda[None, :]) @ fnp.transpose(cp4_u)
    )
    paired_full = (
        (u2 * cp4_lambda[None, :]) @ fnp.transpose(u2)
    )
    directed64 = fnp.asarray(directed_full, dtype=fnp.float64)
    paired64 = fnp.asarray(paired_full, dtype=fnp.float64)
    return (
        fnp.diag(directed64),
        _off_diagonal(directed64),
        _off_diagonal(paired64),
    )


def _cp4_read_diagonal(cp4_lambda, cp4_u):
    """W16 CP4 specialization: marginal K4 only, with no dense plane."""
    u2 = cp4_u * cp4_u
    return fnp.asarray(
        (u2 * u2) @ cp4_lambda,
        dtype=fnp.float64,
    )


def _cp4_reset(
    cp4_lambda,
    pushed_u,
    gate,
    pre_provider_planes,
    k4_target,
    identity,
):
    """Gate, append exact diagonal birth, top-energy cap, subtract drop."""
    gate32 = fnp.asarray(gate, dtype=fnp.float32)
    gate64 = fnp.asarray(gate32, dtype=fnp.float64)
    inherited_u = gate32[:, None] * pushed_u
    _pre_d, pre_a, pre_b = pre_provider_planes
    gate2 = gate64 * gate64
    if int(cp4_lambda.shape[0]) == 0:
        inherited_d = fnp.zeros_like(k4_target)
    else:
        # Reread the stored gated factors so the coordinate birth closes the
        # actual fp32 carrier diagonal, not the analytic gate approximation.
        inherited_d = _cp4_read_diagonal(cp4_lambda, inherited_u)
    inherited_a = _off_diagonal(
        gate2[:, None] * gate64[:, None] * gate64[None, :] * pre_a
    )
    inherited_b = _off_diagonal(
        gate2[:, None] * gate2[None, :] * pre_b
    )

    birth_lambda = fnp.asarray(k4_target - inherited_d, dtype=fnp.float32)
    stored_birth = fnp.asarray(birth_lambda, dtype=fnp.float64)
    candidate_lambda = fnp.concatenate([cp4_lambda, birth_lambda])
    candidate_u = fnp.concatenate([inherited_u, identity], axis=1)
    candidate_planes = (inherited_d + stored_birth, inherited_a, inherited_b)
    if int(candidate_lambda.shape[0]) <= _CP4_CAP:
        return candidate_lambda, candidate_u, candidate_planes

    norm2 = fnp.sum(candidate_u * candidate_u, axis=0)
    energy = (
        candidate_lambda * candidate_lambda
        * (norm2 * norm2)
        * (norm2 * norm2)
    )
    order = fnp.argsort(-energy, stable=True)
    keep = order[:_CP4_CAP]
    drop = order[_CP4_CAP:]
    retained_lambda = candidate_lambda[keep]
    retained_u = candidate_u[:, keep]
    dropped_planes = _cp4_read_planes(
        candidate_lambda[drop],
        candidate_u[:, drop],
    )
    retained_planes = (
        candidate_planes[0] - dropped_planes[0],
        _off_diagonal(candidate_planes[1] - dropped_planes[1]),
        _off_diagonal(candidate_planes[2] - dropped_planes[2]),
    )
    return retained_lambda, retained_u, retained_planes


def _push_cubic(star_h, weight32):
    diagonal = fnp.diag(star_h)
    directed = _off_diagonal(star_h)
    weight2 = weight32 * weight32
    transported = directed @ weight32
    pushed_diagonal = fnp.sum(
        diagonal[:, None] * weight2 * weight32
        + 3.0 * weight2 * transported,
        axis=0,
    )
    pushed_k21 = (
        fnp.transpose(weight2) @ (diagonal[:, None] * weight32)
        + 2.0 * fnp.transpose(weight32 * transported) @ weight32
        + fnp.transpose(weight2) @ transported
    )
    return fnp.asarray(
        _replace_diagonal(pushed_k21, pushed_diagonal),
        dtype=fnp.float64,
    )


def _push_cubic_diagonal(star_h, weight32):
    """W16 specialization: one residual-star GEMM and only pushed K3."""
    diagonal = fnp.diag(star_h)
    directed = _off_diagonal(star_h)
    weight2 = weight32 * weight32
    transported = directed @ weight32
    return fnp.asarray(
        fnp.sum(
            diagonal[:, None] * weight2 * weight32
            + 3.0 * weight2 * transported,
            axis=0,
        ),
        dtype=fnp.float64,
    )


def _push_quartic(star_h, pairing_alpha, weight32, pre_covariance):
    diagonal = fnp.diag(star_h)
    directed = _off_diagonal(star_h)
    weight2 = weight32 * weight32
    weight3 = weight2 * weight32
    transported = directed @ weight32
    mixed = fnp.transpose(weight2) @ (weight32 * transported)

    star_diagonal = fnp.sum(
        diagonal[:, None] * weight2 * weight2
        + 4.0 * weight3 * transported,
        axis=0,
    )
    star_k31 = (
        fnp.transpose(weight3) @ (diagonal[:, None] * weight32)
        + fnp.transpose(weight3) @ transported
        + 3.0 * fnp.transpose(weight2 * transported) @ weight32
    )
    star_k22 = (
        fnp.transpose(weight2) @ (diagonal[:, None] * weight2)
        + 2.0 * (mixed + fnp.transpose(mixed))
    )

    star_diagonal64 = fnp.asarray(star_diagonal, dtype=fnp.float64)
    star_k3164 = fnp.asarray(star_k31, dtype=fnp.float64)
    star_k2264 = fnp.asarray(star_k22, dtype=fnp.float64)
    variance = fnp.diag(pre_covariance)
    pairing_diagonal = 3.0 * pairing_alpha * variance * variance
    pairing_k31 = 3.0 * pairing_alpha * variance[:, None] * pre_covariance
    pairing_k22 = pairing_alpha * (
        fnp.outer(variance, variance) + 2.0 * pre_covariance * pre_covariance
    )
    return (
        star_diagonal64 + pairing_diagonal,
        _off_diagonal(star_k3164 + pairing_k31),
        _off_diagonal(star_k2264 + pairing_k22),
    )


def _push_quartic_diagonal(star_h, pairing_alpha, weight32, pre_covariance):
    """W16 specialization: one residual-star GEMM and only pushed K4."""
    diagonal = fnp.diag(star_h)
    directed = _off_diagonal(star_h)
    weight2 = weight32 * weight32
    weight3 = weight2 * weight32
    transported = directed @ weight32
    star_diagonal = fnp.asarray(
        fnp.sum(
            diagonal[:, None] * weight2 * weight2
            + 4.0 * weight3 * transported,
            axis=0,
        ),
        dtype=fnp.float64,
    )
    variance = fnp.diag(pre_covariance)
    return star_diagonal + 3.0 * pairing_alpha * variance * variance


def _assemble_quartic_provider(host_planes, provider_planes):
    """Add the active coherent CP4 provider exactly once before Q32."""
    if provider_planes is None:
        return host_planes
    return (
        host_planes[0] + provider_planes[0],
        host_planes[1] + provider_planes[1],
        host_planes[2] + provider_planes[2],
    )


def _quartic_residual(k4, k31, k22, provider_planes):
    """Subtract the active CP4 planes once before the H4 residual fit."""
    if provider_planes is None:
        return k4, k31, k22
    return (
        k4 - provider_planes[0],
        k31 - provider_planes[1],
        k22 - provider_planes[2],
    )


def _reset_stars(
    k21_target,
    cp_post_k21,
    k4,
    k31,
    k22,
    covariance,
    quartic_provider_planes=None,
):
    cubic = fnp.asarray(k21_target - cp_post_k21, dtype=fnp.float32)

    k4, k31, k22 = _quartic_residual(
        k4,
        k31,
        k22,
        quartic_provider_planes,
    )
    variance = fnp.diag(covariance)
    pairing_plane = _off_diagonal(
        fnp.outer(variance, variance) + 2.0 * covariance * covariance
    )
    symmetric_k22 = 0.5 * (k22 + fnp.transpose(k22))
    # For symmetric operands the full off-diagonal inner products are exactly
    # twice their strict-upper counterparts, so their ratio is identical.
    numerator = fnp.sum(symmetric_k22 * pairing_plane)
    denominator = fnp.sum(pairing_plane * pairing_plane)
    pairing_alpha = numerator / denominator
    quartic_diagonal = k4 - 3.0 * pairing_alpha * variance * variance
    quartic_directed = _off_diagonal(
        k31 - 3.0 * pairing_alpha * variance[:, None] * covariance
    )
    quartic = fnp.asarray(
        quartic_directed + fnp.diag(quartic_diagonal),
        dtype=fnp.float32,
    )
    return cubic, quartic, pairing_alpha


def _exact_first_relu(pre_covariance):
    variance = fnp.maximum(fnp.diag(pre_covariance), _VAR_FLOOR)
    sigma = fnp.sqrt(variance)
    correlation = fnp.clip(
        pre_covariance / fnp.outer(sigma, sigma),
        -1.0,
        1.0,
    )
    root = fnp.sqrt(fnp.maximum(1.0 - correlation * correlation, 0.0))
    raw_second = (
        fnp.outer(sigma, sigma)
        * (root + (_PI - fnp.arccos(correlation)) * correlation)
        * _INV_TWO_PI
    )
    mean = sigma * _INV_SQRT_2PI
    covariance = raw_second - fnp.outer(mean, mean)
    return mean, covariance


def _initialize_h1(pre_covariance, identity32):
    width = int(pre_covariance.shape[0])
    pre_mean = fnp.zeros(width, dtype=fnp.float64)
    mean, covariance = _exact_first_relu(pre_covariance)
    coefficients, gaussian_raw, sigma, alpha, pdf = _q38_coefficients(
        pre_mean,
        fnp.diag(pre_covariance),
    )
    context = _q32_context(coefficients, sigma, pre_covariance)
    raw2 = gaussian_raw[2]
    raw3 = gaussian_raw[3]
    raw4 = gaussian_raw[4]
    m11 = covariance + fnp.outer(mean, mean)
    zero_vector = fnp.zeros(width, dtype=fnp.float64)
    zero_plane = fnp.zeros((width, width), dtype=fnp.float64)
    del alpha, pdf

    m21 = _replace_diagonal(_gaussian_pair(2, 1, context), raw3)
    m31 = _replace_diagonal(_gaussian_pair(3, 1, context), raw4)
    m22 = _replace_diagonal(_gaussian_pair(2, 2, context), raw4)
    k21, k4, k31, k22 = _connected_planes(
        mean, covariance, m11, m21, m31, m22, raw2, raw3, raw4
    )

    gate = flops.stats.norm.cdf(pre_mean / sigma)
    birth_lambda = fnp.exp(-0.5 * (pre_mean / sigma) ** 2) * _INV_SQRT_2PI / sigma
    cp_post_k21 = _cp_post_read(zero_plane, gate, pre_covariance, birth_lambda)
    cp_lambda, cp_p, cp_q = _cp_grow(
        fnp.zeros(0, dtype=fnp.float32),
        fnp.zeros((width, 0), dtype=fnp.float32),
        fnp.zeros((width, 0), dtype=fnp.float32),
        gate,
        pre_covariance,
        birth_lambda,
        identity32,
    )
    cp4_lambda, cp4_u, cp4_provider_planes = _cp4_reset(
        fnp.zeros(0, dtype=fnp.float32),
        fnp.zeros((width, 0), dtype=fnp.float32),
        gate,
        (zero_vector, zero_plane, zero_plane),
        k4,
        identity32,
    )
    cubic, quartic, pairing_alpha = _reset_stars(
        k21,
        cp_post_k21,
        k4,
        k31,
        k22,
        covariance,
        cp4_provider_planes,
    )
    return (
        mean,
        covariance,
        cp_lambda,
        cp_p,
        cp_q,
        cp4_lambda,
        cp4_u,
        cubic,
        quartic,
        pairing_alpha,
    )


def _full_relu_reset(
    pre_mean,
    pre_covariance,
    pre_k21,
    pre_k4,
    pre_k31,
    pre_k22,
    coefficients,
    gaussian_raw,
    sigma,
    alpha,
    pdf,
):
    k3 = fnp.diag(pre_k21)
    raw1, raw2, raw3, raw4 = _univariate_edgeworth(
        coefficients,
        gaussian_raw,
        sigma,
        alpha,
        pdf,
        k3,
        pre_k4,
    )
    context = _q32_context(coefficients, sigma, pre_covariance)
    m11 = _replace_diagonal(
        _pair_edgeworth(1, 1, context, k3, pre_k21, pre_k4, pre_k31, pre_k22),
        raw2,
    )
    m21 = _replace_diagonal(
        _pair_edgeworth(2, 1, context, k3, pre_k21, pre_k4, pre_k31, pre_k22),
        raw3,
    )
    m31 = _replace_diagonal(
        _pair_edgeworth(3, 1, context, k3, pre_k21, pre_k4, pre_k31, pre_k22),
        raw4,
    )
    m22 = _replace_diagonal(
        _pair_edgeworth(2, 2, context, k3, pre_k21, pre_k4, pre_k31, pre_k22),
        raw4,
    )

    raw_covariance = m11 - fnp.outer(raw1, raw1)
    variance = fnp.maximum(raw2 - raw1 * raw1, _VAR_FLOOR)
    covariance = _repair_covariance(raw_covariance, variance)
    m11 = covariance + fnp.outer(raw1, raw1)
    k21, k4, k31, k22 = _connected_planes(
        raw1, covariance, m11, m21, m31, m22, raw2, raw3, raw4
    )
    return raw1, covariance, k21, k4, k31, k22, alpha, pdf, sigma


def _predict(mlp: MLP):
    width = mlp.width
    identity32 = fnp.eye(width, dtype=fnp.float32)
    first_weight64 = fnp.asarray(mlp.weights[0], dtype=fnp.float64)
    first_pre_covariance = fnp.transpose(first_weight64) @ first_weight64
    (
        mean,
        covariance,
        cp_lambda,
        cp_p,
        cp_q,
        cp4_lambda,
        cp4_u,
        cubic,
        quartic,
        pairing_alpha,
    ) = _initialize_h1(first_pre_covariance, identity32)
    rows = [mean]

    for layer_index in range(1, mlp.depth):
        weight32 = fnp.asarray(mlp.weights[layer_index], dtype=fnp.float32)
        weight64 = fnp.asarray(mlp.weights[layer_index], dtype=fnp.float64)
        pre_mean = fnp.transpose(weight64) @ mean
        pre_covariance = (
            fnp.transpose(weight64) @ covariance
        ) @ weight64

        cp_lambda, pushed_p, pushed_q = _cp_push(
            cp_lambda, cp_p, cp_q, weight32
        )
        cp4_lambda, pushed_u = _cp4_push(cp4_lambda, cp4_u, weight32)
        if layer_index + 1 == mlp.depth:
            final_k3 = (
                _cp_k3(cp_lambda, pushed_p, pushed_q)
                + _push_cubic_diagonal(cubic, weight32)
            )
            final_k4 = (
                _push_quartic_diagonal(
                    quartic,
                    pairing_alpha,
                    weight32,
                    pre_covariance,
                )
                + _cp4_read_diagonal(cp4_lambda, pushed_u)
            )
            coefficients, gaussian_raw, sigma, alpha, pdf = _q38_coefficients(
                pre_mean,
                fnp.diag(pre_covariance),
            )
            final_mean = _univariate_edgeworth(
                coefficients,
                gaussian_raw,
                sigma,
                alpha,
                pdf,
                final_k3,
                final_k4,
            )[0]
            rows.append(final_mean)
            break

        cp_pre_k21 = _cp_read(cp_lambda, pushed_p, pushed_q)
        cubic_pre_k21 = _push_cubic(cubic, weight32)
        pre_k21 = cp_pre_k21 + cubic_pre_k21
        cp4_pre_provider_planes = _cp4_read_planes(cp4_lambda, pushed_u)
        pre_k4, pre_k31, pre_k22 = _assemble_quartic_provider(
            _push_quartic(
                quartic,
                pairing_alpha,
                weight32,
                pre_covariance,
            ),
            cp4_pre_provider_planes,
        )

        coefficients, gaussian_raw, sigma, alpha, pdf = _q38_coefficients(
            pre_mean,
            fnp.diag(pre_covariance),
        )
        (
            mean,
            covariance,
            k21_target,
            k4_target,
            k31_target,
            k22_target,
            alpha,
            pdf,
            sigma,
        ) = _full_relu_reset(
            pre_mean,
            pre_covariance,
            pre_k21,
            pre_k4,
            pre_k31,
            pre_k22,
            coefficients,
            gaussian_raw,
            sigma,
            alpha,
            pdf,
        )
        rows.append(mean)

        gate = flops.stats.norm.cdf(alpha)
        birth_lambda = pdf / sigma
        cp_post_k21 = _cp_post_read(
            cp_pre_k21,
            gate,
            pre_covariance,
            birth_lambda,
        )
        cp4_lambda, cp4_u, cp4_post_provider_planes = _cp4_reset(
            cp4_lambda,
            pushed_u,
            gate,
            cp4_pre_provider_planes,
            k4_target,
            identity32,
        )
        cubic, quartic, pairing_alpha = _reset_stars(
            k21_target,
            cp_post_k21,
            k4_target,
            k31_target,
            k22_target,
            covariance,
            cp4_post_provider_planes,
        )
        cp_lambda, cp_p, cp_q = _cp_grow(
            cp_lambda,
            pushed_p,
            pushed_q,
            gate,
            pre_covariance,
            birth_lambda,
            identity32,
        )

    output = fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)
    if not bool(fnp.all(fnp.isfinite(output))):
        raise FloatingPointError("FB-EC4-32 produced a nonfinite reported mean")
    return output


class Estimator(BaseEstimator):
    """Source-only FB-EC4-32 closure; no target-dependent state exists."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        return _predict(mlp)
