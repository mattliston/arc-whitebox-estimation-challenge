"""CP1792/Hermite host plus an exact-mean nonlinear RQMC control.

This is a target-free Phase 2 resource/science probe.  It retains the current
cap-1792 CP3 and Hermite-10 analytic host, and spends only the remaining 0.1
score-floor headroom on two independently shifted lattice blocks.  The sampled
quantity is the residual between the true ReLU path and a four-ridge Gaussian
response surrogate.

For layer ``l``, let ``T = X V`` for four orthonormal, fixed Walsh directions
``V`` and let ``R_l`` be the host-gated linear response.  Conditional on the
network weights, every row of ``T`` is N(0, I_4) under the random shift, hence

    G_l = mu_l + T R_l

is marginally Gaussian with known diagonal variance.  If ``q_l = relu(G_l)``
and ``qbar_l = E[q_l]`` is the closed-form Gaussian ReLU mean, define

    S_l = A_l + q_l - qbar_l,

where ``A_l`` is the CP1792 reported mean.  Then E[S_l] = A_l exactly, and

    A_l + Q_N[f_l - S_l] = qbar_l + Q_N[f_l - q_l]

is unbiased under the randomized-lattice law.  The identity also exposes the
important limitation: the deterministic CP correction cancels from any
unbiased constant-shift control.  The Hermite host remains useful only through
the stochastic response trajectory (mu, covariance-derived gates, and R).
No target, score, fitted coefficient, or dataset identity is used.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP


_HERMITE_ORDER = 10
_CP_CAP = 1792
_RQMC_BLOCKS = (128, 32)
_RESPONSE_RANK = 4
_VAR_FLOOR = 1.0e-30
_PPF_EPS = 2.0**-32
_UINT32_SCALE = 2.0**-32
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _seed(mlp: MLP) -> int:
    """Domain-separate this probe from every existing sampling lane."""

    return (int(mlp.seed) ^ 0x3B7A49D15C6EF820) & ((1 << 63) - 1)


def _reverse_bits_u32(value):
    value = fnp.bitwise_or(
        fnp.bitwise_and(fnp.right_shift(value, 1), fnp.uint32(0x55555555)),
        fnp.left_shift(fnp.bitwise_and(value, fnp.uint32(0x55555555)), 1),
    )
    value = fnp.bitwise_or(
        fnp.bitwise_and(fnp.right_shift(value, 2), fnp.uint32(0x33333333)),
        fnp.left_shift(fnp.bitwise_and(value, fnp.uint32(0x33333333)), 2),
    )
    value = fnp.bitwise_or(
        fnp.bitwise_and(fnp.right_shift(value, 4), fnp.uint32(0x0F0F0F0F)),
        fnp.left_shift(fnp.bitwise_and(value, fnp.uint32(0x0F0F0F0F)), 4),
    )
    value = fnp.bitwise_or(
        fnp.bitwise_and(fnp.right_shift(value, 8), fnp.uint32(0x00FF00FF)),
        fnp.left_shift(fnp.bitwise_and(value, fnp.uint32(0x00FF00FF)), 8),
    )
    return fnp.bitwise_or(
        fnp.right_shift(value, 16),
        fnp.left_shift(value, 16),
    )


def _odd_lattice_multipliers(width: int):
    value = fnp.arange(width, dtype=fnp.uint32) + fnp.uint32(0x9E3779B9)
    value = fnp.bitwise_xor(value, fnp.right_shift(value, 16))
    value = value * fnp.uint32(0x7FEB352D)
    value = fnp.bitwise_xor(value, fnp.right_shift(value, 15))
    value = value * fnp.uint32(0x846CA68B)
    value = fnp.bitwise_xor(value, fnp.right_shift(value, 16))
    return fnp.bitwise_or(value, fnp.uint32(1))


def _shifted_lattice_normals(count: int, width: int, rng):
    """One independently shifted, nested base-two rank-1 lattice block."""

    indices = fnp.arange(count, dtype=fnp.uint32)
    reversed_indices = _reverse_bits_u32(indices)
    multipliers = _odd_lattice_multipliers(width)
    residues = reversed_indices[:, None] * multipliers[None, :]
    unit = fnp.asarray(residues, dtype=fnp.float64) * _UINT32_SCALE
    shift = rng.uniform(0.0, 1.0, size=(1, width))
    unit = fnp.remainder(unit + shift, 1.0)
    unit = fnp.maximum(fnp.minimum(unit, 1.0 - _PPF_EPS), _PPF_EPS)
    return fnp.asarray(flops.stats.norm.ppf(unit), dtype=fnp.float32)


def _walsh_directions(width: int):
    """Four fixed orthonormal Walsh columns for widths divisible by four."""

    if width % _RESPONSE_RANK != 0:
        raise ValueError("the four-ridge control requires width divisible by four")
    index = fnp.arange(width, dtype=fnp.uint32)
    scale = 1.0 / math.sqrt(float(width))
    column0 = fnp.ones(width, dtype=fnp.float64) * scale
    column1 = fnp.where(
        fnp.bitwise_and(index, fnp.uint32(1)) == 0,
        scale,
        -scale,
    )
    column2 = fnp.where(
        fnp.bitwise_and(index, fnp.uint32(2)) == 0,
        scale,
        -scale,
    )
    column3 = (column1 * column2) / scale
    return fnp.stack((column0, column1, column2, column3), axis=1)


def _hermite_coefficients(alpha, pdf, cdf):
    """Orthonormal-Hermite coefficients for ReLU(alpha + Z)."""

    coefficients = [pdf + alpha * cdf, cdf, pdf / math.sqrt(2.0)]
    he_previous = fnp.ones_like(alpha)
    he_current = -alpha
    factorial = 2.0
    for order in range(3, _HERMITE_ORDER + 1):
        factorial *= order
        coefficients.append(he_current * pdf / math.sqrt(factorial))
        he_previous, he_current = (
            he_current,
            (-alpha) * he_current - (order - 2) * he_previous,
        )
    return coefficients


def _cp_k3(cp_lambda, cp_p, cp_q):
    return 3.0 * ((cp_p * cp_p * cp_q) @ cp_lambda)


def _cp_grow(
    cp_lambda,
    pushed_p,
    pushed_q,
    gate,
    pre_covariance,
    sigma,
    density,
    identity,
):
    post_p = fnp.concatenate(
        [gate[:, None] * pushed_p, gate[:, None] * pre_covariance],
        axis=1,
    )
    post_q = fnp.concatenate(
        [gate[:, None] * pushed_q, identity],
        axis=1,
    )
    post_lambda = fnp.concatenate([cp_lambda, density / sigma])

    if int(post_lambda.shape[0]) <= _CP_CAP:
        return post_lambda, post_p, post_q

    p_norm2 = fnp.sum(post_p * post_p, axis=0)
    q_norm2 = fnp.sum(post_q * post_q, axis=0)
    p_dot_q = fnp.sum(post_p * post_q, axis=0)
    energy = (post_lambda * post_lambda) * (
        3.0 * p_norm2 * p_norm2 * q_norm2
        + 6.0 * p_norm2 * p_dot_q * p_dot_q
    )
    keep = fnp.argsort(-energy)[:_CP_CAP]
    return post_lambda[keep], post_p[:, keep], post_q[:, keep]


def _gaussian_relu_mean(location, response):
    """Exact E[relu(location + T @ response)] for T ~ N(0, I)."""

    variance = fnp.sum(response * response, axis=0)
    positive = variance > 0.0
    safe_variance = fnp.where(positive, variance, 1.0)
    sigma = fnp.sqrt(safe_variance)
    alpha = location / sigma
    pdf = fnp.exp(-0.5 * alpha * alpha) / _SQRT_2PI
    cdf = flops.stats.norm.cdf(alpha)
    gaussian = sigma * pdf + location * cdf
    return fnp.where(positive, gaussian, fnp.maximum(location, 0.0))


class Estimator(BaseEstimator):
    """Current CP1792 host plus a 160-point nonlinear residual RQMC lane."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        width = mlp.width
        rng = fnp.random.default_rng(_seed(mlp))
        blocks = [
            _shifted_lattice_normals(count, width, rng)
            for count in _RQMC_BLOCKS
        ]
        actual = fnp.concatenate(blocks, axis=0)

        directions = _walsh_directions(width)
        latent = fnp.asarray(actual, dtype=fnp.float64) @ directions
        response = fnp.transpose(directions)

        analytic_identity = fnp.eye(width, dtype=fnp.float64)
        cp_identity = fnp.eye(width, dtype=fnp.float32)
        mean = fnp.zeros(width, dtype=fnp.float64)
        covariance = analytic_identity
        cp_lambda = fnp.zeros(0, dtype=fnp.float32)
        cp_p = fnp.zeros((width, 0), dtype=fnp.float32)
        cp_q = fnp.zeros((width, 0), dtype=fnp.float32)
        rows = []

        for layer0, weights_raw in enumerate(mlp.weights):
            weights32 = fnp.asarray(weights_raw, dtype=fnp.float32)
            weights64 = fnp.asarray(weights_raw, dtype=fnp.float64)

            pre_mean = fnp.transpose(weights64) @ mean
            pre_covariance = fnp.einsum(
                "ij,ia,jb->ab",
                covariance,
                weights64,
                weights64,
            )
            pre_variance = fnp.maximum(fnp.diag(pre_covariance), _VAR_FLOOR)
            pre_sigma = fnp.sqrt(pre_variance)
            alpha = pre_mean / pre_sigma
            pdf = fnp.exp(-0.5 * alpha * alpha) / _SQRT_2PI
            cdf = flops.stats.norm.cdf(alpha)

            coefficients = _hermite_coefficients(alpha, pdf, cdf)
            gaussian_mean = pre_sigma * coefficients[0]

            response_pre = response @ weights64
            if int(cp_lambda.shape[0]) == 0:
                pushed_p = cp_p
                pushed_q = cp_q
                k3 = fnp.zeros(width, dtype=fnp.float32)
            else:
                pushed_p = fnp.transpose(weights32) @ cp_p
                pushed_q = fnp.transpose(weights32) @ cp_q
                k3 = _cp_k3(cp_lambda, pushed_p, pushed_q)

            k3_64 = fnp.asarray(k3, dtype=fnp.float64)
            third_derivative = -(alpha * pdf) / pre_variance
            reported_mean = fnp.maximum(
                gaussian_mean + (k3_64 / 6.0) * third_derivative,
                0.0,
            )

            pre_actual = actual @ weights32
            next_actual = fnp.maximum(pre_actual, 0.0)
            if layer0 == 0:
                # The first preactivation is itself exactly Gaussian.  Reuse
                # it as the control, making the layer-one residual identically
                # zero without another matrix multiply.
                proxy_post = fnp.asarray(next_actual, dtype=fnp.float64)
                proxy_expected = gaussian_mean
            else:
                proxy_pre = pre_mean[None, :] + latent @ response_pre
                proxy_post = fnp.maximum(proxy_pre, 0.0)
                proxy_expected = _gaussian_relu_mean(pre_mean, response_pre)
            surrogate = (
                reported_mean[None, :]
                + proxy_post
                - proxy_expected[None, :]
            )
            correction = fnp.mean(
                fnp.asarray(next_actual, dtype=fnp.float64) - surrogate,
                axis=0,
                dtype=fnp.float64,
            )
            rows.append(reported_mean + correction)

            correlation = pre_covariance / fnp.outer(pre_sigma, pre_sigma)
            covariance = fnp.zeros((width, width), dtype=fnp.float64)
            correlation_power = correlation
            for order in range(1, _HERMITE_ORDER + 1):
                coefficient = coefficients[order]
                covariance = (
                    covariance
                    + fnp.outer(coefficient, coefficient) * correlation_power
                )
                correlation_power = correlation_power * correlation
            covariance = covariance * fnp.outer(pre_sigma, pre_sigma)

            exact_variance = fnp.maximum(
                pre_variance
                * ((1.0 + alpha * alpha) * cdf + alpha * pdf)
                - gaussian_mean * gaussian_mean,
                _VAR_FLOOR,
            )
            diagonal_tail = fnp.maximum(
                exact_variance - fnp.diag(covariance),
                0.0,
            )
            covariance = covariance + fnp.diag(diagonal_tail)
            mean = gaussian_mean
            actual = next_actual
            response = response_pre * cdf[None, :]

            if layer0 + 1 < mlp.depth:
                cp_lambda, cp_p, cp_q = _cp_grow(
                    cp_lambda,
                    pushed_p,
                    pushed_q,
                    fnp.asarray(cdf, dtype=fnp.float32),
                    fnp.asarray(pre_covariance, dtype=fnp.float32),
                    fnp.asarray(pre_sigma, dtype=fnp.float32),
                    fnp.asarray(pdf, dtype=fnp.float32),
                    cp_identity,
                )

        return fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)
