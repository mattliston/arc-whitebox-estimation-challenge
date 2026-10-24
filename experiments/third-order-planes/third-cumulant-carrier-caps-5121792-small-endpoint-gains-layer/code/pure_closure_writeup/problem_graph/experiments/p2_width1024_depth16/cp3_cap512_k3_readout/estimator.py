"""Phase 2 analytic Hermite-10 host with a capped CP3 K3 readout.

The float64 mean/covariance host is the source-audited Phase 2 analytic
baseline.  A separate float32 CP state carries a first-order approximation to
the third cumulant.  Only its diagonal K3 read is used, as an open-loop
Edgeworth correction to reported means.  It does not alter the analytic state.

There is deliberately no K21 materialization, K4 synthesis, TT residual,
future-weight geometry, fitted coefficient, or spectral PSD repair.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP


_HERMITE_ORDER = 10
_CP_CAP = 512
_VAR_FLOOR = 1.0e-30
_SQRT_2PI = math.sqrt(2.0 * math.pi)


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
    """Read diag(sum_r lambda_r * sym3(p_r, p_r, q_r))."""
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
    """Gate inherited factors, append the leading Gaussian star block, cap."""
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

    # Squared Frobenius norm of one lambda*sym3(p,p,q) component:
    # lambda^2 * (3 ||p||^4 ||q||^2 + 6 ||p||^2 <p,q>^2).
    # This replaces Phase 1's fitted 0.64/0.88 energy exponents with the exact
    # single-component tensor norm.  Cross-component cancellation is not used.
    p_norm2 = fnp.sum(post_p * post_p, axis=0)
    q_norm2 = fnp.sum(post_q * post_q, axis=0)
    p_dot_q = fnp.sum(post_p * post_q, axis=0)
    energy = (post_lambda * post_lambda) * (
        3.0 * p_norm2 * p_norm2 * q_norm2
        + 6.0 * p_norm2 * p_dot_q * p_dot_q
    )
    keep = fnp.argsort(-energy)[:_CP_CAP]
    return post_lambda[keep], post_p[:, keep], post_q[:, keep]


class Estimator(BaseEstimator):
    """Open-loop CP3 K3 correction on the PSD Hermite covariance host."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        width = mlp.width

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

            if int(cp_lambda.shape[0]) == 0:
                pushed_p = cp_p
                pushed_q = cp_q
                k3 = fnp.zeros(width, dtype=fnp.float32)
            else:
                pushed_p = fnp.transpose(weights32) @ cp_p
                pushed_q = fnp.transpose(weights32) @ cp_q
                k3 = _cp_k3(cp_lambda, pushed_p, pushed_q)

            # d^3/dm^3 E[ReLU(N(m,v))] = -a*phi(a)/v.
            # The CP correction is reported but not fed back into the analytic
            # mean/covariance recursion, preventing an untrained state shift.
            k3_64 = fnp.asarray(k3, dtype=fnp.float64)
            third_derivative = -(alpha * pdf) / pre_variance
            reported_mean = fnp.maximum(
                gaussian_mean + (k3_64 / 6.0) * third_derivative,
                0.0,
            )
            rows.append(reported_mean)

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
