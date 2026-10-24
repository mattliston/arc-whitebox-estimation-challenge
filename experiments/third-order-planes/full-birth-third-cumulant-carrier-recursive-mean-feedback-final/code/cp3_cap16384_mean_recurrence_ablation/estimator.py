"""Target-blind cap-16384 mean-recurrence ablation.

Both candidates retain the exact full-birth CP rank schedule and the existing
Hermite covariance/CP update laws.  The first arm carries only the layer-2
K3-corrected reported mean into W3.  The second carries every reported mean
into the next layer.  The open-loop cap-16384 source remains a separate
control and is not modified by this file.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP


_HERMITE_ORDER = 10
_CP_CAP = 16384
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
    cap,
):
    """Gate inherited factors, append the Gaussian-star block, then cap."""
    post_p = fnp.concatenate(
        [gate[:, None] * pushed_p, gate[:, None] * pre_covariance],
        axis=1,
    )
    post_q = fnp.concatenate(
        [gate[:, None] * pushed_q, identity],
        axis=1,
    )
    post_lambda = fnp.concatenate([cp_lambda, density / sigma])

    if int(post_lambda.shape[0]) <= cap:
        return post_lambda, post_p, post_q

    p_norm2 = fnp.sum(post_p * post_p, axis=0)
    q_norm2 = fnp.sum(post_q * post_q, axis=0)
    p_dot_q = fnp.sum(post_p * post_q, axis=0)
    energy = (post_lambda * post_lambda) * (
        3.0 * p_norm2 * p_norm2 * q_norm2
        + 6.0 * p_norm2 * p_dot_q * p_dot_q
    )
    keep = fnp.argsort(-energy)[:cap]
    return post_lambda[keep], post_p[:, keep], post_q[:, keep]


def _predict(mlp: MLP, mean_mode: str) -> fnp.ndarray:
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
        if mean_mode == "all_reported" or (
            mean_mode == "layer2_only" and layer0 == 1
        ):
            mean = reported_mean
        else:
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
                _CP_CAP,
            )

    return fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)


class _MeanRecurrenceEstimator(BaseEstimator):
    MEAN_MODE = ""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        return _predict(mlp, self.MEAN_MODE)


class EstimatorLayer2MeanCarry(_MeanRecurrenceEstimator):
    """Carry only h2's reported mean into W3, then resume Gaussian means."""

    MEAN_MODE = "layer2_only"


class EstimatorRecursiveMeanCarry(_MeanRecurrenceEstimator):
    """Carry every K3-corrected reported mean into the following layer."""

    MEAN_MODE = "all_reported"
