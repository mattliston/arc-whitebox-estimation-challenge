"""CP16384 recursive-mean host with deployment-faithful raw20/sigma tape.

The prediction recurrence is the exact spelling from
``cp3_cap16384_mean_recurrence_ablation/estimator.py``.  This D3
instrumentation adds only target-blind per-layer observables computed from
the same live CP16384 state.  ``predict`` still returns only the host, while
``predict_with_tape`` exposes the observables needed by the frozen dynamic4
correction.
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
_ENERGY_EPS = 2.0**-40


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


def _edge_matrices(weights32, width):
    square = weights32 * weights32
    a1 = weights32 * (1.0 / math.sqrt(2.0))
    a2 = (square - (2.0 / width)) * math.sqrt(width / 8.0)
    a3 = (square * weights32) * (width / math.sqrt(120.0))
    a4 = (square * square - (12.0 / (width * width))) * (
        width**1.5 / math.sqrt(1536.0)
    )
    return tuple(
        fnp.asarray(value, dtype=fnp.float32) for value in (a1, a2, a3, a4)
    )


def _raw_features(
    layer0,
    width,
    cp_rank,
    alpha,
    sigma,
    cdf,
    pdf,
    coefficients,
    gaussian_mean,
    exact_variance,
    pre_variance,
    k3,
    reported,
    correlation,
    edges,
):
    centered_log_sigma = fnp.log(sigma) - fnp.sum(fnp.log(sigma)) / width
    tail = fnp.zeros_like(alpha)
    total = fnp.zeros_like(alpha)
    for order in range(1, 11):
        square = coefficients[order] * coefficients[order]
        total = total + square
        if order >= 5:
            tail = tail + square
    hermite_tail = tail / fnp.maximum(total, _ENERGY_EPS)
    row_mean = fnp.sum(correlation, axis=1) / width
    row_rms = fnp.sqrt(fnp.sum(correlation * correlation, axis=1) / width)
    row_abs = fnp.sum(fnp.abs(correlation), axis=1) / width
    depth32 = fnp.ones(width, dtype=fnp.float32) * (2.0 * layer0 / 15.0 - 1.0)
    rank_feature = fnp.full(width, cp_rank / float(_CP_CAP), dtype=fnp.float64)
    # Full-birth CP16384 enters layers 0..15 at ranks 0..15360 and never
    # truncates, so the exact retained fraction is one at every layer.
    retained_feature = fnp.ones(width, dtype=fnp.float64)
    values = (
        fnp.asarray(depth32, dtype=fnp.float64),
        rank_feature,
        retained_feature,
        alpha,
        centered_log_sigma,
        cdf,
        pdf,
        gaussian_mean / sigma,
        reported / sigma,
        (reported - gaussian_mean) / sigma,
        fnp.asarray(k3, dtype=fnp.float64) / (sigma * sigma * sigma),
        exact_variance / pre_variance,
        hermite_tail,
        row_mean,
        row_rms,
        row_abs,
        fnp.asarray(fnp.sum(edges[0], axis=0), dtype=fnp.float64),
        fnp.asarray(fnp.sum(edges[1], axis=0), dtype=fnp.float64),
        fnp.asarray(fnp.sum(edges[2], axis=0), dtype=fnp.float64),
        fnp.asarray(fnp.sum(edges[3], axis=0), dtype=fnp.float64),
    )
    return fnp.stack(values, axis=1)


def predict_with_tape(mlp: MLP):
    width = mlp.width

    analytic_identity = fnp.eye(width, dtype=fnp.float64)
    cp_identity = fnp.eye(width, dtype=fnp.float32)
    mean = fnp.zeros(width, dtype=fnp.float64)
    covariance = analytic_identity
    cp_lambda = fnp.zeros(0, dtype=fnp.float32)
    cp_p = fnp.zeros((width, 0), dtype=fnp.float32)
    cp_q = fnp.zeros((width, 0), dtype=fnp.float32)
    rows = []
    raw_rows = []
    sigma_rows = []

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

        edges = _edge_matrices(weights32, width)
        raw_rows.append(
            _raw_features(
                layer0,
                width,
                int(cp_lambda.shape[0]),
                alpha,
                pre_sigma,
                cdf,
                pdf,
                coefficients,
                gaussian_mean,
                exact_variance,
                pre_variance,
                k3,
                reported_mean,
                correlation,
                edges,
            )
        )
        sigma_rows.append(pre_sigma)

        mean = reported_mean
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

    prediction = fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)
    raw20 = fnp.asarray(fnp.stack(raw_rows, axis=0), dtype=fnp.float32)
    sigma = fnp.asarray(fnp.stack(sigma_rows, axis=0), dtype=fnp.float32)
    return prediction, raw20, sigma


class EstimatorInstrumentedRecursiveMeanCarry(BaseEstimator):
    """Deployment form: exact recursive host plus its own raw20/sigma tape."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        prediction, _, _ = predict_with_tape(mlp)
        return prediction
