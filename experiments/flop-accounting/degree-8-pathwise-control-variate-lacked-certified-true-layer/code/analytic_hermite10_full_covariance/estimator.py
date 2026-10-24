"""Sampling-free Phase 2 full-covariance Hermite/Mehler baseline.

The affine mean/covariance transport is exact.  The ReLU covariance is the
order-10 Mehler expansion of the bivariate Gaussian positive-part covariance,
with exact marginal mean and variance on every coordinate.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP


_VAR_FLOOR = 1.0e-30
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _hermite_coefficients(alpha, pdf, cdf, maximum_order: int):
    """Orthonormal-Hermite coefficients for ReLU(alpha + Z)."""
    coefficients = [pdf + alpha * cdf, cdf]
    if maximum_order == 1:
        return coefficients
    coefficients.append(pdf / math.sqrt(2.0))
    he_previous = fnp.ones_like(alpha)
    he_current = -alpha
    factorial = 2.0
    for order in range(3, maximum_order + 1):
        factorial *= order
        coefficients.append(he_current * pdf / math.sqrt(factorial))
        he_previous, he_current = (
            he_current,
            (-alpha) * he_current - (order - 2) * he_previous,
        )
    return coefficients


class _HermiteEstimator(BaseEstimator):
    """Shared implementation for the preregistered target-blind order ladder."""

    HERMITE_ORDER = 10

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        width = mlp.width
        identity = fnp.eye(width, dtype=fnp.float64)
        mean = fnp.zeros(width, dtype=fnp.float64)
        covariance = identity
        rows = []

        for weights_raw in mlp.weights:
            weights = fnp.asarray(weights_raw, dtype=fnp.float64)

            pre_mean = fnp.transpose(weights) @ mean
            pre_covariance = fnp.einsum(
                "ij,ia,jb->ab",
                covariance,
                weights,
                weights,
            )
            pre_variance = fnp.maximum(fnp.diag(pre_covariance), _VAR_FLOOR)
            pre_sigma = fnp.sqrt(pre_variance)
            alpha = pre_mean / pre_sigma
            pdf = fnp.exp(-0.5 * alpha * alpha) / _SQRT_2PI
            cdf = flops.stats.norm.cdf(alpha)

            coefficients = _hermite_coefficients(
                alpha,
                pdf,
                cdf,
                self.HERMITE_ORDER,
            )
            mean = fnp.maximum(pre_sigma * coefficients[0], 0.0)

            # The transported covariance is PSD, so this normalized matrix is
            # a genuine correlation matrix.  Avoid entrywise clipping: that
            # operation can destroy PSD and invalidate the Schur-power proof.
            correlation = pre_covariance / fnp.outer(pre_sigma, pre_sigma)
            covariance = fnp.zeros((width, width), dtype=fnp.float64)
            correlation_power = correlation
            for order in range(1, self.HERMITE_ORDER + 1):
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
                - mean * mean,
                _VAR_FLOOR,
            )
            # Every retained Schur-power term is PSD.  The omitted Hermite
            # tail is added on the diagonal, also PSD, while restoring the
            # exact one-dimensional ReLU variance.
            diagonal_tail = fnp.maximum(
                exact_variance - fnp.diag(covariance),
                0.0,
            )
            covariance = covariance + fnp.diag(diagonal_tail)
            rows.append(mean)

        return fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)


class EstimatorOrder1(_HermiteEstimator):
    HERMITE_ORDER = 1


class EstimatorOrder2(_HermiteEstimator):
    HERMITE_ORDER = 2


class EstimatorOrder4(_HermiteEstimator):
    HERMITE_ORDER = 4


class EstimatorOrder6(_HermiteEstimator):
    HERMITE_ORDER = 6


class EstimatorOrder10(_HermiteEstimator):
    HERMITE_ORDER = 10


class Estimator(EstimatorOrder10):
    """Default submission candidate: the order-10 member of the frozen ladder."""
