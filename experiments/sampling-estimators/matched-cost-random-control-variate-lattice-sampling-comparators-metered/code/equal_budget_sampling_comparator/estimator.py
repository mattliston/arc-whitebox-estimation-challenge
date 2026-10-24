"""Target-free equal-bill sampling comparators for Phase 2 P2R0G.

The three public classes deliberately share the same forward spelling and
return every layer mean.  Their sample counts are fixed before any Phase 2
target is opened and are chosen to sit at or immediately below the frozen
CP1792 flopscope-0.12 bill.

``EstimatorIID``
    Plain seeded IID Gaussian Monte Carlo.
``EstimatorRQMC``
    A nested base-two rank-1 lattice with independent continuous random shifts.
``EstimatorAffineCV``
    IID Monte Carlo with a coupled affine path whose expectation is known
    exactly.  The diagonal-Gaussian quantities choose the affine path but do
    not need to be correct for the control's expectation identity to hold.

No target, score, Phase-1 checkpoint, fitted coefficient, or dataset identity
is imported or embedded here.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP


_IID_SAMPLES = 6_372
_RQMC_BLOCKS = (2_048, 2_048, 2_048, 128, 32, 16, 8, 4, 2)
_AFFINE_CV_SAMPLES = 3_183
_VAR_FLOOR = 1.0e-20
_PPF_EPS = 2.0**-32
_UINT32_SCALE = 2.0**-32
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _seed(mlp: MLP, lane: int) -> int:
    """Domain-separate estimator randomness from the grader-supplied seed."""

    return (int(mlp.seed) ^ int(lane)) & ((1 << 63) - 1)


def _forward_layer_means(samples, weights):
    """Propagate one sample bank and retain float64 reductions only."""

    rows = []
    state = samples
    for weights_raw in weights:
        weights32 = fnp.asarray(weights_raw, dtype=fnp.float32)
        state = fnp.maximum(state @ weights32, 0.0)
        rows.append(fnp.mean(state, axis=0, dtype=fnp.float64))
    return fnp.stack(rows, axis=0)


def _reverse_bits_u32(value):
    """Vectorized 32-bit reversal used by the nested lattice prefixes."""

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
    """Produce a fixed source-only odd vector with a 32-bit avalanche map."""

    value = fnp.arange(width, dtype=fnp.uint32) + fnp.uint32(0x9E3779B9)
    value = fnp.bitwise_xor(value, fnp.right_shift(value, 16))
    value = value * fnp.uint32(0x7FEB352D)
    value = fnp.bitwise_xor(value, fnp.right_shift(value, 15))
    value = value * fnp.uint32(0x846CA68B)
    value = fnp.bitwise_xor(value, fnp.right_shift(value, 16))
    return fnp.bitwise_or(value, fnp.uint32(1))


def _shifted_lattice_normals(count: int, width: int, rng):
    """One unbiased random-shifted, nested base-two lattice replicate."""

    indices = fnp.arange(count, dtype=fnp.uint32)
    reversed_indices = _reverse_bits_u32(indices)
    multipliers = _odd_lattice_multipliers(width)
    # uint32 multiplication intentionally supplies arithmetic modulo 2**32.
    residues = reversed_indices[:, None] * multipliers[None, :]
    unit = fnp.asarray(residues, dtype=fnp.float64) * _UINT32_SCALE
    shift = rng.uniform(0.0, 1.0, size=(1, width))
    unit = fnp.remainder(unit + shift, 1.0)
    unit = fnp.maximum(fnp.minimum(unit, 1.0 - _PPF_EPS), _PPF_EPS)
    return fnp.asarray(flops.stats.norm.ppf(unit), dtype=fnp.float32)


class EstimatorIID(BaseEstimator):
    """Plain IID reference at the frozen CP1792 comparison bill."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        rng = fnp.random.default_rng(_seed(mlp, 0x31A5C85C97CB3127))
        samples = rng.standard_normal(
            (_IID_SAMPLES, mlp.width),
            dtype=fnp.float32,
        )
        rows = _forward_layer_means(samples, mlp.weights)
        return fnp.asarray(rows, dtype=fnp.float32)


class EstimatorRQMC(BaseEstimator):
    """Independent random shifts of frozen nested rank-1 lattice blocks."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        rng = fnp.random.default_rng(_seed(mlp, 0x6C8E9CF570932BD5))
        weighted_sum = fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float64)
        total = 0
        for count in _RQMC_BLOCKS:
            samples = _shifted_lattice_normals(count, mlp.width, rng)
            rows = _forward_layer_means(samples, mlp.weights)
            weighted_sum = weighted_sum + float(count) * rows
            total += count
        return fnp.asarray(weighted_sum / float(total), dtype=fnp.float32)


class EstimatorAffineCV(BaseEstimator):
    """IID estimator with an exact-expectation coupled affine control path."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        width = mlp.width
        rng = fnp.random.default_rng(_seed(mlp, 0x15D14E2F5A4D3C69))
        actual = rng.standard_normal(
            (_AFFINE_CV_SAMPLES, width),
            dtype=fnp.float32,
        )
        affine = actual.copy()
        anchor_mean = fnp.zeros(width, dtype=fnp.float32)
        anchor_variance = fnp.ones(width, dtype=fnp.float32)
        rows = []

        for weights_raw in mlp.weights:
            weights = fnp.asarray(weights_raw, dtype=fnp.float32)
            weights2 = weights * weights
            pre_anchor_mean = anchor_mean @ weights
            pre_anchor_variance = fnp.maximum(anchor_variance @ weights2, _VAR_FLOOR)
            pre_anchor_sigma = fnp.sqrt(pre_anchor_variance)
            alpha = pre_anchor_mean / pre_anchor_sigma
            pdf = fnp.exp(-0.5 * alpha * alpha) / _SQRT_2PI
            gate = fnp.asarray(flops.stats.norm.cdf(alpha), dtype=fnp.float32)
            next_anchor_mean = pre_anchor_sigma * pdf + pre_anchor_mean * gate
            next_anchor_second = (
                (pre_anchor_variance + pre_anchor_mean * pre_anchor_mean) * gate
                + pre_anchor_mean * pre_anchor_sigma * pdf
            )
            next_anchor_variance = fnp.maximum(
                next_anchor_second - next_anchor_mean * next_anchor_mean,
                _VAR_FLOOR,
            )

            pushed = fnp.concatenate((actual, affine), axis=0) @ weights
            next_actual = fnp.maximum(pushed[:_AFFINE_CV_SAMPLES], 0.0)
            next_affine = next_anchor_mean + gate * (
                pushed[_AFFINE_CV_SAMPLES:] - pre_anchor_mean
            )
            correction = fnp.mean(
                next_actual - next_affine,
                axis=0,
                dtype=fnp.float64,
            )
            rows.append(fnp.asarray(next_anchor_mean, dtype=fnp.float64) + correction)

            actual = next_actual
            affine = next_affine
            anchor_mean = fnp.asarray(next_anchor_mean, dtype=fnp.float32)
            anchor_variance = fnp.asarray(next_anchor_variance, dtype=fnp.float32)

        return fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)


class Estimator(EstimatorRQMC):
    """Neutral default for interface tests; promotion requires sealed scoring."""
