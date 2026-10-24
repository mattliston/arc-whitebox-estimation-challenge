"""Combined CP1792 host plus V3 observer paths for authoritative metering.

The embedded checkpoint is target-free and has an identically zero mean head,
so these classes exercise the complete local/recurrent deployment path while
returning the untouched CP1792 host prediction.  A fitted checkpoint may
replace only the frozen parameter literals after every V3 release/parity gate.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP


_HERMITE_ORDER = 10
_CP_CAP = 1792
_VAR_FLOOR = 1.0e-30
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_ENERGY_EPS = 2.0**-40
_MESSAGE_EPS = 2.0**-24
_NORM_SCALE_FLOOR = 2.0**-20
_LOCAL_CLIP = 8.0
_MESSAGE_CLIP = 5.0


def _sigmoid(value):
    return 1.0 / (1.0 + fnp.exp(-value))


def _silu(value):
    return value * _sigmoid(value)


def _hermite_coefficients(alpha, pdf, cdf):
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
        [gate[:, None] * pushed_p, gate[:, None] * pre_covariance], axis=1
    )
    post_q = fnp.concatenate([gate[:, None] * pushed_q, identity], axis=1)
    post_lambda = fnp.concatenate([cp_lambda, density / sigma])
    clean_diagnostic = fnp.asarray([0.0, 1.0, 1.0, 1.0], dtype=fnp.float64)
    if int(post_lambda.shape[0]) <= _CP_CAP:
        return post_lambda, post_p, post_q, fnp.asarray(1.0, dtype=fnp.float64), clean_diagnostic

    p_norm2 = fnp.sum(post_p * post_p, axis=0)
    q_norm2 = fnp.sum(post_q * post_q, axis=0)
    p_dot_q = fnp.sum(post_p * post_q, axis=0)
    energy = (post_lambda * post_lambda) * (
        3.0 * p_norm2 * p_norm2 * q_norm2
        + 6.0 * p_norm2 * p_dot_q * p_dot_q
    )
    keep = fnp.argsort(-energy)[:_CP_CAP]
    energy64 = fnp.asarray(energy, dtype=fnp.float64)
    total = fnp.sum(energy64)
    kept = fnp.sum(energy64[fnp.sort(keep)])
    fraction = fnp.where(
        total <= _ENERGY_EPS,
        fnp.asarray(1.0, dtype=fnp.float64),
        fnp.clip(kept / total, 0.0, 1.0),
    )
    diagnostic = fnp.stack((fnp.min(energy64), total, kept, fraction))
    return post_lambda[keep], post_p[:, keep], post_q[:, keep], fraction, diagnostic


def _edge_matrices(weights32, width):
    square = weights32 * weights32
    a1 = weights32 * (1.0 / math.sqrt(2.0))
    a2 = (square - (2.0 / width)) * math.sqrt(width / 8.0)
    a3 = (square * weights32) * (width / math.sqrt(120.0))
    a4 = (square * square - (12.0 / (width * width))) * (
        width ** 1.5 / math.sqrt(1536.0)
    )
    return tuple(fnp.asarray(value, dtype=fnp.float32) for value in (a1, a2, a3, a4))


def _meter_parameters():
    # Complete 12,041-parameter shape.  A nonzero candidate bias exercises
    # h1->h2 transport; the mean head is exactly zero for host byte parity.
    core1_weight = fnp.zeros((64, 56), dtype=fnp.float32)
    core1_bias = fnp.zeros(64, dtype=fnp.float32)
    core2_weight = fnp.zeros((64, 64), dtype=fnp.float32)
    core2_bias = fnp.zeros(64, dtype=fnp.float32)
    candidate_weight = fnp.zeros((32, 64), dtype=fnp.float32)
    candidate_bias = fnp.asarray(
        [0.1 if index % 2 == 0 else -0.1 for index in range(32)],
        dtype=fnp.float32,
    )
    gate_weight = fnp.zeros((32, 64), dtype=fnp.float32)
    gate_bias = fnp.zeros(32, dtype=fnp.float32)
    mean_weight = fnp.zeros((1, 64), dtype=fnp.float32)
    mean_bias = fnp.zeros(1, dtype=fnp.float32)
    source_logits = fnp.zeros(8, dtype=fnp.float32)
    centers = fnp.zeros((16, 19), dtype=fnp.float64)
    scales = fnp.ones((16, 19), dtype=fnp.float64)
    return (
        core1_weight,
        core1_bias,
        core2_weight,
        core2_bias,
        candidate_weight,
        candidate_bias,
        gate_weight,
        gate_bias,
        mean_weight,
        mean_bias,
        source_logits,
        centers,
        scales,
    )


def _raw_features(
    layer0,
    width,
    cp_rank,
    retained_fraction,
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
    retained_feature = fnp.ones(width, dtype=fnp.float64) * retained_fraction
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


def _standardize(raw, layer0, centers, scales):
    tail = fnp.clip(
        (raw[:, 1:] - centers[layer0][None, :]) / scales[layer0][None, :],
        -_LOCAL_CLIP,
        _LOCAL_CLIP,
    )
    depth = fnp.asarray(raw[:, :1], dtype=fnp.float32)
    return fnp.concatenate((depth, fnp.asarray(tail, dtype=fnp.float32)), axis=1)


def _messages(edges, h_previous):
    groups = []
    log_rms = []
    rhos = []
    for power in range(4):
        prior = h_previous[:, power * 8 : (power + 1) * 8]
        transported = fnp.transpose(edges[power]) @ prior
        mean_square = fnp.sum(transported * transported) / (transported.shape[0] * 8)
        rho = fnp.sqrt(mean_square + _MESSAGE_EPS)
        groups.append(fnp.clip(transported / rho, -_MESSAGE_CLIP, _MESSAGE_CLIP))
        log_rms.append(fnp.clip(fnp.log(rho), -8.0, 8.0))
        rhos.append(rho)
    return fnp.concatenate(groups, axis=1), fnp.stack(log_rms), fnp.stack(rhos)


def _gate_locals(local, source_gates):
    gate_vector = fnp.ones(20, dtype=fnp.float32)
    # Assignment is intentionally avoided: build the declared exact order.
    gate_vector = fnp.stack(
        (
            gate_vector[0],
            source_gates[1], source_gates[1],
            gate_vector[3], gate_vector[4], gate_vector[5], gate_vector[6], gate_vector[7],
            source_gates[1], source_gates[1], source_gates[1],
            gate_vector[11],
            source_gates[0],
            source_gates[2], source_gates[2], source_gates[2],
            source_gates[3], source_gates[3], source_gates[3], source_gates[3],
        )
    )
    return local * gate_vector[None, :]


def _observer_cell(local, messages, log_rms, parameters):
    (
        core1_weight,
        core1_bias,
        core2_weight,
        core2_bias,
        candidate_weight,
        candidate_bias,
        gate_weight,
        gate_bias,
        mean_weight,
        mean_bias,
        source_logits,
        _,
        _,
    ) = parameters
    source_gates = _sigmoid(source_logits)
    gated_local = _gate_locals(local, source_gates)
    gated_groups = []
    gated_logs = []
    for power in range(4):
        gate = source_gates[4 + power]
        gated_groups.append(messages[:, power * 8 : (power + 1) * 8] * gate)
        gated_logs.append(log_rms[power] * gate)
    broadcast_logs = fnp.broadcast_to(fnp.stack(gated_logs)[None, :], (local.shape[0], 4))
    cell_input = fnp.concatenate((gated_local, broadcast_logs, fnp.concatenate(gated_groups, axis=1)), axis=1)
    q1 = _silu(cell_input @ fnp.transpose(core1_weight) + core1_bias)
    q2 = _silu(q1 @ fnp.transpose(core2_weight) + core2_bias)
    candidate = fnp.tanh(q2 @ fnp.transpose(candidate_weight) + candidate_bias)
    update_gate = _sigmoid(q2 @ fnp.transpose(gate_weight) + gate_bias)
    hidden = fnp.tanh(messages + update_gate * (candidate - messages))
    unit = fnp.tanh(q2 @ fnp.transpose(mean_weight) + mean_bias)[:, 0]
    return cell_input, hidden, unit


def _one_scalar_latch(
    raw,
    standardized,
    edges,
    messages,
    log_rms,
    rhos,
    cell_input,
    hidden,
    unit,
    delta,
    prediction,
    sigma,
    cp_diagnostic,
    parameters,
):
    parameter_values = fnp.concatenate([fnp.ravel(value) for value in parameters[:11]])
    diagnostic_values = fnp.concatenate(
        [
            fnp.ravel(raw), fnp.ravel(standardized),
            *[fnp.ravel(edge) for edge in edges],
            fnp.ravel(messages), fnp.ravel(log_rms), fnp.ravel(cell_input),
            fnp.ravel(hidden), fnp.ravel(unit), fnp.ravel(delta), fnp.ravel(prediction),
            fnp.ravel(cp_diagnostic), parameter_values,
        ]
    )
    positive_values = fnp.concatenate((fnp.ravel(sigma), fnp.ravel(rhos)))
    checks = fnp.concatenate(
        (
            fnp.ravel(fnp.isfinite(diagnostic_values)),
            fnp.ravel(fnp.isfinite(positive_values)),
            fnp.ravel(positive_values > 0.0),
            fnp.ravel(cp_diagnostic[:1] >= 0.0),
        )
    )
    # Exactly one client-visible scalar per active layer.  Once false, the
    # Python branch permanently skips every later observer call.
    return bool(fnp.all(checks))


def _predict(mlp: MLP, *, recurrent: bool):
    width = mlp.width
    analytic_identity = fnp.eye(width, dtype=fnp.float64)
    cp_identity = fnp.eye(width, dtype=fnp.float32)
    mean = fnp.zeros(width, dtype=fnp.float64)
    covariance = analytic_identity
    cp_lambda = fnp.zeros(0, dtype=fnp.float32)
    cp_p = fnp.zeros((width, 0), dtype=fnp.float32)
    cp_q = fnp.zeros((width, 0), dtype=fnp.float32)
    retained_fraction = fnp.asarray(1.0, dtype=fnp.float64)
    cp_diagnostic = fnp.asarray([0.0, 1.0, 1.0, 1.0], dtype=fnp.float64)
    parameters = _meter_parameters()
    centers, scales = parameters[-2], parameters[-1]
    h_previous = fnp.zeros((width, 32), dtype=fnp.float32)
    observer_active = True
    rows = []

    for layer0, weights_raw in enumerate(mlp.weights):
        weights32 = fnp.asarray(weights_raw, dtype=fnp.float32)
        weights64 = fnp.asarray(weights_raw, dtype=fnp.float64)
        pre_mean = fnp.transpose(weights64) @ mean
        pre_covariance = fnp.einsum("ij,ia,jb->ab", covariance, weights64, weights64)
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
        host32 = fnp.asarray(reported_mean, dtype=fnp.float32)

        correlation = pre_covariance / fnp.outer(pre_sigma, pre_sigma)
        exact_variance = fnp.maximum(
            pre_variance * ((1.0 + alpha * alpha) * cdf + alpha * pdf)
            - gaussian_mean * gaussian_mean,
            _VAR_FLOOR,
        )
        edges = _edge_matrices(weights32, width)

        if observer_active:
            raw = _raw_features(
                layer0, width, int(cp_lambda.shape[0]), retained_fraction,
                alpha, pre_sigma, cdf, pdf, coefficients, gaussian_mean,
                exact_variance, pre_variance, k3, reported_mean, correlation, edges,
            )
            standardized = _standardize(raw, layer0, centers, scales)
            if layer0 == 0 or not recurrent:
                messages = fnp.zeros((width, 32), dtype=fnp.float32)
                log_rms = fnp.zeros(4, dtype=fnp.float32)
                rhos = fnp.ones(4, dtype=fnp.float32)
            else:
                messages, log_rms, rhos = _messages(edges, h_previous)
            cell_input, hidden, unit = _observer_cell(standardized, messages, log_rms, parameters)
            if layer0 == 0:
                delta = fnp.zeros(width, dtype=fnp.float32)
            else:
                delta = fnp.asarray(pre_sigma, dtype=fnp.float32) * (1.0 / 32.0) * unit
            candidate_row = fnp.maximum(host32 + delta, fnp.asarray(0.0, dtype=fnp.float32))
            current_valid = _one_scalar_latch(
                raw, standardized, edges, messages, log_rms, rhos, cell_input,
                hidden, unit, delta, candidate_row, pre_sigma, cp_diagnostic, parameters,
            )
            if current_valid:
                rows.append(candidate_row)
                h_previous = hidden if recurrent else fnp.zeros_like(h_previous)
            else:
                observer_active = False
                h_previous = fnp.zeros_like(h_previous)
                rows.append(host32)
        else:
            rows.append(host32)

        covariance = fnp.zeros((width, width), dtype=fnp.float64)
        correlation_power = correlation
        for order in range(1, _HERMITE_ORDER + 1):
            coefficient = coefficients[order]
            covariance = covariance + fnp.outer(coefficient, coefficient) * correlation_power
            correlation_power = correlation_power * correlation
        covariance = covariance * fnp.outer(pre_sigma, pre_sigma)
        diagonal_tail = fnp.maximum(exact_variance - fnp.diag(covariance), 0.0)
        covariance = covariance + fnp.diag(diagonal_tail)
        mean = gaussian_mean

        if layer0 + 1 < mlp.depth:
            cp_lambda, cp_p, cp_q, retained_fraction, cp_diagnostic = _cp_grow(
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


class EstimatorRecurrent(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        return _predict(mlp, recurrent=True)


class EstimatorLocal(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        return _predict(mlp, recurrent=False)
