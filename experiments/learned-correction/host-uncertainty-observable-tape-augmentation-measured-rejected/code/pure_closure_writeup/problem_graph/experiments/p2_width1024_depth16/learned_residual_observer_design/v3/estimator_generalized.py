"""Additive zero-head Flopscope deployment body for V3 launch capacities.

The frozen H32/C64 source in ``estimator.py`` is imported as the exact host
and feature authority.  Only observer tensor/channel sizes are generalized.
No dataset, target, checkpoint, or fitted statistic is read.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import flopscope.numpy as fnp
from whestbench.domain import MLP


_SOURCE = Path(__file__).with_name("estimator.py")
_SPEC = importlib.util.spec_from_file_location("v3_observer_frozen_h32_body", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load frozen V3 host/feature body")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

_CP_CAP = 1536
_HIDDEN = 128
_CELL_WIDTH = 256
_POWERS = 4


def _meter_parameters():
    if _HIDDEN <= 0 or _HIDDEN % _POWERS or _CELL_WIDTH <= 0:
        raise RuntimeError("invalid generalized observer capacity")
    inputs = 20 + _POWERS + _HIDDEN
    core1_weight = fnp.zeros((_CELL_WIDTH, inputs), dtype=fnp.float32)
    core1_bias = fnp.zeros(_CELL_WIDTH, dtype=fnp.float32)
    core2_weight = fnp.zeros((_CELL_WIDTH, _CELL_WIDTH), dtype=fnp.float32)
    core2_bias = fnp.zeros(_CELL_WIDTH, dtype=fnp.float32)
    candidate_weight = fnp.zeros((_HIDDEN, _CELL_WIDTH), dtype=fnp.float32)
    candidate_bias = fnp.asarray(
        [0.1 if index % 2 == 0 else -0.1 for index in range(_HIDDEN)],
        dtype=fnp.float32,
    )
    gate_weight = fnp.zeros((_HIDDEN, _CELL_WIDTH), dtype=fnp.float32)
    gate_bias = fnp.zeros(_HIDDEN, dtype=fnp.float32)
    mean_weight = fnp.zeros((1, _CELL_WIDTH), dtype=fnp.float32)
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


def _messages(edges, h_previous):
    channels = _HIDDEN // _POWERS
    groups = []
    log_rms = []
    rhos = []
    for power in range(_POWERS):
        prior = h_previous[:, power * channels : (power + 1) * channels]
        transported = fnp.transpose(edges[power]) @ prior
        mean_square = fnp.sum(transported * transported) / (
            transported.shape[0] * channels
        )
        rho = fnp.sqrt(mean_square + _BASE._MESSAGE_EPS)
        groups.append(
            fnp.clip(transported / rho, -_BASE._MESSAGE_CLIP, _BASE._MESSAGE_CLIP)
        )
        log_rms.append(fnp.clip(fnp.log(rho), -8.0, 8.0))
        rhos.append(rho)
    return fnp.concatenate(groups, axis=1), fnp.stack(log_rms), fnp.stack(rhos)


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
    source_gates = _BASE._sigmoid(source_logits)
    gated_local = _BASE._gate_locals(local, source_gates)
    channels = _HIDDEN // _POWERS
    gated_groups = []
    gated_logs = []
    for power in range(_POWERS):
        gate = source_gates[4 + power]
        gated_groups.append(
            messages[:, power * channels : (power + 1) * channels] * gate
        )
        gated_logs.append(log_rms[power] * gate)
    broadcast_logs = fnp.broadcast_to(
        fnp.stack(gated_logs)[None, :], (local.shape[0], _POWERS)
    )
    cell_input = fnp.concatenate(
        (gated_local, broadcast_logs, fnp.concatenate(gated_groups, axis=1)),
        axis=1,
    )
    q1 = _BASE._silu(cell_input @ fnp.transpose(core1_weight) + core1_bias)
    q2 = _BASE._silu(q1 @ fnp.transpose(core2_weight) + core2_bias)
    candidate = fnp.tanh(q2 @ fnp.transpose(candidate_weight) + candidate_bias)
    update_gate = _BASE._sigmoid(q2 @ fnp.transpose(gate_weight) + gate_bias)
    hidden = fnp.tanh(messages + update_gate * (candidate - messages))
    unit = fnp.tanh(q2 @ fnp.transpose(mean_weight) + mean_bias)[:, 0]
    return cell_input, hidden, unit


def _predict(mlp: MLP, *, recurrent: bool):
    _BASE._CP_CAP = _CP_CAP
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
    h_previous = fnp.zeros((width, _HIDDEN), dtype=fnp.float32)
    observer_active = True
    rows = []

    for layer0, weights_raw in enumerate(mlp.weights):
        weights32 = fnp.asarray(weights_raw, dtype=fnp.float32)
        weights64 = fnp.asarray(weights_raw, dtype=fnp.float64)
        pre_mean = fnp.transpose(weights64) @ mean
        pre_covariance = fnp.einsum("ij,ia,jb->ab", covariance, weights64, weights64)
        pre_variance = fnp.maximum(fnp.diag(pre_covariance), _BASE._VAR_FLOOR)
        pre_sigma = fnp.sqrt(pre_variance)
        alpha = pre_mean / pre_sigma
        pdf = fnp.exp(-0.5 * alpha * alpha) / _BASE._SQRT_2PI
        cdf = _BASE.flops.stats.norm.cdf(alpha)
        coefficients = _BASE._hermite_coefficients(alpha, pdf, cdf)
        gaussian_mean = pre_sigma * coefficients[0]

        if int(cp_lambda.shape[0]) == 0:
            pushed_p = cp_p
            pushed_q = cp_q
            k3 = fnp.zeros(width, dtype=fnp.float32)
        else:
            pushed_p = fnp.transpose(weights32) @ cp_p
            pushed_q = fnp.transpose(weights32) @ cp_q
            k3 = _BASE._cp_k3(cp_lambda, pushed_p, pushed_q)
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
            _BASE._VAR_FLOOR,
        )
        edges = _BASE._edge_matrices(weights32, width)

        if observer_active:
            raw = _BASE._raw_features(
                layer0,
                width,
                int(cp_lambda.shape[0]),
                retained_fraction,
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
            standardized = _BASE._standardize(raw, layer0, centers, scales)
            if layer0 == 0 or not recurrent:
                messages = fnp.zeros((width, _HIDDEN), dtype=fnp.float32)
                log_rms = fnp.zeros(_POWERS, dtype=fnp.float32)
                rhos = fnp.ones(_POWERS, dtype=fnp.float32)
            else:
                messages, log_rms, rhos = _messages(edges, h_previous)
            cell_input, hidden, unit = _observer_cell(
                standardized, messages, log_rms, parameters
            )
            if layer0 == 0:
                delta = fnp.zeros(width, dtype=fnp.float32)
            else:
                delta = (
                    fnp.asarray(pre_sigma, dtype=fnp.float32)
                    * (1.0 / 32.0)
                    * unit
                )
            candidate_row = fnp.maximum(
                host32 + delta, fnp.asarray(0.0, dtype=fnp.float32)
            )
            current_valid = _BASE._one_scalar_latch(
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
                candidate_row,
                pre_sigma,
                cp_diagnostic,
                parameters,
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
        for order in range(1, _BASE._HERMITE_ORDER + 1):
            coefficient = coefficients[order]
            covariance = covariance + fnp.outer(coefficient, coefficient) * correlation_power
            correlation_power = correlation_power * correlation
        covariance = covariance * fnp.outer(pre_sigma, pre_sigma)
        diagonal_tail = fnp.maximum(
            exact_variance - fnp.diag(covariance), 0.0
        )
        covariance = covariance + fnp.diag(diagonal_tail)
        mean = gaussian_mean

        if layer0 + 1 < mlp.depth:
            cp_lambda, cp_p, cp_q, retained_fraction, cp_diagnostic = _BASE._cp_grow(
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
