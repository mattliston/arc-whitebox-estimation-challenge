#!/usr/bin/env python3
"""Pure V3 selection/promotion rules; deliberately contains no target loader."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


T_975_DF99 = np.float64(1.9842169515086827)
SIMPLICITY_ORDER = ("host", "local_no_recurrence", "recurrent")
HOST_CAP_ORDER = (512, 1536, 1792)
PLANNED_LAUNCH_CAP_ORDER = (1536, 1792)
CHECKPOINT_EPOCHS = tuple(range(0, 161, 4))
PATIENCE_CHECKS = 24


def select_checkpoint(checks: list[tuple[int, float]]) -> dict[str, Any]:
    """Frozen strict-improvement/earliest-tie checkpoint rule."""

    if not checks:
        raise ValueError("checkpoint metric list is empty")
    expected = CHECKPOINT_EPOCHS[: len(checks)]
    observed = tuple(int(epoch) for epoch, _ in checks)
    if observed != expected:
        raise ValueError(f"checkpoint epochs must be an exact schedule prefix: {expected}")
    if any(not math.isfinite(float(metric)) or float(metric) < 0.0 for _, metric in checks):
        raise FloatingPointError("checkpoint MSE must be nonnegative finite")
    best_epoch, best_mse = int(checks[0][0]), float(checks[0][1])
    non_improving = 0
    stopped = False
    for epoch, metric_value in checks[1:]:
        metric = float(metric_value)
        if metric < best_mse:
            best_epoch, best_mse = int(epoch), metric
            non_improving = 0
        else:
            non_improving += 1
        if non_improving == PATIENCE_CHECKS:
            stopped = True
            if int(epoch) != observed[-1]:
                raise ValueError("metrics were supplied after patience stopping")
            break
    if observed[-1] == CHECKPOINT_EPOCHS[-1]:
        stopped = True
    if not stopped:
        raise ValueError("checkpoint prefix ends before the frozen stopping rule")
    return {
        "best_epoch": best_epoch,
        "best_final_mse": best_mse,
        "last_checked_epoch": observed[-1],
        "non_improving_checks_at_stop": non_improving,
        "strict_improvement_only": True,
        "exact_equality_keeps_earlier": True,
    }


def _validate(name: str, value: np.ndarray, networks: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != (networks, 16, 1024):
        raise ValueError(f"{name} must be float32[{networks},16,1024]")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} is nonfinite")
    return array


def _ordered_mse(prediction: np.ndarray, target: np.ndarray, layer0: int) -> np.float64:
    """One scalar float64 sum in network-position then neuron order."""

    total = np.float64(0.0)
    networks, _, width = prediction.shape
    for position in range(networks):
        for neuron in range(width):
            error = np.float64(prediction[position, layer0, neuron]) - np.float64(target[position, layer0, neuron])
            total = np.float64(total + np.float64(error * error))
    return np.float64(total / np.float64(networks * width))


def _per_network_gain(
    host: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    layer0: int,
) -> np.ndarray:
    gains = np.empty(host.shape[0], dtype=np.float64)
    for position in range(host.shape[0]):
        host_sum = np.float64(0.0)
        candidate_sum = np.float64(0.0)
        for neuron in range(host.shape[2]):
            host_error = np.float64(host[position, layer0, neuron]) - np.float64(target[position, layer0, neuron])
            candidate_error = np.float64(candidate[position, layer0, neuron]) - np.float64(target[position, layer0, neuron])
            host_sum = np.float64(host_sum + np.float64(host_error * host_error))
            candidate_sum = np.float64(candidate_sum + np.float64(candidate_error * candidate_error))
        gains[position] = np.float64((host_sum - candidate_sum) / np.float64(host.shape[2]))
    return gains


def _mean_sample_sd_ci(values: np.ndarray, t_value: np.float64) -> dict[str, float]:
    count = int(values.size)
    total = np.float64(0.0)
    for value in values:
        total = np.float64(total + np.float64(value))
    mean = np.float64(total / np.float64(count))
    squared = np.float64(0.0)
    for value in values:
        centered = np.float64(value) - mean
        squared = np.float64(squared + np.float64(centered * centered))
    sd = np.float64(math.sqrt(float(squared / np.float64(count - 1))))
    half = np.float64(t_value * sd / np.float64(math.sqrt(count)))
    return {
        "mean": float(mean),
        "sample_sd": float(sd),
        "two_sided_95_ci_low": float(np.float64(mean - half)),
        "two_sided_95_ci_high": float(np.float64(mean + half)),
    }


def selection_decision(
    predictions: Mapping[str, np.ndarray],
    target: np.ndarray,
) -> dict[str, Any]:
    """Choose among the only three eligible arms on exactly 88 networks."""

    networks = 88
    if tuple(predictions) != SIMPLICITY_ORDER:
        raise ValueError(f"prediction keys must be exact simplicity order {SIMPLICITY_ORDER}")
    target = _validate("target", target, networks)
    validated = {name: _validate(name, predictions[name], networks) for name in SIMPLICITY_ORDER}
    host_l2 = _ordered_mse(validated["host"], target, 1)
    rows: dict[str, dict[str, Any]] = {}
    for name in SIMPLICITY_ORDER:
        layer2 = _ordered_mse(validated[name], target, 1)
        final = _ordered_mse(validated[name], target, 15)
        gains = _per_network_gain(validated["host"], validated[name], target, 1)
        rows[name] = {
            "layer2_mse": float(layer2),
            "final_mse": float(final),
            "layer2_non_regression": bool(layer2 <= host_l2),
            "layer2_paired_gain_diagnostic": _mean_sample_sd_ci(gains, np.float64(1.9876082815890742)),
        }
    eligible = [name for name in SIMPLICITY_ORDER if rows[name]["layer2_non_regression"]]
    winner = eligible[0]
    for name in eligible[1:]:
        if rows[name]["final_mse"] < rows[winner]["final_mse"]:
            winner = name
    return {
        "winner": winner,
        "simplicity_order": list(SIMPLICITY_ORDER),
        "final_metric": "aggregate raw final-layer MSE",
        "strict_final_improvement_replaces_incumbent": True,
        "exact_final_equality_keeps_simpler": True,
        "layer2_equality_passes": True,
        "rows": rows,
    }


def host_ladder_selection(
    predictions_by_cap: Mapping[int, Mapping[str, np.ndarray]],
    target: np.ndarray,
) -> dict[str, Any]:
    """Deterministically choose one of the three paired host/observer trios."""

    if tuple(predictions_by_cap) != HOST_CAP_ORDER:
        raise ValueError(f"host caps must be supplied in exact order {HOST_CAP_ORDER}")
    per_cap = {
        str(cap): selection_decision(predictions_by_cap[cap], target)
        for cap in HOST_CAP_ORDER
    }
    # Lower cap, then host/local/recurrent is the frozen simplicity order.
    winner_cap = HOST_CAP_ORDER[0]
    winner_arm = per_cap[str(winner_cap)]["winner"]
    winner_mse = per_cap[str(winner_cap)]["rows"][winner_arm]["final_mse"]
    for cap in HOST_CAP_ORDER[1:]:
        arm = per_cap[str(cap)]["winner"]
        mse = per_cap[str(cap)]["rows"][arm]["final_mse"]
        if mse < winner_mse:
            winner_cap, winner_arm, winner_mse = cap, arm, mse
    return {
        "winner_host_cap": winner_cap,
        "winner_arm": winner_arm,
        "winner_final_mse": winner_mse,
        "tie_order": [
            f"cp{cap}_{arm}" for cap in HOST_CAP_ORDER for arm in SIMPLICITY_ORDER
        ],
        "per_cap": per_cap,
    }


def planned_launch_selection(
    predictions_by_cap: Mapping[int, Mapping[str, np.ndarray]],
    target: np.ndarray,
) -> dict[str, Any]:
    """Select only between the frozen CP1536 primary and CP1792 control."""

    if tuple(predictions_by_cap) != PLANNED_LAUNCH_CAP_ORDER:
        raise ValueError(f"launch caps must be supplied in exact order {PLANNED_LAUNCH_CAP_ORDER}")
    per_cap = {
        str(cap): selection_decision(predictions_by_cap[cap], target)
        for cap in PLANNED_LAUNCH_CAP_ORDER
    }
    winner_cap = PLANNED_LAUNCH_CAP_ORDER[0]
    winner_arm = per_cap[str(winner_cap)]["winner"]
    winner_mse = per_cap[str(winner_cap)]["rows"][winner_arm]["final_mse"]
    for cap in PLANNED_LAUNCH_CAP_ORDER[1:]:
        arm = per_cap[str(cap)]["winner"]
        mse = per_cap[str(cap)]["rows"][arm]["final_mse"]
        if mse < winner_mse:
            winner_cap, winner_arm, winner_mse = cap, arm, mse
    return {
        "winner_host_cap": winner_cap,
        "winner_arm": winner_arm,
        "winner_final_mse": winner_mse,
        "tie_order": [
            f"cp{cap}_{arm}" for cap in PLANNED_LAUNCH_CAP_ORDER for arm in SIMPLICITY_ORDER
        ],
        "cp512_is_not_launch_eligible": True,
        "per_cap": per_cap,
    }


def paired_promotion(
    host: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Identical one-use audit/lockbox law on exactly 100 networks."""

    networks = 100
    host = _validate("host", host, networks)
    candidate = _validate("candidate", candidate, networks)
    target = _validate("target", target, networks)
    host_l2 = _ordered_mse(host, target, 1)
    candidate_l2 = _ordered_mse(candidate, target, 1)
    l2_gains = _per_network_gain(host, candidate, target, 1)
    l2_diagnostic = _mean_sample_sd_ci(l2_gains, T_975_DF99)

    final_gains = _per_network_gain(host, candidate, target, 15)
    final_diagnostic = _mean_sample_sd_ci(final_gains, T_975_DF99)
    host_final = _ordered_mse(host, target, 15)
    candidate_final = _ordered_mse(candidate, target, 15)
    ratios: dict[str, float] = {}
    for layer0 in range(12, 16):
        host_mse = _ordered_mse(host, target, layer0)
        candidate_mse = _ordered_mse(candidate, target, layer0)
        if not np.isfinite(host_mse) or host_mse <= 0.0:
            raise FloatingPointError("deep-layer host denominator is not positive finite")
        ratios[str(layer0 + 1)] = float(np.float64(candidate_mse / host_mse))
    adjacent = [
        math.log(max(ratios[str(layer + 1)], 2.0**-52))
        - math.log(max(ratios[str(layer)], 2.0**-52))
        for layer in (13, 14, 15)
    ]
    gates = {
        "layer2_non_regression": bool(candidate_l2 <= host_l2),
        "final_mean_gain_at_least_1e_minus_8": bool(final_diagnostic["mean"] >= 1.0e-8),
        "candidate_final_at_most_0p99_host": bool(candidate_final <= np.float64(0.99) * host_final),
        "final_paired_95_lcb_strictly_positive": bool(final_diagnostic["two_sided_95_ci_low"] > 0.0),
        "all_layer13_to16_ratios_strictly_below_one": all(value < 1.0 for value in ratios.values()),
        "maximum_adjacent_log_ratio_increase_at_most_log_1p10": bool(max(adjacent) <= math.log(1.10)),
    }
    return {
        "promote": all(gates.values()),
        "failure_or_ambiguous_boundary_action": "permanently select host; no retry",
        "layer2_equality_passes": True,
        "layer2_host_mse": float(host_l2),
        "layer2_candidate_mse": float(candidate_l2),
        "layer2_paired_gain_diagnostic": l2_diagnostic,
        "final_host_mse": float(host_final),
        "final_candidate_mse": float(candidate_final),
        "final_paired_gain": final_diagnostic,
        "deep_candidate_to_host_ratios": ratios,
        "maximum_adjacent_log_ratio_increase": float(max(adjacent)),
        "gates": gates,
    }
