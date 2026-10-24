#!/usr/bin/env python3
"""Target-free Q24 reference parity and Q24-versus-Q32 disagreement gate."""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import flopscope as flops
import numpy as np


HERE = Path(__file__).resolve().parent
PARENT_PATH = HERE.parent / "fullbudget_k21_k4_closure_v4" / "estimator.py"
CHILD_PATH = HERE / "estimator.py"
WIDTH = 8
DEPTH = 16
Q24_GRID = tuple(tuple(range(derivative, derivative + 25)) for derivative in range(7))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_mlp(weights):
    value = type("SyntheticMLP", (), {})()
    value.width = WIDTH
    value.depth = DEPTH
    value.weights = weights
    return value


def fixtures():
    rng = np.random.default_rng(20260823)
    near_identity = [
        (np.eye(WIDTH) + rng.normal(scale=0.02, size=(WIDTH, WIDTH))).astype(np.float32)
        for _ in range(DEPTH)
    ]
    rng = np.random.default_rng(20260823)
    orthogonal = []
    for _ in range(DEPTH):
        q, _ = np.linalg.qr(rng.normal(size=(WIDTH, WIDTH)))
        orthogonal.append((0.35 * q).astype(np.float64))
    rng = np.random.default_rng(17)
    canonical_contractive = [
        rng.normal(scale=0.12, size=(WIDTH, WIDTH)).astype(np.float32)
        for _ in range(DEPTH)
    ]
    rng = np.random.default_rng(20260823)
    identity_plus_qr = []
    for _ in range(DEPTH):
        q, _ = np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))
        identity_plus_qr.append((np.eye(WIDTH) + 0.35 * q).astype(np.float64))
    rng = np.random.default_rng(17)
    alternate_contractive = [
        (
            0.88 * np.eye(WIDTH)
            + 0.12 * rng.standard_normal((WIDTH, WIDTH)) / np.sqrt(WIDTH)
        ).astype(np.float32)
        for _ in range(DEPTH)
    ]
    return (
        ("identity", [np.eye(WIDTH, dtype=np.float32) for _ in range(DEPTH)]),
        ("near_identity_seed20260823_scale0p02_float32", near_identity),
        ("qr_seed20260823_scale0p35_float64", orthogonal),
        ("canonical_pure_normal_seed17_scale0p12_float32", canonical_contractive),
        ("alternate_identity_plus_qr_seed20260823_scale0p35_float64", identity_plus_qr),
        ("alternate_0p88_identity_plus_normal_seed17_scale0p12_over_sqrt8_float32", alternate_contractive),
    )


def install_independent_q24_reference(module):
    fnp = module.fnp

    def reference_q24_context(
        coefficients, sigma, covariance, fusion_context, inverse_context=None,
    ):
        width = int(sigma.shape[0])
        if inverse_context is None:
            inverse_context = module._inverse_sigma_context(sigma)
        inverse_sigma, inv1, _inv2, _inv3, _inv4, _inv6 = inverse_context
        correlation = fnp.clip(
            covariance * inv1[:, None] * inv1[None, :],
            -1.0,
            1.0,
        )
        # Deliberately use V4's 33-plane arenas and slice them here.  This is
        # independent of the child's 25-plane source spelling.
        powers = fusion_context[14][:25]
        fnp.copyto(powers[0], 1.0)
        fnp.cumprod(
            fnp.broadcast_to(correlation, (24, width, width)),
            axis=0,
            out=powers[1:],
        )
        coefficient_table = coefficients
        features = (
            coefficient_table[:, Q24_GRID, :]
            * inverse_sigma[None, :, None, :]
        )
        return (
            features,
            features[module._PAIR_LEFT_GRID, module._LEFT_DERIVATIVE_GRID],
            features[module._PAIR_RIGHT_GRID, module._RIGHT_DERIVATIVE_GRID],
            powers,
            fusion_context[6][:25],
        )

    module._q32_context = reference_q24_context


def install_action_trace(module):
    module._q24_trace = None
    original_connected = module._connected_planes
    original_repair = module._repair_covariance

    def connected_wrapper(*args, **kwargs):
        result = original_connected(*args, **kwargs)
        if module._q24_trace is not None:
            module._q24_trace["connected"].append(
                tuple(np.array(value, copy=True) for value in result[:4])
            )
        return result

    def repair_wrapper(*args, **kwargs):
        result = original_repair(*args, **kwargs)
        if module._q24_trace is not None:
            module._q24_trace["covariance"].append(np.array(result, copy=True))
        return result

    module._connected_planes = connected_wrapper
    module._repair_covariance = repair_wrapper


def run(module, mlp, capture=False):
    if capture:
        module._q24_trace = {"connected": [], "covariance": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=10**12, quiet=True) as budget:
            output = np.asarray(module._predict(mlp))
    operations = budget.summary_dict()["operations"]
    trace = module._q24_trace if capture else None
    module._q24_trace = None
    return output, budget.flops_used, sum(item["calls"] for item in operations.values()), trace


def sequence_envelope(parent_sequence, child_sequence):
    assert len(parent_sequence) == len(child_sequence)
    mse = []
    max_abs = []
    relative_rms = []
    changed = []
    for index, (parent, child) in enumerate(zip(parent_sequence, child_sequence)):
        parent64 = np.asarray(parent, dtype=np.float64)
        child64 = np.asarray(child, dtype=np.float64)
        delta = child64 - parent64
        mse.append(float(np.mean(delta * delta)))
        max_abs.append(float(np.max(np.abs(delta))))
        denominator = max(float(np.sqrt(np.mean(parent64 * parent64))), np.finfo(np.float64).tiny)
        relative_rms.append(float(np.sqrt(np.mean(delta * delta)) / denominator))
        if np.any(delta != 0.0):
            changed.append(index + 1)
    return {
        "count": len(parent_sequence),
        "first_changed_reset": changed[0] if changed else None,
        "max_mse": max(mse, default=0.0),
        "max_mse_reset": mse.index(max(mse)) + 1 if mse else None,
        "max_abs": max(max_abs, default=0.0),
        "max_abs_reset": max_abs.index(max(max_abs)) + 1 if max_abs else None,
        "max_relative_rms": max(relative_rms, default=0.0),
        "max_relative_rms_reset": relative_rms.index(max(relative_rms)) + 1 if relative_rms else None,
        "last_mse": mse[-1] if mse else 0.0,
        "last_max_abs": max_abs[-1] if max_abs else 0.0,
        "last_relative_rms": relative_rms[-1] if relative_rms else 0.0,
    }


def main() -> None:
    q32 = load("q24_gate_parent_q32", PARENT_PATH)
    reference = load("q24_gate_parent_reference", PARENT_PATH)
    install_independent_q24_reference(reference)
    child = load("q24_gate_child", CHILD_PATH)
    install_action_trace(q32)
    install_action_trace(child)

    rows = []
    for name, weights in fixtures():
        mlp = make_mlp(weights)
        q32_output, q32_flops, q32_calls, q32_trace = run(q32, mlp, capture=True)
        reference_output, reference_flops, reference_calls, _ = run(reference, mlp)
        child_output, child_flops, child_calls, child_trace = run(child, mlp, capture=True)
        if not np.array_equal(reference_output, child_output):
            raise AssertionError(f"child/reference Q24 mismatch: {name}")
        if not np.isfinite(child_output).all():
            raise AssertionError(f"nonfinite Q24 output: {name}")

        delta = child_output.astype(np.float64) - q32_output.astype(np.float64)
        per_layer_mse = np.mean(delta * delta, axis=1)
        per_layer_max_abs = np.max(np.abs(delta), axis=1)
        parent_layer_rms = np.sqrt(np.mean(q32_output.astype(np.float64) ** 2, axis=1))
        per_layer_relative_rms = np.sqrt(per_layer_mse) / np.maximum(
            parent_layer_rms, np.finfo(np.float64).tiny
        )
        changed = np.flatnonzero(np.any(delta != 0.0, axis=1))
        assert q32_trace is not None and child_trace is not None
        connected_names = ("K21", "K4", "K31", "K22")
        action_envelope = {
            key: sequence_envelope(
                [event[index] for event in q32_trace["connected"]],
                [event[index] for event in child_trace["connected"]],
            )
            for index, key in enumerate(connected_names)
        }
        action_envelope["repaired_covariance"] = sequence_envelope(
            q32_trace["covariance"], child_trace["covariance"]
        )
        rows.append(
            {
                "fixture": name,
                "reference_child_bitwise_equal": True,
                "all_q24_values_finite": True,
                "first_changed_layer": int(changed[0] + 1) if changed.size else None,
                "final_prediction_disagreement_mse": float(per_layer_mse[-1]),
                "all_layer_prediction_disagreement_mse": float(np.mean(delta * delta)),
                "max_abs_prediction_disagreement": float(np.max(np.abs(delta))),
                "per_layer_prediction_disagreement_mse": [float(value) for value in per_layer_mse],
                "per_layer_prediction_disagreement_max_abs": [float(value) for value in per_layer_max_abs],
                "per_layer_prediction_disagreement_relative_rms": [float(value) for value in per_layer_relative_rms],
                "action_envelope": action_envelope,
                "tiny_flops": {
                    "q32": q32_flops,
                    "q24_reference": reference_flops,
                    "q24_child": child_flops,
                },
                "tiny_calls": {
                    "q32": q32_calls,
                    "q24_reference": reference_calls,
                    "q24_child": child_calls,
                },
            }
        )

    print(
        json.dumps(
            {
                "status": "PASS_Q24_INDEPENDENT_REFERENCE_AND_TARGET_FREE_DISAGREEMENT",
                "fixture_width": WIDTH,
                "fixture_depth": DEPTH,
                "rows": rows,
                "claim_boundary": (
                    "Reference parity proves the source spelling. Q24-versus-Q32 "
                    "synthetic disagreement is target-free supporting evidence only."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
