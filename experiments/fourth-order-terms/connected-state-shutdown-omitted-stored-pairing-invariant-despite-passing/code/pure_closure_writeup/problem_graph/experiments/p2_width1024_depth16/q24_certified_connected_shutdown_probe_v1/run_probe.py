#!/usr/bin/env python3
"""Target-free probe of certified-invalid connected-state shutdown."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "fullbudget_k21_k4_closure_q24_manual_rho_v1" / "estimator.py"
PARENT_SHA256 = "7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00"
FIXTURE_HELPER = HERE.parent / "q24_two_point_moment_repair_probe_v1" / "run_probe.py"
FIXTURE_HELPER_SHA256 = "2c37c363155ac2b709de00a0d7a949a33a43106592e0df88daa7509bf2468b2d"
PREREGISTER = HERE / "PREREGISTER.md"
OUTPUT = HERE / "runs" / "probe_v1" / "results.json"
PROBE_BUDGET = 10**15
TWO_U = 2.220446049250313e-16


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def up(nonnegative):
    return fnp.nextafter(nonnegative, fnp.inf)


def add_sub_error(error_x, error_y, result):
    inherited = up(error_x + error_y)
    local = up(TWO_U * fnp.abs(result))
    return up(inherited + local)


def multiply_error(x, error_x, y, error_y, result):
    left = up(fnp.abs(x) * error_y)
    right = up(fnp.abs(y) * error_x)
    inherited = up(left + right)
    cross = up(error_x * error_y)
    inherited = up(inherited + cross)
    local = up(TWO_U * fnp.abs(result))
    return up(inherited + local)


def certified_invalid(raw1, raw2, raw3, raw4):
    """Certify A/B negativity for the represented fp64 marginal tuple."""
    zero = fnp.zeros_like(raw1)
    raw1_squared = raw1 * raw1
    error_raw1_squared = multiply_error(
        raw1, zero, raw1, zero, raw1_squared
    )
    variance = raw2 - raw1_squared
    error_variance = add_sub_error(zero, error_raw1_squared, variance)
    raw2_squared = raw2 * raw2
    error_raw2_squared = multiply_error(
        raw2, zero, raw2, zero, raw2_squared
    )
    a_left = raw1 * raw3
    error_a_left = multiply_error(raw1, zero, raw3, zero, a_left)
    margin_a = a_left - raw2_squared
    error_margin_a = add_sub_error(
        error_a_left, error_raw2_squared, margin_a
    )
    raw1_raw2 = raw1 * raw2
    error_raw1_raw2 = multiply_error(
        raw1, zero, raw2, zero, raw1_raw2
    )
    centered_21 = raw3 - raw1_raw2
    error_centered_21 = add_sub_error(
        zero, error_raw1_raw2, centered_21
    )
    raw4_minus_raw2_squared = raw4 - raw2_squared
    error_raw4_minus_raw2_squared = add_sub_error(
        zero, error_raw2_squared, raw4_minus_raw2_squared
    )
    b_left = raw4_minus_raw2_squared * variance
    error_b_left = multiply_error(
        raw4_minus_raw2_squared,
        error_raw4_minus_raw2_squared,
        variance,
        error_variance,
        b_left,
    )
    centered_21_squared = centered_21 * centered_21
    error_centered_21_squared = multiply_error(
        centered_21,
        error_centered_21,
        centered_21,
        error_centered_21,
        centered_21_squared,
    )
    margin_b = b_left - centered_21_squared
    error_margin_b = add_sub_error(
        error_b_left, error_centered_21_squared, margin_b
    )
    return fnp.bitwise_or(
        margin_a < -error_margin_a,
        margin_b < -error_margin_b,
    )


def run_shutdown(label: str, weights: list[np.ndarray], helper) -> dict:
    module = load(f"q24_connected_shutdown_{label}", PARENT)
    holder = {"invalid": None}
    counts: list[tuple[object, object]] = []

    parent_univariate = module._univariate_edgeworth
    parent_full_reset = module._full_relu_reset
    parent_cp_post_read = module._cp_post_read
    parent_cp4_reset = module._cp4_reset
    parent_cp_grow = module._cp_grow

    def observed_univariate(*args, **kwargs):
        raw = parent_univariate(*args, **kwargs)
        holder["invalid"] = certified_invalid(*raw)
        return raw

    def shutdown_full_reset(*args, **kwargs):
        result = list(parent_full_reset(*args, **kwargs))
        invalid = holder["invalid"]
        if invalid is None:
            raise RuntimeError("missing certified-invalid mask")
        active = fnp.invert(invalid)
        active_plane = fnp.bitwise_and(active[:, None], active[None, :])
        result[2] = fnp.where(active_plane, result[2], 0.0)
        result[3] = fnp.where(active, result[3], 0.0)
        result[4] = fnp.where(active_plane, result[4], 0.0)
        result[5] = fnp.where(active_plane, result[5], 0.0)
        invalid_plane = fnp.invert(active_plane)
        violation_count = fnp.count_nonzero(
            fnp.where(invalid_plane, result[2], 0.0)
        )
        violation_count = violation_count + fnp.count_nonzero(
            fnp.where(invalid, result[3], 0.0)
        )
        violation_count = violation_count + fnp.count_nonzero(
            fnp.where(invalid_plane, result[4], 0.0)
        )
        violation_count = violation_count + fnp.count_nonzero(
            fnp.where(invalid_plane, result[5], 0.0)
        )
        counts.append((fnp.count_nonzero(invalid), violation_count))
        return tuple(result)

    def masked_vector(value):
        invalid = holder["invalid"]
        if invalid is None:
            return value
        return fnp.where(invalid, 0.0, value)

    def shutdown_cp_post_read(
        pre_k21, gate, pre_covariance, pre_variance, birth_lambda,
    ):
        return parent_cp_post_read(
            pre_k21,
            masked_vector(gate),
            pre_covariance,
            pre_variance,
            masked_vector(birth_lambda),
        )

    def shutdown_cp4_reset(
        cp4_lambda, pushed_u, gate, pre_provider_planes, k4_target,
        identity, diagonal_context,
    ):
        return parent_cp4_reset(
            cp4_lambda,
            pushed_u,
            masked_vector(gate),
            pre_provider_planes,
            k4_target,
            identity,
            diagonal_context,
        )

    def shutdown_cp_grow(
        cp_lambda, pushed_p, pushed_q, gate32, pre_covariance,
        birth_lambda, identity, active_p_bank, active_q_bank, lambda_bank,
    ):
        return parent_cp_grow(
            cp_lambda,
            pushed_p,
            pushed_q,
            gate32,
            pre_covariance,
            masked_vector(birth_lambda),
            identity,
            active_p_bank,
            active_q_bank,
            lambda_bank,
        )

    module._univariate_edgeworth = observed_univariate
    module._full_relu_reset = shutdown_full_reset
    module._cp_post_read = shutdown_cp_post_read
    module._cp4_reset = shutdown_cp4_reset
    module._cp_grow = shutdown_cp_grow

    width = int(weights[0].shape[0])
    mlp = MLP(width=width, depth=16, weights=weights, seed=0, name=label)
    try:
        with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
            output = module._predict(mlp)
        array = np.asarray(output)
        summary = budget.summary_dict(by_namespace=False)
        result = {
            "outcome": "FINITE" if bool(np.isfinite(array).all()) else "NONFINITE",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "output_sha256": helper.digest_array(array),
            "flops": int(budget.flops_used),
            "calls": sum(int(value.get("calls", 0))
                         for value in summary.get("operations", {}).values()),
        }
    except Exception as exc:
        array = None
        result = {
            "outcome": "EXCEPTION",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    result["shutdown_counts_by_reset"] = [int(np.asarray(x[0])) for x in counts]
    result["post_mask_nonzero_incident_counts_by_reset"] = [
        int(np.asarray(x[1])) for x in counts
    ]
    return result | {"_array": array}


def scale_comparison(parent: np.ndarray, child: np.ndarray) -> dict:
    parent64 = parent.astype(np.float64)
    delta = child.astype(np.float64) - parent64
    parent_rms = float(np.sqrt(np.mean(parent64 * parent64)))
    delta_rms = float(np.sqrt(np.mean(delta * delta)))
    parent_max = float(np.max(np.abs(parent64)))
    delta_max = float(np.max(np.abs(delta)))
    return {
        "all_layer_mse": float(np.mean(delta * delta)),
        "final_layer_mse": float(np.mean(delta[-1] * delta[-1])),
        "maximum_absolute_difference": delta_max,
        "parent_rms": parent_rms,
        "disagreement_rms": delta_rms,
        "parent_maximum_absolute_value": parent_max,
        "rms_scale_gate": delta_rms <= parent_rms if parent_rms > 0.0
                          else delta_rms == 0.0,
        "maximum_scale_gate": delta_max <= parent_max if parent_max > 0.0
                              else delta_max == 0.0,
    }


def main() -> int:
    if digest(PARENT) != PARENT_SHA256:
        raise SystemExit("parent hash mismatch")
    if digest(FIXTURE_HELPER) != FIXTURE_HELPER_SHA256:
        raise SystemExit("fixture helper hash mismatch")
    if OUTPUT.exists() or OUTPUT.parent.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT.parent}")
    helper = load("q24_frozen_fixture_helper", FIXTURE_HELPER)
    rows = []
    for label, weights in helper.fixture_rows():
        parent = helper.run_arm(label, weights, repaired=False)
        shutdown = run_shutdown(label, weights, helper)
        parent_array = parent.pop("_array")
        shutdown_array = shutdown.pop("_array")
        comparison = None
        if parent_array is not None and shutdown_array is not None:
            comparison = scale_comparison(parent_array, shutdown_array)
        rows.append(
            {"fixture": label, "parent": parent, "shutdown": shutdown,
             "finite_pair_disagreement": comparison}
        )
        print(label, parent["outcome"], shutdown["outcome"], flush=True)

    no_regression = all(
        not (row["parent"]["outcome"] == "FINITE"
             and row["shutdown"]["outcome"] != "FINITE")
        for row in rows
    )
    exact_zero = all(
        all(value == 0 for value in row["shutdown"].get(
            "post_mask_nonzero_incident_counts_by_reset", []
        ))
        for row in rows
    )
    scale_gate = all(
        comparison is None
        or (comparison["rms_scale_gate"] and comparison["maximum_scale_gate"])
        for comparison in (row["finite_pair_disagreement"] for row in rows)
    )
    activations = sum(
        sum(row["shutdown"].get("shutdown_counts_by_reset", []))
        for row in rows
    )
    recovered = sum(
        row["parent"]["outcome"] != "FINITE"
        and row["shutdown"]["outcome"] == "FINITE"
        for row in rows
    )
    passed = bool(no_regression and exact_zero and scale_gate and activations > 0)
    receipt = {
        "schema": "q24-certified-connected-shutdown-probe-v1",
        "status": "PASS_CONSTRUCTION_PROBE__SOURCE_SCIENCE_HOLD"
                  if passed else "REJECT_CONSTRUCTION_PROBE",
        "bindings": {
            "parent_estimator_sha256": PARENT_SHA256,
            "preregister_sha256": digest(PREREGISTER),
            "runner_sha256": digest(Path(__file__)),
            "fixture_helper_sha256": FIXTURE_HELPER_SHA256,
        },
        "scope": {
            "targets": 0,
            "truth": 0,
            "dataset_rows": 0,
            "full_width_1024_runs": 0,
            "packages": 0,
            "submissions": 0,
        },
        "gates": {
            "activation_count": activations,
            "no_parent_finite_regression": no_regression,
            "all_invalid_incident_connected_targets_exact_zero": exact_zero,
            "all_finite_pairs_within_parent_scale": scale_gate,
            "parent_failure_recovered_count": recovered,
            "construction_pass": passed,
        },
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(receipt["gates"], sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
