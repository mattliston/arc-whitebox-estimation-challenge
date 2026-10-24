#!/usr/bin/env python3
"""Frozen target-free source probe for certified connected-carrier shutdown."""

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
ROOT = HERE.parent
SOURCE = HERE / "estimator.py"
SOURCE_SHA256 = "4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779"
PARENT = ROOT / "fullbudget_k21_k4_closure_q24_manual_rho_v1" / "estimator.py"
PARENT_SHA256 = "7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00"
FIXTURE_HELPER = ROOT / "q24_two_point_moment_repair_probe_v1" / "run_probe.py"
FIXTURE_HELPER_SHA256 = "2c37c363155ac2b709de00a0d7a949a33a43106592e0df88daa7509bf2468b2d"
PREREGISTER = HERE / "PREREGISTER.md"
PREREGISTER_SHA256 = "e83d1bfd6578556853af6ad1b6a9de3f35a2f45869a1262b99241d12653928c0"
OUTPUT = HERE / "runs" / "source_probe_v1" / "results.json"
PROBE_BUDGET = 10**15


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operation_count(budget) -> int:
    return sum(
        int(value.get("calls", 0))
        for value in budget.summary_dict(by_namespace=False)
        .get("operations", {}).values()
    )


def run_candidate(label: str, weights, helper) -> dict:
    module = load(f"q24_cs2_{label}", SOURCE)
    holder = {"invalid": None, "cp_post": None, "cp4_provider": None}
    records: list[tuple[object, object, object, object, object]] = []
    original_certificate = module._certified_invalid_marginals
    original_cp_post = module._cp_post_read
    original_cp4_reset = module._cp4_reset
    original_reset = module._reset_stars

    def observed_certificate(*args, **kwargs):
        invalid = original_certificate(*args, **kwargs)
        holder["invalid"] = invalid
        return invalid

    def observed_cp_post(*args, **kwargs):
        value = original_cp_post(*args, **kwargs)
        holder["cp_post"] = value
        return value

    def observed_cp4_reset(*args, **kwargs):
        value = original_cp4_reset(*args, **kwargs)
        if holder["invalid"] is not None:
            holder["cp4_provider"] = value[2]
        return value

    def observed_reset(*args, **kwargs):
        value = original_reset(*args, **kwargs)
        invalid = holder["invalid"]
        if invalid is None:
            return value
        cubic, quartic, pairing_alpha = value
        valid = fnp.invert(invalid)
        valid_plane = fnp.bitwise_and(valid[:, None], valid[None, :])
        invalid_plane = fnp.invert(valid_plane)
        affected = fnp.count_nonzero(invalid) > 0
        alpha_violation = fnp.count_nonzero(
            fnp.where(affected, pairing_alpha, 0.0)
        )
        star_violation = fnp.count_nonzero(
            fnp.where(invalid_plane, cubic, 0.0)
        )
        star_violation = star_violation + fnp.count_nonzero(
            fnp.where(invalid_plane, quartic, 0.0)
        )
        cp_post = holder["cp_post"]
        provider_d, provider_a, provider_b = holder["cp4_provider"]
        carrier_violation = fnp.count_nonzero(
            fnp.where(invalid_plane, cp_post, 0.0)
        )
        carrier_violation = carrier_violation + fnp.count_nonzero(
            fnp.where(invalid, provider_d, 0.0)
        )
        carrier_violation = carrier_violation + fnp.count_nonzero(
            fnp.where(invalid_plane, provider_a, 0.0)
        )
        carrier_violation = carrier_violation + fnp.count_nonzero(
            fnp.where(invalid_plane, provider_b, 0.0)
        )
        combined_violation = fnp.count_nonzero(
            fnp.where(invalid_plane, cubic + cp_post, 0.0)
        )
        combined_violation = combined_violation + fnp.count_nonzero(
            fnp.where(invalid, fnp.diag(quartic) + provider_d, 0.0)
        )
        combined_violation = combined_violation + fnp.count_nonzero(
            fnp.where(invalid_plane, quartic + provider_a, 0.0)
        )
        combined_violation = combined_violation + fnp.count_nonzero(
            fnp.where(invalid_plane, provider_b, 0.0)
        )
        records.append(
            (
                fnp.count_nonzero(invalid),
                alpha_violation,
                star_violation,
                carrier_violation,
                combined_violation,
            )
        )
        return value

    module._certified_invalid_marginals = observed_certificate
    module._cp_post_read = observed_cp_post
    module._cp4_reset = observed_cp4_reset
    module._reset_stars = observed_reset

    width = int(weights[0].shape[0])
    mlp = MLP(width=width, depth=16, weights=weights, seed=0, name=label)
    try:
        with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
            output = module._predict(mlp)
        array = np.asarray(output)
        result = {
            "outcome": "FINITE" if bool(np.isfinite(array).all()) else "NONFINITE",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "output_sha256": helper.digest_array(array),
            "flops": int(budget.flops_used),
            "calls": operation_count(budget),
        }
    except Exception as exc:
        array = None
        result = {
            "outcome": "EXCEPTION",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    labels = (
        "shutdown_counts_by_reset",
        "pairing_alpha_violation_counts_by_reset",
        "star_incident_nonzero_counts_by_reset",
        "cp_carrier_incident_nonzero_counts_by_reset",
        "combined_carrier_incident_nonzero_counts_by_reset",
    )
    for index, key in enumerate(labels):
        result[key] = [int(np.asarray(row[index])) for row in records]
    return result | {"_array": array}


def comparison(parent: np.ndarray, child: np.ndarray) -> dict:
    parent64 = parent.astype(np.float64)
    delta = child.astype(np.float64) - parent64
    parent_rms = float(np.sqrt(np.mean(parent64 * parent64)))
    disagreement_rms = float(np.sqrt(np.mean(delta * delta)))
    parent_max = float(np.max(np.abs(parent64)))
    disagreement_max = float(np.max(np.abs(delta)))
    return {
        "all_layer_mse": float(np.mean(delta * delta)),
        "final_layer_mse": float(np.mean(delta[-1] * delta[-1])),
        "maximum_absolute_difference": disagreement_max,
        "parent_rms": parent_rms,
        "disagreement_rms": disagreement_rms,
        "parent_maximum_absolute_value": parent_max,
        "rms_scale_gate": disagreement_rms <= parent_rms if parent_rms > 0.0
                          else disagreement_rms == 0.0,
        "maximum_scale_gate": disagreement_max <= parent_max if parent_max > 0.0
                              else disagreement_max == 0.0,
    }


def run_plain(path: Path, tag: str, width: int) -> tuple[int, int]:
    module = load(tag, path)
    eye = np.eye(width, dtype=np.float32)
    mlp = MLP(
        width=width, depth=16, weights=[eye.copy() for _ in range(16)],
        seed=0, name=f"identity_width{width}",
    )
    with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
        module._predict(mlp)
    return int(budget.flops_used), operation_count(budget)


def main() -> int:
    assert digest(SOURCE) == SOURCE_SHA256
    assert digest(PARENT) == PARENT_SHA256
    assert digest(FIXTURE_HELPER) == FIXTURE_HELPER_SHA256
    assert digest(PREREGISTER) == PREREGISTER_SHA256
    if OUTPUT.exists() or OUTPUT.parent.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT.parent}")
    helper = load("q24_cs2_fixture_helper", FIXTURE_HELPER)
    rows = []
    for label, weights in helper.fixture_rows():
        parent = helper.run_arm(label, weights, repaired=False)
        child = run_candidate(label, weights, helper)
        parent_array = parent.pop("_array")
        child_array = child.pop("_array")
        delta = None
        if parent_array is not None and child_array is not None:
            delta = comparison(parent_array, child_array)
        rows.append(
            {"fixture": label, "parent": parent, "shutdown": child,
             "finite_pair_disagreement": delta}
        )
        print(label, parent["outcome"], child["outcome"], flush=True)

    resource_rows = []
    for width in (4, 8, 16):
        parent_value = run_plain(PARENT, f"q24_cs2_parent_{width}", width)
        child_value = run_plain(SOURCE, f"q24_cs2_child_{width}", width)
        resource_rows.append(
            {
                "width": width,
                "parent_flops": parent_value[0],
                "child_flops": child_value[0],
                "flop_delta": child_value[0] - parent_value[0],
                "parent_calls": parent_value[1],
                "child_calls": child_value[1],
                "call_delta": child_value[1] - parent_value[1],
            }
        )

    no_regression = all(
        not (row["parent"]["outcome"] == "FINITE"
             and row["shutdown"]["outcome"] != "FINITE")
        for row in rows
    )
    activations = sum(
        sum(row["shutdown"].get("shutdown_counts_by_reset", []))
        for row in rows
    )
    invariant_keys = (
        "pairing_alpha_violation_counts_by_reset",
        "star_incident_nonzero_counts_by_reset",
        "cp_carrier_incident_nonzero_counts_by_reset",
        "combined_carrier_incident_nonzero_counts_by_reset",
    )
    carrier_gate = all(
        all(value == 0 for key in invariant_keys
            for value in row["shutdown"].get(key, []))
        for row in rows
    )
    scale_gate = all(
        row["finite_pair_disagreement"] is None
        or (
            row["finite_pair_disagreement"]["rms_scale_gate"]
            and row["finite_pair_disagreement"]["maximum_scale_gate"]
        )
        for row in rows
    )
    no_activation_parity = all(
        sum(row["shutdown"].get("shutdown_counts_by_reset", [])) != 0
        or row["finite_pair_disagreement"] is not None
        and row["finite_pair_disagreement"]["all_layer_mse"] == 0.0
        for row in rows
    )
    resource_gate = all(
        row["flop_delta"] == 350 * row["width"] * row["width"]
        + 4634 * row["width"] + 98
        and row["call_delta"] == 2296
        for row in resource_rows
    )
    recovered = sum(
        row["parent"]["outcome"] != "FINITE"
        and row["shutdown"]["outcome"] == "FINITE"
        for row in rows
    )
    passed = bool(
        activations > 0 and no_regression and carrier_gate and scale_gate
        and no_activation_parity and resource_gate
    )
    result = {
        "schema": "q24-certified-connected-carrier-shutdown-source-probe-v2",
        "status": "PASS_SOURCE_PROBE__FULLWIDTH_AND_ACCURACY_HOLD"
                  if passed else "REJECT_SOURCE_PROBE",
        "bindings": {
            "source_sha256": SOURCE_SHA256,
            "parent_estimator_sha256": PARENT_SHA256,
            "preregister_sha256": PREREGISTER_SHA256,
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
            "stored_carrier_invariants": carrier_gate,
            "no_activation_output_parity": no_activation_parity,
            "all_finite_pairs_within_parent_scale": scale_gate,
            "small_width_resource_law": resource_gate,
            "parent_failure_recovered_count": recovered,
            "source_probe_pass": passed,
        },
        "resource_rows": resource_rows,
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(result["gates"], sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
