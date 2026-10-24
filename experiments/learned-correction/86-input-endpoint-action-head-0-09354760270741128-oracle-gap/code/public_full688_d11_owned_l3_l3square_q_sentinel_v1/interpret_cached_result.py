#!/usr/bin/env python3
"""Interpret a frozen four-field result without opening targets or feature caches.

The interpreter separates three logically different receipts:

1. the already-frozen exact-q capacity reference;
2. the four-field source-head gain, including its paired source-zero control;
3. fixed-board composition, which this sentinel does not measure.

Only a caller-supplied terminal JSON is read.  No model, target, feature row,
estimator, or network resource is opened.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "owned-l3-l3square-fold0-result-v1"
PRETARGET_SCHEMA = "owned-l3-l3square-fold0-pretarget-terminal-closure-v1"
FAILURE_SCHEMA = "owned-l3-l3square-fold0-terminal-failure-v1"

PARENT82 = 3.0023717906766396e-8
ORACLE = 6.1455295488600225e-9
FOLDJ = 3.0923227278552953e-8
MIN_RECOVERY = 0.08
THRESHOLD = 2.8113462838133887e-8
HELD = 172
EXPECTED_ARCHITECTURE = {
    "inputs": 86,
    "local22": 22,
    "sealed_parent_messages": 60,
    "appended_four_fields": 4,
    "hidden": 256,
    "hidden_layers": 3,
    "heads": 6,
    "parameters": 155_398,
}
FIELD_ORDER = [
    "mean_l3",
    "preactivation_variance_l3",
    "mean_l3_squared",
    "preactivation_variance_l3_squared",
]
EXPECTED_BINDINGS = {
    "release_sha256": "16715386a00d64347a20d1d4e85ecf5410d2c4d86abe79179c6dab78d3ec622b",
    "runner_sha256": "a6c891cd19b62e1d48280308d4607fefd2a9f8d6d2eaf4a0277dc230c5af6cc5",
    "feature_adapter_sha256": "5c85616fdd102986cae4b97309d1ddedee5564d3c490b0ebf198fe7ac56fbda7",
    "parent_result_sha256": "87624c36e90e3db7e5592ff8eb56acf5696a37d9ad0f2dd7ed656394301f98b4",
    "frozen_generator_sha256": "5f4ed4a777f5ab80141bc2e3d1ced73b973f7a60b9f694dd330ca461b545c086",
    "h4_runner_audit_sha256": "a5010c9595885ff8a944464c0acb16ea82b2c8f2af8aebfc0c6ad34cc138391a",
    "cubic_runner_audit_sha256": "341785c61a6746d17760d3fb5e1969552420fe3c4db0f2d2beec313643f7e739",
    "target_bundle_sha256": "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7",
}


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-20)


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def paired_z(values: list[float]) -> float:
    average = mean(values)
    variance = math.fsum((value - average) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    if standard_error == 0.0:
        return math.inf if average > 0.0 else 0.0
    return average / standard_error


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def terminal_report(payload: dict[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema")
    errors: list[str] = []
    if schema == PRETARGET_SCHEMA:
        require(errors, payload.get("claim_present") is False, "pretarget closure reports a claim")
        require(errors, payload.get("target_values_opened_or_decoded") == 0, "pretarget closure opened targets")
        require(errors, payload.get("training_or_scoring_runs") == 0, "pretarget closure ran training/scoring")
        require(errors, payload.get("derived_feature_cache_written") is False, "pretarget closure wrote a derived cache")
        require(
            errors,
            payload.get("decision") == "CLOSE_FOUR_FIELD_BUNDLE__NO_RETRY_NO_FOLDS_1_TO_3",
            "pretarget closure decision changed",
        )
        phase = "PRETARGET_CANDIDATE_FALSIFIER"
    elif schema == FAILURE_SCHEMA:
        require(
            errors,
            payload.get("status") == "TERMINAL_POST_CLAIM_FAILURE__NO_RETRY",
            "post-claim failure status changed",
        )
        require(
            errors,
            payload.get("decision") == "STOP_NO_RETRY_NO_FOLDS_1_TO_3",
            "post-claim failure decision changed",
        )
        phase = "POST_CLAIM_TERMINAL_FAILURE"
    else:
        return {
            "schema": "owned-l3-l3square-cached-result-interpretation-v1",
            "integrity": "FAIL",
            "errors": [f"unsupported terminal schema: {schema!r}"],
        }
    return {
        "schema": "owned-l3-l3square-cached-result-interpretation-v1",
        "integrity": "PASS" if not errors else "FAIL",
        "terminal_phase": phase,
        "candidate_decision": "TERMINAL_CLOSE__NO_RETRY_NO_REMAINING_FOLDS",
        "oracle_capacity": "NOT_EVALUATED_BY_THIS_TERMINAL_ARTIFACT",
        "source_head_gain": "NOT_EVALUATED_BY_THIS_TERMINAL_ARTIFACT",
        "board_composition": "UNMEASURED__NO_FIXED_BOARD_DELTA_OR_CROSS_TERM",
        "errors": errors,
    }


def interpret_result(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        return terminal_report(payload)

    errors: list[str] = []
    require(errors, payload.get("fold") == 0, "result is not fold 0")
    require(errors, payload.get("fit_rows") == 516, "fit-row count changed")
    require(errors, payload.get("architecture") == EXPECTED_ARCHITECTURE, "architecture changed")

    bindings = payload.get("bindings", {})
    for key, expected in EXPECTED_BINDINGS.items():
        require(errors, bindings.get(key) == expected, f"binding changed: {key}")
    require(errors, valid_sha256(bindings.get("claim_sha256")), "claim hash missing or malformed")
    require(errors, valid_sha256(bindings.get("models_sha256")), "models hash missing or malformed")

    rows = payload.get("per_network")
    require(errors, isinstance(rows, list) and len(rows) == HELD, "held-row payload is not 172 rows")
    if not isinstance(rows, list) or len(rows) != HELD:
        rows = []
    role_ranks = [row.get("role_rank") for row in rows if isinstance(row, dict)]
    require(errors, len(role_ranks) == HELD and len(set(role_ranks)) == HELD, "held role ranks are not unique")
    require(errors, payload.get("held_rows") == role_ranks, "held-row order differs from per-network rows")

    row_keys = {
        "source_on_mse",
        "source_zero_mse",
        "parent82_mse",
        "exact_q_oracle_mse",
        "foldJ_base_mse",
    }
    require(
        errors,
        all(
            isinstance(row, dict)
            and row_keys.issubset(row)
            and all(finite_number(row[key]) for key in row_keys)
            for row in rows
        ),
        "per-network metrics are missing or nonfinite",
    )

    aggregate = payload.get("aggregate", {})
    metric_keys = (
        "source_on_local_q_mse",
        "source_zero_local_q_mse",
        "source_on_endpoint_mse",
        "source_zero_endpoint_mse",
        "parent82_endpoint_mse",
        "exact_q_oracle_endpoint_mse",
        "foldJ_base_endpoint_mse",
    )
    require(errors, all(finite_number(aggregate.get(key)) for key in metric_keys), "aggregate metrics are missing or nonfinite")
    if rows and all(all(finite_number(row.get(key)) for key in row_keys) for row in rows):
        recomputed_means = {
            "source_on_endpoint_mse": mean([float(row["source_on_mse"]) for row in rows]),
            "source_zero_endpoint_mse": mean([float(row["source_zero_mse"]) for row in rows]),
            "parent82_endpoint_mse": mean([float(row["parent82_mse"]) for row in rows]),
            "exact_q_oracle_endpoint_mse": mean([float(row["exact_q_oracle_mse"]) for row in rows]),
            "foldJ_base_endpoint_mse": mean([float(row["foldJ_base_mse"]) for row in rows]),
        }
        for key, value in recomputed_means.items():
            require(errors, finite_number(aggregate.get(key)) and close(aggregate[key], value), f"aggregate does not match rows: {key}")
        improvements = [float(row["source_zero_mse"]) - float(row["source_on_mse"]) for row in rows]
        win_fraction = mean([float(value > 0.0) for value in improvements])
        z_value = paired_z(improvements)
    else:
        recomputed_means = {}
        improvements = []
        win_fraction = math.nan
        z_value = math.nan

    on = float(aggregate.get("source_on_endpoint_mse", math.nan))
    zero = float(aggregate.get("source_zero_endpoint_mse", math.nan))
    on_local = float(aggregate.get("source_on_local_q_mse", math.nan))
    zero_local = float(aggregate.get("source_zero_local_q_mse", math.nan))
    oracle = float(aggregate.get("exact_q_oracle_endpoint_mse", math.nan))
    recovery = (PARENT82 - on) / (PARENT82 - ORACLE)
    isolated = (zero - on) / (zero - oracle) if zero > oracle else -math.inf
    expected_threshold = PARENT82 - MIN_RECOVERY * (PARENT82 - ORACLE)
    require(errors, expected_threshold == THRESHOLD, "interpreter threshold constant is inconsistent")

    fixed_receipts = {
        "frozen_parent82_fold0_endpoint_mse": PARENT82,
        "frozen_parent82_fold0_exact_q_oracle_endpoint_mse": ORACLE,
        "parent82_endpoint_mse": PARENT82,
        "exact_q_oracle_endpoint_mse": ORACLE,
        "foldJ_base_endpoint_mse": FOLDJ,
        "minimum_required_gap_recovery": MIN_RECOVERY,
        "frozen_endpoint_threshold": THRESHOLD,
    }
    for key, expected in fixed_receipts.items():
        require(errors, finite_number(aggregate.get(key)) and close(aggregate[key], expected), f"frozen aggregate changed: {key}")

    derived_receipts = {
        "source_on_gain_vs_parent82_endpoint": PARENT82 - on,
        "source_on_gain_vs_source_zero_endpoint": zero - on,
        "source_on_gain_vs_source_zero_local_q": zero_local - on_local,
        "parent82_to_exact_q_oracle_gap_recovery": recovery,
        "isolated_source_zero_to_oracle_gap_recovery": isolated,
        "network_win_fraction_source_on_vs_zero": win_fraction,
        "paired_network_z_source_on_vs_zero": z_value,
    }
    for key, expected in derived_receipts.items():
        require(
            errors,
            finite_number(expected)
            and finite_number(aggregate.get(key))
            and close(aggregate[key], expected),
            f"derived aggregate changed: {key}",
        )

    finite_gate = all(
        math.isfinite(value)
        for value in (
            on,
            zero,
            on_local,
            zero_local,
            oracle,
            recovery,
            isolated,
            win_fraction,
            z_value,
        )
    )
    recomputed_gates = {
        "parent_fold0_exact_q_oracle_strictly_below_1e_8": oracle < 1.0e-8,
        "source_on_endpoint_at_most_2_8113462838133887e_8": on <= THRESHOLD,
        "parent82_to_exact_q_oracle_gap_recovery_at_least_0_08": recovery >= MIN_RECOVERY,
        "source_on_strictly_beats_source_zero_endpoint": on < zero,
        "source_on_strictly_beats_source_zero_local_q": on_local < zero_local,
        "all_scored_metrics_finite": finite_gate,
    }
    require(errors, payload.get("gates") == recomputed_gates, "reported gate map differs from recomputation")
    passed = all(recomputed_gates.values())
    require(errors, payload.get("passed_all_gates") is passed, "passed_all_gates differs from recomputation")
    expected_status = (
        "PASS_FOLD0_FOUR_FIELD_SENTINEL__HOLD_NO_REMAINING_FOLD_AUTHORITY"
        if passed
        else "FAIL_FOLD0_FOUR_FIELD_SENTINEL__TERMINAL_NO_RETRY"
    )
    expected_decision = (
        "HOLD_AND_REQUEST_SEPARATE_AUTHORITY_FOR_UNTOUCHED_FOLDS_1_TO_3"
        if passed
        else "CLOSE_FOUR_FIELD_BUNDLE__DO_NOT_RUN_FOLDS_1_TO_3"
    )
    require(errors, payload.get("status") == expected_status, "terminal status differs from recomputation")
    require(errors, payload.get("decision") == expected_decision, "terminal decision differs from recomputation")

    diagnostic = payload.get("feature_diagnostic", {})
    rank = diagnostic.get("target_free_conditional_rank", {})
    require(errors, diagnostic.get("four_field_order") == FIELD_ORDER, "four-field order changed")
    require(
        errors,
        rank.get("conditioning_features") == "constant_plus_exact_local22_plus_fit_normalized_clipped_parent60",
        "conditional-rank denominator is not the full parent82 input",
    )
    require(errors, rank.get("new_features") == FIELD_ORDER, "conditional-rank numerator changed")
    require(errors, isinstance(rank.get("minimum_layer_rank"), int) and rank["minimum_layer_rank"] > 0, "conditional new4 rank is zero")
    require(errors, rank.get("passes_nonzero_conditional_rank_gate") is True, "conditional-rank gate did not pass")
    layers = rank.get("layers")
    require(
        errors,
        isinstance(layers, list)
        and [row.get("layer") for row in layers] == [10, 11, 12, 13, 14, 15]
        and all(isinstance(row.get("numerical_rank"), int) and row["numerical_rank"] > 0 for row in layers),
        "per-layer conditional-rank receipt changed",
    )

    paired = payload.get("paired_training", {})
    for key in ("same_initial_state", "same_row_order", "same_node_samples", "same_q_rms"):
        require(errors, paired.get(key) is True, f"paired training diverged: {key}")
    for arm in ("source_on", "source_zero"):
        receipt = paired.get(arm, {})
        require(errors, receipt.get("epochs") == 13, f"{arm} epoch count changed")
        require(errors, receipt.get("samples_per_layer_per_network_per_epoch") == 256, f"{arm} sampler changed")
    require(errors, paired.get("source_on", {}).get("source_zero_coordinates_zero_after_normalization") is False, "source-on receipt is mislabeled")
    require(errors, paired.get("source_zero", {}).get("source_zero_coordinates_zero_after_normalization") is True, "source-zero receipt is mislabeled")

    access = payload.get("access", {})
    require(errors, access.get("target_role") == "observer_fit", "target role changed")
    require(errors, access.get("target_columns") == ["mlp_id", "all_layer_means"], "target columns changed")
    require(errors, access.get("derived_feature_cache_written") is False, "derived feature cache was written")
    require(errors, access.get("remaining_folds_run") == [], "remaining folds were run")
    require(errors, access.get("feature_subset_or_ablation_runs") == 0, "feature ablation was run")
    for key in ("all_data_fit", "integration", "meter", "package", "submission", "observer_selection_opened", "names_scores_board_opened"):
        require(errors, access.get(key) is False, f"unauthorized access/action: {key}")

    integrity = "PASS" if not errors else "FAIL"
    candidate_decision = (
        "PASS_FOLD0_ONLY__HOLD_FOR_SEPARATE_AUTHORITY"
        if integrity == "PASS" and passed
        else "TERMINAL_CLOSE__NO_RETRY_NO_REMAINING_FOLDS"
    )
    return {
        "schema": "owned-l3-l3square-cached-result-interpretation-v1",
        "integrity": integrity,
        "candidate_decision": candidate_decision,
        "oracle_capacity": {
            "kind": "frozen_parent82_exact_q_suffix_capacity_reference",
            "endpoint_mse": oracle,
            "strictly_below_1e_8": oracle < 1.0e-8,
            "boundary": "capacity reference only; not evidence that the four-field source head recovers it",
        },
        "source_head_gain": {
            "source_on_endpoint_mse": on,
            "source_zero_endpoint_mse": zero,
            "gain_vs_frozen_parent82": PARENT82 - on,
            "frozen_parent82_to_oracle_gap_recovery": recovery,
            "paired_new4_gain_endpoint": zero - on,
            "paired_new4_gain_local_q": zero_local - on_local,
            "isolated_source_zero_to_oracle_recovery_diagnostic": isolated,
            "network_win_fraction_on_vs_zero": win_fraction,
            "paired_network_z_on_vs_zero": z_value,
            "passes_frozen_eight_percent_and_paired_sign_gates": passed,
        },
        "board_composition": {
            "status": "UNMEASURED__NO_FIXED_BOARD_DELTA_OR_CROSS_TERM_IN_SENTINEL",
            "why": "Scalar candidate, parent, and board MSEs do not identify the cross term needed to compose a correction with the fixed board.",
            "required_frozen_test_if_fold0_passes": {
                "vectors": "on the same held nodes, freeze board error e=y-board and paired differential d=prediction_on-prediction_zero",
                "sufficient_statistics": ["A=<e,e>", "B=<e,d>", "C=<d,d>"],
                "coefficient_one_mse": "A-2B+C",
                "diagnostic_optimal_scalar": "B/C when C>0; report only, never refit without separate authority",
                "rule": "Do not infer composition from per-network scalar MSEs; acquire the aligned vectors or the exact B cross term.",
            },
        },
        "errors": errors,
    }


def synthetic_result() -> dict[str, Any]:
    on, zero, on_local, zero_local = 2.70e-8, 2.90e-8, 4.0e-9, 5.0e-9
    rows: list[dict[str, Any]] = []
    for index in range(HELD):
        perturbation = (index - (HELD - 1) / 2.0) * 1.0e-13
        rows.append(
            {
                "role_rank": index,
                "source_on_mse": on + perturbation,
                "source_zero_mse": zero - 0.5 * perturbation,
                "parent82_mse": PARENT82,
                "exact_q_oracle_mse": ORACLE,
                "foldJ_base_mse": FOLDJ,
            }
        )
    on_mean = mean([row["source_on_mse"] for row in rows])
    zero_mean = mean([row["source_zero_mse"] for row in rows])
    improvements = [row["source_zero_mse"] - row["source_on_mse"] for row in rows]
    recovery = (PARENT82 - on_mean) / (PARENT82 - ORACLE)
    isolated = (zero_mean - on_mean) / (zero_mean - ORACLE)
    gates = {
        "parent_fold0_exact_q_oracle_strictly_below_1e_8": True,
        "source_on_endpoint_at_most_2_8113462838133887e_8": True,
        "parent82_to_exact_q_oracle_gap_recovery_at_least_0_08": True,
        "source_on_strictly_beats_source_zero_endpoint": True,
        "source_on_strictly_beats_source_zero_local_q": True,
        "all_scored_metrics_finite": True,
    }
    return {
        "schema": SCHEMA,
        "status": "PASS_FOLD0_FOUR_FIELD_SENTINEL__HOLD_NO_REMAINING_FOLD_AUTHORITY",
        "decision": "HOLD_AND_REQUEST_SEPARATE_AUTHORITY_FOR_UNTOUCHED_FOLDS_1_TO_3",
        "fold": 0,
        "fit_rows": 516,
        "held_rows": list(range(HELD)),
        "bindings": {
            **EXPECTED_BINDINGS,
            "claim_sha256": "1" * 64,
            "models_sha256": "2" * 64,
        },
        "architecture": dict(EXPECTED_ARCHITECTURE),
        "aggregate": {
            "source_on_local_q_mse": on_local,
            "source_zero_local_q_mse": zero_local,
            "source_on_endpoint_mse": on_mean,
            "source_zero_endpoint_mse": zero_mean,
            "parent82_endpoint_mse": PARENT82,
            "exact_q_oracle_endpoint_mse": ORACLE,
            "foldJ_base_endpoint_mse": FOLDJ,
            "frozen_parent82_fold0_endpoint_mse": PARENT82,
            "frozen_parent82_fold0_local_q_mse": 5.276099810147771e-9,
            "frozen_parent82_fold0_exact_q_oracle_endpoint_mse": ORACLE,
            "source_on_gain_vs_parent82_endpoint": PARENT82 - on_mean,
            "source_on_gain_vs_source_zero_endpoint": zero_mean - on_mean,
            "source_on_gain_vs_source_zero_local_q": zero_local - on_local,
            "parent82_to_exact_q_oracle_gap_recovery": recovery,
            "isolated_source_zero_to_oracle_gap_recovery": isolated,
            "minimum_required_gap_recovery": MIN_RECOVERY,
            "frozen_endpoint_threshold": THRESHOLD,
            "network_win_fraction_source_on_vs_zero": mean([float(value > 0.0) for value in improvements]),
            "paired_network_z_source_on_vs_zero": paired_z(improvements),
            "maximum_process_rss_bytes": 1,
        },
        "feature_diagnostic": {
            "four_field_order": list(FIELD_ORDER),
            "target_free_conditional_rank": {
                "conditioning_features": "constant_plus_exact_local22_plus_fit_normalized_clipped_parent60",
                "new_features": list(FIELD_ORDER),
                "minimum_layer_rank": 4,
                "passes_nonzero_conditional_rank_gate": True,
                "layers": [{"layer": layer, "numerical_rank": 4} for layer in range(10, 16)],
            },
        },
        "paired_training": {
            "same_initial_state": True,
            "same_row_order": True,
            "same_node_samples": True,
            "same_q_rms": True,
            "source_on": {
                "epochs": 13,
                "samples_per_layer_per_network_per_epoch": 256,
                "source_zero_coordinates_zero_after_normalization": False,
            },
            "source_zero": {
                "epochs": 13,
                "samples_per_layer_per_network_per_epoch": 256,
                "source_zero_coordinates_zero_after_normalization": True,
            },
        },
        "gates": gates,
        "passed_all_gates": True,
        "per_network": rows,
        "access": {
            "target_role": "observer_fit",
            "target_columns": ["mlp_id", "all_layer_means"],
            "derived_feature_cache_written": False,
            "remaining_folds_run": [],
            "feature_subset_or_ablation_runs": 0,
            "all_data_fit": False,
            "integration": False,
            "meter": False,
            "package": False,
            "submission": False,
            "observer_selection_opened": False,
            "names_scores_board_opened": False,
        },
    }


def self_test() -> None:
    good = interpret_result(synthetic_result())
    if good["integrity"] != "PASS" or not good["candidate_decision"].startswith("PASS_FOLD0_ONLY"):
        raise AssertionError(good)
    bad_payload = synthetic_result()
    bad_payload["architecture"]["inputs"] = 26
    bad = interpret_result(bad_payload)
    if bad["integrity"] != "FAIL" or "architecture changed" not in bad["errors"]:
        raise AssertionError(bad)
    pretarget = terminal_report(
        {
            "schema": PRETARGET_SCHEMA,
            "claim_present": False,
            "target_values_opened_or_decoded": 0,
            "training_or_scoring_runs": 0,
            "derived_feature_cache_written": False,
            "decision": "CLOSE_FOUR_FIELD_BUNDLE__NO_RETRY_NO_FOLDS_1_TO_3",
        }
    )
    if pretarget["integrity"] != "PASS":
        raise AssertionError(pretarget)
    print(json.dumps({"status": "PASS_SELF_TEST", "tests": 3}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    if arguments.result is None:
        parser.error("a cached terminal JSON is required unless --self-test is used")
    payload = json.loads(arguments.result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("terminal artifact must contain one JSON object")
    report = interpret_result(payload)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report.get("integrity") != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
