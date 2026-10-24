#!/usr/bin/env python3
"""Interpret a future frozen transpose-sentinel aggregate without target I/O.

The program reads only the RESULT JSON path explicitly supplied by the caller.
It keeps three logically different questions separate:

1. start-5 exact-q oracle capacity;
2. incremental source-on versus source-zero head value;
3. fixed-board coefficient-one composition potential.

Missing board cross-term fields are reported as UNMEASURED.  A source-on raw
score is never silently treated as a board-composition score.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


ORACLE_THRESHOLD = 1.0e-8
SOURCE_ON_ENDPOINT_THRESHOLD = 2.79555972111794e-8
INDEPENDENT256_ENDPOINT = 3.0220317427127064e-8
MINIMUM_WIN_FRACTION = 0.60
MINIMUM_PAIRED_Z = 3.0
CURRENT_BOARD_RAW_REFERENCE = 2.9409071835573285e-8
CURRENT_BOARD_ESTIMATOR_SHA256 = (
    "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
)
CROSS_TERM_ABSOLUTE_TOLERANCE = 5.0e-22


PATHS: dict[str, tuple[str, ...]] = {
    "oracle": (
        "pretraining_oracle.start5_exact_q_oracle_endpoint_mse",
        "pretraining_oracle_gate.start5_exact_q_oracle_endpoint_mse",
        "aggregate.pretraining_start5_direct_transport_oracle_endpoint_mse",
        "aggregate.exact_q_oracle_endpoint_mse",
        "oracle.start5_exact_q_oracle_endpoint_mse",
        "start5_exact_q_oracle_endpoint_mse",
        "start5_exact_q_oracle_mse",
    ),
    "base_endpoint": (
        "sentinel.fold_j_base_endpoint_mse",
        "pretraining_oracle_gate.base_foldJ_endpoint_mse",
        "aggregate.base_foldJ_endpoint_mse",
        "fold_j_base_endpoint_mse",
        "base_endpoint_mse",
        "fold_j_base_mse",
    ),
    "on_endpoint": (
        "sentinel.source_on.endpoint_mse",
        "aggregate.source_on_endpoint_mse",
        "source_on.endpoint_mse",
        "source_on_endpoint_mse",
    ),
    "zero_endpoint": (
        "sentinel.source_zero.endpoint_mse",
        "aggregate.source_zero_endpoint_mse",
        "source_zero.endpoint_mse",
        "source_zero_endpoint_mse",
    ),
    "on_local": (
        "sentinel.source_on.local_q_mse",
        "aggregate.source_on_local_q_mse",
        "source_on.local_q_mse",
        "source_on_local_q_mse",
    ),
    "zero_local": (
        "sentinel.source_zero.local_q_mse",
        "aggregate.source_zero_local_q_mse",
        "source_zero.local_q_mse",
        "source_zero_local_q_mse",
    ),
    "win_fraction": (
        "sentinel.source_on.network_win_fraction_vs_fold_j_base",
        "aggregate.network_win_fraction_vs_foldJ_base",
        "source_on.network_win_fraction_vs_fold_j_base",
        "network_win_fraction_vs_fold_j_base",
        "network_win_fraction",
    ),
    "paired_z": (
        "sentinel.source_on.paired_network_z_vs_fold_j_base",
        "aggregate.paired_network_z_vs_foldJ_base",
        "source_on.paired_network_z_vs_fold_j_base",
        "paired_network_z_vs_fold_j_base",
        "paired_network_z",
    ),
    "board_base": (
        "board_composition.fixed_board_endpoint_mse",
        "board_composition.board_base_endpoint_mse",
    ),
    "board_composed": (
        "board_composition.board_plus_source_delta_endpoint_mse",
        "board_composition.composed_endpoint_mse",
    ),
    "board_delta_energy": (
        "board_composition.source_delta_mean_square",
        "board_composition.delta_energy",
    ),
    "board_residual_delta_inner": (
        "board_composition.board_residual_source_delta_inner_product",
        "board_composition.residual_delta_inner_product",
    ),
}


def _nested(document: dict[str, Any], path: str) -> Any:
    value: Any = document
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _first(document: dict[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        value = _nested(document, path)
        if value is not None:
            return value
    return None


def _finite_float(value: Any, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        if optional:
            return None
        raise ValueError(f"missing/non-numeric required metric: {name}") from error
    if not math.isfinite(result):
        raise ValueError(f"nonfinite metric: {name}")
    return result


def _bool(document: dict[str, Any], path: str) -> bool | None:
    value = _nested(document, path)
    return value if isinstance(value, bool) else None


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def interpret(document: dict[str, Any]) -> dict[str, Any]:
    expected_schema = "public-full688-d11-cubic-action-transpose-fold0-result-v1"
    schema = document.get("schema")
    fold = document.get("fold")
    schema_ok = schema == expected_schema
    fold_ok = fold == 0
    values = {
        name: _finite_float(_first(document, paths), name, optional=True)
        for name, paths in PATHS.items()
    }

    oracle = values["oracle"]
    capacity: dict[str, Any] = {
        "meaning": "truth-derived start-5 exact local-q capacity; not model performance",
        "endpoint_mse": oracle,
        "threshold_strictly_below": ORACLE_THRESHOLD,
    }
    if oracle is None:
        capacity["status"] = "MISSING__DO_NOT_INTERPRET_SOURCE_HEAD"
    else:
        capacity["status"] = (
            "PASS_START5_Q_ORACLE_CAPACITY"
            if oracle < ORACLE_THRESHOLD
            else "FAIL_START5_Q_ORACLE_CAPACITY__TRAINING_SHOULD_NOT_RUN"
        )

    required_source = (
        "on_endpoint",
        "zero_endpoint",
        "on_local",
        "zero_local",
        "win_fraction",
        "paired_z",
    )
    missing_source = [name for name in required_source if values[name] is None]
    source: dict[str, Any] = {
        "meaning": "joint incremental predictive value of the three appended coordinates",
        "missing_metrics": missing_source,
        "source_on_endpoint_mse": values["on_endpoint"],
        "source_zero_endpoint_mse": values["zero_endpoint"],
        "source_on_local_q_mse": values["on_local"],
        "source_zero_local_q_mse": values["zero_local"],
        "network_win_fraction_vs_fold_j_base": values["win_fraction"],
        "paired_network_z_vs_fold_j_base": values["paired_z"],
    }
    if missing_source:
        source["status"] = "MISSING_SOURCE_HEAD_METRICS"
    else:
        on_endpoint = float(values["on_endpoint"])
        zero_endpoint = float(values["zero_endpoint"])
        on_local = float(values["on_local"])
        zero_local = float(values["zero_local"])
        endpoint_gain = zero_endpoint - on_endpoint
        local_gain = zero_local - on_local
        source.update(
            {
                "endpoint_gain_source_zero_minus_on": endpoint_gain,
                "endpoint_relative_gain": _ratio(endpoint_gain, zero_endpoint),
                "local_q_gain_source_zero_minus_on": local_gain,
                "local_q_relative_gain": _ratio(local_gain, zero_local),
                "beats_frozen_independent256_endpoint": on_endpoint
                < INDEPENDENT256_ENDPOINT,
                "passes_endpoint_threshold": on_endpoint
                <= SOURCE_ON_ENDPOINT_THRESHOLD,
                "passes_network_win_fraction": float(values["win_fraction"])
                >= MINIMUM_WIN_FRACTION,
                "passes_paired_z": float(values["paired_z"]) >= MINIMUM_PAIRED_Z,
            }
        )
        source["passes_source_on_vs_zero_endpoint"] = endpoint_gain > 0.0
        source["passes_source_on_vs_zero_local_q"] = local_gain > 0.0
        paired_initial = _bool(document, "paired_training.same_initial_state")
        paired_order = _bool(document, "paired_training.same_row_order")
        paired_samples = _bool(document, "paired_training.same_node_samples")
        paired_training_pass = all(
            value is True for value in (paired_initial, paired_order, paired_samples)
        )
        source["paired_training_contract"] = {
            "same_initial_state": paired_initial,
            "same_row_order": paired_order,
            "same_node_samples": paired_samples,
            "passes": paired_training_pass,
        }
        oracle_gate_pass = oracle is not None and oracle < ORACLE_THRESHOLD
        source["passes_start5_q_oracle_pretraining_gate"] = oracle_gate_pass
        source_pass = all(
            (
                oracle_gate_pass,
                schema_ok,
                fold_ok,
                paired_training_pass,
                source["beats_frozen_independent256_endpoint"],
                source["passes_endpoint_threshold"],
                source["passes_network_win_fraction"],
                source["passes_paired_z"],
                source["passes_source_on_vs_zero_endpoint"],
                source["passes_source_on_vs_zero_local_q"],
            )
        )
        source["status"] = (
            "PASS_FROZEN_SENTINEL_SOURCE_HEAD_GATE"
            if source_pass
            else "FAIL_FROZEN_SENTINEL_SOURCE_HEAD_GATE__STOP"
        )
        reported_pass = _bool(document, "passed_all_gates")
        source["reported_passed_all_gates"] = reported_pass
        source["reported_matches_recomputed"] = (
            reported_pass is None or reported_pass is source_pass
        )

        base = values["base_endpoint"]
        if base is not None and oracle is not None:
            exact_gap = float(base) - float(oracle)
            source["base_to_start5_oracle_gap"] = exact_gap
            source["total_head_fraction_of_exact_q_gap_closed"] = _ratio(
                float(base) - on_endpoint, exact_gap
            )
            source["source_only_fraction_of_exact_q_gap_closed"] = _ratio(
                endpoint_gain, exact_gap
            )

    board_base = values["board_base"]
    board_composed = values["board_composed"]
    same_rows = _bool(document, "board_composition.same_held_rows")
    coefficient_one = _bool(document, "board_composition.coefficient_one")
    held_oof = _bool(document, "board_composition.source_delta_is_held_oof")
    delta_definition = _nested(document, "board_composition.delta_definition")
    board_sha256 = _nested(document, "board_composition.fixed_board_estimator_sha256")
    board_role = _nested(document, "board_composition.role")
    expected_definition = "prediction_source_on_minus_prediction_source_zero"
    board: dict[str, Any] = {
        "meaning": "fixed-board plus coefficient-one isolated source-head delta on identical held rows",
        "current_board_global_raw_reference_not_a_fold_comparator": CURRENT_BOARD_RAW_REFERENCE,
        "fixed_board_endpoint_mse": board_base,
        "board_plus_source_delta_endpoint_mse": board_composed,
        "same_held_rows": same_rows,
        "coefficient_one": coefficient_one,
        "source_delta_is_held_oof": held_oof,
        "delta_definition": delta_definition,
        "fixed_board_estimator_sha256": board_sha256,
        "role": board_role,
        "deployment_status_regardless_of_diagnostic": "NO_DEPLOYMENT_MODEL_OR_METER_FROM_SENTINEL",
    }
    board_measured = (
        board_base is not None
        and board_composed is not None
        and same_rows is True
        and coefficient_one is True
        and held_oof is True
        and delta_definition == expected_definition
        and board_sha256 == CURRENT_BOARD_ESTIMATOR_SHA256
        and board_role == "observer_fit"
    )
    if not board_measured:
        board["status"] = (
            "UNMEASURED__DO_NOT_INFER_FROM_SOURCE_HEAD_OR_GLOBAL_BOARD_RAW"
        )
    else:
        gain = float(board_base) - float(board_composed)
        board["fixed_coefficient_one_gain"] = gain
        board["fixed_coefficient_one_relative_gain"] = _ratio(gain, float(board_base))
        board["status"] = (
            "POSITIVE_HELD_BOARD_COMPOSITION_POTENTIAL__NOT_DEPLOYABLE"
            if gain > 0.0
            else "NO_POSITIVE_HELD_BOARD_COMPOSITION_POTENTIAL"
        )
        energy = values["board_delta_energy"]
        inner = values["board_residual_delta_inner"]
        if energy is not None and inner is not None:
            identity_change = float(energy) - 2.0 * float(inner)
            observed_change = float(board_composed) - float(board_base)
            board["cross_term_identity"] = {
                "delta_energy_minus_two_residual_delta_inner": identity_change,
                "observed_mse_change": observed_change,
                "absolute_error": abs(identity_change - observed_change),
                "absolute_tolerance": CROSS_TERM_ABSOLUTE_TOLERANCE,
                "passes": abs(identity_change - observed_change)
                <= CROSS_TERM_ABSOLUTE_TOLERANCE,
            }

    return {
        "schema": "public-full688-d11-cubic-action-transpose-result-interpretation-v1",
        "result_contract": {
            "input_schema": schema,
            "expected_schema": expected_schema,
            "schema_matches": schema_ok,
            "fold": fold,
            "fold_is_zero": fold_ok,
        },
        "start5_q_oracle_capacity": capacity,
        "source_head_gain": source,
        "board_composition_potential": board,
        "hard_boundary": (
            "c_and_b_are_old-tape-derived_predictors; no direct additive source credit"
        ),
    }


def _synthetic_self_test() -> None:
    document = {
        "schema": "public-full688-d11-cubic-action-transpose-fold0-result-v1",
        "fold": 0,
        "aggregate": {
            "pretraining_start5_direct_transport_oracle_endpoint_mse": 7.0e-9,
            "base_foldJ_endpoint_mse": 3.1e-8,
            "source_on_endpoint_mse": 2.7e-8,
            "source_zero_endpoint_mse": 2.9e-8,
            "source_on_local_q_mse": 4.0e-9,
            "source_zero_local_q_mse": 4.5e-9,
            "network_win_fraction_vs_foldJ_base": 0.7,
            "paired_network_z_vs_foldJ_base": 4.0,
        },
        "paired_training": {
            "same_initial_state": True,
            "same_row_order": True,
            "same_node_samples": True,
        },
        "passed_all_gates": True,
    }
    result = interpret(document)
    assert result["start5_q_oracle_capacity"]["status"].startswith("PASS")
    assert result["source_head_gain"]["status"].startswith("PASS")
    assert result["board_composition_potential"]["status"].startswith("UNMEASURED")

    board_base = 3.0e-8
    energy = 1.0e-9
    inner = 8.0e-10
    document["board_composition"] = {
        "fixed_board_endpoint_mse": board_base,
        "board_plus_source_delta_endpoint_mse": board_base + energy - 2.0 * inner,
        "same_held_rows": True,
        "coefficient_one": True,
        "source_delta_is_held_oof": True,
        "delta_definition": "prediction_source_on_minus_prediction_source_zero",
        "fixed_board_estimator_sha256": CURRENT_BOARD_ESTIMATOR_SHA256,
        "role": "observer_fit",
        "source_delta_mean_square": energy,
        "board_residual_source_delta_inner_product": inner,
    }
    result = interpret(document)
    assert result["board_composition_potential"]["status"].startswith("POSITIVE")
    assert result["board_composition_potential"]["cross_term_identity"]["passes"]

    terminal_oracle_failure = {
        "schema": "public-full688-d11-cubic-action-transpose-fold0-result-v1",
        "fold": 0,
        "pretraining_oracle_gate": {
            "base_foldJ_endpoint_mse": 3.1e-8,
            "start5_exact_q_oracle_endpoint_mse": 1.1e-8,
        }
    }
    result = interpret(terminal_oracle_failure)
    assert result["start5_q_oracle_capacity"]["status"].startswith("FAIL")
    assert result["source_head_gain"]["status"] == "MISSING_SOURCE_HEAD_METRICS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        _synthetic_self_test()
        print("PASS")
        return
    if arguments.result is None:
        parser.error("RESULT JSON path required unless --self-test is used")
    document = json.loads(arguments.result.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("RESULT root must be a JSON object")
    print(json.dumps(interpret(document), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
