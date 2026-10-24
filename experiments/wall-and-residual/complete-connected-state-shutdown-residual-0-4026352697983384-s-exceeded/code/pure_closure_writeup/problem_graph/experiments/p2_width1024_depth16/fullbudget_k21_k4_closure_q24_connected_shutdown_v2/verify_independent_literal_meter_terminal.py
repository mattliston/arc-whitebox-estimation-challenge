#!/usr/bin/env python3
"""Verify the immutable, one-use Q24-CS2 literal-meter terminal result.

This verifier is deliberately read-only.  It does not import or execute the
candidate estimator, reconstruct targets, or rerun the meter.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN_NAME = "fb_ec4_q24_cs2_v2_4efb2c40_n1_s8_seed20260824"
RUN = ROOT / "meter_protocol/runs" / RUN_NAME

SOURCE = HERE / "estimator.py"
SOURCE_HOLD = HERE / "SOURCE_HOLD_V2.json"
STRICT_AUDIT = HERE / "INDEPENDENT_STRICT_AUDIT_CS2.json"
COMPARATOR_AUDIT = HERE / "INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_V1.json"
COMPARATOR_RECEIPT = HERE / "runs/fullwidth_official_he_seeds20260824_20260826_v1/receipt.json"
RELEASE = HERE / "ROOT_LITERAL_METER_RELEASE_CS2_V1.json"
BOARD_PREREGISTER = (
    ROOT
    / "fullbudget_k21_k4_closure_q24_connected_shutdown_v2_board_promotion_v1/PREREGISTER.md"
)
DRIVER = ROOT / "meter_protocol/run_synthetic_meter.py"
METER_VERIFIER = ROOT / "meter_protocol/verify_meter_run.py"
LOCK = ROOT / "data_gate/requirements.lock.txt"
SCORING = ROOT / "data_gate/.venv/lib/python3.11/site-packages/whestbench/scoring.py"
INVOCATION = RUN / "invocation.json"
RAW_STDOUT = RUN / "raw_whest_stdout.json"
RAW_STDERR = RUN / "raw_whest_stderr.txt"
RAW_RSS = RUN / "raw_process_tree_rss.jsonl"
RECEIPT = RUN / "receipt.json"

EXPECTED = {
    SOURCE: "4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779",
    SOURCE_HOLD: "c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c",
    STRICT_AUDIT: "cba8bfbb437f32aa377ca72da74b9504894e6380429e2acf0e5a059d7ef8afcb",
    COMPARATOR_AUDIT: "beaa6e1dc7d51e028250edf6ae6730bdb454b99a6bc297bf04f5e8b498d79421",
    COMPARATOR_RECEIPT: "151de92da658f15de016b256c74cb00255bd1d4aea8c4d8ef3934bab10665645",
    RELEASE: "3070fa8fd7f2444de4b16b5bdf6e22c8eeb5f307752749951ca1fd8a7e6f6cb6",
    BOARD_PREREGISTER: "e837715025dfd314afa296681a619ffae5e3b2829956342e9a9ed8c28e0a6d88",
    DRIVER: "279e2491ad7bf48b150ad670948c03fdcd2006ae3dae25e733930f4800e5cb34",
    METER_VERIFIER: "91471a7d543020460359acf7db8548969182734e9469ec02194002e3008d37d8",
    LOCK: "dbd429c8eeb64c5809f7a045fc782173e476c943fbd6ce7d2616c94433481f98",
    SCORING: "e6d05ca38e3a69daf6d25e3024e33a79e5bb9571e03b9dc53580d727c64384e3",
    INVOCATION: "9c54afd90e1971d1f7452d5e344435c7c0f56f2948eb6ba6e0cc977ea7e00af0",
    RAW_STDOUT: "0f52d9ad802ce0ef15e5b4151586160163734a3132c7f3fdad47f56f0c036196",
    RAW_STDERR: "b79fbe5b7bb5bfec0f368e96f31b2e138ef374d8fe0878ec5a56f8b1117e8912",
    RAW_RSS: "c7168e6ee868f02dd0d3d338e96c82cc2d8bd5b50aa82b590905e6f15368351c",
    RECEIPT: "93ccaef3026a4d8c0e6e9d373ea77b442b4834805c838ac37e564c367fc71cf3",
}

FLOP_BUDGET = 2**41
WALL_LIMIT_S = 120.0
RESIDUAL_LIMIT_S = 0.4
MEMORY_LIMIT_BYTES = 8 * 1024**3
EXPECTED_FLOPS = 2_131_652_533_303
EXPECTED_CALLS = 9_504
EXPECTED_WALL_S = 25.368777791969478
EXPECTED_BACKEND_S = 24.245027110911906
EXPECTED_OVERHEAD_S = 0.721115411259234
EXPECTED_RESIDUAL_S = 0.4026352697983384
EXPECTED_DESCENDANT_RSS = 4_589_715_456
EXPECTED_WEIGHT_SHA = "4073521961bbac02491a2d1ba620c568fea73ed21bf052c73e42c807286a56ec"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2.0e-15, abs_tol=2.0e-15)


def find_literal_dict_binding(tree: ast.AST, key_text: str, name_text: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == key_text
                and isinstance(value, ast.Name)
                and value.id == name_text
            ):
                return True
    return False


def main() -> int:
    for path, expected in EXPECTED.items():
        assert path.is_file(), path
        assert digest(path) == expected, path

    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    invocation = json.loads(INVOCATION.read_text(encoding="utf-8"))
    raw = json.loads(RAW_STDOUT.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    comparator = json.loads(COMPARATOR_RECEIPT.read_text(encoding="utf-8"))

    # The frozen authorization remains immutable; the surviving fixed namespace
    # is the event that consumes its one use.
    assert release["status"] == "ROOT_AUTHORIZED_ONE_TARGET_FREE_LITERAL_METER__NOT_YET_CONSUMED"
    assert release["authorization"]["authorized"] is True
    assert release["authorization"]["one_use"] is True
    assert release["authorization"]["scope_id"] == (
        "fb-ec4-q24-cs2-v2-4efb2c40-literal-meter-20260824"
    )
    assert release["execution"]["label"] == RUN_NAME
    assert release["execution"]["argv"][2:] == invocation["driver_argv"]
    assert Path(release["execution"]["output_namespace_must_be_absent_before_launch"]).name == RUN_NAME
    assert RUN.is_dir() and RECEIPT.is_file()
    assert release["execution"]["one_use_law"].startswith("any surviving fixed meter output namespace")
    assert release["outcome_law"]["failure_is_terminal_for_this_literal_meter_release"] is True
    assert release["outcome_law"]["accuracy_authority"] == "none"
    assert release["bindings"]["candidate_estimator"]["sha256"] == EXPECTED[SOURCE]
    assert release["bindings"]["source_hold"]["sha256"] == EXPECTED[SOURCE_HOLD]
    assert release["bindings"]["independent_strict_audit"]["sha256"] == EXPECTED[STRICT_AUDIT]
    assert release["bindings"]["independent_fullwidth_comparator_audit"]["sha256"] == EXPECTED[COMPARATOR_AUDIT]

    assert receipt["schema"] == "phase2-target-blind-synthetic-meter-receipt-v1"
    assert receipt["status"] == "FAIL_PHASE2_SYNTHETIC_METER_ASSERTION"
    assert receipt["estimator"]["sha256"] == EXPECTED[SOURCE]
    assert receipt["invocation"] == invocation
    assert receipt["provenance"]["meter_driver_sha256"] == EXPECTED[DRIVER]
    assert receipt["provenance"]["receipt_verifier_sha256"] == EXPECTED[METER_VERIFIER]
    assert receipt["provenance"]["requirements_lock_sha256"] == EXPECTED[LOCK]
    assert receipt["provenance"]["installed_source_sha256"]["whestbench/scoring.py"] == EXPECTED[SCORING]
    artifact_paths = {
        "invocation.json": INVOCATION,
        "raw_whest_stdout.json": RAW_STDOUT,
        "raw_whest_stderr.txt": RAW_STDERR,
        "raw_process_tree_rss.jsonl": RAW_RSS,
    }
    for name, path in artifact_paths.items():
        record = receipt["raw_artifacts"][name]
        assert record["sha256"] == EXPECTED[path]
        assert record["bytes"] == path.stat().st_size

    # Target-free binding and official deterministic seed/weight law.
    assert invocation["dataset_argument"] is None
    assert "--dataset" not in invocation["whest_argv"]
    assert receipt["scope"]["accuracy_scored"] is False
    for key in (
        "phase1_checkpoints_accessed",
        "prior_scores_accessed",
        "public_phase2_target_or_truth_values_accessed",
        "teacher_states_accessed",
    ):
        assert receipt["scope"][key] == 0
    fixture = receipt["deterministic_synthetic_fixture"]
    assert fixture["seed"] == 20260824
    assert fixture["width"] == 1024 and fixture["depth"] == 16
    assert fixture["all_weights_reconstruct_exactly"] is True
    assert fixture["aggregate_weights_sha256"] == EXPECTED_WEIGHT_SHA
    comparator_seed_row = next(
        row for row in comparator["rows"] if row["fixture"]["seed"] == 20260824
    )
    assert comparator_seed_row["fixture"]["aggregate_weights_sha256"] == EXPECTED_WEIGHT_SHA
    assert comparator_seed_row["fixture"]["estimator_seed"] == fixture["per_mlp"][0]["estimator_seed"]

    config = raw["run_config"]
    assert config == {
        "n_mlps": 1,
        "width": 1024,
        "depth": 16,
        "seed": 20260824,
        "flop_budget": FLOP_BUDGET,
        "setup_timeout_s": 5.0,
        "wall_time_limit_s": WALL_LIMIT_S,
        "residual_wall_time_limit_s": RESIDUAL_LIMIT_S,
        "lambda_flops_per_second": 0.0,
    }
    results = raw["results"]
    assert results["n_failed_mlps"] == 1
    assert results["failure_breakdown"] == {
        "budget_exhausted": 0,
        "time_exhausted": 0,
        "residual_wall_time_exhausted": 1,
        "combined_budget_exhausted": 0,
        "error": 0,
    }
    assert len(results["per_mlp"]) == 1
    row = results["per_mlp"][0]
    assert row["budget_exhausted"] is False
    assert row["time_exhausted"] is False
    assert row["residual_wall_time_exhausted"] is True
    assert row["combined_budget_exhausted"] is False
    assert row["traceback"] is None

    assert row["flops_used"] == EXPECTED_FLOPS
    assert close(row["wall_time_s"], EXPECTED_WALL_S)
    assert close(row["flopscope_backend_time_s"], EXPECTED_BACKEND_S)
    assert close(row["flopscope_overhead_time_s"], EXPECTED_OVERHEAD_S)
    assert close(row["residual_wall_time_s"], EXPECTED_RESIDUAL_S)
    assert close(
        row["wall_time_s"],
        row["flopscope_backend_time_s"]
        + row["flopscope_overhead_time_s"]
        + row["residual_wall_time_s"],
    )
    assert row["flops_used"] <= FLOP_BUDGET
    assert row["wall_time_s"] <= WALL_LIMIT_S
    assert row["residual_wall_time_s"] > RESIDUAL_LIMIT_S

    estimator_breakdown = results["breakdowns"]["estimator"]
    namespace = estimator_breakdown["by_namespace"]["estimator.estimator-client"]
    assert namespace["flops_used"] == EXPECTED_FLOPS
    assert namespace["calls"] == EXPECTED_CALLS
    assert sum(item["flop_cost"] for item in namespace["operations"].values()) == EXPECTED_FLOPS
    assert sum(item["calls"] for item in namespace["operations"].values()) == EXPECTED_CALLS
    assert close(
        sum(item["flopscope_backend_time_s"] for item in namespace["operations"].values()),
        EXPECTED_BACKEND_S,
    )
    assert close(
        sum(item["flopscope_overhead_time_s"] for item in namespace["operations"].values()),
        namespace["flopscope_overhead_time_s"],
    )
    assert estimator_breakdown["flops_used"] == EXPECTED_FLOPS
    assert close(estimator_breakdown["flopscope_overhead_time_s"], EXPECTED_OVERHEAD_S)
    assert estimator_breakdown["flopscope_overhead_time_s"] > namespace["flopscope_overhead_time_s"]

    measured = receipt["measured"]
    assert measured["flops_used_per_mlp"] == [EXPECTED_FLOPS]
    assert measured["wall_time_s_per_mlp"] == [EXPECTED_WALL_S]
    assert measured["flopscope_backend_time_s_per_mlp"] == [EXPECTED_BACKEND_S]
    assert measured["flopscope_overhead_time_s_per_mlp"] == [EXPECTED_OVERHEAD_S]
    assert measured["residual_wall_time_s_per_mlp"] == [EXPECTED_RESIDUAL_S]
    assert sum(item["flop_cost"] for item in measured["operation_totals"].values()) == EXPECTED_FLOPS
    assert sum(item["calls"] for item in measured["operation_totals"].values()) == EXPECTED_CALLS
    assert measured["n_failed_mlps"] == 1
    assert receipt["assertions"]["no_failed_mlps"] is False
    memory = measured["memory"]
    assert memory["memory_limit_bytes"] == MEMORY_LIMIT_BYTES
    assert memory["sampler"]["peak_descendants_aggregate_rss_bytes"] == EXPECTED_DESCENDANT_RSS
    assert memory["os_rusage_children"]["normalized_peak_bytes"] == EXPECTED_DESCENDANT_RSS
    assert EXPECTED_DESCENDANT_RSS < MEMORY_LIMIT_BYTES
    assert memory["sampled_descendant_peak_below_memory_limit"] is True

    # Audit the receipt's misleading finiteness label.  The driver binds it
    # directly to the aggregate failure bit, so residual failure alone makes it
    # false; it is not a direct observation of prediction values.
    driver_text = DRIVER.read_text(encoding="utf-8")
    driver_tree = ast.parse(driver_text, filename=str(DRIVER))
    assert find_literal_dict_binding(driver_tree, "all_values_finite", "no_failures")
    assert 'no_failures = int(results.get("n_failed_mlps", -1)) == 0' in driver_text
    assert measured["output_contract"] == {
        "shape": [16, 1024],
        "materialized_scored_dtype": "float32",
        "all_values_finite": False,
        "authority": (
            "whestbench.scoring.materialise_predictions and validate_predictions; "
            "a shape or finiteness violation is recorded as an MLP failure"
        ),
    }

    # Exact whestbench source proves normal-path validation happens before the
    # post-predict residual check, and the latter substitutes finite zeros.
    scoring_text = SCORING.read_text(encoding="utf-8")
    validate_start = scoring_text.index("def validate_predictions(")
    validate_end = scoring_text.index("\ndef _predict_stats_to_dict", validate_start)
    validate_text = scoring_text[validate_start:validate_end]
    assert "_check_prediction_shape(predictions, depth=depth, width=width)" in validate_text
    assert "pred_np = fnp.asarray(predictions, dtype=fnp.float32)" in validate_text
    assert "if not fnp.all(fnp.isfinite(pred_np)):" in validate_text
    assert "raise exc" in validate_text
    assert "return pred_np" in validate_text
    evaluation_start = scoring_text.index("def evaluate_estimator(")
    evaluation_text = scoring_text[evaluation_start:]
    validation_index = evaluation_text.index(
        "predictions = validate_predictions(raw_predictions, depth=spec.depth, width=spec.width)"
    )
    residual_marker_index = evaluation_text.index(
        "# Post-predict check: residual wall-time limit"
    )
    residual_fallback_index = evaluation_text.index(
        "predictions = fnp.zeros((spec.depth, spec.width))", residual_marker_index
    )
    assert validation_index < residual_marker_index < residual_fallback_index

    # A shape/non-finite validation failure would take the generic error path;
    # this result instead has error=0 and only the post-validation residual bit.
    assert results["failure_breakdown"]["error"] == 0
    assert results["failure_breakdown"]["residual_wall_time_exhausted"] == 1

    assert release["board_promotion_boundary"]["board_promotion_status"] == (
        "REJECTED_BEFORE_PACKAGE_OR_SUBMISSION"
    )
    assert release["board_promotion_boundary"]["gate_fail_count"] == 3
    assert release["access_boundary"]["package_authorized"] is False
    assert release["access_boundary"]["submission_authorized"] is False

    print(
        json.dumps(
            {
                "status": "PASS_INDEPENDENT_TERMINAL_AUDIT__SOURCE_FINITE__RESOURCE_FAIL_RESIDUAL_ONLY",
                "release_consumed": True,
                "source_sha256": EXPECTED[SOURCE],
                "receipt_sha256": EXPECTED[RECEIPT],
                "flops": EXPECTED_FLOPS,
                "operation_calls": EXPECTED_CALLS,
                "residual_wall_time_s": EXPECTED_RESIDUAL_S,
                "residual_over_limit_s": EXPECTED_RESIDUAL_S - RESIDUAL_LIMIT_S,
                "pre_fallback_candidate_tensor": "validated finite float32 [16,1024]",
                "scored_tensor": "finite zero fallback",
                "terminal_resource_gate": "FAIL_RESIDUAL_WALL_TIME",
                "retry_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
