#!/usr/bin/env python3
"""Independent read-only audit of the frozen Q24-CS2 full-width comparator."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "estimator.py"
SOURCE_HOLD = HERE / "SOURCE_HOLD_V2.json"
STRICT_AUDIT = HERE / "INDEPENDENT_STRICT_AUDIT_CS2.json"
PREREGISTER = HERE / "PREREGISTER_FULLWIDTH_COMPARATOR_AND_METER_V1.json"
RUNNER = HERE / "run_target_free_fullwidth_comparator.py"
RELEASE = HERE / "ROOT_FULLWIDTH_COMPARATOR_RELEASE_SCAFFOLD_V1.json"
CLAIM = HERE / "runs/fullwidth_official_he_seeds20260824_20260826_v1/CLAIM.json"
RECEIPT = HERE / "runs/fullwidth_official_he_seeds20260824_20260826_v1/receipt.json"
METER_SCAFFOLD = HERE / "ROOT_LITERAL_METER_RELEASE_SCAFFOLD_V1.json"
PARENT = ROOT / "fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py"
PARENT_METER = ROOT / "meter_protocol/runs/fb_ec4_q24_manual_rho_v1_7d0c9e30_n1_s8_seed20260824/receipt.json"
METER_DRIVER = ROOT / "meter_protocol/run_synthetic_meter.py"
PROMOTION_PREREGISTER = ROOT / "fullbudget_k21_k4_closure_q24_connected_shutdown_v2_board_promotion_v1/PREREGISTER.md"

EXPECTED = {
    SOURCE: "4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779",
    SOURCE_HOLD: "c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c",
    STRICT_AUDIT: "cba8bfbb437f32aa377ca72da74b9504894e6380429e2acf0e5a059d7ef8afcb",
    PREREGISTER: "e06a68a92a139c670b0fe2fcb38615e96fa8d96a0019b3eef58bc2e995d36dd0",
    RUNNER: "c4251eabce9341c9f60da441a37d583fdea47c54102e54a8b89ece816d23b2f2",
    RELEASE: "c5dc4408c69ac6eac36b05f139a54407366d23ccfcc1330c73f5f3e4ba98b3ab",
    CLAIM: "9d74f6f1bd3b65d7b8fe8579208dfbea93e459b6fa3c57e13bc35d4b1a0c1bcd",
    RECEIPT: "151de92da658f15de016b256c74cb00255bd1d4aea8c4d8ef3934bab10665645",
    METER_SCAFFOLD: "747bec4fa4d4c2b2dab216288eac0f1819ab7d2cbad5fb91ead2b366e227e1be",
    PARENT: "7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00",
    PARENT_METER: "4ad5d4bb8e70a3f57ef7559f5dceffe0d92fc7b77626088cff87431bdbf18160",
    METER_DRIVER: "279e2491ad7bf48b150ad670948c03fdcd2006ae3dae25e733930f4800e5cb34",
    PROMOTION_PREREGISTER: "e837715025dfd314afa296681a619ffae5e3b2829956342e9a9ed8c28e0a6d88",
}
SEEDS = (20260824, 20260825, 20260826)
WIDTH = 1024
DEPTH = 16
BOARD_CHANGE_SIZE_LIMIT = 1.0e-8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    value = hashlib.sha256()
    value.update(contiguous.dtype.str.encode("ascii"))
    value.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    value.update(memoryview(contiguous).cast("B"))
    return value.hexdigest()


def combine_named_digests(items: list[tuple[str, str]]) -> str:
    value = hashlib.sha256()
    for name, item_digest in items:
        encoded = name.encode("utf-8")
        value.update(len(encoded).to_bytes(8, "big"))
        value.update(encoded)
        value.update(item_digest.encode("ascii"))
    return value.hexdigest()


def reconstruct_fixture(seed: int) -> dict[str, object]:
    streams = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(streams[0])
    scale = float(np.sqrt(2.0 / WIDTH))
    layers: list[str] = []
    named: list[tuple[str, str]] = []
    for layer in range(DEPTH):
        weight = (rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32)
        layer_digest = array_digest(weight)
        layers.append(layer_digest)
        named.append((f"mlp0/layer{layer}", layer_digest))
        del weight
    return {
        "aggregate_weights_sha256": combine_named_digests(named),
        "layer_weight_sha256": layers,
        "estimator_seed": int(streams[2].generate_state(1)[0]),
    }


def contract_pass(telemetry: dict[str, object]) -> bool:
    return bool(
        telemetry["completed"]
        and telemetry["shape"] == [DEPTH, WIDTH]
        and telemetry["dtype"] == "float32"
        and telemetry["all_finite"]
    )


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2.0e-15, abs_tol=0.0)


def main() -> int:
    for path, expected in EXPECTED.items():
        assert path.is_file() and digest(path) == expected, path

    runner_tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    imported_modules = set()
    for node in ast.walk(runner_tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module)
    forbidden_roots = {
        "datasets", "pandas", "requests", "scipy", "sklearn", "torch",
        "urllib", "whestbench.dataset", "whestbench.scoring",
        "whestbench.simulation",
    }
    assert not any(
        module == forbidden or module.startswith(forbidden + ".")
        for module in imported_modules for forbidden in forbidden_roots
    )
    fixture_function = next(
        node for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fixture"
    )
    assert [argument.arg for argument in fixture_function.args.args] == ["seed"]
    assert not any(
        isinstance(node, ast.Name) and node.id in {"dataset", "target", "truth"}
        for node in ast.walk(fixture_function)
    )

    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    claim = json.loads(CLAIM.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert preregister["fullwidth_comparator"]["seeds_exact_in_order"] == list(SEEDS)
    assert release["status"] == "ROOT_AUTHORIZED_ONE_TARGET_FREE_FULLWIDTH_COMPARATOR__NOT_YET_CONSUMED"
    assert release["authorization"]["authorized"] is True
    assert release["authorization"]["one_use"] is True
    assert claim["status"] == "CLAIMED_ONE_USE_RELEASE__NO_RETRY_IF_PROCESS_CRASHES"
    assert claim["bindings"]["runner_sha256"] == EXPECTED[RUNNER]
    assert claim["bindings"]["fullwidth_release_sha256"] == EXPECTED[RELEASE]
    assert claim["fixed_execution"]["seeds_exact_in_order"] == list(SEEDS)
    assert claim["scope"] == {
        "dataset_arguments": 0,
        "dataset_rows_accessed": 0,
        "fits": 0,
        "packages": 0,
        "submissions": 0,
        "target_values_accessed": 0,
        "truth_computations": 0,
    }
    assert receipt["bindings"]["one_use_claim_sha256"] == EXPECTED[CLAIM]
    assert receipt["bindings"]["fullwidth_release_sha256"] == EXPECTED[RELEASE]
    assert receipt["bindings"]["runner_sha256"] == EXPECTED[RUNNER]
    assert receipt["scope"]["seeds_executed_exact_in_order"] == list(SEEDS)
    assert receipt["scope"]["seed_sweep"] is False
    for key in (
        "dataset_arguments", "dataset_rows_accessed", "public_target_values_accessed",
        "private_target_values_accessed", "truth_computations", "fits",
        "samples_used_by_estimator", "packages", "submissions", "graph_edits",
    ):
        assert receipt["scope"][key] == 0

    weight_rows = []
    activation_count = 0
    affected_count = 0
    aggregate_violations = {
        "pairing_alpha_violation_count": 0,
        "connected_star_incident_nonzero_count": 0,
        "cp_carrier_incident_nonzero_count": 0,
        "combined_carrier_incident_nonzero_count": 0,
    }
    comparator_parent_contract = True
    comparator_child_contract = True
    comparator_scale_gate = True
    comparator_parity_gate = True
    board_rows = []
    for expected_seed, row in zip(SEEDS, receipt["rows"], strict=True):
        fixture = row["fixture"]
        assert fixture["seed"] == expected_seed
        assert fixture["width"] == WIDTH and fixture["depth"] == DEPTH
        assert fixture["seed_tree"] == "numpy.random.SeedSequence(seed).spawn(3)"
        assert fixture["weights_stream"] == 0 and fixture["estimator_seed_stream"] == 2
        assert fixture["dataset_rows_accessed"] == 0
        assert fixture["target_values_accessed"] == 0
        assert fixture["truth_computed"] is False
        reconstructed = reconstruct_fixture(expected_seed)
        assert reconstructed["aggregate_weights_sha256"] == fixture["aggregate_weights_sha256"]
        assert reconstructed["layer_weight_sha256"] == fixture["layer_weight_sha256"]
        assert reconstructed["estimator_seed"] == fixture["estimator_seed"]
        weight_rows.append({
            "seed": expected_seed,
            "aggregate_weights_sha256": fixture["aggregate_weights_sha256"],
            "estimator_seed": fixture["estimator_seed"],
        })

        parent = row["parent"]
        child = row["child"]
        comparator_parent_contract &= contract_pass(parent)
        comparator_child_contract &= contract_pass(child)
        assert parent["flops_used"] == 2_131_280_786_389
        assert child["flops_used"] == 2_131_652_533_303
        assert child["flops_used"] - parent["flops_used"] == 371_746_914
        observer = child["shutdown_observer"]
        assert observer["certificate_reset_count"] == 14
        assert len(observer["by_reset"]) == 14
        local_activation = sum(item["invalid_unit_count"] for item in observer["by_reset"])
        local_affected = sum(item["invalid_unit_count"] > 0 for item in observer["by_reset"])
        assert local_activation == observer["invalid_unit_reset_activation_count"]
        assert local_affected == observer["affected_reset_count"]
        activation_count += local_activation
        affected_count += local_affected
        for key in aggregate_violations:
            local = sum(item[key] for item in observer["by_reset"])
            assert local == observer["violation_totals"][key]
            aggregate_violations[key] += local

        comparison = row["comparison"]
        assert comparison is not None
        assert close(
            comparison["all_layer_disagreement_rms"] ** 2,
            comparison["all_layer_disagreement_mse"],
        )
        assert close(
            comparison["final_layer_disagreement_rms"] ** 2,
            comparison["final_layer_disagreement_mse"],
        )
        recomputed_rms_gate = (
            comparison["all_layer_disagreement_rms"] <= comparison["parent_all_layer_rms"]
            if comparison["parent_all_layer_rms"] > 0.0
            else comparison["all_layer_disagreement_rms"] == 0.0
        )
        recomputed_max_gate = (
            comparison["all_layer_disagreement_maxabs"] <= comparison["parent_all_layer_maxabs"]
            if comparison["parent_all_layer_maxabs"] > 0.0
            else comparison["all_layer_disagreement_maxabs"] == 0.0
        )
        assert recomputed_rms_gate == comparison["rms_parent_scale_gate"]
        assert recomputed_max_gate == comparison["maxabs_parent_scale_gate"]
        comparator_scale_gate &= recomputed_rms_gate and recomputed_max_gate
        if local_activation == 0:
            comparator_parity_gate &= comparison["raw_byte_parity"]
        final_mse = comparison["final_layer_disagreement_mse"]
        board_rows.append({
            "seed": expected_seed,
            "final_layer_disagreement_mse": final_mse,
            "passes_le_1e_minus_8": bool(final_mse <= BOARD_CHANGE_SIZE_LIMIT),
            "multiple_of_limit": final_mse / BOARD_CHANGE_SIZE_LIMIT,
        })

    assert weight_rows[0]["aggregate_weights_sha256"] == "4073521961bbac02491a2d1ba620c568fea73ed21bf052c73e42c807286a56ec"
    parent_meter = json.loads(PARENT_METER.read_text(encoding="utf-8"))
    serialized_meter = json.dumps(parent_meter, sort_keys=True)
    assert weight_rows[0]["aggregate_weights_sha256"] in serialized_meter
    assert activation_count == 51 and affected_count == 18
    assert all(value == 0 for value in aggregate_violations.values())
    comparator_pass = bool(
        comparator_parent_contract and comparator_child_contract
        and activation_count > 0 and all(value == 0 for value in aggregate_violations.values())
        and comparator_scale_gate and comparator_parity_gate
    )
    assert comparator_pass is True
    assert receipt["gates"] == {
        "parent_contracts_all_three": True,
        "child_contracts_all_three": True,
        "invalid_unit_reset_activation_count": 51,
        "at_least_one_activation": True,
        "carrier_violation_totals": aggregate_violations,
        "carrier_invariants": True,
        "comparisons_available_all_three": True,
        "zero_activation_rows_have_raw_byte_parity": True,
        "all_rows_within_parent_output_scale": True,
        "fullwidth_comparator_pass": True,
    }
    assert receipt["status"] == "PASS_TARGET_FREE_FULLWIDTH_STABILITY__LITERAL_METER_AND_ACCURACY_HOLD"

    promotion_text = PROMOTION_PREREGISTER.read_text(encoding="utf-8")
    assert "final-layer\n   parent-versus-CS2 disagreement MSE must be finite and `<= 1e-8`" in promotion_text
    board_change_size_pass = all(row["passes_le_1e_minus_8"] for row in board_rows)
    assert board_change_size_pass is False
    assert sum(row["passes_le_1e_minus_8"] for row in board_rows) == 0

    meter_output = ROOT / "meter_protocol/runs/fb_ec4_q24_cs2_v2_4efb2c40_n1_s8_seed20260824"
    assert not meter_output.exists()
    summary = {
        "bindings": {str(path.relative_to(ROOT)): value for path, value in EXPECTED.items()},
        "runner_static_no_target_gate": True,
        "claim_and_one_use_chain": True,
        "weight_rows": weight_rows,
        "activation_count": activation_count,
        "affected_reset_count": affected_count,
        "carrier_violation_totals": aggregate_violations,
        "comparator_source_stability_pass": comparator_pass,
        "board_promotion_change_size_rows": board_rows,
        "board_promotion_change_size_pass": board_change_size_pass,
        "meter_output_namespace_absent": True,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
