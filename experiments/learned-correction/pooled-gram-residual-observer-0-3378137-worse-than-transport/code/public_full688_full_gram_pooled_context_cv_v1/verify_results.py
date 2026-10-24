#!/Users/matt/anaconda3/bin/python3
"""Independent artifact checks for frozen P2R0F2K four-fold CV."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPECTED_PREREG = "45d2f5cca1c8dfa5c3d0bde4f5bd8e4841350bd73208a24d5d9bde687211d757"
EXPECTED_PROGRAM = "f9700d8f589249112a1cbcf743f05d0cdaf0759e43cd314564a8f622af81c3d2"
EXPECTED_MANIFEST = "77607a7546aad379296527d3677441e64450289f657beff92dd894b5241b0935"
EXPECTED_SEAL = "b46dd6570f1bfa87a2c99b61ab999de850b890b1a9e9051a318072d1671323e1"
EXPECTED_J_DECOMPOSITION = "c709e2c04c61f427aa0b41c57b37dd0313138a81e8e6438610d430976d38ea0f"
EXPECTED_CV = "2a8a569599f3d9b8d0efe373a659367447dc652a0e24e9785a8d72db52abcd44"
TAIL_RANKS = [55, 71, 106, 317, 339, 458, 485, 507, 540, 575, 602]
EXPOSURE = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
MEAN_GATE = 2.9672528546945005e-8
MAXIMUM_GATE = 1.0e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float, *, atol: float = 1.0e-20) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1.0e-13, abs_tol=atol):
        raise AssertionError((left, right))


def main() -> None:
    assert sha256(HERE / "PREREGISTRATION.json") == EXPECTED_PREREG
    assert sha256(HERE / "run_experiment.py") == EXPECTED_PROGRAM
    assert sha256(HERE / "RUN_MANIFEST.json") == EXPECTED_MANIFEST
    assert sha256(HERE / "PREREGISTRATION_SEAL.json") == EXPECTED_SEAL
    assert sha256(HERE / "CV_RESULTS.json") == EXPECTED_CV
    seal = load(HERE / "PREREGISTRATION_SEAL.json")
    assert seal["preregistration_sha256"] == EXPECTED_PREREG
    assert seal["program_sha256"] == EXPECTED_PROGRAM
    assert seal["run_manifest_sha256"] == EXPECTED_MANIFEST
    assert seal["p2r0f2j_decomposition_sha256"] == EXPECTED_J_DECOMPOSITION
    assert seal["data_gate"]["observer_selection_targets_allowed"] is False

    cv = load(HERE / "CV_RESULTS.json")
    assert cv["schema"] == "public-full688-full-gram-pooled-context-cv-v1"
    assert cv["node"] == "P2R0F2K"
    assert cv["status"] == "complete"
    assert cv["observer_fit_networks"] == 688
    assert cv["observer_selection_targets_opened"] == 0
    assert cv["training"]["epochs"] == 6
    assert cv["training"]["own_primary_state_exposure_schedule"] == EXPOSURE
    assert cv["training"]["fixed_final_checkpoint"] is True
    architecture = cv["architecture"]
    assert architecture["parameters"] == 34048
    assert architecture["local_input_features"] == 42
    assert architecture["context_input_features"] == 84
    assert architecture["offdiagonal_pair_features"] == 6
    assert architecture["channels"] == 4
    assert architecture["signed_transport"] == "Phi*(s@W)"
    assert architecture["full_gram_observable"] == "Phi^2*((s_c*s_d)@W^2)"
    assert architecture["deployment_truth_inputs"] == 0

    all_valid: list[int] = []
    weighted_host = 0.0
    weighted_autonomous = 0.0
    weighted_host_layers = [0.0] * 16
    weighted_autonomous_layers = [0.0] * 16
    per_network: list[dict[str, object]] = []
    total = 0
    for summary in cv["fold_results"]:
        fold = int(summary["fold"])
        result_path = HERE / "folds" / f"fold_{fold}.json"
        model_path = HERE / "folds" / f"fold_{fold}.pt"
        assert sha256(result_path) == summary["result_sha256"]
        assert sha256(model_path) == summary["model_sha256"]
        record = load(result_path)
        assert record["schema"] == "public-full688-full-gram-pooled-context-fold-v1"
        assert record["status"] == "complete"
        assert record["fold"] == fold
        assert record["observer_selection_targets_opened"] == 0
        assert record["preregistration_sha256"] == EXPECTED_PREREG
        assert record["program_sha256"] == EXPECTED_PROGRAM
        assert record["model_sha256"] == sha256(model_path)
        assert record["readiness"]["ready"] is True
        assert record["readiness"]["available"] == 688
        assert record["readiness"]["unpatched_tail_ranks"] == []
        assert record["readiness"]["data_derived_tail_ranks"] == TAIL_RANKS
        assert record["readiness"]["repaired_tail_ranks"] == TAIL_RANKS
        trace = record["training"]["trace"]
        assert [row["epoch"] for row in trace] == [1, 2, 3, 4, 5, 6]
        assert [row["own_primary_state_exposure"] for row in trace] == EXPOSURE
        assert all(
            math.isfinite(float(row[key]))
            for row in trace
            for key in (
                "fit_objective",
                "transport_delta_max_abs",
                "mix_delta_max_abs",
                "context_weight_max_abs",
            )
        )
        valid = [int(value) for value in record["valid_role_ranks"]]
        train = [int(value) for value in record["train_role_ranks"]]
        assert len(valid) == record["valid_networks"]
        assert len(train) == record["train_networks"]
        assert not set(valid).intersection(train)
        assert sorted(valid + train) == list(range(688))
        assert record["autonomous"]["finite_full_rollout"] is True
        assert math.isfinite(float(record["autonomous"]["maximum_absolute_correction"]))
        assert len(record["autonomous"]["per_network"]) == len(valid)
        assert [int(row["role_rank"]) for row in record["autonomous"]["per_network"]] == valid
        all_valid.extend(valid)
        per_network.extend(record["autonomous"]["per_network"])
        count = len(valid)
        total += count
        weighted_host += count * float(record["host"]["final_mse"])
        weighted_autonomous += count * float(record["autonomous"]["final_mse"])
        for layer in range(16):
            weighted_host_layers[layer] += count * float(record["host"]["layer_mse"][layer])
            weighted_autonomous_layers[layer] += count * float(record["autonomous"]["layer_mse"][layer])
        close(summary["host_final_mse"], record["host"]["final_mse"])
        close(summary["autonomous_final_mse"], record["autonomous"]["final_mse"])
        close(
            summary["maximum_per_network_final_mse"],
            max(float(row["final_mse"]) for row in record["autonomous"]["per_network"]),
        )

    assert total == 688
    assert sorted(all_valid) == list(range(688))
    assert len(set(all_valid)) == 688
    close(cv["oof"]["host_final_mse"], weighted_host / total)
    close(cv["oof"]["autonomous_final_mse"], weighted_autonomous / total)
    for layer in range(16):
        close(cv["oof"]["host_layer_mse"][layer], weighted_host_layers[layer] / total)
        close(cv["oof"]["autonomous_layer_mse"][layer], weighted_autonomous_layers[layer] / total)
    close(cv["oof"]["autonomous_final_mse"], 3.2444190155738505e-8)
    maximum = max(per_network, key=lambda row: float(row["final_mse"]))
    close(cv["oof"]["maximum_per_network_final_mse"], maximum["final_mse"])
    assert cv["oof"]["maximum_role_rank"] == maximum["role_rank"]
    assert cv["oof"]["maximum_mlp_id"] == maximum["mlp_id"]

    recomputed_gate = {
        "conjunctive_pass": False,
        "finite_all_autonomous_rollouts": True,
        "maximum_per_network_below_1e_7": float(maximum["final_mse"]) < MAXIMUM_GATE,
        "mean_below_2_9672528546945005e_8": weighted_autonomous / total < MEAN_GATE,
    }
    recomputed_gate["conjunctive_pass"] = all(
        value for key, value in recomputed_gate.items() if key != "conjunctive_pass"
    )
    assert cv["decision_gate"] == recomputed_gate
    cost = cv["cost"]
    assert cost["parameter_count"] == 34048
    assert cost["correction_source_structure_upper_estimate"] == 1_341_549_952
    assert cost["host_plus_correction_upper_estimate"] == 2_128_319_009_085
    assert cost["headroom_under_upper_estimate"] == 71_680_990_915
    assert cost["within_phase2_budget_by_upper_estimate"] is True
    print(
        json.dumps(
            {
                "status": "PASS",
                "oof_networks": total,
                "oof_autonomous_final_mse": cv["oof"]["autonomous_final_mse"],
                "maximum_per_network_final_mse": maximum["final_mse"],
                "decision_gate_pass": cv["decision_gate"]["conjunctive_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
