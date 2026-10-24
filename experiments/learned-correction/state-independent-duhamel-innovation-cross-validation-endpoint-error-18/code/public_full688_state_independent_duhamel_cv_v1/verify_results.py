#!/Users/matt/anaconda3/bin/python3
"""Independent artifact-level checks for frozen P2R0F2H four-fold CV."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPECTED_PREREG = "0824bef80a047d9bbd3eeb62fb57a6c410aeb784693017a5b1da325dc24e82db"
EXPECTED_PROGRAM = "d1472e602cc0a7cd612da8ee4341050c907875e0261bbc117aa78580aceeb551"
EXPECTED_MANIFEST = "efa5eb3516eaa15e2883b437e71463f08dbb26b1cc691bcd2f350f7253e43b47"
EXPECTED_SEAL = "682647e8c6e1d5f0ef8ef42b9568e3a5ede5a5ec73bd0e5dbb97dc58cb384197"
EXPECTED_CV = "55fd5ec3c50cbe70fad9c9f83e88625afc374cb5dd4e32261c1a1a1518727104"
TAIL_RANKS = [55, 71, 106, 317, 339, 458, 485, 507, 540, 575, 602]
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
    assert seal["data_gate"]["observer_selection_targets_allowed"] is False

    cv = load(HERE / "CV_RESULTS.json")
    assert cv["schema"] == "public-full688-state-independent-duhamel-cv-v1"
    assert cv["node"] == "P2R0F2H"
    assert cv["status"] == "complete"
    assert cv["observer_fit_networks"] == 688
    assert cv["observer_selection_targets_opened"] == 0
    assert cv["training"]["epochs"] == 6
    assert cv["architecture"]["parameters"] == 12880
    assert cv["architecture"]["predicted_state_inputs_to_source"] == 0

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
        assert len(trace) == 6
        assert all(math.isfinite(float(row["fit_objective"])) for row in trace)
        valid = [int(value) for value in record["valid_role_ranks"]]
        train = [int(value) for value in record["train_role_ranks"]]
        assert len(valid) == record["valid_networks"]
        assert len(train) == record["train_networks"]
        assert not set(valid).intersection(train)
        assert sorted(valid + train) == list(range(688))
        assert record["autonomous"]["finite_full_rollout"] is True
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
        close(
            cv["oof"]["autonomous_layer_mse"][layer],
            weighted_autonomous_layers[layer] / total,
        )
    close(cv["oof"]["autonomous_final_mse"], 3.908100560611721e-8)
    maximum = max(per_network, key=lambda row: float(row["final_mse"]))
    close(cv["oof"]["maximum_per_network_final_mse"], maximum["final_mse"])
    assert cv["oof"]["maximum_role_rank"] == maximum["role_rank"]
    assert cv["oof"]["maximum_mlp_id"] == maximum["mlp_id"]

    recomputed_gate = {
        "conjunctive_pass": False,
        "finite_all_autonomous_rollouts": True,
        "maximum_per_network_below_1e_7": float(maximum["final_mse"]) < MAXIMUM_GATE,
        "mean_below_2_9672528546945005e_8": (
            weighted_autonomous / total < MEAN_GATE
        ),
    }
    recomputed_gate["conjunctive_pass"] = all(
        value for key, value in recomputed_gate.items() if key != "conjunctive_pass"
    )
    assert cv["decision_gate"] == recomputed_gate
    assert cv["cost"]["host_plus_correction_upper_estimate"] == 2_127_770_171_325
    assert cv["cost"]["headroom_under_upper_estimate"] == 72_229_828_675
    assert cv["cost"]["within_phase2_budget_by_upper_estimate"] is True
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
