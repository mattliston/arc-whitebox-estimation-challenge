#!/Users/matt/anaconda3/bin/python3
"""Independent artifact-level checks for frozen P2R0F2I four-fold CV."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPECTED_PREREG = "fdbe5d236aa9d80c4e339d72a773ccea8d74847f97270ccba8fbc270275ee00e"
EXPECTED_PROGRAM = "9c1c83ed0fbe841798568800b0123f6a88bac68629cf7bb8598e7ed29fce2616"
EXPECTED_MANIFEST = "dab3dfc0967053d3b9cf098b1e6c38edd14fe5107b44987bd2049a25301a1850"
EXPECTED_SEAL = "9ed5648bd945266977c23759506d0acdafcdeed99ec54b4abb0ae7eee06634ae"
EXPECTED_CV = "ab810134e83924b698eca0647f97a44ba3ef884653bb3fb3e6516862f00b445f"
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
    assert seal["data_gate"]["observer_selection_targets_allowed"] is False

    cv = load(HERE / "CV_RESULTS.json")
    assert cv["schema"] == "public-full688-typed-diagonal-error-variance-cv-v1"
    assert cv["node"] == "P2R0F2I"
    assert cv["status"] == "complete"
    assert cv["observer_fit_networks"] == 688
    assert cv["observer_selection_targets_opened"] == 0
    assert cv["training"]["epochs"] == 6
    assert cv["training"]["own_mean_state_exposure_schedule"] == EXPOSURE
    assert cv["training"]["variance_state_exposure"] == "fully autonomous"
    assert cv["training"]["variance_loss_weight"] == 0.05
    assert cv["architecture"]["parameters"] == 16368
    assert cv["architecture"]["input_features"] == 24

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
        assert record["schema"] == "public-full688-typed-diagonal-error-variance-fold-v1"
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
        assert [row["own_mean_state_exposure"] for row in trace] == EXPOSURE
        assert all(
            math.isfinite(float(row[key]))
            for row in trace
            for key in ("fit_objective", "primary_objective", "variance_score")
        )
        valid = [int(value) for value in record["valid_role_ranks"]]
        train = [int(value) for value in record["train_role_ranks"]]
        assert len(valid) == record["valid_networks"]
        assert len(train) == record["train_networks"]
        assert not set(valid).intersection(train)
        assert sorted(valid + train) == list(range(688))
        assert record["autonomous"]["finite_full_rollout"] is True
        assert record["autonomous"]["maximum_variance_state"] >= 0.0
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
    close(cv["oof"]["autonomous_final_mse"], 3.5165106535091496e-8)
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
    assert cv["cost"]["parameter_count"] == 16368
    assert cv["cost"]["correction_source_structure_upper_estimate"] == 842_887_168
    assert cv["cost"]["host_plus_correction_upper_estimate"] == 2_127_820_346_301
    assert cv["cost"]["headroom_under_upper_estimate"] == 72_179_653_699
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
