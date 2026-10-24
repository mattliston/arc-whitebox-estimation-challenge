#!/usr/bin/env python3
"""Lightweight structural verifier for the bounded OOF ensemble screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    result = json.loads((HERE / "RESULT.json").read_text(encoding="utf-8"))
    assert result["schema"] == "public-observer-fit-oof-ensemble-screen-v1"
    assert result["status"] == "complete"
    scope = result["scope"]
    assert scope == {
        "board_opened": 0,
        "group_key": "mlp_id",
        "names_opened": 0,
        "networks": 688,
        "new_q24_targets_opened": 0,
        "observer_selection_targets_opened": 0,
        "outer_folds": 4,
        "role": "observer_fit",
    }
    assert result["candidate_order"] == ["D", "J", "I", "K", "M"]
    assert result["alignment"]["fold_counts"] == [147, 172, 171, 198]
    assert result["alignment"]["all_candidates_same_role_rank_partition"] is True
    assert result["alignment"]["all_candidates_same_mlp_id_partition"] is True

    artifact = HERE / result["prediction_artifact"]["path"]
    assert artifact.is_file()
    assert sha256(artifact) == result["prediction_artifact"]["sha256"]
    assert result["prediction_artifact"]["shape"] == [5, 688, 16, 1024]
    assert result["prediction_artifact"]["dtype"] == "float32"

    for label, record in result["inventory"].items():
        directory = P2 / record["directory"]
        assert sha256(directory / "run_experiment.py") == record["program_sha256"]
        assert sha256(directory / "CV_RESULTS.json") == record["cv_results_sha256"]
        assert record["saved_unit_level_oof_predictions_before_this_screen"] is False
        assert len(record["folds"]) == 4
        reproduction = result["reproduction"][label]
        assert reproduction["absolute_difference"] < 2e-15

    for record in result["held_methods"].values():
        assert len(record["fold_final_mse"]) == 4
        assert record["raw_final_mse"] > 0.0
        assert record["maximum_per_network_final_mse"] >= record["raw_final_mse"]
        assert record["deployment_cost"][
            "within_phase2_budget_by_conservative_upper_estimate"
        ] is True

    decision = result["decision"]
    held_best = min(
        record["raw_final_mse"] for record in result["held_methods"].values()
    )
    assert abs(held_best - decision["best_held_raw_final_mse"]) < 1e-20
    assert held_best >= decision["material_gate"]
    assert decision["promote_deployable_form"] is False
    assert decision["family_status"] == "closed_measured"
    assert result["oracle"]["nondeployable_reason"]
    print(
        json.dumps(
            {
                "status": "pass",
                "best_held_method": decision["best_held_method"],
                "best_held_raw_final_mse": held_best,
                "family_status": decision["family_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
