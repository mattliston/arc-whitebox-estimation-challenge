#!/usr/bin/env python3
"""Score exact D1 CP4 caps on the frozen disjoint 16-network panel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREGISTRATION = HERE / "PREREGISTRATION.json"
ROSTER_AUTHORITY = P2 / "public_q24_cp4_cap_disjoint16_validation_v1/PREREGISTRATION.json"
BASE_SCORER = P2 / "public_q24_cp4_cap_disjoint16_validation_v1/score_panel.py"
RESULT = HERE / "RESULT.json"
PREREGISTRATION_SHA256 = "aa93dc769d3388512a6262d6ff77b10438b9796c183369ec6a946b19e40477fd"
ROSTER_AUTHORITY_SHA256 = "2cbaedbf506d345234e744ca1e5521faf6823a922659ea177f04781e847a6c51"
BASE_SCORER_SHA256 = "d12b3cd8f01fc9ca727a6a7e32fddc29bcd651ead51209a5ae74858002d47ba5"
ARM_RUNNER_SHA256 = "ff30f1a901ce363425846c073a7e788d98df11fc58533e5c96c6ee0c9b9557fc"
ARMS = ("d1", "cap768", "cap512")
SOURCE_SHA256 = {
    "d1": "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e",
    "cap768": "5e631aea5fcfba83ccae15a97f820c82762a55b70400f61fbf4ef9e225216a11",
    "cap512": "bfae86342955e8a4795a0e4de974d607057b81466509d6f227bc89d5f392f0c7",
}
EXPECTED_SAVING = {"cap768": 103_776_627_200, "cap512": 125_843_312_640}
MINIMUM_NETWORK_WINS = 10
BOARD_RAW = 2.9409214512554626e-8
BOARD_ADJUSTED = 2.1258043062525995e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    if sha256(BASE_SCORER) != BASE_SCORER_SHA256:
        raise AssertionError("audited panel scorer changed")
    spec = importlib.util.spec_from_file_location("d1_cp4_disjoint16_base_scorer", BASE_SCORER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_SCORER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise AssertionError("D1 cap preregistration changed")
    if sha256(ROSTER_AUTHORITY) != ROSTER_AUTHORITY_SHA256:
        raise AssertionError("roster authority changed")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_PANEL_TARGET_OPEN":
        raise AssertionError("preregistration status changed")
    roster_prereg = json.loads(ROSTER_AUTHORITY.read_text(encoding="utf-8"))
    base = load_base()
    ranks, mlp_ids, weight_hashes = base.validate_roster(roster_prereg)
    base.HERE = HERE
    base.PREREGISTRATION_SHA256 = PREREGISTRATION_SHA256
    base.ARM_RUNNER_SHA256 = ARM_RUNNER_SHA256
    base.ARMS = ARMS
    base.SOURCE_SHA256 = SOURCE_SHA256
    values, receipts = base.load_all_outputs(ranks, weight_hashes)
    # The inherited loader performs the only target read after all 48 receipts.
    target = base.load_targets_after_output_verification(ranks, mlp_ids)
    baseline = base.network_final_mse(values["d1"], target)
    arms: dict[str, dict[str, object]] = {}
    passing = []
    for arm in ("cap768", "cap512"):
        candidate = base.network_final_mse(values[arm], target)
        gain = baseline - candidate
        wins = int(np.sum(gain > 0.0))
        mean_gate = float(np.mean(candidate)) < float(np.mean(baseline))
        majority_gate = wins >= MINIMUM_NETWORK_WINS
        baseline_bills = np.asarray(
            [int(receipts["d1"][rank]["literal_meter"]["flops_used"]) for rank in ranks],
            dtype=np.float64,
        )
        candidate_bills = np.asarray(
            [int(receipts[arm][rank]["literal_meter"]["flops_used"]) for rank in ranks],
            dtype=np.float64,
        )
        savings = (baseline_bills - candidate_bills).astype(np.int64)
        # The literal meter reports several discrete structural classes across
        # this panel, so paired absolute savings need not equal the nominal
        # dense-fixture saving.  The frozen source relation is still a single
        # cap-literal edit; require it to save work on every paired network and
        # retain every measured bill/saving for audit.
        if not np.all(savings > 0):
            raise AssertionError(f"{arm} failed to save FLOPs on every rank")
        raw_ratio = float(np.mean(candidate) / np.mean(baseline))
        compute_ratio = float(np.max(candidate_bills / baseline_bills))
        standard_error = float(np.std(gain, ddof=1) / math.sqrt(len(gain)))
        record = {
            "mean_network_final_mse": float(np.mean(candidate)),
            "median_network_final_mse": float(np.median(candidate)),
            "baseline_mean_network_final_mse": float(np.mean(baseline)),
            "raw_ratio": raw_ratio,
            "relative_mean_mse_gain": float(np.mean(gain) / np.mean(baseline)),
            "mean_absolute_mse_gain": float(np.mean(gain)),
            "median_absolute_mse_gain": float(np.median(gain)),
            "network_win_count": wins,
            "network_loss_count": int(np.sum(gain < 0.0)),
            "network_tie_count": int(np.sum(gain == 0.0)),
            "per_network_final_mse": [float(value) for value in candidate],
            "per_network_absolute_gain": [float(value) for value in gain],
            "paired_gain_standard_error": standard_error,
            "paired_gain_t_statistic": float(np.mean(gain) / standard_error) if standard_error > 0 else None,
            "mean_mse_gate": mean_gate,
            "clear_majority_gate": majority_gate,
            "conjunctive_pass": mean_gate and majority_gate,
            "nominal_expected_structural_flop_saving": EXPECTED_SAVING[arm],
            "observed_structural_flop_saving_by_rank": {
                str(rank): int(saving) for rank, saving in zip(ranks, savings)
            },
            "minimum_observed_structural_flop_saving": int(np.min(savings)),
            "maximum_observed_structural_flop_saving": int(np.max(savings)),
            "mean_observed_structural_flop_saving": float(np.mean(savings)),
            "conservative_max_compute_ratio": compute_ratio,
            "projected_board_raw_mse": BOARD_RAW * raw_ratio,
            "projected_board_adjusted_score": BOARD_ADJUSTED * raw_ratio * compute_ratio,
            "literal_flops_by_rank": {
                str(rank): int(receipts[arm][rank]["literal_meter"]["flops_used"])
                for rank in ranks
            },
        }
        arms[arm] = record
        if record["conjunctive_pass"]:
            passing.append(arm)
    selected = min(
        passing,
        key=lambda arm: (float(arms[arm]["mean_network_final_mse"]), int(arm[3:])),
        default=None,
    )
    result = {
        "schema": "q24-d1-cp4-cap-exact-integrated-disjoint16-result-v1",
        "status": "PASS_CAP_VALIDATION" if passing else "NO_CAP_PASSES_VALIDATION",
        "role": "observer_fit",
        "role_ranks": ranks,
        "mlp_ids": mlp_ids,
        "baseline_d1": {
            "mean_network_final_mse": float(np.mean(baseline)),
            "median_network_final_mse": float(np.median(baseline)),
            "per_network_final_mse": [float(value) for value in baseline],
        },
        "arms": arms,
        "selection": {
            "passing_arms": passing,
            "selected_arm": selected,
            "minimum_network_wins": MINIMUM_NETWORK_WINS,
            "direct_cap_integration_licensed": selected is not None,
            "board_submission": False,
        },
        "target_columns_opened": ["mlp_id", "all_layer_means"],
        "bindings": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "roster_authority_sha256": sha256(ROSTER_AUTHORITY),
            "runner_sha256": ARM_RUNNER_SHA256,
            "scoring_program_sha256": sha256(Path(__file__)),
            "observer_fit_targets_sha256": base.sha256(base.TARGETS),
            "arm_receipt_sha256": {
                arm: {
                    str(rank): sha256(HERE / "outputs" / f"{arm}_rank{rank}" / "RECEIPT.json")
                    for rank in ranks
                }
                for arm in ARMS
            },
        },
    }
    temporary = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESULT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
