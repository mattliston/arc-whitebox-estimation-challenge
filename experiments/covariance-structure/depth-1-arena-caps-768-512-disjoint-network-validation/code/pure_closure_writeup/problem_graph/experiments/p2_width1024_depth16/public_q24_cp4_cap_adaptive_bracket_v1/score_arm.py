#!/usr/bin/env python3
"""Score one completed step of the frozen adaptive CP4 bracket."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREGISTRATION = HERE / "PREREGISTRATION.json"
PREDECESSOR_RESULT = P2 / "public_q24_cp4_cap_co_design_sign_screen_v1/RESULT.json"
BASE_SCORER = P2 / "public_q24_cp4_cap_co_design_sign_screen_v1/score_screen.py"
PREREGISTRATION_SHA256 = "6cb74cff24f014ef5fe101c4f8ddebff6ec4eb8a083bac116237e64aaec38f35"
PREDECESSOR_RESULT_SHA256 = "2c8b489633e180b8dfed90bd39d8f216f0c660d3b660ec39b9b85049436a90f7"
BASE_SCORER_SHA256 = "10e2bc0fceaa834d92dc2ea33b9ee1997efda52b8afff741d4644b8e4fc52e34"
ARM_RUNNER_SHA256 = "5783c596f4ac191cd3d26cf2148f16323a869b3a6a1d8e7351565bce51c0202d"
ARMS = ("cap640", "cap512", "cap384", "cap256")
RANKS = (35, 279)
SOURCE_SHA256 = {
    "cap640": "5b89058711507a327a08c48a7be508d5af9b0d7c6af54b2b4df59045598be080",
    "cap512": "3246a0238f397ad7b56f37551e6263120141152ae9affd7f22e3764da7603388",
    "cap384": "b2327145d0f7e1aceba70fd7a7b39c6caa9330cc79badaa98c148bf6135fea39",
    "cap256": "5356ea650cef99d0c8efecef4c99d8762dd3b472c99f1dccfc929ae903f6249f",
}
WEIGHTS_SHA256 = {
    35: "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2",
    279: "46b1e740e32de76c0b5d4d855ac307e8ba1509497303456f2dfc42ece2f34298",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    if sha256(BASE_SCORER) != BASE_SCORER_SHA256:
        raise AssertionError("audited scorer changed")
    spec = importlib.util.spec_from_file_location("cp4_adaptive_base_scorer", BASE_SCORER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_SCORER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def predecessor(arm: str) -> tuple[str, dict[str, object], str]:
    if arm in ("cap512", "cap640"):
        record = json.loads(PREDECESSOR_RESULT.read_text(encoding="utf-8"))
        return "cap768", record["arms"]["cap768"], sha256(PREDECESSOR_RESULT)
    previous = "cap512" if arm == "cap384" else "cap384"
    path = HERE / f"SCORE_{previous}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") not in ("STEP_ACCEPTED", "STEP_REVERSED"):
        raise AssertionError("predecessor step status changed")
    if record.get("arm") != previous:
        raise AssertionError("predecessor step identity changed")
    return previous, record["metrics"], sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    arguments = parser.parse_args()
    arm = arguments.arm
    result_path = HERE / f"SCORE_{arm}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise AssertionError("adaptive preregistration changed")
    if sha256(PREDECESSOR_RESULT) != PREDECESSOR_RESULT_SHA256:
        raise AssertionError("cap768 predecessor result changed")
    base = load_base()
    cap_values = []
    receipts = {}
    for rank in RANKS:
        directory = HERE / "outputs" / f"{arm}_rank{rank}"
        receipt_path = directory / "RECEIPT.json"
        prediction_path = directory / "prediction.npy"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "PASS_TARGET_FREE_ARM":
            raise AssertionError("arm receipt status changed")
        if receipt.get("arm") != arm or int(receipt.get("role_rank", -1)) != rank:
            raise AssertionError("arm receipt identity changed")
        expected_bindings = {
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_sha256": SOURCE_SHA256[arm],
            "weights_sha256": WEIGHTS_SHA256[rank],
            "runner_sha256": ARM_RUNNER_SHA256,
        }
        if receipt.get("bindings") != expected_bindings:
            raise AssertionError("arm receipt bindings changed")
        if sha256(prediction_path) != receipt["prediction_file_sha256"]:
            raise AssertionError("arm prediction file changed")
        value = np.load(prediction_path, allow_pickle=False)
        if value.dtype != np.float32 or value.shape != (16, 1024) or not np.isfinite(value).all():
            raise AssertionError("arm prediction schema changed")
        if base.array_sha256(value) != receipt["prediction_array_sha256"]:
            raise AssertionError("arm prediction contents changed")
        cap_values.append(np.ascontiguousarray(value))
        receipts[rank] = receipt

    g11b = np.stack([base.load_fixed(*base.G11B[rank]) for rank in RANKS])
    canonical = np.stack([base.load_fixed(*base.CANONICAL[rank]) for rank in RANKS])
    # Both new outputs are verified before this first and only target read.
    target, mlp_ids = base.load_targets_after_all_arm_checks()
    cap_output = np.stack(cap_values)
    delta = cap_output.astype(np.float64) - canonical.astype(np.float64)
    candidate = np.maximum(g11b.astype(np.float64) + delta, 0.0).astype(np.float32)
    metrics = base.metrics(candidate, target)
    prior_arm, prior, prior_hash = predecessor(arm)
    rank_improvements = [
        float(metrics["per_rank_final_layer_mse"][index])
        < float(prior["per_rank_final_layer_mse"][index])
        for index in range(2)
    ]
    pooled_improvement = (
        float(metrics["pooled_final_layer_mse"])
        < float(prior["pooled_final_layer_mse"])
    )
    accepted = all(rank_improvements) and pooled_improvement
    metrics.update(
        {
            "predecessor_arm": prior_arm,
            "absolute_pooled_gain_vs_predecessor": float(prior["pooled_final_layer_mse"])
            - float(metrics["pooled_final_layer_mse"]),
            "relative_pooled_gain_vs_predecessor": (
                float(prior["pooled_final_layer_mse"])
                - float(metrics["pooled_final_layer_mse"])
            ) / float(prior["pooled_final_layer_mse"]),
            "per_rank_strict_improvement": rank_improvements,
            "pooled_strict_improvement": pooled_improvement,
            "final_delta_rms": float(np.sqrt(np.mean(delta[:, -1] * delta[:, -1]))),
            "final_delta_maximum_absolute": float(np.max(np.abs(delta[:, -1]))),
            "literal_flops_by_rank": {
                str(rank): int(receipts[rank]["literal_meter"]["flops_used"])
                for rank in RANKS
            },
            "candidate_array_sha256": base.array_sha256(candidate),
        }
    )
    result = {
        "schema": "q24-cp4-cap-adaptive-two-rank-step-v1",
        "status": "STEP_ACCEPTED" if accepted else "STEP_REVERSED",
        "arm": arm,
        "role_ranks": list(RANKS),
        "mlp_ids": mlp_ids,
        "metrics": metrics,
        "step_accepted": accepted,
        "target_columns_opened": ["mlp_id", "all_layer_means"],
        "board_submission": False,
        "bindings": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "predecessor_result_sha256": prior_hash,
            "arm_runner_sha256": ARM_RUNNER_SHA256,
            "scoring_program_sha256": sha256(Path(__file__)),
            "arm_receipt_sha256": {
                str(rank): sha256(HERE / "outputs" / f"{arm}_rank{rank}" / "RECEIPT.json")
                for rank in RANKS
            },
        },
    }
    temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
