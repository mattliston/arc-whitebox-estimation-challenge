#!/usr/bin/env python3
"""Score the frozen two-rank CP4 cap co-design sign screen once."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREGISTRATION = HERE / "PREREGISTRATION.json"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
RESULT = HERE / "RESULT.json"
PREREGISTRATION_SHA256 = "9d5bb3045cea6d170c906155889b11d00fa3153d5f97e939c5a609f66c6c88cb"
TARGETS_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
ARM_RUNNER_SHA256 = "b4fe8329e81b91db4670b3c5d62d7abb5ead0697d7f60d345164673c1621bdb8"
ARMS = ("cap1792", "cap1024", "cap768")
RANKS = (35, 279)
SOURCE_SHA256 = {
    "cap1792": "c0e68e11af37e118ee9ff906bd607280e4353cf6d378e22e41fffdfe726414c6",
    "cap1024": "7a1f2e3f3a51ae3c44cc4717c56fd70d0f6bef11c4b2fabf27f3937de6900e0e",
    "cap768": "33c2f01770c95b1490c611a5b6d62c5e0568cf49ee8722cf52381c07ed088e26",
}
WEIGHTS_SHA256 = {
    35: "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2",
    279: "46b1e740e32de76c0b5d4d855ac307e8ba1509497303456f2dfc42ece2f34298",
}
G11B = {
    35: (
        P2 / "public_q24_d1_alpha075_qhead_bank_v1/integrated_outputs/candidate_0_rank35.npy",
        "db123f80b51c83085c93bd9b1c3197144676a11e2af38f2ae4f2da230d4ac653",
    ),
    279: (
        P2 / "public_q24_d1_alpha075_qhead_bank_v1/integrated_outputs/candidate_1_rank279.npy",
        "f10cf2721862835f313ce2a5ac82e7f23c01e1e543418710f5976a579170fe0b",
    ),
}
CANONICAL = {
    35: (
        P2 / "p2r0c1d5_cp3_winograd_hybrid_v1/reference_rank35_first.npy",
        "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
    ),
    279: (
        P2 / "p2r0c1d5_cp3_winograd_hybrid_v1/reference_rank279.npy",
        "f6970b298c3397b512635846c4fa75bb377ee6af0ca4ff0e6241db2dcd5015ba",
    ),
}
ROW_MANIFEST = {
    35: (
        P2 / "learned_residual_observer_design/v4_feature_store_v3/"
        "sealed_target_blind_feature_store_v3/rows/observer_fit/0035/row_manifest.json",
        "5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809",
        89,
    ),
    279: (
        P2 / "learned_residual_observer_design/v4_feature_store_v3/"
        "sealed_target_blind_feature_store_v3/rows/observer_fit/0279/row_manifest.json",
        "96850100a37a82803f4dfc0536230e7cf556bc9979024d7ada08e2c1c3108c6f",
        534,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def load_fixed(path: Path, expected: str) -> np.ndarray:
    if sha256(path) != expected:
        raise AssertionError(f"fixed array changed: {path}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (16, 1024) or not np.isfinite(value).all():
        raise AssertionError(f"fixed array schema changed: {path}")
    return np.ascontiguousarray(value)


def load_arms() -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, dict[str, object]]]]:
    values: dict[str, dict[int, np.ndarray]] = {}
    receipts: dict[str, dict[int, dict[str, object]]] = {}
    for arm in ARMS:
        values[arm] = {}
        receipts[arm] = {}
        for rank in RANKS:
            directory = HERE / "outputs" / f"{arm}_rank{rank}"
            receipt_path = directory / "RECEIPT.json"
            prediction_path = directory / "prediction.npy"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "PASS_TARGET_FREE_ARM":
                raise AssertionError("arm receipt status changed")
            if receipt.get("arm") != arm or int(receipt.get("role_rank", -1)) != rank:
                raise AssertionError("arm receipt identity changed")
            bindings = receipt.get("bindings", {})
            if bindings != {
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "source_sha256": SOURCE_SHA256[arm],
                "weights_sha256": WEIGHTS_SHA256[rank],
                "runner_sha256": ARM_RUNNER_SHA256,
            }:
                raise AssertionError("arm binding changed")
            if sha256(prediction_path) != receipt["prediction_file_sha256"]:
                raise AssertionError("arm prediction file changed")
            value = np.load(prediction_path, allow_pickle=False)
            if value.dtype != np.float32 or value.shape != (16, 1024) or not np.isfinite(value).all():
                raise AssertionError("arm prediction schema changed")
            if array_sha256(value) != receipt["prediction_array_sha256"]:
                raise AssertionError("arm prediction contents changed")
            values[arm][rank] = np.ascontiguousarray(value)
            receipts[arm][rank] = receipt
    return values, receipts


def load_targets_after_all_arm_checks() -> tuple[np.ndarray, list[int]]:
    if sha256(TARGETS) != TARGETS_SHA256:
        raise AssertionError("observer-fit target bundle changed")
    expected_ids = []
    for rank in RANKS:
        path, expected_hash, expected_id = ROW_MANIFEST[rank]
        if sha256(path) != expected_hash:
            raise AssertionError("row manifest changed")
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record["role_rank"]) != rank or int(record["mlp_id"]) != expected_id:
            raise AssertionError("row identity changed")
        expected_ids.append(expected_id)
    # This is deliberately the first and only target read in the program.
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    means = table.column("all_layer_means").combine_chunks()
    storage = getattr(means, "storage", means)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    all_targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)
    if ids.dtype != np.int32 or all_targets.dtype != np.float32:
        raise AssertionError("target schema changed")
    if [int(ids[rank]) for rank in RANKS] != expected_ids:
        raise AssertionError("target/weight identity alignment changed")
    return np.ascontiguousarray(all_targets[list(RANKS)], dtype=np.float32), expected_ids


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    return {
        "pooled_all_layer_mse": float(np.mean(error * error)),
        "pooled_final_layer_mse": float(np.mean(error[:, -1] * error[:, -1])),
        "per_rank_final_layer_mse": [
            float(np.mean(error[index, -1] * error[index, -1]))
            for index in range(len(RANKS))
        ],
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise AssertionError("preregistration changed")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_TARGET_OPEN":
        raise AssertionError("preregistration status changed")
    arm_values, receipts = load_arms()
    g11b = np.stack([load_fixed(*G11B[rank]) for rank in RANKS])
    canonical = np.stack([load_fixed(*CANONICAL[rank]) for rank in RANKS])
    # Only after all six target-free rows and fixed anchors pass verification.
    target, mlp_ids = load_targets_after_all_arm_checks()
    baseline = metrics(g11b, target)
    arms: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        cap_output = np.stack([arm_values[arm][rank] for rank in RANKS])
        delta = cap_output.astype(np.float64) - canonical.astype(np.float64)
        candidate = np.maximum(
            g11b.astype(np.float64) + delta, 0.0
        ).astype(np.float32)
        score = metrics(candidate, target)
        baseline_final = float(baseline["pooled_final_layer_mse"])
        candidate_final = float(score["pooled_final_layer_mse"])
        score.update(
            {
                "absolute_final_mse_gain": baseline_final - candidate_final,
                "relative_final_mse_gain": (baseline_final - candidate_final) / baseline_final,
                "final_delta_rms": float(np.sqrt(np.mean(delta[:, -1] * delta[:, -1]))),
                "final_delta_maximum_absolute": float(np.max(np.abs(delta[:, -1]))),
                "literal_flops_by_rank": {
                    str(rank): int(receipts[arm][rank]["literal_meter"]["flops_used"])
                    for rank in RANKS
                },
                "output_array_sha256": array_sha256(candidate),
            }
        )
        arms[arm] = score
    improving = [
        arm for arm in ARMS
        if float(arms[arm]["pooled_final_layer_mse"])
        < float(baseline["pooled_final_layer_mse"])
    ]
    selected = min(
        improving,
        key=lambda arm: (float(arms[arm]["pooled_final_layer_mse"]), arm),
        default=None,
    )
    result = {
        "schema": "q24-cp4-cap-co-design-two-rank-sign-screen-v1",
        "status": "COMPLETE_POSITIVE_SIGN" if improving else "COMPLETE_NO_POSITIVE_SIGN",
        "role": "observer_fit",
        "role_ranks": list(RANKS),
        "mlp_ids": mlp_ids,
        "target_columns_opened": ["mlp_id", "all_layer_means"],
        "baseline_g11b": baseline,
        "arms": arms,
        "promotion": {
            "improving_arms": improving,
            "selected_lowest_raw_mse_arm": selected,
            "larger_panel_licensed": bool(improving),
            "cap512_licensed": "cap768" in improving,
            "board_submission": False,
        },
        "bindings": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "scoring_program_sha256": sha256(Path(__file__)),
            "arm_runner_sha256": ARM_RUNNER_SHA256,
            "observer_fit_targets_sha256": sha256(TARGETS),
            "arm_receipt_sha256": {
                arm: {
                    str(rank): sha256(HERE / "outputs" / f"{arm}_rank{rank}" / "RECEIPT.json")
                    for rank in RANKS
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
