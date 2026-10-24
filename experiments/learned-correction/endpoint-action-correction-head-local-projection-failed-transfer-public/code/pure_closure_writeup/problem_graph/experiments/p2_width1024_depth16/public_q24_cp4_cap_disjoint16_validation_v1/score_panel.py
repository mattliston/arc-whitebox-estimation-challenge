#!/usr/bin/env python3
"""One-shot score of the frozen exact-integrated disjoint 16-panel."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREGISTRATION = HERE / "PREREGISTRATION.json"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
FEATURE_STORE = (
    P2 / "learned_residual_observer_design/v4_feature_store_v3/"
    "sealed_target_blind_feature_store_v3"
)
FEATURE_MANIFEST = FEATURE_STORE / "manifest.json"
RESULT = HERE / "RESULT.json"
PREREGISTRATION_SHA256 = "2cbaedbf506d345234e744ca1e5521faf6823a922659ea177f04781e847a6c51"
TARGETS_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
ARM_RUNNER_SHA256 = "f4bf3a882fca125b4023eefe605cbd5ba286406c726c3bed80f0047125fc23e6"
ARMS = ("g11b", "cap768", "cap512")
SOURCE_SHA256 = {
    "g11b": "a6e332c5372839409340b100484d81e423f23239849e0c9de11b72669f9367ea",
    "cap768": "91dce2514914c8b9b930f709e5c87185ce35197ebcfc4b3d4b1815f22b5065ed",
    "cap512": "b8ce15b72ad68356b5f46cfd56ee118afa3e8d1e911815a5d5370a88868f159c",
}
EXPECTED_SAVING = {"cap768": 103_776_627_200, "cap512": 125_843_312_640}
MINIMUM_NETWORK_WINS = 10


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


def validate_roster(prereg: dict[str, object]) -> tuple[list[int], list[int], dict[int, str]]:
    if sha256(FEATURE_MANIFEST) != FEATURE_MANIFEST_SHA256:
        raise AssertionError("feature manifest changed")
    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    eligible = []
    for rank, entry in enumerate(entries):
        if 35 <= rank <= 234 or rank == 279:
            continue
        mlp_id = int(entry["mlp_id"])
        digest = hashlib.blake2s(
            f"q24-cp4-cap-disjoint16-v1:{mlp_id}".encode("ascii"), digest_size=16
        ).hexdigest()
        eligible.append((digest, rank, mlp_id, str(entry["sha256"])))
    selected = sorted(eligible)[:16]
    roster = prereg["roster"]
    expected = [
        (str(row["hash"]), int(row["role_rank"]), int(row["mlp_id"]), str(row["row_manifest_sha256"]))
        for row in roster
    ]
    observed = [(digest, rank, mlp_id, row_hash) for digest, rank, mlp_id, row_hash in selected]
    if observed != expected:
        raise AssertionError("hash-selected roster changed")
    weight_hashes = {}
    for row in roster:
        rank = int(row["role_rank"])
        entry = entries[rank]
        row_path = FEATURE_STORE / str(entry["path"])
        if sha256(row_path) != str(row["row_manifest_sha256"]):
            raise AssertionError("row manifest binding changed")
        row_record = json.loads(row_path.read_text(encoding="utf-8"))
        if row_record["files"]["weights.npy"]["sha256"] != row["weights_sha256"]:
            raise AssertionError("weight binding changed")
        weight_hashes[rank] = str(row["weights_sha256"])
    return [item[1] for item in selected], [item[2] for item in selected], weight_hashes


def load_all_outputs(
    ranks: list[int], weight_hashes: dict[int, str]
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, dict[str, object]]]]:
    values: dict[str, np.ndarray] = {}
    receipts: dict[str, dict[int, dict[str, object]]] = {}
    for arm in ARMS:
        arm_values = []
        receipts[arm] = {}
        for rank in ranks:
            directory = HERE / "outputs" / f"{arm}_rank{rank}"
            receipt_path = directory / "RECEIPT.json"
            prediction_path = directory / "prediction.npy"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "PASS_TARGET_FREE_ARM":
                raise AssertionError("arm receipt status changed")
            if receipt.get("arm") != arm or int(receipt.get("role_rank", -1)) != rank:
                raise AssertionError("arm receipt identity changed")
            if receipt.get("bindings") != {
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "source_sha256": SOURCE_SHA256[arm],
                "weights_sha256": weight_hashes[rank],
                "runner_sha256": ARM_RUNNER_SHA256,
            }:
                raise AssertionError("arm receipt binding changed")
            if sha256(prediction_path) != receipt["prediction_file_sha256"]:
                raise AssertionError("prediction file changed")
            value = np.load(prediction_path, allow_pickle=False)
            if value.dtype != np.float32 or value.shape != (16, 1024) or not np.isfinite(value).all():
                raise AssertionError("prediction schema changed")
            if array_sha256(value) != receipt["prediction_array_sha256"]:
                raise AssertionError("prediction contents changed")
            arm_values.append(np.ascontiguousarray(value))
            receipts[arm][rank] = receipt
        values[arm] = np.stack(arm_values)
    return values, receipts


def load_targets_after_output_verification(ranks: list[int], mlp_ids: list[int]) -> np.ndarray:
    if sha256(TARGETS) != TARGETS_SHA256:
        raise AssertionError("target bundle changed")
    # This is deliberately the only target read and occurs after all 48 receipts.
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    means = table.column("all_layer_means").combine_chunks()
    storage = getattr(means, "storage", means)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    target = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)
    if ids.dtype != np.int32 or target.dtype != np.float32:
        raise AssertionError("target schema changed")
    if [int(ids[rank]) for rank in ranks] != mlp_ids:
        raise AssertionError("target/weight identity alignment changed")
    return np.ascontiguousarray(target[ranks], dtype=np.float32)


def network_final_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = prediction[:, -1].astype(np.float64) - target[:, -1].astype(np.float64)
    return np.mean(error * error, axis=1)


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise AssertionError("preregistration changed")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_PANEL_TARGET_OPEN":
        raise AssertionError("preregistration status changed")
    ranks, mlp_ids, weight_hashes = validate_roster(prereg)
    values, receipts = load_all_outputs(ranks, weight_hashes)
    target = load_targets_after_output_verification(ranks, mlp_ids)
    baseline = network_final_mse(values["g11b"], target)
    arms: dict[str, dict[str, object]] = {}
    passing = []
    for arm in ("cap768", "cap512"):
        candidate = network_final_mse(values[arm], target)
        gain = baseline - candidate
        wins = int(np.sum(gain > 0.0))
        mean_gate = float(np.mean(candidate)) < float(np.mean(baseline))
        majority_gate = wins >= MINIMUM_NETWORK_WINS
        saving = [
            int(receipts["g11b"][rank]["literal_meter"]["flops_used"])
            - int(receipts[arm][rank]["literal_meter"]["flops_used"])
            for rank in ranks
        ]
        if saving != [EXPECTED_SAVING[arm]] * 16:
            raise AssertionError(f"{arm} structural bill changed")
        standard_error = float(np.std(gain, ddof=1) / math.sqrt(len(gain)))
        record = {
            "mean_network_final_mse": float(np.mean(candidate)),
            "median_network_final_mse": float(np.median(candidate)),
            "baseline_mean_network_final_mse": float(np.mean(baseline)),
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
            "exact_structural_flop_saving": EXPECTED_SAVING[arm],
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
        "schema": "q24-cp4-cap-exact-integrated-disjoint16-validation-result-v1",
        "status": "PASS_CAP_VALIDATION" if passing else "NO_CAP_PASSES_VALIDATION",
        "role": "observer_fit",
        "role_ranks": ranks,
        "mlp_ids": mlp_ids,
        "baseline_g11b": {
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
            "runner_sha256": ARM_RUNNER_SHA256,
            "scoring_program_sha256": sha256(Path(__file__)),
            "observer_fit_targets_sha256": sha256(TARGETS),
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
