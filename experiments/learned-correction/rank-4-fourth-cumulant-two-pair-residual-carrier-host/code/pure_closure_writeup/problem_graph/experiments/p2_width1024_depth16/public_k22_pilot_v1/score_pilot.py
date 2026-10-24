#!/usr/bin/env python3
"""Stage-2 paired raw scoring for the K22-carrier pilot (truth opens here).

Refuses to run unless stage-1 coverage is complete for BOTH arms (400/400
receipts with the sealed source shas). Truth: sealed observer_fit targets,
rows 35..234, final layer only for the primary metric.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ROSTER = list(range(35, 235))
ARM_SHAS = {
    "base": "e4d250563279a8c99dceb72bd71428855bd798d651730fca9c57d6577aa6f62d",
    "k22r4": "702e971c410ed688355af93c075aa0e13b38cc6462d0864c6c663ba51b4bb593",
}


def verify_coverage():
    ids = {}
    for arm, expected_sha in ARM_SHAS.items():
        arm_ids = []
        for rank in ROSTER:
            receipt_path = HERE / "rows" / arm / f"row_{rank:04d}" / "result.json"
            assert receipt_path.exists(), f"missing {arm} rank {rank}"
            receipt = json.loads(receipt_path.read_text())
            assert receipt["status"] == "complete-target-free-k22pilot"
            assert receipt["source_sha256"] == expected_sha, f"{arm} {rank} source drift"
            assert receipt["targets_read"] == 0
            arm_ids.append(int(receipt["mlp_id"]))
        ids[arm] = arm_ids
    assert ids["base"] == ids["k22r4"], "arm row alignment broken"
    return ids["base"]


def load_truth(expected_ids):
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    mlp_ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)
    assert np.array_equal(mlp_ids[35:235], np.asarray(expected_ids, dtype=mlp_ids.dtype))
    return np.ascontiguousarray(targets[35:235], dtype=np.float64)


def load_final(arm):
    rows = [np.load(HERE / "rows" / arm / f"row_{rank:04d}" / "host.npy",
                    allow_pickle=False)[15] for rank in ROSTER]
    return np.asarray(np.stack(rows), dtype=np.float64)


def main():
    expected_ids = verify_coverage()
    truth_final = load_truth(expected_ids)[:, 15, :]
    per_row = {}
    for arm in ARM_SHAS:
        final = load_final(arm)
        per_row[arm] = np.mean((final - truth_final) ** 2, axis=1)
    base = per_row["base"]
    cand = per_row["k22r4"]
    delta = cand - base
    rel = float((cand.mean() - base.mean()) / base.mean())
    result = {
        "schema": "k22pilot-raw-result-v1",
        "node": "P2R0F3A",
        "rows": len(ROSTER),
        "base_mean_final_mse": float(base.mean()),
        "k22r4_mean_final_mse": float(cand.mean()),
        "relative_change": rel,
        "candidate_wins": int((delta < 0).sum()),
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_sd": float(delta.std(ddof=1)),
        "paired_t": float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)))),
        "per_row_base_min_max": [float(base.min()), float(base.max())],
        "per_row_k22r4_min_max": [float(cand.min()), float(cand.max())],
        "licensing": (
            "combined_card_build" if rel <= -0.10 else
            "iteration_needs_new_prereg" if rel < 0.0 else
            "close_r4_pivot_fifth_order"
        ),
        "truth_rows_opened": "observer_fit 35..234 final layer",
        "arm_shas": ARM_SHAS,
    }
    out = HERE / "PILOT_RAW_RESULT.json"
    assert not out.exists(), "append-only"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1))
    print("receipt_sha256:", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
