#!/usr/bin/env python3
"""Stage-2 paired scoring: combined card vs the sealed K22-pilot base rows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
BASE_ROWS = P2 / "public_k22_pilot_v1/rows/base"
ROSTER = list(range(35, 235))
COMBINED_SHA = "14c9288da98fbc80f009f0880eb87d73d2ce60016161aa4e5c022e1c942d1b76"


def verify_coverage():
    ids = []
    flops_seen = []
    for rank in ROSTER:
        receipt_path = HERE / "rows/combined" / f"row_{rank:04d}" / "result.json"
        assert receipt_path.exists(), f"missing combined rank {rank}"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "complete-target-free-combined"
        assert receipt["source_sha256"] == COMBINED_SHA
        assert receipt["targets_read"] == 0
        ids.append(int(receipt["mlp_id"]))
        flops_seen.append(int(receipt["flops_used"]))
        base_receipt = json.loads(
            (BASE_ROWS / f"row_{rank:04d}" / "result.json").read_text())
        assert int(base_receipt["mlp_id"]) == ids[-1]
    return ids, flops_seen


def load_truth(expected_ids):
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    mlp_ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)
    assert np.array_equal(mlp_ids[35:235], np.asarray(expected_ids, dtype=mlp_ids.dtype))
    return np.ascontiguousarray(targets[35:235, 15], dtype=np.float64)


def main():
    ids, flops_seen = verify_coverage()
    truth_final = load_truth(ids)
    base_final = np.stack([
        np.load(BASE_ROWS / f"row_{rank:04d}/host.npy", allow_pickle=False)[15]
        for rank in ROSTER]).astype(np.float64)
    combined_final = np.stack([
        np.load(HERE / "rows/combined" / f"row_{rank:04d}/host.npy",
                allow_pickle=False)[15]
        for rank in ROSTER]).astype(np.float64)
    base = np.mean((base_final - truth_final) ** 2, axis=1)
    cand = np.mean((combined_final - truth_final) ** 2, axis=1)
    delta = cand - base
    rel = float((cand.mean() - base.mean()) / base.mean())
    budget = 2**41
    result = {
        "schema": "combined-pilot-raw-result-v1",
        "node": "P2R0F3A",
        "rows": len(ROSTER),
        "base_mean_final_mse": float(base.mean()),
        "combined_mean_final_mse": float(cand.mean()),
        "relative_change": rel,
        "combined_wins": int((delta < 0).sum()),
        "paired_t": float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)))),
        "flops_min_max": [min(flops_seen), max(flops_seen)],
        "worst_budget_margin": budget - max(flops_seen),
        "licensing": (
            "fire" if rel <= -0.25 else
            "fire_if_projection_clears_2.6e-8" if rel <= -0.10 else
            "hold_decompose"
        ),
        "base_arm_note": "sealed K22-pilot base rows (W3 spelling, 0.01 percent substitution bias accepted in the preregistration)",
        "combined_sha256": COMBINED_SHA,
    }
    out = HERE / "PILOT_RAW_RESULT.json"
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1))
    print("receipt_sha256:", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
