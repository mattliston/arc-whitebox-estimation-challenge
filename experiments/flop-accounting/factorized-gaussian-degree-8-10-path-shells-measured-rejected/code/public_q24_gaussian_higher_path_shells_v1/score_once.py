#!/usr/bin/env python3
"""Open the rank-0 target once after the higher-shell replay is frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
REFERENCE = P2 / "public_q24_cp3_age_retirement_discriminator_v1/runs/rank0000_full.npz"
DIAGNOSTIC = P2 / "public_phase1_kerdock_hidden_snap_v1/runs/row_0000/DIAGNOSTIC_ARRAYS_BASE.npz"
WEIGHTS = (
    P2 / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0000/weights.npy"
)
LEADER_ADJUSTED = 7.67408396767286e-9
DEPTH = 16
CANDIDATES = (
    "d8_134",
    "d8_224",
    "d8_paths",
    "d10_145",
    "d10_235",
    "d10_paths",
    "d8_d10_paths",
    "d6_d8_d10_paths",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mse(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.mean(value * value))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1).copy()
    b = np.asarray(right, dtype=np.float64).reshape(-1).copy()
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def local_q(state: np.ndarray, phi: np.ndarray, weights: np.ndarray) -> np.ndarray:
    q = np.zeros_like(state, dtype=np.float32)
    q[2] = state[2]
    for layer in range(3, DEPTH):
        q[layer] = state[layer] - phi[layer] * (state[layer - 1] @ weights[layer])
    return q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_replay.resolve()
    source_receipt_path = source.with_suffix(".json")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_receipt = json.loads(source_receipt_path.read_text())
    if source_receipt.get("status") != "COMPLETE_TARGET_BLIND_SOURCE_REPLAY":
        raise RuntimeError(source_receipt.get("status"))
    if source_receipt["output"]["sha256"] != sha256(source):
        raise RuntimeError("source replay hash mismatch")
    with np.load(source, allow_pickle=False) as tape:
        anchor_j = np.ascontiguousarray(tape["anchor_native_j"], dtype=np.float32)
        candidate_j = {
            label: np.ascontiguousarray(tape[f"{label}_native_j"], dtype=np.float32)
            for label in CANDIDATES
        }
    with np.load(REFERENCE, allow_pickle=False) as reference:
        canonical_host = np.ascontiguousarray(reference["host"], dtype=np.float32)
        canonical_j = np.ascontiguousarray(reference["native_j_output"], dtype=np.float32)
        canonical_raw = np.ascontiguousarray(reference["native_raw"], dtype=np.float32)
    # Sole target-bearing open, after the frozen source hash check above.
    with np.load(DIAGNOSTIC, allow_pickle=False) as diagnostic:
        target = np.ascontiguousarray(diagnostic["target"], dtype=np.float32)
    weights = np.ascontiguousarray(np.load(WEIGHTS, allow_pickle=False), dtype=np.float32)
    phi = canonical_raw[:, :, 5]
    canonical_correction = canonical_j - canonical_host
    truth_state = target - canonical_host
    q_truth = local_q(truth_state, phi, weights)
    q_bank = local_q(canonical_correction, phi, weights)
    q_gap = q_truth - q_bank
    canonical_raw_mse = mse(
        canonical_j[-1].astype(np.float64) - target[-1].astype(np.float64)
    )
    metrics: dict[str, object] = {"canonical": {"raw_mse": canonical_raw_mse}}
    positive = []
    leader = []
    for label, replay_j in candidate_j.items():
        delta = replay_j.astype(np.float64) - anchor_j.astype(np.float64)
        reanchored = canonical_j.astype(np.float64) + delta
        delta_q = local_q(delta.astype(np.float32), phi, weights)
        costs = source_receipt["cost_by_candidate"][label]
        raw = mse(reanchored[-1] - target[-1].astype(np.float64))
        passes_positive = raw < float(costs["adjusted_positive_raw_below"])
        passes_leader = raw <= float(costs["leader_raw_mse_gate"])
        if passes_positive:
            positive.append(label)
        if passes_leader:
            leader.append(label)
        layer_rows = []
        for layer in range(10, DEPTH):
            layer_rows.append(
                {
                    "layer_zero_based": layer,
                    "delta_q_mse": mse(delta_q[layer]),
                    "truth_gap_mse": mse(q_gap[layer]),
                    "delta_q_truth_gap_correlation": correlation(delta_q[layer], q_gap[layer]),
                    "remaining_gap_mse": mse(q_gap[layer] - delta_q[layer]),
                }
            )
        metrics[label] = {
            "canonical_reanchored_raw_mse": raw,
            "projected_adjusted": raw * int(costs["combined_flops"]) / (2**41),
            "passes_adjusted_positive_breakeven": passes_positive,
            "passes_leader_raw_gate": passes_leader,
            "endpoint_delta_mse": mse(delta[-1]),
            "endpoint_delta_truth_improvement_correlation": correlation(
                delta[-1], target[-1].astype(np.float64) - canonical_j[-1].astype(np.float64)
            ),
            "late_q_flat_correlation": correlation(delta_q[10:], q_gap[10:]),
            "layers10_15": layer_rows,
        }
    if leader:
        status = "PASS_LEADER"
    elif positive:
        status = "PASS_ADJUSTED_POSITIVE_COMPONENT"
    else:
        status = "FAIL_TERMINAL"
    receipt = {
        "schema": "q24-gaussian-higher-path-shells-rank0-score-v1",
        "status": status,
        "passing_adjusted_positive_arms": positive,
        "passing_leader_arms": leader,
        "leader_adjusted": LEADER_ADJUSTED,
        "metrics": metrics,
        "cost_by_candidate": source_receipt["cost_by_candidate"],
        "bindings": {
            "source_replay_sha256": sha256(source),
            "source_receipt_sha256": sha256(source_receipt_path),
            "reference_sha256": sha256(REFERENCE),
            "target_artifact_sha256": sha256(DIAGNOSTIC),
            "weights_sha256": sha256(WEIGHTS),
            "scorer_sha256": sha256(Path(__file__)),
        },
        "access": {
            "target_opened_only_after_frozen_source_hash_check": True,
            "fitting": False,
            "selection": False,
            "sampling": False,
            "submission": False,
        },
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
