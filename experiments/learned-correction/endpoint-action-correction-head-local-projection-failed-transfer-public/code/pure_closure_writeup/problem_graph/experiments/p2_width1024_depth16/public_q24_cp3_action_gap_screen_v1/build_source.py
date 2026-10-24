#!/usr/bin/env python3
"""Build the target-blind source matrix for the rank-0 CP3 action screen."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
QUARANTINE = P2 / "public_q24_cp3_full_snapshot_rank0_v1/runs/row_0000"
REFERENCE = P2 / "public_q24_cp3_age_retirement_discriminator_v1/runs/rank0000_full.npz"
WEIGHTS = (
    P2 / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0000/weights.npy"
)
PREREGISTER = HERE / "PREREGISTER.md"
SOURCE = HERE / "SOURCE.npz"
RECEIPT = HERE / "SOURCE_RECEIPT.json"
LAYERS = tuple(range(10, 16))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def layer_array(layer: int, name: str) -> np.ndarray:
    return np.load(
        QUARANTINE / f"layer_{layer:02d}" / f"{name}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )


def coordinate_folds() -> np.ndarray:
    folds = np.empty(1024, dtype=np.int8)
    for coordinate in range(1024):
        digest = hashlib.blake2s(
            f"p2-cp3-action-gap-v1:{coordinate}".encode("ascii"), digest_size=4
        ).digest()
        folds[coordinate] = int.from_bytes(digest, "little") % 5
    counts = np.bincount(folds, minlength=5)
    if int(np.min(counts)) < 170 or int(np.max(counts)) > 230:
        raise AssertionError(counts)
    return folds


def final_cp_k3() -> np.ndarray:
    lam = np.asarray(layer_array(15, "lambda_pre"), dtype=np.float32)
    p = np.asarray(layer_array(15, "pushed_p"), dtype=np.float32)
    q = np.asarray(layer_array(15, "pushed_q"), dtype=np.float32)
    if p.shape != q.shape or p.shape != (1024, lam.shape[0]):
        raise AssertionError((lam.shape, p.shape, q.shape))
    active = np.empty_like(p)
    np.multiply(p, p, out=active)
    np.multiply(active, q, out=active)
    return np.asarray(3.0 * (active @ lam), dtype=np.float64)


def main() -> None:
    if SOURCE.exists() or RECEIPT.exists():
        raise FileExistsError("source output already exists")
    weights = np.load(WEIGHTS, mmap_mode="r", allow_pickle=False)
    with np.load(REFERENCE, allow_pickle=False) as reference:
        # Deliberately target-blind named access only.  q_truth/q_bank are not opened.
        raw20 = np.ascontiguousarray(reference["native_raw"], dtype=np.float64)
        autonomous = np.ascontiguousarray(
            reference["native_j_correction"], dtype=np.float64
        )
    if raw20.shape != (16, 1024, 20) or autonomous.shape != (16, 1024):
        raise AssertionError((raw20.shape, autonomous.shape))
    qj = np.zeros_like(autonomous)
    for layer in range(2, 16):
        transported = np.asarray(autonomous[layer - 1] @ weights[layer], dtype=np.float64)
        qj[layer] = autonomous[layer] - raw20[layer, :, 5] * transported

    cp_k3 = np.empty((6, 1024), dtype=np.float64)
    sigma = np.empty_like(cp_k3)
    for local, layer in enumerate(LAYERS[:-1]):
        cp_k3[local] = np.diag(np.asarray(layer_array(layer, "cp_pre_k21")))
        sigma[local] = np.asarray(layer_array(layer, "sigma"), dtype=np.float64)
    cp_k3[-1] = final_cp_k3()
    sigma[-1] = np.asarray(layer_array(15, "sigma"), dtype=np.float64)
    cp3_action = cp_k3 / (sigma * sigma * sigma)
    if not bool(np.all(np.isfinite(cp3_action))):
        raise FloatingPointError("nonfinite CP3 action feature")
    baseline = np.concatenate(
        (
            raw20[np.asarray(LAYERS)],
            autonomous[np.asarray(LAYERS), :, None],
            qj[np.asarray(LAYERS), :, None],
        ),
        axis=2,
    )
    folds = coordinate_folds()
    temporary = SOURCE.with_name(f".{SOURCE.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        baseline=baseline,
        cp3_action=cp3_action,
        autonomous_qj=qj[np.asarray(LAYERS)],
        coordinate_fold=folds,
        layers=np.asarray(LAYERS, dtype=np.int8),
    )
    temporary.replace(SOURCE)
    receipt = {
        "schema": "q24-cp3-action-gap-source-v1",
        "status": "COMPLETE_TARGET_BLIND_SOURCE",
        "source_feature": "diag(cp_pre_k21)/sigma^3; final 3*(p*p*q)@lambda/sigma^3",
        "baseline_features": "raw20 + autonomous A + autonomous qJ",
        "cp4_available": False,
        "fold_counts": np.bincount(folds, minlength=5).tolist(),
        "diagnostics": {
            "cp3_action_rms_by_layer": np.sqrt(np.mean(cp3_action**2, axis=1)).tolist(),
            "cp3_action_max_abs_by_layer": np.max(np.abs(cp3_action), axis=1).tolist(),
        },
        "access": {
            "reference_members_opened": ["native_raw", "native_j_correction"],
            "q_truth_opened": False,
            "observer_selection_opened": False,
            "sampling": False,
        },
        "bindings": {
            "preregister_sha256": sha256(PREREGISTER),
            "program_sha256": sha256(Path(__file__)),
            "quarantine_manifest_sha256": sha256(QUARANTINE / "QUARANTINE_MANIFEST.json"),
            "failed_byte_parity_sha256": sha256(QUARANTINE / "FAILED_BYTE_PARITY.json"),
            "reference_sha256": sha256(REFERENCE),
            "weights_sha256": sha256(WEIGHTS),
            "source_sha256": sha256(SOURCE),
        },
    }
    write_json_once(RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
