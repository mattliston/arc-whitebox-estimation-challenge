#!/Users/matt/anaconda3/bin/python3
"""Assemble the target-blind rank-0 Kerdock/CV feature interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
ROW = HERE / "runs/row_0000"
NATIVE = (
    P2
    / "public_q24_native_transport_energy_final_v1"
    / "runs/native_rows_remaining_v1/row_0000"
)
BASE = ROW / "KERDOCK_TAPE.npz"
SNAP = ROW / "KERDOCK_TAPE_ALL_LAYER_SNAP_HERMITE_CV_D12_DEGREES.npz"
NATIVE_DIAGNOSTICS = ROW / "DIAGNOSTIC_ARRAYS_BASE_HERMITE_CV_D12_DEGREES.npz"
DEGREE_ACTIONS = ROW / "DEGREE_SHRINK_ARRAYS.npz"
OUTPUT = ROW / "TERMINAL_TARGET_BLIND_FEATURES.npz"
MANIFEST = ROW / "TERMINAL_TARGET_BLIND_FEATURES.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    with np.load(BASE, allow_pickle=False) as base, np.load(
        SNAP, allow_pickle=False
    ) as snap:
        shared = (
            "mean_blocks",
            "gate_blocks",
            "cv_d2_mean_blocks",
            "cv_d4_mean_blocks",
            "cv_d6_mean_blocks",
            "cv_d8_mean_blocks",
            "cv_d12_mean_blocks",
            "cv_degree_correction_blocks",
            "pre_snap_empirical_mean_blocks",
            "pre_snap_empirical_var_blocks",
            "pre_snap_scale_blocks",
            "pre_snap_target_mean",
            "pre_snap_target_var",
        )
        for name in shared:
            arrays[f"base_{name}"] = np.asarray(base[name])
            arrays[f"all_snap_{name}"] = np.asarray(snap[name])
    arrays["host_mean"] = np.load(NATIVE / "host.npy", allow_pickle=False)
    arrays["host_raw20"] = np.load(NATIVE / "raw20.npy", allow_pickle=False)
    arrays["host_sigma"] = np.load(NATIVE / "sigma.npy", allow_pickle=False)
    arrays["host_phi"] = np.asarray(arrays["host_raw20"][:, :, 5])
    with np.load(NATIVE_DIAGNOSTICS, allow_pickle=False) as diagnostics:
        # These keys are source-only frozen-J quantities.  Deliberately exclude
        # target, truth_error, q_truth, endpoint_truth, and all scored arrays.
        arrays["native_j_primary"] = np.asarray(diagnostics["native_j_autonomous"])
        arrays["native_j_full_states"] = np.asarray(
            diagnostics["native_j_full_states"]
        )
        arrays["native_j_q_bank"] = np.asarray(diagnostics["native_j_q_bank"])
    with np.load(DEGREE_ACTIONS, allow_pickle=False) as actions:
        for name in (
            "base_endpoint_base_blocks",
            "base_endpoint_degree_actions",
            "base_endpoint_feature_noise_covariance",
            "base_endpoint_feature_cross_covariance",
            "all_layer_snap_endpoint_base_blocks",
            "all_layer_snap_endpoint_degree_actions",
            "all_layer_snap_endpoint_feature_noise_covariance",
            "all_layer_snap_endpoint_feature_cross_covariance",
            "joint_controls",
        ):
            arrays[name] = np.asarray(actions[name])
    np.savez_compressed(OUTPUT, **arrays)
    schema = {
        name: {"dtype": value.dtype.str, "shape": list(value.shape)}
        for name, value in sorted(arrays.items())
    }
    manifest = {
        "schema": "phase1-kerdock-rank0-target-blind-features-v1",
        "status": "complete",
        "role": "observer_fit",
        "role_rank": 0,
        "mlp_id": 383,
        "target_arrays_in_artifact": [],
        "explicit_exclusions": [
            "target",
            "truth_error",
            "native_j_q_truth",
            "native_j_endpoint_truth",
            "truth-derived coefficients",
        ],
        "layer_indexing": "zero-based arrays; late suffix q10:q15 means indices 10:16",
        "degree_axis": "cv_degree_correction_blocks[degree, block, layer, coordinate], degree=0:12",
        "block_axis": "block 0 uses GF32 chirps a=0:7; block 1 uses a=8:15; each has 16384 paths",
        "cv_mean_semantics": (
            "cv_dD_mean_blocks is base sampled post-ReLU mean plus cumulative exact-"
            "host Hermite control corrections through D; only layers 9:15 are finite."
        ),
        "degree_correction_semantics": (
            "Adding sum_d weight[d]*cv_degree_correction_blocks[d] to mean_blocks "
            "forms a degreewise control without another path propagation."
        ),
        "endpoint_action_semantics": (
            "endpoint_degree_actions is the physical q10:q15 propagation of each "
            "degree correction around the frozen native-J primary bank."
        ),
        "source_bindings": {
            "base_tape": {"path": BASE.name, "sha256": sha256(BASE)},
            "all_snap_tape": {"path": SNAP.name, "sha256": sha256(SNAP)},
            "native_diagnostics": {
                "path": NATIVE_DIAGNOSTICS.name,
                "sha256": sha256(NATIVE_DIAGNOSTICS),
            },
            "degree_actions": {
                "path": DEGREE_ACTIONS.name,
                "sha256": sha256(DEGREE_ACTIONS),
            },
        },
        "artifact": {"path": OUTPUT.name, "sha256": sha256(OUTPUT)},
        "arrays": schema,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(OUTPUT), "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
