#!/usr/bin/env python3
"""Generate an independent target-free recurrence authority on rank 35.

Run this with the training Python environment (NumPy 1.x plus PyTorch).  It
does not import FlopScope or the deployable estimator and opens no target.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
Q_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
J_ARTIFACT = P2 / "public_q24_native_j_checkpoint_interpolation_v1/INTERPOLATED_MODEL.pt"
Q_ARTIFACT = P2 / "public_q24_native_j_q10q15_residual_head_v1/Q_SUFFIX_HEAD.pt"
TAPE = P2 / "public_q24_native_transport_energy_pilot_v1/runs/native_rows_v1/row_0035"
WEIGHTS = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0035/weights.npy"
)
ARRAYS = HERE / "TORCH_AUTHORITY_RANK35.npz"
MANIFEST = HERE / "TORCH_AUTHORITY_RANK35.json"

EXPECTED = {
    "j_artifact_sha256": "fc65528720c7e1378ed7011b7b201f4086217a544b092de208ea1064656d9ef5",
    "q_artifact_sha256": "5fb57a510d933940f19ddd5ab11650c3a966c2039c6bd12f73988457c2ce91b8",
    "raw20_sha256": "0768502bbd8beb26b85ffdd6b347f0312e1f59e7d4416c849efac8c2dec88365",
    "sigma_sha256": "c22932b89b08a4ad6be821245a275b1126ef8410cebdc7d75335b62359e55ff5",
    "weights_sha256": "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if ARRAYS.exists() or MANIFEST.exists():
        raise FileExistsError("authority outputs already exist")
    bindings = {
        "j_artifact_sha256": sha256(J_ARTIFACT),
        "q_artifact_sha256": sha256(Q_ARTIFACT),
        "raw20_sha256": sha256(TAPE / "raw20.npy"),
        "sigma_sha256": sha256(TAPE / "sigma.npy"),
        "weights_sha256": sha256(WEIGHTS),
    }
    if bindings != EXPECTED:
        raise AssertionError({"expected": EXPECTED, "observed": bindings})
    torch.manual_seed(0xD1A075)
    np.random.seed(0xD1A075)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    raw = np.load(TAPE / "raw20.npy", allow_pickle=False)
    sigma = np.load(TAPE / "sigma.npy", allow_pickle=False)
    weights = np.load(WEIGHTS, allow_pickle=False)
    j_module = load_module("d1_alpha075_authority_j", J_SOURCE)
    q_module = load_module("d1_alpha075_authority_q", Q_SOURCE)
    j_state = torch.load(J_ARTIFACT, map_location="cpu", weights_only=False)["state_dict"]
    q_state = torch.load(Q_ARTIFACT, map_location="cpu", weights_only=False)["state_dict"]
    j_model = j_module.TransportEnergyMultiStateResidual(
        j_state["centers"].numpy(),
        j_state["scales"].numpy(),
        j_state["layer_rms"].numpy(),
    )
    j_model.load_state_dict(j_state, strict=True)
    j_model.eval()
    q_model = q_module.QSuffixHead(j_model.centers, j_model.scales, j_model.layer_rms)
    q_model.load_state_dict(q_state, strict=True)
    q_model.eval()

    raw_t = torch.from_numpy(np.ascontiguousarray(raw))[None]
    sigma_t = torch.from_numpy(np.ascontiguousarray(sigma))[None]
    weights_t = torch.from_numpy(np.ascontiguousarray(weights))[None]
    with torch.no_grad():
        j_value = j_model(raw_t, sigma_t, weights_t, exposure=1.0)[0]
        q_state_value = torch.zeros(1024, dtype=torch.float32)
        q_layers = [torch.zeros(1024, dtype=torch.float32) for _ in range(10)]
        for layer in range(10, 16):
            message = raw_t[0, layer, :, 5] * (q_state_value @ weights_t[0, layer])
            j_message = raw_t[0, layer, :, 5] * (
                j_value[layer - 1] @ weights_t[0, layer]
            )
            base_q = j_value[layer] - j_message
            source = q_model.source(
                raw_t[0, layer], sigma_t[0, layer], j_value[layer], base_q, layer
            )
            q_state_value = message + source
            q_layers.append(q_state_value)
        q_value = torch.stack(q_layers, dim=0)
    j_array = np.ascontiguousarray(j_value.numpy(), dtype=np.float32)
    q_array = np.ascontiguousarray(q_value.numpy(), dtype=np.float32)
    if j_array.shape != (16, 1024) or q_array.shape != (16, 1024):
        raise AssertionError("authority shape changed")
    if not np.isfinite(j_array).all() or not np.isfinite(q_array).all():
        raise FloatingPointError("authority recurrence is nonfinite")
    np.savez(ARRAYS, j_correction=j_array, q_correction=q_array)
    result = {
        "schema": "q24-d1-alpha075-qhead-torch-authority-rank35-v1",
        "status": "PASS_TARGET_FREE_AUTHORITY_GENERATION",
        "role": "observer_fit",
        "role_rank": 35,
        "target_columns_opened": [],
        "observer_selection_opened": False,
        "board_resource_opened": False,
        "bindings": bindings,
        "arrays_file_sha256": sha256(ARRAYS),
        "j_array_sha256": array_sha256(j_array),
        "q_array_sha256": array_sha256(q_array),
    }
    MANIFEST.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
