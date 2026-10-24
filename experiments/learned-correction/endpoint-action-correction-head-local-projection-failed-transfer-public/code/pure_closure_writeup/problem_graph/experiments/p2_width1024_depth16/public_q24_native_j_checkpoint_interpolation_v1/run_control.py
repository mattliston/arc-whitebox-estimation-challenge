#!/usr/bin/env python3
"""Zero-incremental-cost epoch6/continued-epoch9 interpolation control."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PILOT = P2 / "public_q24_native_transport_energy_pilot_v1/PILOT_MODEL.pt"
CONT = P2 / "public_q24_native_j_autonomous_continuation_v1/CONTINUED_MODEL.pt"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
Q_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
Q_MODEL = P2 / "public_q24_native_j_q10q15_residual_head_v1/Q_SUFFIX_HEAD.pt"
Q_RESULT = P2 / "public_q24_native_j_q10q15_residual_head_v1/RESULT.json"
OUT = HERE / "RESULT.json"
MODEL = HERE / "INTERPOLATED_MODEL.pt"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def artifact_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def model_for(j, left, right, alpha: float):
    state = {}
    for key in left:
        if key not in right or left[key].shape != right[key].shape:
            raise AssertionError(key)
        state[key] = left[key] + float(alpha) * (right[key] - left[key])
    model = j.TransportEnergyMultiStateResidual(
        state["centers"].numpy(), state["scales"].numpy(), state["layer_rms"].numpy()
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    if OUT.exists() or MODEL.exists():
        raise FileExistsError("interpolation output already exists")
    torch.manual_seed(0x49504F4C)
    np.random.seed(0x49504F4C)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    j = load_module("native_j_interp_exact_j", J_SOURCE)
    q = load_module("native_j_interp_q_head", Q_SOURCE)
    dataset = q.Dataset()
    target, host = q.targets_and_host()
    error = np.ascontiguousarray(target - host, dtype=np.float32)
    _, inner, protected = q.split()
    left = artifact_state(PILOT)
    right = artifact_state(CONT)
    inner_records = []
    selected = None
    selected_model = None
    for alpha in ALPHAS:
        model = model_for(j, left, right, alpha)
        score = j.score(model, dataset, inner, error)
        row = {"alpha": alpha, "inner_autonomous_final_mse": float(score["final_mse"])}
        inner_records.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        candidate = (float(score["final_mse"]), alpha)
        if selected is None or candidate < selected:
            selected = candidate
            selected_model = model
    assert selected is not None and selected_model is not None
    selected_mse, selected_alpha = selected
    # Alpha is now frozen; both protected diagnostics are one-shot.
    protected_base = j.score(selected_model, dataset, protected, error)
    q_artifact = torch.load(Q_MODEL, map_location="cpu", weights_only=False)
    q_head = q.QSuffixHead(
        selected_model.centers, selected_model.scales, selected_model.layer_rms
    )
    q_head.load_state_dict(q_artifact["state_dict"], strict=True)
    prepared = q.prepare(j, selected_model, dataset, protected, error)
    protected_with_frozen_q = q.autonomous_metrics(q_head, dataset, prepared)

    artifact = {
        "schema": "q24-native-j-interpolated-checkpoint-v1",
        "state_dict": selected_model.state_dict(),
        "alpha": selected_alpha,
        "left_sha256": sha256(PILOT),
        "right_sha256": sha256(CONT),
    }
    temporary = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary)
    temporary.replace(MODEL)
    result = {
        "schema": "q24-native-j-checkpoint-interpolation-control-v1",
        "status": "complete",
        "observer_fit_only": True,
        "observer_selection_targets_opened": 0,
        "board_submissions": 0,
        "alpha_law": "theta(alpha)=epoch6+alpha*(continued_epoch9-epoch6)",
        "alpha_grid": list(ALPHAS),
        "inner32": inner_records,
        "selection": {
            "selected_alpha": selected_alpha,
            "selected_inner_autonomous_final_mse": selected_mse,
            "metric": "native-J-only autonomous final MSE",
        },
        "protected_once": {
            "selected_interpolation_native_j_final_mse": float(protected_base["final_mse"]),
            "selected_interpolation_plus_frozen_q_head": protected_with_frozen_q,
            "alpha1_continued_j_final_mse": 3.095364057782614e-8,
            "alpha1_continued_j_plus_q_head_final_mse": json.loads(Q_RESULT.read_text())["protected_once"]["candidate_final_mse"],
        },
        "deployability": {
            "native_j_incremental_flops": 0,
            "state_schema_unchanged": True,
            "q_head_retrained": False,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "left_checkpoint_sha256": sha256(PILOT),
            "right_checkpoint_sha256": sha256(CONT),
            "q_head_sha256": sha256(Q_MODEL),
            "interpolated_checkpoint_sha256": sha256(MODEL),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["protected_once"], sort_keys=True))


if __name__ == "__main__":
    main()
