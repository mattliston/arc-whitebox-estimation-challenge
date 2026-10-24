#!/Users/matt/anaconda3/bin/python3
"""Frozen one-shot exact-Gaussian-secant transport discriminator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PARENT = P2 / "public_exposure_trained_residual_state_v1"
PARENT_PROGRAM = PARENT / "run_experiment.py"
PARENT_MODEL = PARENT / "MODEL.pt"
PARENT_RESULT = PARENT / "RESULT.json"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RESULT = HERE / "RESULT.json"

PROGRAM_SHA256 = "5c16a68bb907b0e3c945bce883a877c99743fb40fffca3d8039054fd2eb566e1"
MODEL_SHA256 = "c998c4410bf36d3b4f1152c45d86fd4c654c7e05abe85e297f07b15f16d58cc7"
PARENT_RESULT_SHA256 = "fe2f2f3c8cfa890d416b520f443fbbe475ea8fabafbd16956006a9154b100440"
PARENT_FINAL_MSE = 3.4709697955756664e-8
PRIMARY_GATE = 3.297421305796883e-8
STRETCH_GATE = 1.0e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parent_module():
    spec = importlib.util.spec_from_file_location("frozen_exposure_parent", PARENT_PROGRAM)
    if spec is None or spec.loader is None:
        raise ImportError(PARENT_PROGRAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gaussian_mean(alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    cdf = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * alpha.square()) / math.sqrt(2.0 * math.pi)
    return sigma * (pdf + alpha * cdf)


def secant_forward(
    model: torch.nn.Module,
    raw: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = raw.shape[0]
    width = raw.shape[2]
    depth = raw.shape[1]
    state = torch.zeros(batch, width, dtype=raw.dtype, device=raw.device)
    outputs: list[torch.Tensor] = []
    differences: list[torch.Tensor] = []
    gain = model.transport_gain()
    for layer in range(depth):
        if layer < 2:
            state = torch.zeros_like(state)
            outputs.append(state)
            differences.append(state)
            continue
        delta_mu = torch.bmm(state[:, None, :], weights[:, layer]).squeeze(1)
        tangent = raw[:, layer, :, 5] * delta_mu
        alpha = raw[:, layer, :, 3]
        sigma_layer = sigma[:, layer]
        exact = gaussian_mean(alpha + delta_mu / sigma_layer, sigma_layer)
        base = sigma_layer * (
            raw[:, layer, :, 6] + alpha * raw[:, layer, :, 5]
        )
        secant = exact - base
        local = model.innovation(raw[:, layer], sigma_layer, secant, layer)
        state = gain[layer] * secant + local
        outputs.append(state)
        differences.append(secant - tangent)
    return torch.stack(outputs, dim=1), torch.stack(differences, dim=1)


def score(parent, model, panel: dict[str, object], ranks: np.ndarray) -> dict[str, object]:
    raw = panel["raw"]
    sigma = panel["sigma"]
    error = panel["error"]
    paths = panel["weight_paths"]
    assert isinstance(raw, np.ndarray)
    assert isinstance(sigma, np.ndarray)
    assert isinstance(error, np.ndarray)
    assert isinstance(paths, list)
    tangent_sse = np.zeros(parent.DEPTH, dtype=np.float64)
    secant_sse = np.zeros(parent.DEPTH, dtype=np.float64)
    delta_sse = np.zeros(parent.DEPTH, dtype=np.float64)
    correction_delta_sse = np.zeros(parent.DEPTH, dtype=np.float64)
    finite = True
    per_network_tangent: list[float] = []
    per_network_secant: list[float] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranks), 2):
            group = ranks[offset : offset + 2]
            batch_raw = parent.tensor_rows(raw, group)
            batch_sigma = parent.tensor_rows(sigma, group)
            truth = parent.tensor_rows(error, group)
            weights = parent.load_weights(paths, group)
            tangent = model(batch_raw, batch_sigma, weights, exposure=1.0)
            secant, transport_delta = secant_forward(
                model, batch_raw, batch_sigma, weights
            )
            finite = finite and bool(torch.isfinite(secant).all())
            tangent_squared = (tangent - truth).double().square()
            secant_squared = (secant - truth).double().square()
            tangent_sse += tangent_squared.sum(dim=(0, 2)).cpu().numpy()
            secant_sse += secant_squared.sum(dim=(0, 2)).cpu().numpy()
            delta_sse += transport_delta.double().square().sum(dim=(0, 2)).cpu().numpy()
            correction_delta_sse += (
                (secant - tangent).double().square().sum(dim=(0, 2)).cpu().numpy()
            )
            per_network_tangent.extend(
                tangent_squared[:, -1].mean(dim=1).cpu().numpy().tolist()
            )
            per_network_secant.extend(
                secant_squared[:, -1].mean(dim=1).cpu().numpy().tolist()
            )
    denominator = len(ranks) * parent.WIDTH
    tangent_mse = tangent_sse / denominator
    secant_mse = secant_sse / denominator
    transport_delta_mse = delta_sse / denominator
    correction_delta_mse = correction_delta_sse / denominator
    return {
        "finite_full_secant_rollout": finite,
        "tangent_layer_mse": tangent_mse.tolist(),
        "secant_layer_mse": secant_mse.tolist(),
        "transport_secant_minus_tangent_layer_rms": np.sqrt(transport_delta_mse).tolist(),
        "correction_secant_minus_tangent_layer_rms": np.sqrt(correction_delta_mse).tolist(),
        "tangent_final_mse": float(tangent_mse[-1]),
        "secant_final_mse": float(secant_mse[-1]),
        "secant_ratio_to_tangent": float(secant_mse[-1] / tangent_mse[-1]),
        "per_network_tangent_final_mse": per_network_tangent,
        "per_network_secant_final_mse": per_network_secant,
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError("single-run discriminator already has RESULT.json")
    for path, expected in (
        (PARENT_PROGRAM, PROGRAM_SHA256),
        (PARENT_MODEL, MODEL_SHA256),
        (PARENT_RESULT, PARENT_RESULT_SHA256),
    ):
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"frozen parent changed: {path}: {actual}")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen-before-metrics":
        raise AssertionError("preregistration is not frozen")

    parent = load_parent_module()
    parent.configure()
    started = time.perf_counter()
    panel = parent.load_panel()
    raw = panel["raw"]
    error = panel["error"]
    assert isinstance(raw, np.ndarray)
    assert isinstance(error, np.ndarray)
    centers, scales, layer_rms = parent.fit_moments(raw, error)
    model = parent.LocalInnovationTransport(centers, scales, layer_rms)
    payload = torch.load(PARENT_MODEL, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)

    validation = np.arange(parent.FIT_STOP, parent.VALID_STOP)
    metrics = score(parent, model, panel, validation)
    reproduced = float(metrics["tangent_final_mse"])
    if not math.isclose(reproduced, PARENT_FINAL_MSE, rel_tol=0.0, abs_tol=2e-15):
        raise AssertionError(
            f"parent tangent did not reproduce: {reproduced} != {PARENT_FINAL_MSE}"
        )
    secant = float(metrics["secant_final_mse"])
    result = {
        "schema": "public-gaussian-secant-transport-discriminator-result-v1",
        "status": "complete",
        "scope": {
            "role": "observer_fit",
            "development_evaluation": [parent.FIT_STOP, parent.VALID_STOP],
            "observer_selection_targets_opened": 0,
            "observer_fit_targets_at_or_above_rank_200_opened": 0,
            "threads": 2,
            "training_or_parameter_fit": False,
            "validation_retuning": False,
        },
        "intervention": {
            "control": "frozen-parent Phi*(e@W) tangent transport",
            "candidate": "exact Gaussian ReLU secant F(mu+e@W,sigma)-F(mu,sigma)",
            "all_parent_parameters_unchanged": True,
        },
        "metrics": metrics,
        "gates": {
            "finite": bool(metrics["finite_full_secant_rollout"]),
            "primary_5pct_improvement": secant < PRIMARY_GATE,
            "secondary_better_than_parent": secant < PARENT_FINAL_MSE,
            "stretch_below_1e_8": secant < STRETCH_GATE,
        },
        "causal_stop": (
            "PROMOTE_EXACT_SECANT_TO_FULL688"
            if secant < PRIMARY_GATE
            else "STOP_GAUSSIAN_MEAN_CURVATURE_MECHANISM"
        ),
        "provenance": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "program_sha256": sha256(Path(__file__)),
            "parent_program_sha256": PROGRAM_SHA256,
            "parent_model_sha256": MODEL_SHA256,
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "gates": result["gates"], "causal_stop": result["causal_stop"]}, sort_keys=True))


if __name__ == "__main__":
    main()
