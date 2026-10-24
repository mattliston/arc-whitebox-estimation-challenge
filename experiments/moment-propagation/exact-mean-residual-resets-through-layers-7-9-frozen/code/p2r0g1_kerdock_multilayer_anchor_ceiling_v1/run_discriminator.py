#!/Users/matt/anaconda3/bin/python3
"""Frozen exact mean-residual resets at h2:h7 and h2:h9."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "public_exposure_trained_residual_state_v1"
SOURCE_PROGRAM = SOURCE / "run_experiment.py"
SOURCE_MODEL = SOURCE / "MODEL.pt"
SOURCE_RESULT = SOURCE / "RESULT.json"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RESULT = HERE / "RESULT.json"
ANCHOR_SETS = {"h2_through_h7": frozenset(range(1, 7)), "h2_through_h9": frozenset(range(1, 9))}
EXPECTED_MODEL_SHA256 = "c998c4410bf36d3b4f1152c45d86fd4c654c7e05abe85e297f07b15f16d58cc7"
EXPECTED_SOURCE_RESULT_SHA256 = "fe2f2f3c8cfa890d416b520f443fbbe475ea8fabafbd16956006a9154b100440"
EXPECTED_SOURCE_PROGRAM_SHA256 = "5c16a68bb907b0e3c945bce883a877c99743fb40fffca3d8039054fd2eb566e1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_module():
    spec = importlib.util.spec_from_file_location("frozen_residual_source", SOURCE_PROGRAM)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_PROGRAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(model, source, panel: dict[str, object], ranks: np.ndarray, anchors: frozenset[int]) -> dict[str, object]:
    raw, sigma, error, paths = (panel[k] for k in ("raw", "sigma", "error", "weight_paths"))
    assert isinstance(raw, np.ndarray) and isinstance(sigma, np.ndarray)
    assert isinstance(error, np.ndarray) and isinstance(paths, list)
    layer_sse = np.zeros(source.DEPTH, dtype=np.float64)
    per_network: list[float] = []
    maximum_absolute = 0.0
    finite = True
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranks), 2):
            group = ranks[offset : offset + 2]
            batch_raw = source.tensor_rows(raw, group)
            batch_sigma = source.tensor_rows(sigma, group)
            truth = source.tensor_rows(error, group)
            weights = source.load_weights(paths, group)
            state = torch.zeros(len(group), source.WIDTH, dtype=batch_raw.dtype)
            outputs: list[torch.Tensor] = []
            gain = model.transport_gain()
            for layer in range(source.DEPTH):
                if layer < source.FIRST_CORRECTED_LAYER:
                    state = torch.zeros_like(state)
                else:
                    message = torch.bmm(state[:, None, :], weights[:, layer]).squeeze(1)
                    message = batch_raw[:, layer, :, 5] * message
                    local = model.innovation(batch_raw[:, layer], batch_sigma[:, layer], message, layer)
                    state = gain[layer] * message + local
                if layer in anchors:
                    state = truth[:, layer].clone()
                outputs.append(state)
            prediction = torch.stack(outputs, dim=1)
            finite = finite and bool(torch.isfinite(prediction).all())
            maximum_absolute = max(maximum_absolute, float(prediction.abs().max()))
            squared = (prediction - truth).double().square()
            layer_sse += squared.sum(dim=(0, 2)).cpu().numpy()
            per_network.extend(squared[:, -1].mean(dim=1).cpu().numpy().tolist())
    layer_mse = layer_sse / (len(ranks) * source.WIDTH)
    return {
        "anchor_layers_one_based": [value + 1 for value in sorted(anchors)],
        "finite_full_rollout": finite,
        "maximum_absolute_correction": maximum_absolute,
        "final_mse": float(layer_mse[-1]),
        "layer_mse": layer_mse.tolist(),
        "per_network_final_mse": per_network,
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError("single-run result already exists")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen-before-metrics":
        raise AssertionError("preregistration is not frozen")
    expected = ((SOURCE_MODEL, EXPECTED_MODEL_SHA256), (SOURCE_RESULT, EXPECTED_SOURCE_RESULT_SHA256), (SOURCE_PROGRAM, EXPECTED_SOURCE_PROGRAM_SHA256))
    for path, digest in expected:
        if sha256(path) != digest:
            raise AssertionError(f"source changed: {path}")

    started = time.perf_counter()
    source = source_module()
    source.configure()
    panel = source.load_panel()
    raw, error = panel["raw"], panel["error"]
    assert isinstance(raw, np.ndarray) and isinstance(error, np.ndarray)
    centers, scales, layer_rms = source.fit_moments(raw, error)
    model = source.LocalInnovationTransport(centers, scales, layer_rms)
    artifact = torch.load(SOURCE_MODEL, map_location="cpu", weights_only=True)
    model.load_state_dict(artifact["state_dict"], strict=True)
    ranks = np.arange(source.FIT_STOP, source.VALID_STOP)
    metrics = {name: score(model, source, panel, ranks, anchors) for name, anchors in ANCHOR_SETS.items()}

    source_receipt = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    autonomous = float(source_receipt["metrics"]["validation_autonomous"]["final_mse"])
    teacher = float(source_receipt["metrics"]["validation_same_checkpoint_teacher_forced"]["final_mse"])
    gap = autonomous - teacher
    for value in metrics.values():
        mse = float(value["final_mse"])
        value["fraction_of_autonomous_to_teacher_mse_gap_closed"] = (autonomous - mse) / gap
        value["ratio_to_autonomous"] = mse / autonomous
    result = {
        "schema": "p2r0g1-kerdock-multilayer-anchor-ceiling-result-v1",
        "status": "complete",
        "split": {"role": "observer_fit", "development_validation": [160, 200], "observer_selection_targets_opened": 0, "observer_fit_targets_at_or_above_rank_200_opened": 0, "deployment_or_board_truth_opened": 0, "network_names_opened": 0},
        "comparators": {"autonomous_final_mse": autonomous, "all_layer_teacher_forced_final_mse": teacher, "autonomous_to_teacher_mse_gap": gap},
        "metrics": metrics,
        "gates": {name + "_reaches_1e_8": float(value["final_mse"]) < 1e-8 for name, value in metrics.items()},
        "interpretation_boundary": "These are exact full-width mean-residual reset ceilings for the old scalar recurrence. They do not test marginal or joint cumulant re-anchoring into the new resummed suffix, and are not submission-eligible.",
        "provenance": {"preregistration_sha256": sha256(PREREGISTRATION), "program_sha256": sha256(Path(__file__)), "source_model_sha256": sha256(SOURCE_MODEL), "source_result_sha256": sha256(SOURCE_RESULT), "source_program_sha256": sha256(SOURCE_PROGRAM), "elapsed_seconds": time.perf_counter() - started},
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(RESULT), "final_mse": {k: v["final_mse"] for k, v in metrics.items()}, "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
